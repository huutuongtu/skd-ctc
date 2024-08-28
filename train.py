from jiwer import wer
from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor, HubertConfig, HubertModel
import torch, json, os, librosa, transformers, gc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pyctcdecode import build_ctcdecoder
import pandas as pd
from tqdm import tqdm
import warnings
from torch.utils.data import Dataset
import numpy as np
from dataloader import ASR_Dataset
from dataloader import text_to_tensor
from models import Hubert
from torch.optim.lr_scheduler import CosineAnnealingLR


feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, padding_side='right', do_normalize=True, return_attention_mask=False)
min_wer = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def scheduling_func(e, E=200, t=0.3):
    return min(max((e-1)/(E-1), t), 1-t)

def collate_fn(batch):
    with torch.no_grad():
        sr = 16000
        max_col = [-1] * 2
        target_length = []
        for row in batch:
          if row[0].shape[0] > max_col[0]:
              max_col[0] = row[0].shape[0]
          if len(row[1]) > max_col[1]:
              max_col[1] = len(row[1])
        cols = {'waveform':[], 'transcript':[], 'outputlengths':[]}
        
        for row in batch:
            pad_wav = np.concatenate([row[0], np.zeros(max_col[0] - row[0].shape[0])])
            cols['waveform'].append(pad_wav)
            cols['outputlengths'].append(len(row[1]))
            row[1].extend([0] * (max_col[1] - len(row[1])))
            cols['transcript'].append(row[1])
        
        inputs = feature_extractor(cols['waveform'], sampling_rate = 16000)
        input_values = torch.tensor(inputs.input_values, device=device)
        cols['transcript'] = torch.tensor(cols['transcript'], dtype=torch.long, device=device)
        cols['outputlengths'] = torch.tensor(cols['outputlengths'], dtype=torch.long, device=device)
    
    return input_values, cols['transcript'], cols['outputlengths']

#dataset should contain 2 cols, 1 is Path contain absolute path of audio and 1 is Transcript contain text transcript of audio 
df_train = pd.read_csv('./dataset/train.csv')
df_dev = pd.read_csv("./dataset/dev.csv")
train_dataset = ASR_Dataset(df_train)
batch_size = 2
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

model = Hubert.from_pretrained(
    'facebook/hubert-base-ls960',
)

model = model.to(device)

list_vocab = ['', ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', "'"]

decoder_ctc = build_ctcdecoder(
                              labels = list_vocab,
                              )

num_epoch=200
temperature = 1
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
warmup_steps = num_epoch//10
scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch - warmup_steps)
ctc_loss = nn.CTCLoss(blank = 0)


for epoch in range(num_epoch):
  #freeze model first 12.5% of the steps except linear
  if epoch < num_epoch//8:
    model.freeze()
  else:
    model.unfreeze()
  model.train().to(device)
  running_loss = []
  print(f'EPOCH {epoch}:')
  for i, data in tqdm(enumerate(train_loader)):
    acoustic, labels, target_lengths = data
    i_logits, logits= model(acoustic)

    #skd
    teacher_logits_detached = logits.clone().detach() #Stop gradients for the teacher's logits
    l_skd = F.kl_div(F.log_softmax(i_logits/temperature, dim=2), F.softmax(teacher_logits_detached/temperature, dim=2))

    logits = logits.transpose(0,1)
    i_logits = i_logits.transpose(0,1)
    input_lengths = torch.full(size=(logits.shape[1],), fill_value=logits.shape[0], dtype=torch.long, device=device)
    logits = F.log_softmax(logits, dim=2)
    i_logits = F.log_softmax(i_logits, dim=2)

    #ctc and ictc
    l_ctc = ctc_loss(logits, labels, input_lengths, target_lengths)
    l_ictc = ctc_loss(i_logits, labels, input_lengths, target_lengths)

    #alpha and total loss
    alpha = scheduling_func(e=epoch+1, E=num_epoch, t=0.3)
    loss = (1-alpha)*l_ctc + alpha*(l_ictc + l_skd)

    running_loss.append(l_ictc.item())
    loss.backward()
    optimizer.step()

    # linear warmup lr
    if epoch < warmup_steps:
      lr = 3e-5 * (epoch + 1) / warmup_steps
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr  
    else:
      scheduler.step()
    

    optimizer.zero_grad()
    # break

  print(f"Training loss: {sum(running_loss) / len(running_loss)}")
  if sum(running_loss) / len(running_loss)<=1: #ensure decode fast
    with torch.no_grad():
      model.eval().to(device)
      worderrorrate = []
      for point in tqdm(range(len(df_dev))):
        acoustic, _ = librosa.load(df_dev['Path'][point], sr=16000)
        acoustic = feature_extractor(acoustic, sampling_rate = 16000)
        acoustic = torch.tensor(acoustic.input_values, device=device)
        transcript = df_dev['Transcript'][point]
        logits, _ = model(acoustic)
        logits = F.log_softmax(logits.squeeze(0), dim=1)
        x = logits.detach().cpu().numpy()
        hypothesis = decoder_ctc.decode(x).strip()
        # print(hypothesis)
        error = wer(transcript, hypothesis)
        worderrorrate.append(error)
      epoch_wer = sum(worderrorrate)/len(worderrorrate)

      if (epoch_wer < min_wer):
        print("save_checkpoint...")
        min_wer = epoch_wer
        torch.save(model.state_dict(), 'checkpoint/checkpoint.pth')

      print("wer checkpoint " + str(epoch) + ": " + str(epoch_wer))
      print("min_wer: " + str(min_wer))
      
