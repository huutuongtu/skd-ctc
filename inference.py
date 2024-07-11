from jiwer import wer
from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor, HubertConfig, HubertModel
import torch, json, os, librosa, transformers, gc
import torch.nn as nn
import torch.nn.functional as F
from pyctcdecode import build_ctcdecoder
import pandas as pd
from tqdm import tqdm
import warnings
import numpy as np
from models import Hubert

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, padding_side='right', do_normalize=True, return_attention_mask=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gc.collect()
df_dev = pd.read_csv("./dataset/test.csv")

save_path = "./result/result.csv"
model = Hubert.from_pretrained(
    'facebook/hubert-base-ls960',
)
model.load_state_dict(torch.load("checkpoint/checkpoint.pth"))

model.freeze_feature_extractor()
model = model.to(device)

PATH = []
TRANSCRIPT = []
PREDICT = []

list_vocab = ['', ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', "'"]
decoder_ctc = build_ctcdecoder(
                              labels = list_vocab,
                              )

time_start = time.time()
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
    PREDICT.append(hypothesis.strip())
    PATH.append(df_dev['Path'][point])
    TRANSCRIPT.append(df_dev['Transcript'][point])
time_end = time.time()

print(time_end-time_start)

train = pd.DataFrame([PATH, TRANSCRIPT, PREDICT])
train = train.transpose()
train.columns=['Path', 'Transcript', 'Predict'] 
train.to_csv(save_path)
