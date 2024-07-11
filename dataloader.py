import torch
from torch.utils.data import Dataset
import numpy as np
import json
import librosa

with open('./vocab.json') as f:
    dict_vocab = json.load(f)
key_list = list(dict_vocab.keys())
val_list = list(dict_vocab.values())

def text_to_tensor(string_text):
    text = string_text
    text_list = []
    for idex in text:
        text_list.append(dict_vocab[idex])
    return text_list

class ASR_Dataset(Dataset):

    def __init__(self, data):
        self.len_data           = len(data)
        self.path               = list(data['Path'])
        self.transcript         = list(data['Transcript'])

    def __getitem__(self, index):
        waveform, _ = librosa.load(self.path[index], sr=16000)
        transcript  = text_to_tensor(self.transcript[index])
        return waveform, transcript

    def __len__(self):
        return self.len_data
    
