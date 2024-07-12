import torch.nn as nn
import torch     
from transformers import HubertConfig, HubertModel
 
class Hubert(HubertModel):
    def __init__(self, config):
        super().__init__(config,)
        self.s_layer = 8 #layer of student
        self.hubert = HubertModel(config)
        self.post_init()
        self.classifier_t = nn.Linear(config.hidden_size, 29)
        self.classifier_s = nn.Linear(config.hidden_size, 29)

    def freeze(self):
        for param in self.hubert.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        for param in self.hubert.parameters():
            param.requires_grad = True

    def forward(self, audio_input):
        out = self.hubert(audio_input, 
                            attention_mask=None, 
                            output_hidden_states=True,).hidden_states
        
        i_logits = self.classifier_s(out[self.s_layer])
        logits = self.classifier_t(out[-1])

        return i_logits, logits
 