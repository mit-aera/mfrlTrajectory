#!/usr/bin/env python
# coding: utf-8

# https://github.com/hyunwoongko/transformer/tree/master

import numpy as np
import torch
from torch import nn

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        hidden_t = hidden.unsqueeze(1).repeat(1, src_len, 1)
        # encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        # print(hidden.size())
        # print(encoder_outputs.size())
        energy = torch.tanh(self.attn(torch.cat((hidden_t, encoder_outputs.permute(1, 0, 2)), dim = 2))) 
        attention = self.v(energy)
        attention = attention.reshape(hidden.shape[0],-1)        

        return nn.functional.softmax(attention, dim=1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :, :].repeat(1,x.size(1),1)
        return self.dropout(x)


class MLPLayer(nn.Module):
    def __init__(self, hid_dim, drop_prob=0.1):
        super().__init__()
        self.linear = nn.Linear(hid_dim, hid_dim)
        self.relu = nn.ReLU(inplace = True)
        self.dropout = nn.Dropout(p=drop_prob)
        # self.linear2 = nn.Linear(hid_dim, hid_dim)
        # self.norm = nn.LayerNorm(hid_dim)
    
    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        # x = self.norm(self.linear2(x))
        return x


class MLPLayerNorm(nn.Module):
    def __init__(self, hid_dim, drop_prob=0.1):
        super().__init__()
        self.linear = nn.Linear(hid_dim, hid_dim)
        self.relu = nn.ReLU(inplace = True)
        self.dropout = nn.Dropout(p=drop_prob)
        self.linear2 = nn.Linear(hid_dim, hid_dim)
        self.norm = nn.LayerNorm(hid_dim)
    
    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.norm(self.linear2(x))
        return x
