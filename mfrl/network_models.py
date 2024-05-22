#!/usr/bin/env python
# coding: utf-8

import os, sys, io, random, time, yaml, copy
import numpy as np

import torch
import torch.nn as nn
from .transformer_blocks import MLPLayer, MLPLayerNorm, Attention
from torch.distributions import MultivariateNormal, Normal

class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.device = kwargs["device"]
        self.emb_dim = kwargs["emb_dim"]
        self.hid_dim = kwargs["hid_dim"]
        
        self.rnn = nn.GRU(kwargs["emb_hid_dim"], kwargs["hid_dim"], bidirectional = True, dropout=kwargs["dropout"])
        
        emb_module_list = [
            nn.Sequential(
                nn.Linear(kwargs["emb_dim"]-2, kwargs["hid_dim"]),
                nn.ReLU(inplace=True),
                nn.Dropout(p=kwargs["dropout"]))]
        emb_module_list.extend([MLPLayer(kwargs["hid_dim"], kwargs["dropout"]) for _ in range(kwargs["emb_n_layers"])])
        emb_module_list.append(nn.Linear(kwargs["hid_dim"], kwargs["emb_hid_dim"]))
        emb_module_list.append(nn.LayerNorm(kwargs["emb_hid_dim"]))
        self.fc_emb = nn.ModuleList(emb_module_list)
        
        bern_module_list = [
            nn.Sequential(
                nn.Sigmoid(),
                nn.Linear(14, kwargs["hid_dim"]),
                nn.ReLU(inplace=True),
                nn.Dropout(p=kwargs["dropout"])
            )]
        bern_module_list.extend([MLPLayer(kwargs["hid_dim"], kwargs["dropout"]) for _ in range(kwargs["enc_bern_n_layers"])])
        bern_module_list.append(nn.Linear(kwargs["hid_dim"], 2*kwargs["hid_dim"]))
        bern_module_list.append(nn.LayerNorm(2*kwargs["hid_dim"]))
        self.bernstein2hidden = nn.ModuleList(bern_module_list)
        
        self.fc = nn.Linear(kwargs["hid_dim"] * 2, kwargs["hid_dim"])
        
    def forward(self, src, initial=None):
        embedded = src
        for layer in self.fc_emb:
            embedded = layer(embedded)
        
        initial_hidden = initial
        for layer in self.bernstein2hidden:
            initial_hidden = layer(initial_hidden)
        
        initial_hidden_reshaped = initial_hidden.reshape(-1, 2, self.hid_dim)
        initial_hidden_reshaped = initial_hidden_reshaped.permute(1, 0, 2)
        initial_hidden_reshaped = initial_hidden_reshaped.contiguous()
        outputs, hidden = self.rnn(embedded, initial_hidden_reshaped)
        hidden = torch.relu(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
                
        return outputs, hidden


class RewEncoder(Encoder):
    def __init__(self, **kwargs):
        super(RewEncoder, self).__init__(**kwargs)
        
        emb_module_list = [
            nn.Sequential(
                nn.Linear(kwargs["emb_dim"], kwargs["hid_dim"]),
                nn.ReLU(inplace=True),
                nn.Dropout(p=kwargs["dropout"]))]
        emb_module_list.extend([MLPLayer(kwargs["hid_dim"], kwargs["dropout"]) for _ in range(kwargs["emb_n_layers"])])
        emb_module_list.append(nn.Linear(kwargs["hid_dim"], kwargs["emb_hid_dim"]))
        emb_module_list.append(nn.LayerNorm(kwargs["emb_hid_dim"]))
        self.fc_emb = nn.ModuleList(emb_module_list)
        
        fv_module_list = []
        fv_module_list.extend([MLPLayer(kwargs["hid_dim"], kwargs["dropout"]) for _ in range(kwargs["enc_fv_n_layers"])])
        fv_module_list.append(nn.Linear(kwargs["hid_dim"], kwargs["latent_size"]))
        self.hidden2mean = nn.ModuleList(fv_module_list)
    
    def forward(self, src, seq_len, initial=None):
        batch_size = src.shape[1]
        embed = src[:,:,:8].clone()
        embed[0,:,6] = 0
        embed[:,:,6:8] = torch.einsum('ijk, j -> ijk', embed[:,:,6:8], seq_len-1)
        
        embedded = embed
        for layer in self.fc_emb:
            embedded = layer(embedded)
        
        initial_hidden = initial
        for layer in self.bernstein2hidden:
            initial_hidden = layer(initial_hidden)
        
        initial_hidden_reshaped = initial_hidden.reshape(-1, 2, self.hid_dim)
        initial_hidden_reshaped = initial_hidden_reshaped.permute(1, 0, 2)
        initial_hidden_reshaped = initial_hidden_reshaped.contiguous()
        encoder_outputs, hidden = self.rnn(embedded, initial_hidden_reshaped)
        hidden = torch.relu(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        
        mean = hidden
        for layer in self.hidden2mean:
            mean = layer(mean)

        return mean, None

class Decoder(nn.Module):
    def __init__(self, attention, **kwargs):
        super().__init__()
        self.device = kwargs["device"]
        self.emb_dim = kwargs["emb_dim"]
        self.num_fidelity = kwargs["num_fidelity"]
        self.max_seq_len = kwargs["max_seq_len"]
        self.attention = attention
        
        emb_module_list = [
            nn.Sequential(
                nn.Linear(kwargs["emb_dim"], kwargs["hid_dim"]),
                nn.ReLU(inplace=True),
                nn.Dropout(p=kwargs["dropout"]))]
        emb_module_list.extend([MLPLayer(kwargs["hid_dim"], kwargs["dropout"]) for _ in range(kwargs["emb_n_layers"])])
        emb_module_list.append(nn.Linear(kwargs["hid_dim"], kwargs["emb_hid_dim"]))
        emb_module_list.append(nn.LayerNorm(kwargs["emb_hid_dim"]))
        self.fc_emb = nn.ModuleList(emb_module_list)
        
        self.rnn = nn.GRU((kwargs["hid_dim"] * 2) + kwargs["emb_hid_dim"], kwargs["hid_dim"])
        self.fc_out_hid = nn.Sequential(    
            nn.Linear((kwargs["hid_dim"] * 2) + kwargs["hid_dim"] + kwargs["emb_hid_dim"], kwargs["hid_dim"]),
            nn.ReLU(inplace = True),
        )

        wp_module_list = [
            nn.Sequential(
                nn.Linear(kwargs["hid_dim"], kwargs["hid_dim"]),
                nn.ReLU(inplace=True),
                nn.Dropout(p=kwargs["dropout"]))]
        wp_module_list.extend([MLPLayer(kwargs["hid_dim"], kwargs["dropout"]) for _ in range(kwargs["dec_wp_n_layers"])])
        wp_module_list.append(nn.Linear(kwargs["hid_dim"], kwargs["emb_dim"]-2))
        self.fc_out_wp = nn.ModuleList(wp_module_list)
        
        self.fc_out_time = []
        self.fc_out_snapw = []
        
        for f_ii in range(self.num_fidelity):
            t_module_list = [
            nn.Sequential(
                nn.Linear(kwargs["hid_dim"], kwargs["hid_dim"]),
                nn.ReLU(inplace=True),
                nn.Dropout(p=kwargs["dropout"]))]
            t_module_list.extend([MLPLayer(kwargs["hid_dim"], kwargs["dropout"]) for _ in range(kwargs["dec_time_n_layers"])])
            t_module_list.append(nn.Linear(kwargs["hid_dim"], 1))
            fc_out_time_t = nn.ModuleList(t_module_list)
            setattr(self, "fc_out_time_{}".format(f_ii+1), fc_out_time_t)
            self.fc_out_time.append(fc_out_time_t)
            
            s_module_list = [
            nn.Sequential(
                nn.Linear(kwargs["hid_dim"], kwargs["hid_dim"]),
                nn.ReLU(inplace=True),
                nn.Dropout(p=kwargs["dropout"]))]
            s_module_list.extend([MLPLayer(kwargs["hid_dim"], kwargs["dropout"]) for _ in range(kwargs["dec_snapw_n_layers"])])
            s_module_list.append(nn.Linear(kwargs["hid_dim"], 1))
            fc_out_snapw_t = nn.ModuleList(s_module_list)
            setattr(self, "fc_out_snapw_{}".format(f_ii+1), fc_out_snapw_t)
            self.fc_out_snapw.append(fc_out_snapw_t)
    
    def forward(self, trg, hidden, enc_src, seq_len, fidelity=2):
        batch_size = seq_len.size(0)
        
        outputs = torch.zeros(self.max_seq_len, batch_size, self.emb_dim, dtype=torch.float).to(self.device)

        outputs[0,:,:6] = trg[:,:6]
        outputs[0,:,6:] = torch.tensor(float('-inf'))
        for t in range(self.max_seq_len-1):
            hidden = hidden.squeeze()
            if hidden.ndim == 1:
                hidden = hidden.unsqueeze(0)
            
            embedded = trg
            for layer in self.fc_emb:
                embedded = layer(embedded)
            embedded = embedded.unsqueeze(0)
            
            a = self.attention(hidden, enc_src)
            a = a.unsqueeze(1)

            weighted = torch.bmm(a, enc_src.permute(1, 0, 2))
            weighted = weighted.permute(1, 0, 2)

            rnn_input = torch.cat((embedded, weighted), dim = 2)
            # rnn_input = embedded
            
            output_t, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
            embedded_t = embedded.reshape(batch_size, -1)
            output_t = output_t.reshape(batch_size, -1)
            weighted = weighted.reshape(batch_size, -1)
            
            weighted = weighted.squeeze(0)
            # output_hid = self.fc_out_hid(torch.cat((output_t, weighted, input_t), dim = 1))
            # output_points = self.fc_out_points(output_hid)
            # output_time = self.fc_out_time[fidelity-1](output_hid)
            # output_snapw = self.fc_out_snapw[fidelity-1](output_hid)
            # output = torch.cat((output_points, output_time, output_snapw), dim=1)
            
            o_hid = self.fc_out_hid(torch.cat((output_t, weighted, embedded_t), dim = 1))

            o_wp = o_hid.clone()
            for layer in self.fc_out_wp:
                o_wp = layer(o_wp)
            o_time = o_hid.clone()
            for layer in self.fc_out_time[fidelity-1]:
                o_time = layer(o_time)
            o_snapw = o_hid.clone()
            for layer in self.fc_out_snapw[fidelity-1]:
                o_snapw = layer(o_snapw)
            output = torch.cat((o_wp, o_time, o_snapw), dim=1)
        
            outputs[t+1,:,:] = output
            trg = output
        
        mask = torch.zeros(self.max_seq_len, batch_size, self.emb_dim, dtype=torch.bool).to(self.device)
        for k in range(batch_size):
            mask[int(seq_len[k].data):,k,:-2] = 1
        outputs[mask] = 0
        
        mask_t = torch.zeros(self.max_seq_len, batch_size, self.emb_dim, dtype=torch.bool).to(self.device)
        for k in range(batch_size):
            mask_t[0,k,-2:] = 1
            mask_t[int(seq_len[k].data):,k,-2:] = 1
        outputs[mask_t] = torch.tensor(float('-inf')).to(self.device)
        # outputs[mask_t] = 0
        
        return outputs

    def forward_single(self, trg, hidden, enc_src, fidelity=2):
        hidden = hidden.squeeze()
        batch_size = hidden.size(0)
        
        embedded = trg
        for layer in self.fc_emb:
            embedded = layer(embedded)
        embedded = embedded.unsqueeze(0)
        
        a = self.attention(hidden, enc_src)
        a = a.unsqueeze(1)

        weighted = torch.bmm(a, enc_src.permute(1, 0, 2))
        weighted = weighted.permute(1, 0, 2)

        rnn_input = torch.cat((embedded, weighted), dim = 2)     

        output_t, hidden_t = self.rnn(rnn_input, hidden.unsqueeze(0))
        embedded_t = embedded.reshape(batch_size, -1)
        output_t = output_t.reshape(batch_size, -1)
        weighted = weighted.reshape(batch_size, -1)

        weighted = weighted.squeeze(0)

        o_hid = self.fc_out_hid(torch.cat((output_t, weighted, embedded_t), dim = 1))

        o_wp = o_hid.clone()
        for layer in self.fc_out_wp:
            o_wp = layer(o_wp)
        o_time = o_hid.clone()
        for layer in self.fc_out_time[fidelity-1]:
            o_time = layer(o_time)
        o_snapw = o_hid.clone()
        for layer in self.fc_out_snapw[fidelity-1]:
            o_snapw = layer(o_snapw)
        output = torch.cat((o_wp, o_time, o_snapw), dim=1)
        
        return output, hidden_t, output_t


class WaypointsEncDec(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.max_seq_len = kwargs["max_seq_len"]
        self.latent_size = kwargs["latent_size"]
        
        self.device = kwargs["device"]
        self.attn = Attention(kwargs["hid_dim"], kwargs["hid_dim"])
        self.enc = Encoder(**kwargs)
        self.dec = Decoder(self.attn, **kwargs)
        
        # VAE layers
        h2m_module_list = [MLPLayer(kwargs["hid_dim"], kwargs["dropout"]) for _ in range(kwargs["vae_n_layers"])]
        h2m_module_list.append(nn.Linear(kwargs["hid_dim"], kwargs["latent_size"]))
        self.hidden2mean = nn.ModuleList(h2m_module_list)
        h2v_module_list = [MLPLayer(kwargs["hid_dim"], kwargs["dropout"]) for _ in range(kwargs["vae_n_layers"])]
        h2v_module_list.append(nn.Linear(kwargs["hid_dim"], kwargs["latent_size"]))
        self.hidden2logv = nn.ModuleList(h2v_module_list)
        l2h_module_list = [
            nn.Sequential(
                nn.Linear(kwargs["latent_size"], kwargs["hid_dim"]),
                nn.ReLU(inplace=True),
                nn.Dropout(p=kwargs["dropout"]))]
        l2h_module_list.extend([MLPLayer(kwargs["hid_dim"], kwargs["dropout"]) for _ in range(kwargs["vae_n_layers"]-1)])
        l2h_module_list.append(nn.Linear(kwargs["hid_dim"], kwargs["hid_dim"]))
        self.latent2hidden = nn.ModuleList(l2h_module_list)
        
        # self.time_norm = nn.Softmax(dim=0)
        self.time_norm = torch.exp
        self.snapw_norm = nn.Softmax(dim=0)
    
    def denorm_outputs(self, outputs, seq_len):
        o_yaw = nn.functional.normalize(outputs[:,:,3:5], p=2.0, dim=2)
        o_wp = torch.cat((outputs[:,:,:3], o_yaw, outputs[:,:,5:6]), dim = 2)
        o_time = self.time_norm(outputs[:,:,6]) / (seq_len-1).unsqueeze(0)
        o_snapw = self.snapw_norm(outputs[:,:,7])
        
        return o_wp, o_time, o_snapw
    
    def vae_reparam(self, hidden, train=True):
        batch_size = hidden.shape[0]
        
        mean = hidden.clone()
        for layer in self.hidden2mean:
            mean = layer(mean)
        logv = hidden.clone()
        for layer in self.hidden2logv:
            logv = layer(logv)
        
        std = torch.exp(0.5 * logv)
        z = torch.randn([batch_size, self.latent_size]).to(self.device)
        if train:
            z = z * std + mean
        else:
            z = mean
        
        hidden_out = z.clone()
        for layer in self.latent2hidden:
            hidden_out = layer(hidden_out)
        return hidden_out, mean, logv
    
    def forward(self, src, seq_len, initial, fidelity=2, train=True):        
        # initial_t = torch.cat((initial,(src[0,:,6].clone() * (seq_len-1)).unsqueeze(1)), dim=1)
        enc_src, hidden = self.enc(src[:,:,:6], initial)
        
        hidden, mean, logv = self.vae_reparam(hidden, train)
        
        embed = src[0,:,:8].clone()
        embed[:,6] = 0
        outputs = self.dec(embed, hidden, enc_src, seq_len, fidelity)
        o_wp, o_time, o_snapw = self.denorm_outputs(outputs, seq_len)
        
        return o_wp, o_time, o_snapw, mean, logv

    def forward_enc(self, src, seq_len, initial=None):
        # initial_t = torch.cat((initial,(src[0,:,6].clone() * (seq_len-1)).unsqueeze(1)), dim=1)
        enc_src, hidden = self.enc(src[:,:,:6], initial)
        hidden, mean, logv = self.vae_reparam(hidden, train=True)
        return hidden, enc_src, mean, logv
    
    def forward_dec_single(self, trg, hidden, enc_src, seq_len, fidelity=2):
        output, hidden_t, rnn_output = self.dec.forward_single(trg, hidden, enc_src, fidelity)
        return output, hidden_t, rnn_output