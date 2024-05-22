#!/usr/bin/env python
# coding: utf-8

import os, sys, io, random
import json
import time
import torch
import argparse
import numpy as np
import yaml, copy
from multiprocessing import cpu_count
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.autograd import Variable
from collections import OrderedDict, defaultdict
import matplotlib.pyplot as plt
import PIL.Image
from torchvision.transforms import ToTensor
from scipy.stats import norm

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

from scipy import stats
from sklearn.metrics import pairwise_distances
from scipy.spatial.transform import Rotation as R

def get_spatial_dev(data, data_len, points_scale, yaw_dev, psi_dev):
    data_tmp_init = copy.deepcopy(data)
    for i in range(3):
        data_tmp_init[:,i] *= (points_scale[i] / 2.)
    data_tmp = copy.deepcopy(data_tmp_init)
    psi_dev_rad = psi_dev/180.*np.pi
    
    bs = 2000
    max_trial = 100
    while True:
        n_try = 0
        data_tmp = copy.deepcopy(data_tmp_init)
        flag_found_2 = True
        
        for d_ii in range(1, data_len):
            pos_diff = np.linalg.norm(data_tmp_init[d_ii,:3]-data_tmp_init[d_ii-1,:3])
            dev_i = data_tmp_init[d_ii,:3]-data_tmp_init[d_ii-1,:3]
            dev_i /= np.linalg.norm(dev_i)
            vec_z = np.array([0.,0.,1.])
            rot, rssd = R.align_vectors(np.expand_dims(dev_i, axis=0), np.expand_dims(vec_z, axis=0))
            
            while True:
                psi = np.random.random(bs) * 2 * np.pi
                vec_i = np.vstack([np.cos(psi)*np.sin(psi_dev_rad), np.sin(psi)*np.sin(psi_dev_rad), np.ones(bs)*np.cos(psi_dev_rad)]).T
                dev = rot.apply(vec_i)
                dev *= pos_diff
                pos_tmp = dev + np.tile(np.expand_dims(data_tmp[d_ii-1,:3],0),(bs,1))
                for i in range(3):
                    pos_tmp[:,i] /= (points_scale[i] / 2.)
                flag_found = False
                for i in range(bs):
                    if np.all(pos_tmp[i,:] > -1) and np.all(pos_tmp[i,:] < 1):
                        data_tmp[d_ii,:3] = copy.deepcopy(data_tmp[d_ii-1,:3]) + dev[i,:]
                        flag_found = True
                        break
                if flag_found:
                    n_try = 0
                    break
                else:
                    n_try += 1
                    if n_try >= max_trial:
                        flag_found_2 = False
                        break
            if not flag_found_2:
                break
        if flag_found_2:
            break
    dev_t = np.zeros((data_len-1, 4))
    for d_ii in range(1, data_len):
        if np.random.random(1) > 0.5:
            dev_t[d_ii-1, 3] = yaw_dev/180.*np.pi # +/- 30 deg
        else:
            dev_t[d_ii-1, 3] = -yaw_dev/180.*np.pi # +/- 30 deg
    for d_ii in range(1, data_len):
        dev = data_tmp[d_ii,:3]-data_tmp_init[d_ii,:3]
        dev /= (points_scale / 2.)
        dev_t[d_ii-1, :3] = dev
    
    r_ii_set = np.arange(1,data_len).astype(np.int32)
    return r_ii_set, list(dev_t)

def get_constant_dev(data, data_len, points_scale, yaw_dev):
    data_tmp_init = copy.deepcopy(data)
    for i in range(3):
        data_tmp_init[:,i] *= (points_scale[i] / 2.)
    data_tmp = copy.deepcopy(data_tmp_init)
    
    # dev_t = np.zeros((data_len-1, 4))
    # for d_ii in range(1, data_len):
    #     pos_diff = np.linalg.norm(data_tmp_init[d_ii,:3]-data_tmp_init[d_ii-1,:3])
    #     dev = (np.random.random(3)-0.5)
    #     dev /= np.linalg.norm(dev)
    #     dev *= pos_diff
    #     data_tmp[d_ii,:3] = copy.deepcopy(data_tmp[d_ii-1,:3]) + dev
    #     d_i = np.linalg.norm(data_tmp_init[d_ii,:3]-data_tmp_init[d_ii-1,:3])
    #     d_f = np.linalg.norm(data_tmp[d_ii,:3]-data_tmp[d_ii-1,:3])
    #     if np.abs(d_i-d_f) > 1e-3:
    #         print(d_i)
    #         print(data_tmp_init[d_ii,:3]-data_tmp_init[d_ii-1,:3])
    #         print(d_f)
    #         print(data_tmp[d_ii,:3]-data_tmp[d_ii-1,:3])
    #         assert d_i == d_f  
    #     if np.random.random(1) > 0.5:
    #         dev_t[d_ii-1, 3] = yaw_dev/180.*np.pi # +/- 30 deg
    #     else:
    #         dev_t[d_ii-1, 3] = -yaw_dev/180.*np.pi # +/- 30 deg
        
    bs = 2000
    max_trial = 100
    while True:
        n_try = 0
        data_tmp = copy.deepcopy(data_tmp_init)
        flag_found_2 = True
        
        for d_ii in range(1, data_len):
            pos_diff = np.linalg.norm(data_tmp_init[d_ii,:3]-data_tmp_init[d_ii-1,:3])
            while True:
                dev = (np.random.random((bs,3))-0.5)*2.
                for i in range(3):
                    dev[:,i] -= data_tmp[d_ii-1,i] / (points_scale[i]/2.)
                dev /= np.tile(np.expand_dims(np.linalg.norm(dev, axis=1),1),(1,3))
                dev *= pos_diff
                pos_tmp = dev + np.tile(np.expand_dims(data_tmp[d_ii-1,:3],0),(bs,1))
                for i in range(3):
                    pos_tmp[:,i] /= (points_scale[i] / 2.)
                flag_found = False
                for i in range(bs):
                    if np.all(pos_tmp[i,:] > -1) and np.all(pos_tmp[i,:] < 1):
                        data_tmp[d_ii,:3] = copy.deepcopy(data_tmp[d_ii-1,:3]) + dev[i,:]
                        flag_found = True
                        break
                if flag_found:
                    n_try = 0
                    break
                else:
                    n_try += 1
                    if n_try >= max_trial:
                        flag_found_2 = False
                        break
            if not flag_found_2:
                break
        if flag_found_2:
            break
    dev_t = np.zeros((data_len-1, 4))
    for d_ii in range(1, data_len):
        if np.random.random(1) > 0.5:
            dev_t[d_ii-1, 3] = yaw_dev/180.*np.pi # +/- 30 deg
        else:
            dev_t[d_ii-1, 3] = -yaw_dev/180.*np.pi # +/- 30 deg
    for d_ii in range(1, data_len):
        dev = data_tmp[d_ii,:3]-data_tmp_init[d_ii,:3]
        dev /= (points_scale / 2.)
        dev_t[d_ii-1, :3] = dev
    
    # diff_i = np.linalg.norm(np.diff(data_tmp_init[:data_len,:3],axis=0),axis=1)
    # diff_f = np.linalg.norm(np.diff(data_tmp[:data_len,:3],axis=0),axis=1)
    # if not np.all(np.abs(diff_i-diff_f) < 0.01):
    #     print(data_len)
    #     print(data_tmp_init[:,:3])
    #     print(data_tmp[:,:3])
    #     print(diff_i)
    #     print(diff_f)
    #     assert np.all(np.abs(diff_i-diff_f) < 0.01)
    
    r_ii_set = np.arange(1,data_len).astype(np.int32)
    return r_ii_set, list(dev_t)

def get_biased_dev(data, data_len, N_dev, points_scale, pos_dev, yaw_dev):
    r_ii_set = np.sort(np.random.permutation(data_len-1)[:N_dev] + 1)
    r_ii_set = r_ii_set.astype(np.int32)
    dev_t = []
    for d_ii in range(N_dev):
        while True:
            dev = (np.random.random(4)-0.5)
            dev[:3] /= np.linalg.norm(dev[:3])
            dev[:3] *= pos_dev
            dev[:3] /= (points_scale / 2.)
            pos_tmp = data[r_ii_set[d_ii], :3] + dev[:3]
            if np.all(pos_tmp > -1) and np.all(pos_tmp < 1):
                break
        if dev[3] > 0:
            dev[3] = yaw_dev/180.*np.pi # +/- 30 deg
        else:
            dev[3] = -yaw_dev/180.*np.pi # +/- 30 deg
        dev_t.append(dev)
    return r_ii_set, dev_t

def get_bald(m, v):
    C = np.sqrt(np.pi * np.log(2) / 2.)
    p  = norm.cdf(m/np.sqrt(v**2+1))
    bald1 = - p * np.log(p) - (1-p) * np.log(1-p)
    bald2 = C * np.exp(-m**2/2./(v**2 + C**2)) / np.sqrt(v**2 + C**2)
    bald = bald1 - bald2
    return bald

# kmeans ++ initialization
# https://github.com/JordanAsh/badge/blob/master/query_strategies/badge_sampling.py
def kmeans_pp_centers(X, K):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    # print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] >  newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        # print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2)/ sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in indsAll: ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll

def np2cuda(data):
    return torch.tensor(data).float()

def np2cuda2(data):
    return torch.tensor(data).float().cuda()

def plot_output(min_snap, points_i, points_f, t_set, snap_w):
    wp_loss = np.linalg.norm(points_i - points_f)
    print("plot outputs - WP-Loss: %9.6f" % (wp_loss.item()))

    flag_min_snap_success = True
    try:
        t_set, d_ordered, d_ordered_yaw = min_snap.update_traj( \
            points_f[:,:3], t_set, np.ones_like(t_set), \
            snap_w=snap_w, \
            yaw_mode=0, \
            deg_init_min=0, deg_init_max=4, \
            deg_end_min=0, deg_end_max=4, \
            deg_init_yaw_min=0, deg_init_yaw_max=2, \
            deg_end_yaw_min=0, deg_end_yaw_max=2)
        V_t = min_snap.generate_sampling_matrix(t_set, N=min_snap.N_POINTS, der=0, endpoint=True)
        val = V_t.dot(d_ordered)
    except:
        flag_min_snap_success = False
    ##############

    fig = plt.figure(figsize=(5,5))
    fig.canvas.draw()
    ax = fig.add_subplot(111)
    if flag_min_snap_success:
        ax.plot(val[:,0], val[:,1], linewidth=2, zorder=1,c='dimgray')
    ax.plot(points_i[:,0], points_i[:,1], marker='o', color='r')
    ax.scatter(points_i[:,0], points_i[:,1], marker='o', color='r', s=50)
    ax.plot(points_f[:,0], points_f[:,1], marker='o', color='b')
    ax.scatter(points_f[:,0], points_f[:,1], marker='o', color='b', s=50)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    img = PIL.Image.open(buf)
    img = ToTensor()(img)
    plt.close(fig)

    return img

def plot_output_bploy(min_snap, points_i, points_f, t_set, snap_w, bernstein):
    len_i = points_i.shape[0]
    len_f = points_f.shape[0]    
    wp_loss = np.linalg.norm(points_i[len_i-len_f:,:] - points_f)
    print("plot outputs - WP-Loss: %9.6f" % (wp_loss.item()))

    flag_min_snap_success = True
    try:
        t_set, d_ordered, d_ordered_yaw = min_snap.update_traj_new( \
            points_f[:,:3], t_set, np.ones_like(t_set), \
            snap_w=snap_w, \
            yaw_mode=0, \
            deg_init_min=0, deg_init_max=4, \
            deg_end_min=0, deg_end_max=4, \
            deg_init_yaw_min=0, deg_init_yaw_max=2, \
            deg_end_yaw_min=0, deg_end_yaw_max=2, \
            bernstein=bernstein)
        V_t = min_snap.generate_sampling_matrix(t_set, N=min_snap.N_POINTS, der=0, endpoint=True)
        val = V_t.dot(d_ordered)
    except:
        flag_min_snap_success = False
    ##############

    fig = plt.figure(figsize=(5,5))
    fig.canvas.draw()
    ax = fig.add_subplot(111)
    if flag_min_snap_success:
        ax.plot(val[:,0], val[:,1], linewidth=2, zorder=1,c='dimgray')
    ax.plot(points_i[:,0], points_i[:,1], marker='o', color='r')
    ax.scatter(points_i[:,0], points_i[:,1], marker='o', color='r', s=50)
    ax.plot(points_f[:,0], points_f[:,1], marker='o', color='b')
    ax.scatter(points_f[:,0], points_f[:,1], marker='o', color='b', s=50)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    img = PIL.Image.open(buf)
    img = ToTensor()(img)
    plt.close(fig)

    return img

def plot_output_der(min_snap, points_i, points_f, t_set, snap_w, der):
    len_i = points_i.shape[0]
    len_f = points_f.shape[0]    
    wp_loss = np.linalg.norm(points_i[len_i-len_f:,:] - points_f)
    print("plot outputs - WP-Loss: %9.6f" % (wp_loss.item()))

    flag_min_snap_success = True
    try:
        t_set, d_ordered, d_ordered_yaw = min_snap.update_traj_new( \
            points_f[:,:3], t_set, np.ones_like(t_set), \
            snap_w=snap_w, \
            yaw_mode=0, \
            deg_init_min=0, deg_init_max=4, \
            deg_end_min=0, deg_end_max=4, \
            deg_init_yaw_min=0, deg_init_yaw_max=2, \
            deg_end_yaw_min=0, deg_end_yaw_max=2, \
            end_der=der)
        V_t = min_snap.generate_sampling_matrix(t_set, N=min_snap.N_POINTS, der=0, endpoint=True)
        val = V_t.dot(d_ordered)
    except:
        flag_min_snap_success = False
    ##############

    fig = plt.figure(figsize=(5,5))
    fig.canvas.draw()
    ax = fig.add_subplot(111)
    if flag_min_snap_success:
        ax.plot(val[:,0], val[:,1], linewidth=2, zorder=1,c='dimgray')
    ax.plot(points_i[:,0], points_i[:,1], marker='o', color='r')
    ax.scatter(points_i[:,0], points_i[:,1], marker='o', color='r', s=50)
    ax.plot(points_f[:,0], points_f[:,1], marker='o', color='b')
    ax.scatter(points_f[:,0], points_f[:,1], marker='o', color='b', s=50)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    img = PIL.Image.open(buf)
    img = ToTensor()(img)
    plt.close(fig)

    return img

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x