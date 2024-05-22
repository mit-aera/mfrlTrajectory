#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os, sys, time, copy, io
import yaml, h5py, shutil
from os import path
from pyDOE import lhs

sys.path.insert(0, '../')
from pyTrajectoryUtils.pyTrajectoryUtils.utils import *
from .trajSampler import TrajSampler_online

def meta_low_fidelity(poly, \
    alpha_set, t_set_sta, points, points_scale, \
    debug=True, multicore=False, lb=0.1, ub=1.9, snap_init=1.0, \
    yaw_mode=2):
    
    def wrapper_sanity_check(args):
        points = args[0]
        points_scale = args[1]
        t_set = args[2]
        alpha_set = args[3]
        snap_i = args[4]
        
        alpha_t = alpha_set[:t_set.shape[0]]
        if alpha_set.shape[0] == t_set.shape[0]:
            alpha_s = np.ones_like(alpha_t)
        elif alpha_set.shape[0] == 2*t_set.shape[0]:
            alpha_s = alpha_set[t_set.shape[0]:]

        t_set_tmp = t_set * alpha_t
        flag_loop = poly.check_flag_loop_points(t_set, points)
        N_POLY = t_set.shape[0]
        res, d_ordered_tmp = poly.obj_func_acc(points, t_set_tmp, flag_loop, snap_w=alpha_s)
        if yaw_mode == 0:
            res_yaw = 0
        elif yaw_mode == 2:
            res_yaw, d_ordered_yaw_tmp = poly.obj_func_acc(points, t_set_tmp, flag_loop, snap_w=alpha_s, flag_yaw=True)
        elif yaw_mode == 3:
            res_yaw, d_ordered_yaw_tmp = poly.obj_func_acc(points, t_set_tmp, flag_loop, snap_w=alpha_s, flag_yaw=True, flag_direct_yaw=True)
        snap_tmp = res/((np.sum(t_set_sta)/np.sum(t_set_tmp))**7)*(alpha_s.shape[0]/np.sum(alpha_s)) + \
            res_yaw/((np.sum(t_set_sta)/np.sum(t_set_tmp))**3)*(alpha_s.shape[0]/np.sum(alpha_s))

#         t_set_snap = t_set_tmp * (np.sum(t_set_sta)/np.sum(t_set_tmp))
#         alpha_snap =  alpha_s * (alpha_s.shape[0]/np.sum(alpha_s))
#         res, _ = poly.obj_func_acc(points, t_set_snap, flag_loop, snap_w=alpha_snap)
#         res_yaw, _ = poly.obj_func_acc(points, t_set_snap, flag_loop, snap_w=alpha_snap, flag_yaw=True)
#         snap_tmp = res + res_yaw
        
#         text_trap = io.StringIO()
#         sys.stdout = text_trap
#         feas = poly.sanity_check( \
#             t_set_tmp, d_ordered_tmp, d_ordered_yaw_tmp, flag_parallel=True)
#         sys.stdout = sys.__stdout__
        
        snap_tmp /= snap_i
        
        # generate status
        status = np.zeros((poly.N_POINTS*N_POLY,18))
        status[:,:15] = poly.get_status_acc(t_set_tmp, d_ordered_tmp, flag_loop)
        if yaw_mode == 2:
            status_yaw = poly.get_status_acc(t_set_tmp, d_ordered_yaw_tmp, flag_loop, flag_yaw=True)
            status_yaw = np.array(np.split(status_yaw,3,axis=1))
            status_yaw = np.swapaxes(status_yaw,0,1)
            status[:,15:] = poly.get_yaw_der(status_yaw)
        elif yaw_mode == 3:
            status_yaw = poly.get_status_acc(t_set_tmp, d_ordered_yaw_tmp, flag_loop, flag_yaw=True)
            status_yaw = np.array(np.split(status_yaw,3,axis=1))
            status_yaw = np.swapaxes(status_yaw,0,1)
            status[:,15:] = status_yaw[:,:,0]
        
        ws, ctrl = poly._quadModel.getWs_vector(status)
        feas = 1
        if np.any(ws < poly._quadModel.w_min) or np.any(ws > poly._quadModel.w_max):
            feas = 0
        
#         wall_margin = 1.0
#         for d_ii in range(3):
#             if np.any(status[:,d_ii] > wall_margin*0.5*points_scale[d_ii]) or \
#                 np.any(status[:,d_ii] < -wall_margin*0.5*points_scale[d_ii]):
#                 feas = 0
        
        res_t = np.zeros(2)
        res_t[0] = feas
        res_t[1] = snap_tmp
        return res_t
    
    if multicore:
        data_list = []
        for it in range(alpha_set.shape[0]):
            alpha_tmp = alpha_set[it,:]*(ub-lb) + lb
            data_list.append((points, points_scale, t_set_sta, alpha_tmp, snap_init))
        results = parmap(wrapper_sanity_check, data_list)
    else:
        results = []
        for it in range(alpha_set.shape[0]):
            alpha_tmp = alpha_set[it,:]*(ub-lb) + lb
            results.append(wrapper_sanity_check((points, points_scale, t_set_sta, alpha_tmp, snap_init)))

    for it in range(alpha_set.shape[0]):
        if results[it][0]:
            if debug:
                print("Succeed")
        else:
            if debug:
                print("Failed")
    return np.array(results)

def get_dataset_init( \
    name, low_fidelity, \
    N_L=200, t_set_sta=None, model_path="./", \
    lb=0.1, ub=1.9, sampling_mode=1, batch_size=200, \
    flag_multicore=False):
    
    X_L = []
    Y_L = []
    
    sample_name_ = name
    path_dataset_low = "{}/{}/low_fidelity_data_sta_{}_{}_smode{}.h5" \
        .format(model_path,str(sample_name_),np.int(10*lb),np.int(10*ub),sampling_mode)
    
    filedir = '{}/{}'.format(model_path,sample_name_)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    
    flag_generate_dataset = True
    
    t_dim = t_set_sta.shape[0]
    
    X_L_0 = np.empty((0,2*t_dim))
    X_L_1 = np.empty((0,2*t_dim))
    X_L = np.empty((0,2*t_dim))
    Y_L = np.empty(0)
    
    if path.exists(path_dataset_low):
        h5f = h5py.File(path_dataset_low, 'r')
        X_L_0 = np.array(h5f['X_L_0'][:])
        X_L_1 = np.array(h5f['X_L_1'][:])
        h5f.close()
        if X_L_0.shape[0] >= N_L/2 and X_L_1.shape[0] >= N_L/2:
            flag_generate_dataset = False
            X_L = np.concatenate((X_L_0[:np.int(N_L/2),:],X_L_1[:np.int(N_L/2),:]))
            Y_L = np.zeros(N_L)
            Y_L[np.int(N_L/2):] = 1

    if flag_generate_dataset:
        traj_sampler_t = TrajSampler_online(N=t_dim, sigma=0.2, flag_load=False, cov_mode=1, flag_pytorch=False)
        traj_sampler_p = TrajSampler_online(N=t_dim, sigma=0.2, flag_load=False, cov_mode=1, flag_pytorch=False)
#         traj_sampler_p = lambda N_sample: lhs(p_dim, N_sample)
        def sample_data(N_sample):
            res_t = np.concatenate(
                (lhs(t_dim, np.int(N_sample/2)), traj_sampler_t.rsample(N_sample-np.int(N_sample/2))),axis=0)
            perm_idx_t = np.random.permutation(N_sample)
            res_t = res_t[perm_idx_t,:]
            
#             res_p = np.concatenate(
#                 (lhs(p_dim, np.int(N_sample/2)), traj_sampler_p.rsample(N_sample-np.int(N_sample/2))),axis=0)
            res_p = lhs(t_dim, N_sample)
            perm_idx_p = np.random.permutation(N_sample)
            res_p = res_p[perm_idx_p,:]
            
            res = np.concatenate((res_t, res_p), axis=1)
            return res

        N_L_i = 10
        X_L_t = np.ones((2*N_L_i+1, 2*t_dim))
        for i in range(2*N_L_i+1):
            X_L_t[i,:t_dim] *= (0.1 + 0.4*i/N_L_i)
        labels_low = low_fidelity(X_L_t, debug=False, multicore=flag_multicore)
        Y_L_t = 1.0*labels_low
        if np.where(Y_L_t == 0)[0].shape[0] > 0:
            X_L_0 = np.concatenate((X_L_0, X_L_t[np.where(Y_L_t[:,0] == 0)]))
        if np.where(Y_L_t > 0)[0].shape[0] > 0:
            X_L_1 = np.concatenate((X_L_1, X_L_t[np.where(Y_L_t[:,0] > 0)]))
        prYellow("N_L_0: {}, N_L_1: {}".format(X_L_0.shape[0],X_L_1.shape[0]))
        prYellow("---")
        
        while True:
            X_L_t = sample_data(batch_size)
            
            labels_low = low_fidelity(X_L_t, debug=False, multicore=flag_multicore)
            Y_L_t = 1.0*labels_low
            if np.where(Y_L_t == 0)[0].shape[0] > 0:
                X_L_0 = np.concatenate((X_L_0, X_L_t[np.where(Y_L_t[:,0] == 0)]))
            if np.where(Y_L_t > 0)[0].shape[0] > 0:
                X_L_1 = np.concatenate((X_L_1, X_L_t[np.where(Y_L_t[:,0] > 0)]))
            prYellow("N_L_0: {}, N_L_1: {}".format(X_L_0.shape[0],X_L_1.shape[0]))
            
            h5f = h5py.File(path_dataset_low, 'w')
            h5f.create_dataset('X_L_0', data=np.array(X_L_0))
            h5f.create_dataset('X_L_1', data=np.array(X_L_1))
            h5f.close()
           
            if X_L_0.shape[0] >= N_L/2 and X_L_1.shape[0] >= N_L/2:
#                 X_L = np.concatenate((X_L_0[:np.int(N_L/2),:],X_L_1[:np.int(N_L/2),:]))
#                 Y_L = np.zeros(N_L)
#                 Y_L[np.int(N_L/2):] = 1
                N_L_0 = X_L_0.shape[0]
                N_L_1 = X_L_1.shape[0]
                X_L = np.concatenate((X_L_0,X_L_1))
                Y_L = np.zeros(N_L_0+N_L_1)
                Y_L[N_L_0:] = 1
                break

    X_L = np.array(X_L)
    Y_L = np.array(Y_L)

    return X_L, Y_L