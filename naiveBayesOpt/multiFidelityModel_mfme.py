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

def meta_high_fidelity(poly, \
    alpha_set, t_set_sim, points, points_scale, \
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
        flag_direct_yaw = False
        if yaw_mode == 0:
            res_yaw = 0
        elif yaw_mode == 2:
            res_yaw, d_ordered_yaw_tmp = poly.obj_func_acc(points, t_set_tmp, flag_loop, snap_w=alpha_s, flag_yaw=True)
        elif yaw_mode == 3:
            res_yaw, d_ordered_yaw_tmp = poly.obj_func_acc(points, t_set_tmp, flag_loop, snap_w=alpha_s, flag_yaw=True, flag_direct_yaw=True)
            flag_direct_yaw = True
        snap_tmp = res/((np.sum(t_set_sim)/np.sum(t_set_tmp))**7)*(alpha_s.shape[0]/np.sum(alpha_s)) + \
            res_yaw/((np.sum(t_set_sim)/np.sum(t_set_tmp))**3)*(alpha_s.shape[0]/np.sum(alpha_s))

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
        
        debug_array = poly.sim.run_simulation_from_der(
            t_set_tmp, d_ordered_tmp, d_ordered_yaw_tmp, N_trial=1, 
            max_pos_err=0.2, min_pos_err=0.1, 
            max_yaw_err=15., min_yaw_err=5., 
            freq_ctrl=200, freq_sim=400, flag_debug=False, direct_yaw=flag_direct_yaw)
        failure_idx = debug_array[0]["failure_idx"]
        feas = 1
        if failure_idx != -1:
            feas = 0
                
        res_t = np.zeros(2)
        res_t[0] = feas
        res_t[1] = snap_tmp
        return res_t
    
    if multicore:
        data_list = []
        for it in range(alpha_set.shape[0]):
            alpha_tmp = alpha_set[it,:]*(ub-lb) + lb
            data_list.append((points, points_scale, t_set_sim, alpha_tmp, snap_init))
        results = parmap(wrapper_sanity_check, data_list)
    else:
        results = []
        for it in range(alpha_set.shape[0]):
            alpha_tmp = alpha_set[it,:]*(ub-lb) + lb
            results.append(wrapper_sanity_check((points, points_scale, t_set_sim, alpha_tmp, snap_init)))

    for it in range(alpha_set.shape[0]):
        if results[it][0]:
            if debug:
                print("Succeed")
        else:
            if debug:
                print("Failed")
    return np.array(results)

def get_dataset_init( \
    name_suffix, sample_idx, low_fidelity, tmp_high_fidelity, \
    N_L=200, t_set_sta=None, model_path="./", \
    lb=0.1, ub=1.9, sampling_mode=1, batch_size=200, \
    flag_multicore=False):
    
#     name_suffix = "traj_vec_directyaw_numpygd_maxp5"
#     sample_name_ = "traj_vec_directyaw_numpygd_maxp5_{}_{}".format(seqi,bi)
    sample_name_L = "{}_{}".format(name_suffix, sample_idx)
    filedir='bo_data/{}/'.format(sample_name_)
    filename='exp_data_bo_MaxIter30_123.yaml'
    yamlFile = os.path.join(filedir, filename)

    with open(yamlFile, "r") as input_stream:
        yaml_in = yaml.load(input_stream)
    X_L_t = np.array(yaml_in["X_L"])
    Y_L_t = np.array(yaml_in["Y_L"])
    
    sample_name_H = "mfbo_{}_{}".format(name_suffix, sample_idx)
    
    filedir = '{}/{}'.format(model_path,sample_name_H)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    
    path_dataset_low = "{}/{}/low_fidelity_data_sta_{}_{}_smode{}.h5" \
        .format(model_path,str(sample_name_H),np.int(10*lb),np.int(10*ub),sampling_mode)
    
    if not path.exists(path_dataset_low):
        if np.where(Y_L_t == 0)[0].shape[0] > 0:
            X_L_0 = X_L_t[np.where(Y_L_t[:,0] == 0)]
        if np.where(Y_L_t > 0)[0].shape[0] > 0:
            X_L_1 = X_L_t[np.where(Y_L_t[:,0] > 0)]
        prYellow("N_L_0: {}, N_L_1: {}".format(X_L_0.shape[0],X_L_1.shape[0]))
        h5f = h5py.File(path_dataset_low, 'w')
        h5f.create_dataset('X_L_0', data=np.array(X_L_0))
        h5f.create_dataset('X_L_1', data=np.array(X_L_1))
        h5f.close()
    
    path_dataset_high = "{}/{}/high_fidelity_data_sta_{}_{}_smode{}.h5" \
        .format(model_path,str(sample_name_H),np.int(10*lb),np.int(10*ub),sampling_mode)
    
    if not path.exists(path_dataset_high):
        for i in range(len(t_set_sim_i)):
            t_set_sim_i[i] *= alpha_sim
        high_fidelity_i = lambda x: tmp_high_fidelity(x, t_set_sta)
        range_min=0.2
        range_max=3.2
        opt_N_eval=11
        opt_N_step=3
        N_eval = opt_N_eval
        N_step = opt_N_step

        alpha_low = range_min
        alpha_high = range_max
        res_alpha = 1
        for step in range(N_step):
            alpha_set_list = []

            if step == 0:
                alpha_set_list = list(np.linspace(alpha_low,alpha_high,N_eval))
            else:
                alpha_set_list = list(np.linspace(alpha_low,alpha_high,N_eval))[1:-1]
            alpha_set_list_t = []
            N_alpha = len(alpha_set_list)
            res_idx = -1
            for al_ii in range(N_alpha-1,-1,-1):
                alpha_t = alpha_min * alpha_set_list[al_ii]
                alpha_t = norm_alpha(alpha_t)
                res = high_fidelity_i(np.expand_dims(alpha_t,0))
                if res[0][0] == 0:
                    res_idx = al_ii
                    break

            if res_idx == -1 and step == 0:
                res_alpha = alpha_low
                break
            elif res_idx == -1 and step > 0:
                alpha_high = alpha_set_list[0]
            elif res_idx == N_alpha-1 and step == 0:
                res_alpha = alpha_high
                break
            elif res_idx == N_alpha-1 and step > 0:
                alpha_low = alpha_set_list[-1]
            else:
                alpha_low = alpha_set_list[np.int(res_idx)]
                alpha_high = alpha_set_list[np.int(res_idx)+1]

            if step == N_step-1:
                res_alpha = alpha_high
        alpha_sim = copy.deepcopy(res_alpha)
        print("#####################################")
        print("alpha_sim : {}".format(alpha_sim))
        print("#####################################")
        
        t_set_sim = t_set_sta * alpha_sim
        high_fidelity_f = lambda x: tmp_high_fidelity(x, t_set_sim)
        
        alpha_list = list(np.linspace(0.5,1.5,21))
        alpha_set_list = []
        for alpha in alpha_list:
            alpha_set_list.append(np.ones_like(t_set_sta)*alpha_t)
        X_H_t = np.array(alpha_set_list)
        Y_H_t = high_fidelity(X_next_H)
        
        if np.where(Y_H_t == 0)[0].shape[0] > 0:
            X_H_0 = X_H_t[np.where(Y_H_t[:,0] == 0)]
        if np.where(Y_H_t > 0)[0].shape[0] > 0:
            X_H_1 = X_H_t[np.where(Y_H_t[:,0] > 0)]
        prYellow("N_H_0: {}, N_H_1: {}".format(X_H_0.shape[0],X_H_1.shape[0]))
        h5f = h5py.File(path_dataset_high, 'w')
        h5f.create_dataset('alpha_sim', data=np.array([alpha_sim]))
        h5f.create_dataset('X_H_0', data=np.array(X_H_0))
        h5f.create_dataset('X_H_1', data=np.array(X_H_1))
        h5f.close()

    return X_L, Y_L, X_H, Y_L, alpha_sim