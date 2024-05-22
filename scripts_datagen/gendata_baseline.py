#!/usr/bin/env python
# coding: utf-8

import os, sys, io, random
import json, argparse
import time, yaml, copy
import numpy as np
from collections import OrderedDict, defaultdict
from scipy.special import factorial
from pyDOE import lhs
import scipy
from scipy import optimize

sys.path.insert(0, '../')
from pyTrajectoryUtils.pyTrajectoryUtils.minSnapTrajectory import *
from pyTrajectoryUtils.pyTrajectoryUtils.utils import *

N_POINTS = 40
traj_tool = TrajectoryTools(drone_model="STMCFB", N_POINTS=N_POINTS)
min_snap = MinSnapTrajectory(drone_model="STMCFB", N_POINTS=N_POINTS)
points_mean = 10.

def get_min_time(data_t, flag_snapw=True, kt=1.):
    batch_size = len(data_t)

    def wrapper_sanity_check(args):
        points_set_t = args[0]
        t_set, d_ordered, d_ordered_yaw = min_snap.get_min_snap_traj(
            points_set_t[:,:4], alpha_scale=1.0, flag_loop=False, yaw_mode=3, \
            deg_init_min=0, deg_init_max=4, \
            deg_end_min=0, deg_end_max=4, \
            deg_init_yaw_min=0, deg_init_yaw_max=2, \
            deg_end_yaw_min=0, deg_end_yaw_max=2, \
            points_mean=points_mean, \
            kt=kt, kt2=0, mu=0.01,\
            flag_rand_init=False, flag_numpy_opt=True, flag_scipy_opt=False, \
            flag_opt_alpha=True)
        return t_set

    data_list = []
    for b_ii in range(batch_size):
        data_list.append((data_t[b_ii], b_ii))
            
    results = parmap(wrapper_sanity_check, data_list)
    snap_w_list = []
    if flag_snapw:
        for b_ii in range(batch_size):
            # snap_w_t = 1./results[b_ii]
            snap_w_t = np.ones_like(results[b_ii])
            snap_w_t /= np.sum(snap_w_t)
            snap_w_list.append(snap_w_t)
    
    label_sta = min_snap.optimize_alpha_acc_yaw( \
        data_t, results, snap_w_list=snap_w_list, \
        range_min=0.2, range_max=3.2, opt_N_eval=11, opt_N_step=3, direct_yaw=True, flag_nonstop=True)
    results_sta = []
    for b_ii in range(batch_size):
        results_sta.append(results[b_ii] * label_sta[b_ii])
    
    label_sta2 = min_snap.optimize_alpha_acc_yaw( \
        data_t, results_sta, snap_w_list=snap_w_list, \
        range_min=0.2, range_max=3.2, opt_N_eval=11, opt_N_step=3, direct_yaw=True, flag_nonstop=True)
    results_sta2 = []
    for b_ii in range(batch_size):
        results_sta2.append(results_sta[b_ii] * label_sta2[b_ii])
    
    return snap_w_list, results_sta2

def get_min_time_sim(data_t, snap_w_list, results_sta, flag_robust=False):
    batch_size = len(data_t)
    label_sim = min_snap.optimize_alpha_acc_yaw_sim( \
        data_t, results_sta, snap_w_list=snap_w_list, \
        range_min=0.5, range_max=2.5, opt_N_eval=3, opt_N_step=5, direct_yaw=True, flag_robust=flag_robust, flag_nonstop=True)
    results_sim = []
    for b_ii in range(batch_size):
        results_sim.append(results_sta[b_ii] * label_sim[b_ii])
    
    if flag_robust:
        label_sim2 = min_snap.optimize_alpha_acc_yaw_sim( \
            data_t, results_sim, snap_w_list=snap_w_list, \
            range_min=0.8, range_max=1.2, opt_N_eval=3, opt_N_step=11, direct_yaw=True, flag_robust=flag_robust, flag_nonstop=True)
        results_sim2 = []
        for b_ii in range(batch_size):
            results_sim2.append(results_sim[b_ii] * label_sim2[b_ii])
        return results_sim2
    return results_sim

def main(args):
    yaw_ratio = 0.
    
    suffix = ""
    mfrl_datapath = "mfrl_ps993_test"
    
    h5f_filedir = "../dataset/{}{}.h5".format(mfrl_datapath, suffix)
    h5f_write = "../dataset/{}{}_w{}.h5".format(mfrl_datapath, suffix, str(args.reg_w).replace(".", "p"))
    print(h5f_write)
    
    batch_size = args.batch_size
    
    for p_ii in range(args.data_min_dim, args.data_max_dim+1):
        h5f = h5py.File(h5f_filedir, 'r')
        points_arr = np.array(h5f['{}'.format(p_ii)]["points"])
        alpha_sta_i = np.array(h5f['{}'.format(p_ii)]["alpha_sta"])
        alpha_sim_i = np.array(h5f['{}'.format(p_ii)]["alpha_sim"])
        h5f.close()
        N_data = points_arr.shape[0]
        N_batch = int(points_arr.shape[0]/batch_size)

        o_data = []
        o_label_sta = []
        o_snapw_sta = []
        o_results_sta =[]
        o_label_sim = []
        o_compare = []
        for b_ii in range(N_batch):
            data_set = list(points_arr[b_ii*batch_size:(b_ii+1)*batch_size,:6])

            snap_w_list, results_sta = get_min_time(data_set, flag_snapw=True, kt=args.reg_w*0.01)
            o_snapw_sta.extend(snap_w_list)
            o_results_sta.extend(results_sta)
            for d_ii in range(batch_size):
                data_tmp = data_set[d_ii]
                data_tmp[1:,5] = results_sta[d_ii]
                data_tmp[1:,5] /= np.sum(data_tmp[1:,5])
                data_tmp = np.pad(data_tmp, ((0,0),(0,1)), mode='constant', constant_values=0)
                data_tmp[1:,6] = 1./data_tmp[1:,5]
                data_tmp[1:,6] *= 1. / np.sum(data_tmp[1:,6])
                o_data.append(data_tmp)
                o_label_sta.append(np.sum(results_sta[d_ii]))

            data_t = copy.deepcopy(o_data)
            data_set = list(data_t)
            results_sta = list(o_results_sta)
            snap_w_list = list(o_snapw_sta)

            results_sim = get_min_time_sim(data_set, snap_w_list, results_sta, flag_robust=False)
            for d_ii in range(batch_size):
                o_label_sim.append(np.sum(results_sim[d_ii]))
                o_compare.append(np.sum(results_sta[d_ii])/np.sum(results_sim[d_ii]))
            
            print("Seq {}: N_data sta, sim - {}, {}".format(p_ii, len(o_label_sta), len(o_label_sim)))

        o_data = np.array(o_data)
        o_label_sta = np.array(o_label_sta)
        o_label_sim = np.array(o_label_sim)
        o_compare = np.array(o_compare)
        
        print("Seq {}: compare sta, sim - {}, {}".format(p_ii, np.mean(o_label_sta/alpha_sta_i), np.mean(o_label_sim/alpha_sim_i)))

        if p_ii == 5:
            h5f = h5py.File(h5f_write, 'w')
        else:
            h5f = h5py.File(h5f_write, 'a')
        grp = h5f.create_group("{}".format(p_ii))
        grp.create_dataset('points', data=np.array(o_data))        
        grp.create_dataset('alpha_sta', data=np.array(o_label_sta))
        grp.create_dataset('alpha_sim', data=np.array(o_label_sim))
        h5f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--reg_w', type=float, default=1.)
    parser.add_argument('-bs', '--batch_size', type=int, default=50)
    
    # Load training data
    parser.add_argument('-dmin', '--data_min_dim', type=int, default=5)
    parser.add_argument('-dmax', '--data_max_dim', type=int, default=14)

    args = parser.parse_args()

    main(args)
