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

def _get_twice_triangle_area(a, b, c):
    if np.all(a == b) or np.all(b == c) or np.all(c == a):
        return 0
    twice_triangle_area = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

    return twice_triangle_area

def _get_menger_curvature(a, b, c, eps=1e-3):
    if np.linalg.norm(a-b)<eps or np.linalg.norm(b-c)<eps or np.linalg.norm(c-a)<eps:
        return 0
    menger_curvature = (2*_get_twice_triangle_area(a, b, c) /
                        (np.linalg.norm(a-b) * np.linalg.norm(b-c) * np.linalg.norm(c-a)))
    return -menger_curvature

def get_waypoints_profile(points):
    num_wp = points.shape[0]
    wp_len = 0
    wp_curv = 0
    for i in range(num_wp-1):
        wp_len += np.linalg.norm(points[i,:]-points[i+1,:])
    for i in range(num_wp-2):
        wp_curv += np.abs(_get_menger_curvature(points[i,:],points[i+1,:],points[i+2,:]))
    
    return wp_len, wp_curv

def get_yaw(points, t_set):
    t_set_sta, d_ordered, d_ordered_yaw = min_snap.update_traj( \
        points, t_set, np.ones_like(t_set), yaw_mode=0, \
        deg_init_min=0, deg_init_max=4, \
        deg_end_min=0, deg_end_max=4, \
        deg_init_yaw_min=0, deg_init_yaw_max=2, \
        deg_end_yaw_min=0, deg_end_yaw_max=2)

    N_wp = points.shape[0]
    vel_vec = np.zeros((N_wp,2))
    for p_ii in range(1,N_wp-1):
        vel_vec[p_ii,:] = d_ordered[p_ii*min_snap.N_DER+1,:2]

    v0, v1 = min_snap.generate_single_point_matrix(0.01, der=1)
    T2_mat = np.diag(min_snap.generate_basis(t_set_sta[0],min_snap.N_DER-1,0))
    der0 = T2_mat.dot(d_ordered[:min_snap.N_DER,:])
    der1 = T2_mat.dot(d_ordered[min_snap.N_DER:2*min_snap.N_DER,:])
    vel_vec[0,:] = ((v0.dot(der0)+v1.dot(der1))/(t_set_sta[0]))[:2]
    v0, v1 = min_snap.generate_single_point_matrix(0.99, der=1)
    T2_mat = np.diag(min_snap.generate_basis(t_set_sta[-1],min_snap.N_DER-1,0))
    der0 = T2_mat.dot(d_ordered[-2*min_snap.N_DER:-min_snap.N_DER,:])
    der1 = T2_mat.dot(d_ordered[-min_snap.N_DER:,:])
    vel_vec[-1,:] = ((v0.dot(der0)+v1.dot(der1))/(t_set_sta[-1]))[:2]

    yaw_init = np.arctan2(vel_vec[:,1],vel_vec[:,0])
    return yaw_init

def _check_waypoints_single(
    points, t_set_init=None, \
    len_min=0.0, len_max=30.0, \
    curv_min=5.0, curv_max=20.0, \
    b_min=0.0, b_max=1.0):
    
    # res_len, res_curv = get_waypoints_profile(points)
    # if (res_len < len_min) or (res_len > len_max):
    #     return False, None, None
    # if (res_curv < curv_min) or (res_curv > curv_max):
    #     return False, None, None
    
    points_yaw = np.zeros((points.shape[0],4))
    points_yaw[:,:3] = points[:,:3]
    if np.all(t_set_init != None):
        points_yaw[:,3] = get_yaw(points, t_set_init)
    else:
        try:
            t_set_init, d_ordered, d_ordered_yaw = min_snap.get_min_snap_traj(
                points[:,:3], alpha_scale=1.0, flag_loop=False, yaw_mode=0, \
                deg_init_min=0, deg_init_max=4, \
                deg_end_min=0, deg_end_max=4, \
                deg_init_yaw_min=0, deg_init_yaw_max=2, \
                deg_end_yaw_min=0, deg_end_yaw_max=2, \
                flag_rand_init=False, flag_numpy_opt=True, flag_scipy_opt=False)
            points_yaw[:,3] = get_yaw(points, t_set_init)
            points_yaw[:,3] = get_closed_turn_direct_yaw(points_yaw[:,3])
        except:
            return False, None, None

    try:
        t_set, d_ordered, d_ordered_yaw = min_snap.get_min_snap_traj(
            points_yaw, alpha_scale=1.0, flag_loop=False, yaw_mode=2, \
            deg_init_min=0, deg_init_max=4, \
            deg_end_min=0, deg_end_max=4, \
            deg_init_yaw_min=0, deg_init_yaw_max=2, \
            deg_end_yaw_min=0, deg_end_yaw_max=2, \
            t_set_init=t_set_init, \
            flag_rand_init=False, flag_numpy_opt=True, flag_scipy_opt=False)
        V_t = min_snap.generate_sampling_matrix(t_set, N=min_snap.N_POINTS, der=0, endpoint=True)
        val = V_t.dot(d_ordered)
    except:
        return False, None, None
    
    if (min(val[:,0]) < b_min) or (max(val[:,0]) > b_max) or \
        (min(val[:,1]) < b_min) or (max(val[:,1]) > b_max) or \
        (min(val[:,2]) < b_min) or (max(val[:,2]) > b_max):
        return False, None, None
    else:
        return True, t_set, points_yaw

    
def wrapper_check_waypoints_single(args):
    points = args[0]
    if len(args) == 1:
        return _check_waypoints_single(points)
    else:
        t_set_init = args[1]
        return _check_waypoints_single(points, t_set_init=t_set_init)

def check_waypoints_cache(points_set, flag_multicore=False):
    if not flag_multicore:
        label_set = np.zeros(len(points_set))
        t_set_list = []
        points_set_list = []
        for points_idx in range(len(points_set)):
            lbl, t_set, points_yaw = wrapper_check_waypoints_single((points_set[points_idx][:,:3],points_set[points_idx][1:,4]))
            label_set[points_idx] = lbl
            t_set_list.append(t_set)
            points_set_list.append(points_yaw)
    else:
        data_list = []
        for points_idx in range(len(points_set)):
            data_list.append([points_set[points_idx][:,:3],points_set[points_idx][1:,4]])
        res = parmap(wrapper_check_waypoints_single, data_list)
        label_set = np.zeros(len(points_set))
        t_set_list = []
        points_set_list = []
        for i in range(len(points_set)):
            label_set[i] = np.array(res[i][0])
            t_set_list.append(res[i][1])
            points_set_list.append(res[i][2])
    return label_set, t_set_list, points_set_list

def check_waypoints(points_set, flag_multicore=False):
    if not flag_multicore:
        label_set = np.zeros(len(points_set))
        t_set_list = []
        points_set_list = []
        for points_idx in range(len(points_set)):
            lbl, t_set, points_yaw = _check_waypoints_single(points_set[points_idx])
            label_set[points_idx] = lbl
            t_set_list.append(t_set)
            points_set_list.append(points_yaw)
    else:
        data_list = []
        for points_idx in range(len(points_set)):
            data_list.append([points_set[points_idx]])
        res = parmap(wrapper_check_waypoints_single, data_list)
        label_set = np.zeros(len(points_set))
        t_set_list = []
        points_set_list = []
        for i in range(len(points_set)):
            label_set[i] = np.array(res[i][0])
            t_set_list.append(res[i][1])
            points_set_list.append(res[i][2])
        
    return label_set, t_set_list, points_set_list

def get_min_time(data_t, flag_snapw=True):
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
            kt=0., kt2=0, mu=0.01,\
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
        range_min=0.2, range_max=3.2, opt_N_eval=11, opt_N_step=3, direct_yaw=True)
    results_sta = []
    for b_ii in range(batch_size):
        results_sta.append(results[b_ii] * label_sta[b_ii])
    
    label_sta2 = min_snap.optimize_alpha_acc_yaw( \
        data_t, results_sta, snap_w_list=snap_w_list, \
        range_min=0.2, range_max=3.2, opt_N_eval=11, opt_N_step=3, direct_yaw=True)
    results_sta2 = []
    for b_ii in range(batch_size):
        results_sta2.append(results_sta[b_ii] * label_sta2[b_ii])
    
    return snap_w_list, results_sta2

def get_min_time_sim(data_t, snap_w_list, results_sta, flag_robust=False):
    batch_size = len(data_t)
    label_sim = min_snap.optimize_alpha_acc_yaw_sim( \
        data_t, results_sta, snap_w_list=snap_w_list, \
        range_min=0.5, range_max=2.5, opt_N_eval=3, opt_N_step=5, direct_yaw=True, flag_robust=flag_robust)
    results_sim = []
    for b_ii in range(batch_size):
        results_sim.append(results_sta[b_ii] * label_sim[b_ii])
    
    if flag_robust:
        label_sim2 = min_snap.optimize_alpha_acc_yaw_sim( \
            data_t, results_sim, snap_w_list=snap_w_list, \
            range_min=0.8, range_max=1.2, opt_N_eval=3, opt_N_step=11, direct_yaw=True, flag_robust=flag_robust)
        results_sim2 = []
        for b_ii in range(batch_size):
            results_sim2.append(results_sim[b_ii] * label_sim2[b_ii])
        return results_sim2
    return results_sim

def get_data(seq_len, points_scale, batch_size, max_time, flag_sim=False, \
    len_min=0.0, len_max=30.0, \
    curv_min=5.0, curv_max=20.0):
    
    # Get feasible cube
    x_ret = []
    N_bs_scale = 2
    while len(x_ret) <= batch_size:
        points_set_tmp = []
        while len(points_set_tmp) <= N_bs_scale*batch_size:
            for b_idx in range(N_bs_scale*batch_size):
                points = np.zeros((seq_len,3))
                for d_ii in range(3):
                    points[:,d_ii] = np.random.rand(seq_len)
                res_len, res_curv = get_waypoints_profile(points)
                if (res_len >= len_min) and (res_len <= len_max) and (res_curv >= curv_min) and (res_curv <= curv_max):
                    points_set_tmp.append(points)

        labels, t_set_list, points_set_list = check_waypoints(points_set_tmp, True)
        for b_idx in range(N_bs_scale*batch_size):
            if labels[b_idx] == 1:
                points_t = np.zeros((seq_len,6))
                points_t[:,:4] = copy.deepcopy(points_set_list[b_idx])
                points_t[-1,4] = 1.
                t_set_tmp = np.array(t_set_list[b_idx])
                points_t[1:,5] = t_set_tmp / np.sum(t_set_tmp)
                x_ret.append(points_t)
        print("Seq {}: sampling points - {}/{}".format(seq_len, len(x_ret), batch_size))
    
    points_array = np.array(x_ret)
    N_data = points_array.shape[0]

    # Run sta/sim to get data
    data_set = []
    for d_ii in range(N_data):
        data_tmp = points_array[d_ii, :, :]
        for dim_ii in range(3):
            data_tmp[:,dim_ii] -= 0.5
            data_tmp[:,dim_ii] *= points_scale[dim_ii]
        data_tmp[:,3] = get_closed_turn_direct_yaw(data_tmp[:,3])
        data_set.append(data_tmp)

    o_data = []
    o_label_sta = []
    o_snapw_sta = []
    o_results_sta =[]

    snap_w_list, results_sta = get_min_time(data_set, flag_snapw=True)
    o_snapw_sta.extend(snap_w_list)
    o_results_sta.extend(results_sta)
    # print(results_sta)

    for d_ii in range(N_data):
        data_tmp = data_set[d_ii]
        data_tmp[1:,5] = results_sta[d_ii]
        data_tmp[1:,5] /= np.sum(data_tmp[1:,5])
        data_tmp = np.pad(data_tmp, ((0,0),(0,1)), mode='constant', constant_values=0)
        data_tmp[1:,6] = 1./data_tmp[1:,5]
        data_tmp[1:,6] *= 1. / np.sum(data_tmp[1:,6])

        o_data.append(data_tmp)
        o_label_sta.append(np.sum(results_sta[d_ii]))

    o_data = np.array(o_data)
    o_label_sta = np.array(o_label_sta)
    
    if not flag_sim:
        valid = np.where(o_label_sta < max_time)[0]
        # print("  - eval data : {}/{}/{}".format(valid.shape[0], N_data, batch_size))
        return o_data[valid], o_label_sta[valid], None, None
    
    if flag_sim:
        o_data_sim = []
        o_label_sim = []
        o_compare = []

        data_t = copy.deepcopy(o_data)
        data_set = list(data_t)
        results_sta = list(o_results_sta)
        snap_w_list = list(o_snapw_sta)
        results_sim = get_min_time_sim(data_set, snap_w_list, results_sta, flag_robust=False)

        for d_ii in range(N_data):
            data_tmp = data_set[d_ii]
            data_tmp[1:,5] = results_sta[d_ii]
            data_tmp[1:,5] /= np.sum(data_tmp[1:,5])
            data_tmp = np.pad(data_tmp, ((0,0),(0,1)), mode='constant', constant_values=0)
            data_tmp[1:,6] = 1./data_tmp[1:,5]
            data_tmp[1:,6] *= 1. / np.sum(data_tmp[1:,6])

            o_data_sim.append(data_tmp)
            o_label_sim.append(np.sum(results_sim[d_ii]))
            o_compare.append(np.sum(results_sta[d_ii])/np.sum(results_sim[d_ii]))

        o_data_sim = np.array(o_data_sim)
        o_label_sim = np.array(o_label_sim)
        o_compare = np.array(o_compare)
        
        valid = np.where((o_label_sta < max_time) & (o_label_sim < max_time))[0]
        # print("  - eval data : {}/{}/{}".format(valid.shape[0], N_data, batch_size))
        return o_data[valid], o_label_sta[valid], o_label_sim[valid], o_compare[valid]

def main(args):
    # N_data
    # Sta - 10000 per seqlen
    # Sim - 500 per seqlen
    # Test - Sim 100 per seqlen
    
    points_scale = np.array([9.,9.,3.])
    suffix = "_ps{}{}{}".format(int(points_scale[0]),int(points_scale[1]),int(points_scale[2]))
    N_sta = 10000
    N_sim = 500
    N_real = 5
    if args.test:
        suffix += "_test"
        N_sta = 100
        N_sim = 100
    else:
        suffix += "_train"
    
    if args.debug:
        suffix += "_debug"
        N_sta = 10
        N_sim = 10
        
    h5f_write = "../dataset/mfrl{}.h5".format(suffix)
    print(h5f_write)
    
    for p_ii in range(args.data_min_dim, args.data_max_dim+1):
        o_data_sta = []
        o_data_sim = []
        o_label_sta = []
        o_label_sim = []
        o_compare = []
        
        batch_size = args.batch_size_sim
        
        flag_sim = True
        while True:
            if len(o_data_sim) >= N_sim:
                o_data_sim = o_data_sim[:N_sim]
                o_label_sim = o_label_sim[:N_sim]
                o_compare = o_compare[:N_sim]
                flag_sim = False
                if not args.test:
                    batch_size = args.batch_size
                if len(o_data_sta) >= N_sta:
                    o_data_sta = o_data_sta[:N_sta]
                    o_label_sta = o_label_sta[:N_sta]
                    break
            if args.debug:
                batch_size = 10
            
            o_data_t, o_label_sta_t, o_label_sim_t, o_compare_t = get_data(p_ii, points_scale, batch_size, args.max_time, flag_sim)
            o_data_sta.extend(o_data_t)
            o_label_sta.extend(o_label_sta_t)
            if flag_sim:
                o_data_sim.extend(o_data_t)
                o_label_sim.extend(o_label_sim_t)
                o_compare.extend(o_compare_t)
            
            print("Seq {}: N_data sta, sim - {}/{}, {}/{}".format(p_ii, len(o_data_sta), N_sta, len(o_data_sim), N_sim))
        
        if p_ii == 5:
            h5f = h5py.File(h5f_write, 'w')
        else:
            h5f = h5py.File(h5f_write, 'a')
        grp = h5f.create_group("{}".format(p_ii))
        if not args.test:
            grp.create_dataset('points_sta', data=np.array(o_data_sta))
            grp.create_dataset('points_sim', data=np.array(o_data_sim))
            grp.create_dataset('points_real', data=np.array(o_data_sim)[:N_real,:,:])
        else:
            grp.create_dataset('points', data=np.array(o_data_sta))
            grp.create_dataset('rand_idx', data=np.arange(N_real))            
        grp.create_dataset('alpha_sta', data=np.array(o_label_sta))
        grp.create_dataset('alpha_sim', data=np.array(o_label_sim))
        grp.create_dataset('alpha_real', data=np.array(o_label_sim)[:N_real])
        h5f.close()
        
        print(" - Comp: {}, mean: {}, max: {}".format(
            np.mean(np.array(o_compare)),
            np.mean(np.array(o_label_sta)),
            np.max(np.array(o_label_sta))
        ))
        
        print(" - N data: {}, N label: {}".format(
            len(o_data_sta), 
            len(o_label_sta)
        ))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', action='store_true')
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-bs', '--batch_size', type=int, default=500)
    parser.add_argument('-bss', '--batch_size_sim', type=int, default=100)
    parser.add_argument('-mt', '--max_time', type=float, default=300)
    
    # Load training data
    parser.add_argument('-dmin', '--data_min_dim', type=int, default=5)
    parser.add_argument('-dmax', '--data_max_dim', type=int, default=14)

    args = parser.parse_args()

    main(args)
