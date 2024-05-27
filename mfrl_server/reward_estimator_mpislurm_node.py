#!/usr/bin/env python
# coding: utf-8

import os, sys, io, random, shutil
import json
import time
import argparse
import numpy as np
import yaml, copy, h5py

sys.path.insert(0, '../')
import pathlib
p = os.path.dirname(pathlib.Path(__file__).parent.resolve())
sys.path.insert(0, p)

from pyTrajectoryUtils.pyTrajectoryUtils.minSnapTrajectory import *

import pickle
from mpi4py import MPI

from naiveBayesOpt.models import *
from mfrl.training_utils import *

min_snap = MinSnapTrajectory(drone_model="STMCFB", N_POINTS=40)

def eval_offline_L(points_list, t_set_list, snap_w_list, ep):
    res = min_snap.sanity_check_acc_yaw( \
        points_list, t_set_list, snap_w_list, direct_yaw=True, flag_sta=True)
    return [res, 0, ep]
def eval_offline_H(points_list, t_set_list, snap_w_list, ep):
    res = min_snap.sanity_check_acc_yaw( \
        points_list, t_set_list, snap_w_list, direct_yaw=True, flag_sta=False)
    return [res, 0, ep]
def eval_offline_test(points_list, t_set_list, snap_w_list, ep):
    res = min_snap.sanity_check_acc_yaw( \
        points_list, t_set_list, snap_w_list, direct_yaw=True, flag_sta=False)
    return [res, 1, ep]
eval_funcs_offline = [eval_offline_L, eval_offline_H]

def eval_online_L(points_list, idx_new_list, t_set_list, snap_w_list, ep):
    res = min_snap.sanity_check_acc_yaw_online( \
        points_list, idx_new_list, t_set_list, snap_w_list, direct_yaw=True, flag_sta=True)
    return [res, 0, ep]
def eval_online_H(points_list, idx_new_list, t_set_list, snap_w_list, ep):
    res = min_snap.sanity_check_acc_yaw_online( \
        points_list, idx_new_list, t_set_list, snap_w_list, direct_yaw=True, flag_sta=False)
    return [res, 0, ep]
def eval_online_test(points_list, idx_new_list, t_set_list, snap_w_list, ep):
    res = min_snap.sanity_check_acc_yaw_online( \
        points_list, idx_new_list, t_set_list, snap_w_list, direct_yaw=True, flag_sta=False)
    return [res, 1, ep]
def eval_online_test_wp(points_list, idx_new_list, t_set_list, snap_w_list, ep):
    res = min_snap.sanity_check_acc_yaw_online( \
        points_list, idx_new_list, t_set_list, snap_w_list, direct_yaw=True, flag_sta=False, flag_wp_update=True)
    return [res, 1, ep]
eval_funcs_online = [eval_online_L, eval_online_H]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--flag_online', action='store_true')
    parser.add_argument('-t', '--eval_type', type=int, default=-1)
    parser.add_argument('-d', '--tmp_dir', type=str, default='./tmp')
    parser.add_argument('-i', '--idx', type=int, default=0)
    parser.add_argument('-nb', '--num_batch', type=int, default=100)
    args = parser.parse_args()
    
    # print("tmp_dir: {}".format(args.tmp_dir))
    if not os.path.exists(args.tmp_dir):
        sys.exit()
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    
    if rank == 0:
        data = []
        flag_file_exist = True
        for b_ii in range(args.num_batch):
            data_path = "{}/eval_{}_{}.pkl".format(args.tmp_dir, args.eval_type, args.idx + b_ii)
            if os.path.exists(data_path):
                with open(data_path, 'rb') as handle:
                    data_t = pickle.load(handle)
                data.append(data_t)
            else:
                print("Failed to load: {}".format(data_path))
                flag_file_exist = False
        if not flag_file_exist:
            sys.exit()
        print("nprocs: {}/{}/{}".format(nprocs,len(data),args.num_batch))
    else:
        data = None

    data_t = comm.scatter(data, root=0)
            
    if args.flag_online:
        if args.eval_type == -1: # eval_test
            res = eval_online_test(data_t[0], data_t[1], data_t[2], data_t[3], data_t[4])
        elif args.eval_type == -2: # eval_test_wp
            res = eval_online_test_wp(data_t[0], data_t[1], data_t[2], data_t[3], data_t[4])
        else:
            res = eval_funcs_online[args.eval_type](data_t[0], data_t[1], data_t[2], data_t[3], data_t[4])
    else:
        if args.eval_type == -1: # eval_test
            res = eval_offline_test(data_t[0], data_t[1], data_t[2], data_t[3])
        else:
            res = eval_funcs_offline[args.eval_type](data_t[0], data_t[1], data_t[2], data_t[3])
    
    res = comm.gather(res, root=0)

    if rank == 0:
        data_res_path = "{}/eval_{}_{}_res.pkl".format(args.tmp_dir, args.eval_type, args.idx)
        print("Saved data to : {}".format(data_res_path))
        with open(data_res_path, 'wb') as handle:
            pickle.dump(res, handle, protocol=4)
