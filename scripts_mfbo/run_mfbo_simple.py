#!/usr/bin/env python
# coding: utf-8

import os, sys, time, copy
import numpy as np
import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from os import path
import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, '../')
from naiveBayesOpt.utils import *
from naiveBayesOpt.agents_sfme import NaiveBayesOpt
from naiveBayesOpt.multiFidelityModel_sfme import meta_low_fidelity, get_dataset_init
from pyTrajectoryUtils.pyTrajectoryUtils.utils import *
from pyTrajectoryUtils.pyTrajectoryUtils.minSnapTrajectory import MinSnapTrajectory

if __name__ == "__main__":
    yaml_name = 'mfbo_example'
    sample_name = ['half_space_figure8']
    drone_model = "STMCFB"
    model_path = "bo_data"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    rand_seed = [123, 445, 678, 115, 92, 384, 992, 874, 490, 41, 83, 78, 991, 993, 994, 995, 996, 997, 998, 999]
    
    parser = argparse.ArgumentParser(description='mfgpc experiment')
    parser.add_argument('-l', dest='flag_load_exp_data', action='store_true', help='load exp data')
    parser.add_argument('-g', dest='flag_switch_gpu', action='store_true', help='switch gpu to gpu 1')
    parser.add_argument('-t', "--sample_idx", type=int, help="assign model index", default=0)
    parser.add_argument("-s", "--seed_idx", type=int, help="assign seed index", default=0)
    parser.add_argument("-y", "--yaw_mode", type=int, help="assign yaw mode", default=3)
    parser.add_argument("-m", "--max_iter", type=int, help="assign maximum iteration", default=100)
    parser.set_defaults(flag_load_exp_data=False)
    parser.set_defaults(flag_switch_gpu=False)
    args = parser.parse_args()

    if args.flag_switch_gpu:
        torch.cuda.set_device(1)
    else:
        torch.cuda.set_device(0)
    torch.autograd.set_detect_anomaly(True)

    yaw_mode = args.yaw_mode
    sample_name_ = sample_name[args.sample_idx]
    rand_seed_ = rand_seed[args.seed_idx]
    MAX_ITER = np.int(args.max_iter)
    print("MAX_ITER: {}".format(MAX_ITER))
    
    lb = 0.1
    ub = 1.9
    if yaw_mode == 0:
        flag_yaw_zero = True
    else:
        flag_yaw_zero = False
    points_scale = np.array([10., 10., 10.])
    
    print("Trajectory {}".format(sample_name_))
    points, t_set_sta = get_waypoints(yaml_name, sample_name_, flag_t_set=True)
    
    if yaw_mode == 0:
        sample_name_ += "_yaw_zero"
        
    print("Yaw_mode {}".format(yaw_mode))
    poly = MinSnapTrajectory(
        N_POINTS=40,
        drone_model=drone_model, 
        yaw_mode=yaw_mode)
    poly.direct_yaw_ref = True
    
    low_fidelity = lambda x, debug=False, multicore=True: \
        meta_low_fidelity(poly, x, t_set_sta, points, points_scale, debug=debug, lb=lb, ub=ub, multicore=multicore)
        
    X_L, Y_L = \
        get_dataset_init( \
            sample_name_, low_fidelity, N_L=512, 
            t_set_sta=t_set_sta, model_path=model_path, 
            lb=lb, ub=ub, sampling_mode=2, batch_size=32, 
            flag_multicore=True)
    
    print("Seed {}".format(rand_seed_))

    np.random.seed(rand_seed_)
    torch.manual_seed(rand_seed_)
    
    # MFBO
    fileprefix = "bo_MaxIter{}".format(MAX_ITER)
    filedir = './{}/{}'.format(model_path,sample_name_)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    logprefix = '{}/{}/{}'.format(sample_name_, fileprefix, rand_seed_)
    filename_res = 'result_{}_{}.yaml'.format(fileprefix, rand_seed_)
    filename_exp = 'exp_data_{}_{}.yaml'.format(fileprefix, rand_seed_)
    res_path = os.path.join(filedir, filename_res)
    print(res_path)
    flag_check = check_result_data(filedir,filename_res,MAX_ITER)
    if not flag_check:
        bo_model = NaiveBayesOpt( \
            X_L=X_L, Y_L=Y_L, \
            lb=lb, ub=ub, rand_seed=rand_seed_, \
            C_L=1.0, delta_L=0.9, \
            beta=3.0, N_cand=16384, \
            gpu_batch_size=1024, \
            sampling_func=low_fidelity, \
            t_set_sta=t_set_sta, \
            sampling_mode=5, \
            num_eval_L=128, \
            model_prefix=logprefix, \
            model_path="{}/{}/".format(model_path,sample_name_), \
            iter_create_model=200)
        
        path_exp_data = os.path.join(filedir, filename_exp)
        if args.flag_load_exp_data and path.exists(path_exp_data):
            bo_model.load_exp_data(filedir=filedir, filename=filename_exp)

        bo_model.active_learning( \
            N=MAX_ITER, \
            filedir=filedir, \
            filename_result=filename_res, \
            filename_exp=filename_exp)