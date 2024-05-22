#!/usr/bin/env python
# coding: utf-8

import os, sys, time, copy
import numpy as np
import yaml, h5py, shutil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from os import path
import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, '../')
from naiveBayesOpt.utils import *
from naiveBayesOpt.agents_mfme import NaiveBayesOpt
from naiveBayesOpt.multiFidelityModel_sfme import meta_low_fidelity, get_dataset_init
from naiveBayesOpt.multiFidelityModel_mfme import meta_high_fidelity

from pyTrajectoryUtils.pyTrajectoryUtils.minSnapTrajectory import *
from pyTrajectoryUtils.pyTrajectoryUtils.utils import *

class BayesOptCounter():
    def __init__(self):
        self.iter = 0
    def get_iter(self):
        return self.iter
    def set_iter(self, x):
        self.iter = x
        return
    def inc_iter(self):
        self.iter += 1
        return

def meta_high_fidelity_robot( \
    poly, traj_tool, 
    alpha_set, t_set_robot, points, \
    lb=0.6, ub=1.4, counter=None):
    
    counter.inc_iter()
    res = np.zeros((alpha_set.shape[0],2))
    for it in range(alpha_set.shape[0]):
        alpha_tmp = alpha_set[it,:]*(ub-lb) + lb
        
        t_set_tmp, d_ordered_tmp, d_ordered_yaw_tmp, snap_tmp = poly.update_traj( \
            points=points, \
            t_set=t_set_robot, \
            alpha_set=alpha_tmp[:t_set_robot.shape[0]], \
            snap_w=alpha_tmp[t_set_robot.shape[0]:], \
            yaw_mode=3, return_snap=True)
        
        d_ordered_yaw_tmp2 = np.zeros((d_ordered_yaw_tmp.shape[0], 2))
        d_ordered_yaw_tmp2[:,:1] = d_ordered_yaw_tmp
        curr_iter = counter.get_iter()
        traj_tool.save_trajectory_yaml(
            t_set_tmp, d_ordered_tmp, d_ordered_yaw_tmp2, \
            traj_dir="./bo_data_t/robot_exp_traj/", traj_name="{}_iter{}".format(sample_name_, curr_iter))
        traj_tool.save_trajectory_csv(
            t_set_tmp, d_ordered_tmp, d_ordered_yaw_tmp2, \
            traj_dir="./bo_data_t/robot_exp_traj/", traj_name="{}_iter{}".format(sample_name_, curr_iter), freq=200)
        
        while True:
            print('Enter experiment result (success:1/fail:0) :')
            x = input()
            try:
                num_success = np.float(x)
            except ValueError:
                print("Input cannot be converted to float")
                continue
            print("result: {}".format(num_success))
            if num_success == 0 or num_success == 1:
                break
            else:
                print("Wrong result")
        
        if num_success == 1:
            res[it,0] = 1
        else:
            res[it,0] = 0
        res[it,1] = snap_tmp
    
    return res

if __name__ == "__main__":
    yaml_name = 'mfbo_example'
    sample_name = ['half_space_figure8']
    drone_model = "STMCFB"
    model_path = "bo_data"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    rand_seed = [123, 124, 125, 126, 127]
    
    parser = argparse.ArgumentParser(description='mfgpc experiment')
    parser.add_argument('-l', dest='flag_load_exp_data', action='store_true', help='load exp data')
    parser.add_argument('-g', dest='flag_switch_gpu', action='store_true', help='switch gpu to gpu 1')
    parser.add_argument('-r', dest='flag_robot', action='store_true', help='run real_world experiment')
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
    MAX_ITER = int(args.max_iter)
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
    t_set_sim = copy.deepcopy(t_set_sta)*1.0
    t_set_robot = copy.deepcopy(t_set_sta)*1.0
    
    num_eval_H = 32
    if args.flag_robot:
        sample_name_ += "_robot"
        num_eval_H = 1
        
    print("Yaw_mode {}".format(yaw_mode))
    poly = MinSnapTrajectory(
        N_POINTS=40,
        drone_model=drone_model, 
        yaw_mode=yaw_mode)
    traj_tool = TrajectoryTools(
        MAX_POLY_DEG=9, 
        MAX_SYS_DEG=4, 
        N_POINTS=40)
    poly.direct_yaw_ref = True
    traj_tool.direct_yaw_ref = poly.direct_yaw_ref
    
    bo_counter = BayesOptCounter()
    
    low_fidelity = lambda x, debug=False, multicore=True: \
        meta_low_fidelity(poly, x, t_set_sta, points, points_scale, debug=debug, lb=lb, ub=ub, multicore=multicore)
    if not args.flag_robot:
        high_fidelity = lambda x, debug=False, multicore=True: \
            meta_high_fidelity(poly, x, t_set_sim, points, points_scale, debug=debug, lb=lb, ub=ub, multicore=multicore)
    else:
        high_fidelity = lambda x: \
            meta_high_fidelity_robot(poly, traj_tool, x, t_set_robot, points, lb=lb, ub=ub, counter=bo_counter)
    
    X_L, Y_L = \
        get_dataset_init( \
            sample_name_, low_fidelity, N_L=512, 
            t_set_sta=t_set_sta, model_path=model_path, 
            lb=lb, ub=ub, sampling_mode=2, batch_size=32, 
            flag_multicore=True)
    print("X_L shape: ",X_L.shape)
    
    alpha_H = np.array([
        [0.5, 0],
        [0.6, 0],
        [0.7, 0],
        [0.8, 0],
        [0.9, 0],
        [1.0, 1],
        [1.1, 1],
        [1.2, 1],
        [1.3, 1],
        [1.4, 1],
        [1.5, 1],
        [1.6, 1],
        [1.7, 1]
    ])
    Y_H = alpha_H[:,1]
    X_H = []
    for i in range(alpha_H.shape[0]):
        alpha_tmp = (alpha_H[i,0]-lb)/(ub-lb)
        X_H_tmp = np.ones(t_set_sta.shape[0]*2)
        X_H_tmp[:t_set_sta.shape[0]] *= alpha_tmp
        X_H.append(X_H_tmp)
    X_H = np.array(X_H)
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
            X_H=X_H, Y_H=Y_H, \
            lb=lb, ub=ub, rand_seed=rand_seed_, \
            C_L=1.0, C_H=10.0, \
            delta_L=0.9, delta_H=0.6, \
            beta=3.0, N_cand=16384, \
            gpu_batch_size=1024, \
            sampling_func_L=low_fidelity, \
            sampling_func_H=high_fidelity, \
            t_set_sim=t_set_robot, \
            sampling_mode=5, \
            num_eval_L=128, \
            num_eval_H=num_eval_H, \
            model_prefix=logprefix, \
            model_path="{}/{}/".format(model_path,sample_name_), \
            iter_create_model=200)
        
        path_exp_data = os.path.join(filedir, filename_exp)
        if args.flag_load_exp_data and path.exists(path_exp_data):
            bo_model.load_exp_data(filedir=filedir, filename=filename_exp)

        if hasattr(bo_model, 'start_iter'):
            bo_counter.set_iter(bo_model.start_iter)
        prGreen("curr_iter: {}".format(bo_counter.get_iter()))
        
        bo_model.active_learning( \
            N=MAX_ITER, \
            filedir=filedir, \
            filename_result=filename_res, \
            filename_exp=filename_exp)