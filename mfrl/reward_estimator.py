#!/usr/bin/env python
# coding: utf-8

import os, sys, io, random, shutil
import json
import time
import argparse
import numpy as np
import yaml, copy, h5py
from multiprocessing import cpu_count
from tensorboardX import SummaryWriter
from collections import OrderedDict, defaultdict
import matplotlib.pyplot as plt
import PIL.Image
from torchvision.transforms import ToTensor
import pickle

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Normal
from torch.utils.data import DataLoader
from torch.autograd import Variable

from pyTrajectoryUtils.pyTrajectoryUtils.minSnapTrajectory import *
from pyTrajectoryUtils.pyTrajectoryUtils.utils import TrajectoryTools
from mfrl.network_models import RewEncoder as GPEnc
from mfrl.network_models import WaypointsEncDec
from mfrl.training_utils import *

from naiveBayesOpt.models import *
from mfrl_server.reward_estimator_client import RewardEstimatorClient

# from threading import Thread, Lock
# from multiprocessing import Pool, Pipe, TimeoutError, Process
from pathos.pp import ParallelPool
from pathos.multiprocessing import ProcessingPool

traj_tool = TrajectoryTools(drone_model="STMCFB", N_POINTS=40)
min_snap = MinSnapTrajectory(drone_model="STMCFB", N_POINTS=40)
MSE = torch.nn.MSELoss(reduction ='sum')

def eval_L(points_list, idx_new_list, t_set_list, snap_w_list, ep):
    res = min_snap.sanity_check_acc_yaw_online( \
        points_list, idx_new_list, t_set_list, snap_w_list, direct_yaw=True, flag_sta=True, flag_wp_update=True)
    # res = np.random.randint(2, size=len(points_list))
    return [res, 0, ep]
def eval_H(points_list, idx_new_list, t_set_list, snap_w_list, ep):
    res = min_snap.sanity_check_acc_yaw_online( \
        points_list, idx_new_list, t_set_list, snap_w_list, direct_yaw=True, flag_sta=False, flag_wp_update=True)
    # res = np.random.randint(2, size=len(points_list))
    return [res, 0, ep]
def eval_test(points_list, idx_new_list, t_set_list, snap_w_list, ep):
    res = min_snap.sanity_check_acc_yaw_online( \
        points_list, idx_new_list, t_set_list, snap_w_list, direct_yaw=True, flag_sta=False)
    # res = np.random.randint(2, size=len(points_list))
    return [res, 1, ep]
def eval_test_wp(points_list, idx_new_list, t_set_list, snap_w_list, ep):
    res = min_snap.sanity_check_acc_yaw_online( \
        points_list, idx_new_list, t_set_list, snap_w_list, direct_yaw=True, flag_sta=False, flag_wp_update=True)
    return [res, 1, ep]

def eval_R(points_list, idx_new_list, t_set_list, snap_w_list, ep, flag_wp_update=True):
    DATA_DIR = "../logs/mfrl_robot_eval_data"
    TRAJ_DIR_T = "{}/traj_{}".format(DATA_DIR, ep)
    if not os.path.exists(TRAJ_DIR_T):
        os.makedirs(TRAJ_DIR_T)
    for r_ii in range(len(points_list)):
        if flag_wp_update and isinstance(points_list[r_ii], (list)):
            points = points_list[r_ii][0]
            points_new = points_list[r_ii][1]
        else:
            points = points_list[r_ii]
            points_new = copy.deepcopy(points)
        t_set_i = t_set_list[r_ii][0,:]
        if len(snap_w_list) > 0:
            snap_w_i = snap_w_list[r_ii][0,:]
            snap_w_f = snap_w_list[r_ii][1,:]
        else:
            snap_w_i = np.ones_like(t_set)
            snap_w_f = np.ones_like(t_set)

        ##########################################
        _, d_ordered = min_snap.obj_func_acc(points, t_set_i, flag_loop=False, snap_w=snap_w_i)
        _, d_ordered_yaw = min_snap.obj_func_acc(points, t_set_i, flag_loop=False, snap_w=snap_w_i, flag_yaw=True, flag_direct_yaw=True)
        d_init = d_ordered[5*idx_new_list[r_ii]+1:5*(idx_new_list[r_ii]+1),:]
        d_yaw_init = d_ordered_yaw[3*idx_new_list[r_ii]+1:3*(idx_new_list[r_ii]+1),0:1]
        points_opt = points_new[idx_new_list[r_ii]:,:]
        t_set_opt = t_set_list[r_ii][1,idx_new_list[r_ii]:]
        snap_w_opt = snap_w_f[idx_new_list[r_ii]:]

        _, d_ordered_opt = min_snap.obj_func_acc(points_opt, t_set_opt, flag_loop=False, snap_w=snap_w_opt, b_ext_init=d_init)
        _, d_ordered_yaw_opt = min_snap.obj_func_acc(points_opt, t_set_opt, flag_loop=False, snap_w=snap_w_opt, \
                                                   flag_yaw=True, flag_direct_yaw=True, b_ext_init=d_yaw_init)

        t_set = t_set_list[r_ii][1,:]
        d_ordered[5*idx_new_list[r_ii]:,:] = d_ordered_opt
        d_ordered_yaw[3*idx_new_list[r_ii]:,:] = d_ordered_yaw_opt
        V_t = min_snap.generate_sampling_matrix(t_set, N=min_snap.N_POINTS, der=0, endpoint=True)
        val = V_t.dot(d_ordered)
        val_mean = (np.max(val, axis=0)+np.min(val, axis=0))/2.
        d_ordered_yaw_t = np.zeros((d_ordered_yaw.shape[0],2))
        d_ordered_yaw_t[:,0] = d_ordered_yaw[:,0]
        
        for k in range(points.shape[0]):
            d_ordered[5*k, :] -= val_mean
        traj_tool.save_trajectory_yaml(t_set, d_ordered, d_ordered_yaw_t, \
            traj_dir=TRAJ_DIR_T, \
            traj_name="ep_{}_traj_{}".format(ep, r_ii))
        ##########################################
    
    while True:
        print('Continue? (Yes:1/No:0) :')
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
    
    RES_DIR_T = "{}/ep_{}_res.txt".format(DATA_DIR, ep)
    df = pd.read_csv(RES_DIR_T, delim_whitespace=True)
    print(df)
    res = df.to_numpy().reshape(-1)
    return [res, 0, ep]

def eval_flyable(points_list, idx_new_list, t_set_list, snap_w_list, flag_wp_update=True):
    res_flyable = np.zeros(len(points_list))
    room_size = np.array([9, 9, 3])
    xy_margin = 0.2
    for r_ii in range(len(points_list)):
        if flag_wp_update:
            points = points_list[r_ii][0]
            points_new = points_list[r_ii][1]
        else:
            points = points_list[r_ii]
            points_new = copy.deepcopy(points)
        t_set_i = t_set_list[r_ii][0,:]
        if len(snap_w_list) > 0:
            snap_w_i = snap_w_list[r_ii][0,:]
            snap_w_f = snap_w_list[r_ii][1,:]
        else:
            snap_w_i = np.ones_like(t_set)
            snap_w_f = np.ones_like(t_set)

        ##########################################
        _, d_ordered = min_snap.obj_func_acc(points, t_set_i, flag_loop=False, snap_w=snap_w_i)
        _, d_ordered_yaw = min_snap.obj_func_acc(points, t_set_i, flag_loop=False, snap_w=snap_w_i, flag_yaw=True, flag_direct_yaw=True)
        d_init = d_ordered[5*idx_new_list[r_ii]+1:5*(idx_new_list[r_ii]+1),:]
        d_yaw_init = d_ordered_yaw[3*idx_new_list[r_ii]+1:3*(idx_new_list[r_ii]+1),0:1]
        points_opt = points_new[idx_new_list[r_ii]:,:]
        t_set_opt = t_set_list[r_ii][1,idx_new_list[r_ii]:]
        snap_w_opt = snap_w_f[idx_new_list[r_ii]:]

        _, d_ordered_opt = min_snap.obj_func_acc(points_opt, t_set_opt, flag_loop=False, snap_w=snap_w_opt, b_ext_init=d_init)
        _, d_ordered_yaw_opt = min_snap.obj_func_acc(points_opt, t_set_opt, flag_loop=False, snap_w=snap_w_opt, \
                                                   flag_yaw=True, flag_direct_yaw=True, b_ext_init=d_yaw_init)

        t_set = t_set_list[r_ii][1,:]
        d_ordered[5*idx_new_list[r_ii]:,:] = d_ordered_opt
        d_ordered_yaw[3*idx_new_list[r_ii]:,:] = d_ordered_yaw_opt
        V_t = min_snap.generate_sampling_matrix(t_set, N=min_snap.N_POINTS, der=0, endpoint=True)
        val = V_t.dot(d_ordered)
        val_mean = (np.max(val, axis=0)+np.min(val, axis=0))/2.
        val_t = val-val_mean
        if np.all(val_t[:,0] > -room_size[0]/2.+xy_margin) and np.all(val_t[:,0] < room_size[0]/2.-xy_margin) and \
            np.all(val_t[:,1] > -room_size[1]/2.+xy_margin) and np.all(val_t[:,1] < room_size[1]/2.-xy_margin) and \
            np.all(val_t[:,2] > -room_size[2]/2.) and np.all(val_t[:,2] < room_size[2]/2.):
            res_flyable[r_ii] = 1
    return res_flyable

eval_funcs = [eval_L, eval_H, eval_R]

def eval_dataset(
    gp_enc_args, dis_model_args, state_dict, data_path, 
    num_fidelity=2, batch_size=200):
    
    gp_enc = GPEnc(**gp_enc_args).cuda()
    gp_enc.load_state_dict(state_dict[0])
    
    dis_model = MFCondS2SDeepGPC(dis_model_args[0], gp_enc).cuda()
    dis_model.load_state_dict(state_dict[1])
    
    res_data = []
    train_dataset = []
    h5f = h5py.File(data_path, 'r')
    for f_ii in range(num_fidelity):
        train_dataset.append(TensorDataset(
            np2cuda(h5f[str(f_ii)]["X"]), 
            np2cuda(h5f[str(f_ii)]["len"]), 
            np2cuda(h5f[str(f_ii)]["bpoly"])))
    h5f.close()
    
    for f_ii in range(num_fidelity):
        data_loader_tmp = DataLoader(train_dataset[f_ii], batch_size, shuffle=False)
        est_data = [[], [], []]
        for minibatch_i, (x_batch, len_batch, bpoly_batch) in enumerate(data_loader_tmp):
            m, v, pm = dis_model.predict_proba([x_batch.cuda(), len_batch.cuda(), bpoly_batch.cuda()], fidelity=f_ii+1)
            est_data[0].extend(pm[:,1])
            est_data[1].extend(m)
            est_data[2].extend(v)
        est_data = np.array(est_data).T
        res_data.append(est_data)
    return res_data

class RewardEstimator():
    def __init__(self, 
            gp_enc, gp_enc_args=dict(),
            num_fidelity=3,
            latent_size=32, 
            min_seq_len=5, max_seq_len=14,
            batch_size=256,
            num_inducing=128,
            log_path="../logs/tmp",
            points_scale=np.array([10.,10.,10.]),
            load_dataset=False, load_ep=0, 
            coef_wp=1., coef_kl=0.001, coef_gp=0.001,
            flag_eval_zmq=False, zmq_server_ip="tcp://localhost:1234",
            flag_test=False,
            al_type=0, il_type=0,
            coef_reg=0, coef_reg_real=0, bs_est=500,
            rew_max=0.2, rew_min=-1.5,
            rew_bias=0.2, flag_eval_zmq_testonly=False
        ):
        
        self.num_fidelity = num_fidelity
        self.num_inducing = num_inducing
        self.points_scale = points_scale
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.mean_spd = 4.
        self.max_spd = 20.
        self.tags = ["L", "H", "R"]
        self.num_update_ep = 200
        # self.max_bo_data_ratio = 10
        # self.max_bo_data = self.max_bo_data_ratio * self.num_update_ep * self.batch_size
        # self.max_bo_data = 400
        self.max_bo_data = 40000
        self.gp_enc_args = gp_enc_args
        self.flag_data_managing = True
        self.flag_eval_zmq = flag_eval_zmq
        self.flag_eval_zmq_testonly = flag_eval_zmq_testonly
        if self.flag_eval_zmq:
            self.eval_client = RewardEstimatorClient(sever_ip=zmq_server_ip)
        
        self.rew_max = rew_max
        self.rew_min = rew_min
        self.rew_bias = rew_bias
        self.coef_wp = coef_wp
        self.coef_kl = coef_kl
        self.coef_gp = coef_gp
        self.coef_reg = coef_reg
        self.coef_reg_real = coef_reg_real
        
        self.log_path = log_path
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.training_data_path = "{}/est_data.h5".format(self.log_path)
        self.training_data_init_path = "{}/est_data_init.h5".format(self.log_path)
        self.training_data_all_path = "{}/est_data_all.h5".format(self.log_path)
        self.inducing_points_filedir = "{}/inducing_points_data.h5".format(self.log_path)
        self.inducing_points_init_filedir = "{}/inducing_points_data_init.h5".format(self.log_path)
        
        # self.learning_rate = 1e-3
        self.learning_rate = 1e-4
        # self.learning_rate_decay = 0.9995
        self.learning_rate_decay = 1.0
        self.print_every = 50
        self.max_eval_proc = 128
        
        self.al_type = al_type
        self.il_type = il_type
        
        self.ep_idx = 0
        
        self.eval_th_set = []
        for f_ii in range(self.num_fidelity):
            self.eval_th_set.append([])
        self.eval_th_pool = ParallelPool(self.max_eval_proc)
        # self.eval_th_pool = ProcessingPool(self.max_eval_proc)
        
        self.test_th_set = []
        self.test_th_pool = ParallelPool(self.max_eval_proc)
        # self.test_th_pool = ProcessingPool(self.max_eval_proc)
        
        self.eval_dataset_th = None
        self.eval_dataset_th_pool = ParallelPool(self.max_eval_proc)
        
        self.train_data = dict()
        self.train_data_new = dict()
        self.train_data_fail = dict()
        for f_ii in range(self.num_fidelity):
            self.train_data[str(f_ii)] = dict()
            self.train_data_new[str(f_ii)] = dict()
            self.train_data_fail[str(f_ii)] = dict()
        self.test_data = dict()
        self.test_data["rew"] = np.empty(0)
        self.test_data["res"] = np.empty(0)
        self.test_data["fail"] = np.empty(0)
        
        self.load_dataset = load_dataset
        
        self.flag_initialized = False
        self.gp_enc = gp_enc.cuda()
        
        self.bs_est = bs_est
        ###############################################
        if self.load_dataset:
            if not flag_test:
                training_data_path_cp = "{}/est_data_ep{}.h5".format(self.log_path, load_ep)
                shutil.copyfile(training_data_path_cp, self.training_data_path)
            inducing_points_filedir_cp = "{}/inducing_points_data_ep{}.h5".format(self.log_path, load_ep)
            shutil.copyfile(inducing_points_filedir_cp, self.inducing_points_filedir)
            
            self.flag_initialized = True
            self.load_inducing_points()
            self.create_model()
            # self.load_train_data()
            self.init_train_data()
            if self.flag_data_managing and not flag_test and self.il_type == 0:
                self.run_eval_dataset_th()
        ###############################################
        
        self.reset_real_queue()
        
        # self.cov_set = dict()
        # with open('../dataset/min_jerk_cov.pkl', 'rb') as handle:
        #     self.cov_set = pickle.load(handle)
        
        # def get_bpoly_mat(N = 4):
        #     A = np.zeros((2*(N+1),2*(N+1)))
        #     for i in range(N+1):
        #         n = np.ones(i+1) * i
        #         k = np.arange(0,i+1)
        #         A[i,:i+1] = comb(n, k, exact=False)
        #         A[2*(N+1)-i-1,2*(N+1)-i-1:] = A[i,:i+1]
        #     B = - copy.deepcopy(A)
        #     for i in range(N+1):
        #         for j in range(i+1):
        #             A[i,j] *= (-1) ** (i+1-j)
        #             A[2*(N+1)-i-1,2*(N+1)-j-1] *= (-1) ** (i+1-j)
        #     return [B, A]
        # self.alpha_bpoly_pos = get_bpoly_mat(N = 4)
        # self.alpha_bpoly_yaw = get_bpoly_mat(N = 2)
        
        return
    
    def reset_real_queue(self):
        self.eval_real_queue = dict()
        self.eval_real_queue["points"] = []
        self.eval_real_queue["idx_new"] = []
        self.eval_real_queue["t_set"] = []
        self.eval_real_queue["snap_w"] = []
        self.eval_real_queue["acq"] = []
    
    def create_model(self):
        train_z = []
        for f_ii in range(self.num_fidelity):
            train_z.append(np2cuda2(self.ip_data[f_ii]))
        self.dis_model = MFCondS2SDeepGPC(train_z, self.gp_enc).cuda()
        
        opt_params = list(self.gp_enc.parameters()) + list(self.dis_model.parameters())
        # opt_params = list(self.dis_model.parameters())
        self.optimizer = torch.optim.Adam(opt_params, lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=self.learning_rate_decay)
        
        self.mll = VariationalELBO(self.dis_model.likelihood, self.dis_model, self.batch_size)
        return
    
    def copy_train_data(self, ep):
        training_data_path_cp = "{}/est_data_ep{}.h5".format(self.log_path, ep)
        inducing_points_filedir_cp = "{}/inducing_points_data_ep{}.h5".format(self.log_path, ep)
        shutil.copyfile(self.training_data_path, training_data_path_cp)
        shutil.copyfile(self.inducing_points_filedir, inducing_points_filedir_cp)
        
        s_unit = 5
        margin = 20
        for ep_old in range(ep-margin, 0, -s_unit):
            training_data_path_cp_old = "{}/est_data_ep{}.h5".format(self.log_path, ep_old)
            if os.path.isfile(training_data_path_cp_old):
                try:
                    os.remove(training_data_path_cp_old)
                except:
                    print("failed to remove {}".format(training_data_path_cp_old))
        return
    
    def copy_train_data_tmp(self, ep):
        training_data_path_cp = "{}/est_data_ep{}_tmp.h5".format(self.log_path, ep)
        inducing_points_filedir_cp = "{}/inducing_points_data_ep{}_tmp.h5".format(self.log_path, ep)
        shutil.copyfile(self.training_data_path, training_data_path_cp)
        shutil.copyfile(self.inducing_points_filedir, inducing_points_filedir_cp)
        
        s_unit = 5
        margin = 20
        for ep_old in range(ep-margin, 0, -s_unit):
            training_data_path_cp_old = "{}/est_data_ep{}_tmp.h5".format(self.log_path, ep_old)
            if os.path.isfile(training_data_path_cp_old):
                try:
                    os.remove(training_data_path_cp_old)
                except:
                    print("failed to remove {}".format(training_data_path_cp_old))
        return
    
    # Save/Load initial dataset for the inducing points
    def save_inducing_points(self):
        if self.flag_initialized:
            self.ip_data = self.dis_model.get_inducing_points()
        h5f = h5py.File(self.inducing_points_filedir, 'w')
        for f_ii in range(self.num_fidelity):
            h5f.create_dataset("{}".format(f_ii), data=self.ip_data[f_ii], 
                maxshape=(None, self.ip_data[f_ii].shape[1]), chunks=True)
        h5f.close()
        if not self.flag_initialized:
            shutil.copyfile(self.inducing_points_filedir, self.inducing_points_init_filedir)
        return
    
    def build_inducing_points(self):
        self.ip_data = []
        for f_ii in range(self.num_fidelity):
            idx_p = np.where(self.train_data_new[str(f_ii)]["Y"] == 1)[0]
            idx_n = np.where(self.train_data_new[str(f_ii)]["Y"] == 0)[0]
            if idx_p.shape[0] < idx_n.shape[0]:
                num_p = min(idx_p.shape[0], int(self.num_inducing/2.))
                num_n = self.num_inducing - num_p
            else:
                num_n = min(idx_n.shape[0], int(self.num_inducing/2.))
                num_p = self.num_inducing - num_n
            print("Build inducing points X_{} - {}/{}/{}".format(self.tags[f_ii],self.train_data_new[str(f_ii)]["Y"].shape[0], num_p, num_n))
            idx_p_arr = idx_p[np.argsort(self.train_data_new[str(f_ii)]["info"][idx_p,0]).astype(np.int32)[:num_p]]
            idx_n_arr = idx_n[np.argsort(self.train_data_new[str(f_ii)]["info"][idx_n,0]).astype(np.int32)[:num_n]]
            ip_x = np2cuda2(np.concatenate([
                self.train_data_new[str(f_ii)]["X"][idx_p_arr, :, :], 
                self.train_data_new[str(f_ii)]["X"][idx_n_arr, :, :]], axis=0))
            ip_y = np2cuda2(np.concatenate([
                self.train_data_new[str(f_ii)]["Y"][idx_p_arr], 
                self.train_data_new[str(f_ii)]["Y"][idx_n_arr]], axis=0))
            ip_len = np2cuda2(np.concatenate([
                self.train_data_new[str(f_ii)]["len"][idx_p_arr], 
                self.train_data_new[str(f_ii)]["len"][idx_n_arr]], axis=0))
            ip_bpoly = np2cuda2(np.concatenate([
                self.train_data_new[str(f_ii)]["bpoly"][idx_p_arr, :], 
                self.train_data_new[str(f_ii)]["bpoly"][idx_n_arr, :]], axis=0))
            ip_z, _ = self.gp_enc(torch.swapaxes(ip_x, 0, 1), ip_len, ip_bpoly)
            if f_ii > 0:
                ip_z = torch.cat([ip_z, ip_y_prev.unsqueeze(1)], axis=1)
            ip_y_prev = ip_y
            self.ip_data.append(ip_z.cpu().detach().numpy())
        self.save_inducing_points()
        return
    
    def load_inducing_points_pre(self, path):
        self.ip_data = []
        h5f = h5py.File(path, 'r')
        for f_ii in range(self.num_fidelity):
            self.ip_data.append(np.array(h5f["{}".format(f_ii)]))
        h5f.close()
        return
    
    def load_inducing_points(self):
        self.ip_data = []
        h5f = h5py.File(self.inducing_points_filedir, 'r')
        for f_ii in range(self.num_fidelity):
            self.ip_data.append(np.array(h5f["{}".format(f_ii)]))
        h5f.close()
        return
        
    def init_train_data(self):
        # Check train data size        
        h5f = h5py.File(self.training_data_path, 'r')
        for f_ii in range(self.num_fidelity):
            self.train_data[str(f_ii)]["X"] = np.array(h5f[str(f_ii)]["X"])[0:1,:,:]
            self.train_data[str(f_ii)]["Y"] = np.array(h5f[str(f_ii)]["Y"])[0:1]
            self.train_data[str(f_ii)]["bpoly"] = np.array(h5f[str(f_ii)]["bpoly"])[0:1,:]
            self.train_data[str(f_ii)]["len"] = np.array(h5f[str(f_ii)]["len"])[0:1]
            self.train_data[str(f_ii)]["imp"] = np.array(h5f[str(f_ii)]["imp"])[0:1]
        h5f.close()
        
        for f_ii in range(self.num_fidelity):
            self.train_data_new[str(f_ii)]["X"] = np.empty((0, \
                self.train_data[str(f_ii)]["X"].shape[1], \
                self.train_data[str(f_ii)]["X"].shape[2]))
            self.train_data_new[str(f_ii)]["Y"] = np.empty(0)
            self.train_data_new[str(f_ii)]["bpoly"] = np.empty((0, \
                self.train_data[str(f_ii)]["bpoly"].shape[1]))
            self.train_data_new[str(f_ii)]["len"] = np.empty(0)
            self.train_data_new[str(f_ii)]["info"] = np.empty((0,3))
            self.train_data_new[str(f_ii)]["imp"] = np.empty(0)
            
            self.train_data_fail[str(f_ii)]["X"] = np.empty((0, \
                self.train_data[str(f_ii)]["X"].shape[1], \
                self.train_data[str(f_ii)]["X"].shape[2]))
            self.train_data_fail[str(f_ii)]["Y"] = np.empty(0)
            self.train_data_fail[str(f_ii)]["bpoly"] = np.empty((0, \
                self.train_data[str(f_ii)]["bpoly"].shape[1]))
            self.train_data_fail[str(f_ii)]["len"] = np.empty(0)
            self.train_data_fail[str(f_ii)]["info"] = np.empty((0,3))
            self.train_data_fail[str(f_ii)]["imp"] = np.empty(0)
        return
    
    def load_train_data(self):
        h5f = h5py.File(self.training_data_path, 'r')
        for f_ii in range(self.num_fidelity):
            N_data = int(np.array(h5f["N_data"])[f_ii])
            N_data_batch = self.num_update_ep * self.batch_size
            N_data_tmp = min(N_data, N_data_batch)
            perm_idx = np.random.permutation(N_data)[:N_data_tmp]
            self.train_data[str(f_ii)]["X"] = np.array(h5f[str(f_ii)]["X"])[list(perm_idx),:,:]
            self.train_data[str(f_ii)]["Y"] = np.array(h5f[str(f_ii)]["Y"])[list(perm_idx)]
            self.train_data[str(f_ii)]["bpoly"] = np.array(h5f[str(f_ii)]["bpoly"])[list(perm_idx),:]
            self.train_data[str(f_ii)]["len"] = np.array(h5f[str(f_ii)]["len"])[list(perm_idx)]
            self.train_data[str(f_ii)]["imp"] = np.array(h5f[str(f_ii)]["imp"])[list(perm_idx)]
        h5f.close()
        
        self.train_dataset = []
        for f_ii in range(self.num_fidelity):
            self.train_dataset.append(TensorDataset(
                np2cuda(self.train_data[str(f_ii)]["X"]), 
                np2cuda(self.train_data[str(f_ii)]["Y"]), 
                np2cuda(self.train_data[str(f_ii)]["len"]), 
                np2cuda(self.train_data[str(f_ii)]["bpoly"])))
        return
    
    def save_train_data(self, ep=0):
        save_data_info = np.zeros(3*self.num_fidelity)
        if not self.flag_initialized:
            print("Save dataset")
            
            h5f = h5py.File(self.training_data_path, 'w')
            N_data_set = np.zeros(self.num_fidelity)
            for f_ii in range(self.num_fidelity):
                idx_p = np.where(self.train_data_new[str(f_ii)]["Y"] == 1)[0]
                idx_n = np.where(self.train_data_new[str(f_ii)]["Y"] == 0)[0]
                # num_pn = np.array([idx_p.shape[0], idx_n.shape[0]])
                print("save_train_data X_{} - all/pos/neg - {}/{}/{}".format(self.tags[f_ii], self.train_data_new[str(f_ii)]["Y"].shape[0], idx_p.shape[0], idx_n.shape[0]))
                save_data_info[3*f_ii] = idx_p.shape[0]
                save_data_info[3*f_ii+1] = idx_n.shape[0]
                if self.train_data_new[str(f_ii)]["X"].shape[0] > self.max_bo_data and self.il_type < 2:
                    if idx_p.shape[0] < idx_n.shape[0]:
                        num_p = min(idx_p.shape[0], int(self.max_bo_data/2.))
                        num_n = self.max_bo_data - num_p
                    else:
                        num_n = min(idx_n.shape[0], int(self.max_bo_data/2.))
                        num_p = self.max_bo_data - num_n
                    idx_p_arr = idx_p[np.argsort(self.train_data_new[str(f_ii)]["info"][idx_p,0]).astype(np.int32)[:num_p]]
                    idx_n_arr = idx_n[np.argsort(self.train_data_new[str(f_ii)]["info"][idx_n,0]).astype(np.int32)[:num_n]]
                    ip_x = np.concatenate([
                        self.train_data_new[str(f_ii)]["X"][idx_p_arr, :, :], 
                        self.train_data_new[str(f_ii)]["X"][idx_n_arr, :, :]], axis=0)
                    ip_y = np.concatenate([
                        self.train_data_new[str(f_ii)]["Y"][idx_p_arr], 
                        self.train_data_new[str(f_ii)]["Y"][idx_n_arr]], axis=0)
                    ip_bpoly = np.concatenate([
                        self.train_data_new[str(f_ii)]["bpoly"][idx_p_arr, :], 
                        self.train_data_new[str(f_ii)]["bpoly"][idx_n_arr, :]], axis=0)
                    ip_len = np.concatenate([
                        self.train_data_new[str(f_ii)]["len"][idx_p_arr], 
                        self.train_data_new[str(f_ii)]["len"][idx_n_arr]], axis=0)
                    ip_imp = np.concatenate([
                        self.train_data_new[str(f_ii)]["imp"][idx_p_arr], 
                        self.train_data_new[str(f_ii)]["imp"][idx_n_arr]], axis=0)
                else:
                    ip_x = self.train_data_new[str(f_ii)]["X"]
                    ip_y = self.train_data_new[str(f_ii)]["Y"]
                    ip_bpoly = self.train_data_new[str(f_ii)]["bpoly"]
                    ip_len = self.train_data_new[str(f_ii)]["len"]
                    ip_imp = self.train_data_new[str(f_ii)]["imp"]
                
                grp = h5f.create_group("{}".format(f_ii))
                grp.create_dataset('X', data=ip_x, maxshape=(None, ip_x.shape[1], ip_x.shape[2]), chunks=True)
                grp.create_dataset('Y', data=ip_y, maxshape=(None,), chunks=True)
                grp.create_dataset('bpoly', data=ip_bpoly, maxshape=(None, ip_bpoly.shape[1]), chunks=True)
                grp.create_dataset('len', data=ip_len, maxshape=(None,), chunks=True)
                grp.create_dataset('imp', data=ip_imp, maxshape=(None,), chunks=True)
                # grp.create_dataset('N_pn', data=num_pn)
                N_data_set[f_ii] = ip_x.shape[0]
            h5f.create_dataset("N_data", data=N_data_set)
            h5f.close()
            shutil.copyfile(self.training_data_path, self.training_data_init_path)
        else:
            if ep % self.bs_est == 0 and ep > 0 and self.il_type >= 2:
                training_data_path_old = "{}/est_data_old_{}.h5".format(self.log_path, ep-self.bs_est)
                shutil.copyfile(self.training_data_path, training_data_path_old)
                
                h5f = h5py.File(self.training_data_path, 'a')
                for f_ii in range(self.num_fidelity):
                    N_data = int(np.array(h5f["N_data"])[f_ii])
                    N_data_new = self.train_data_new[str(f_ii)]["X"].shape[0]
                    N_data_fail = self.train_data_fail[str(f_ii)]["X"].shape[0]
                    print("training_data: X_{} - saved/new/fail - {}/{}/{}".format(self.tags[f_ii], N_data, N_data_new, N_data_fail))
                    idx_p = np.where(self.train_data_new[str(f_ii)]["Y"] == 1)[0]
                    idx_n = np.where(self.train_data_new[str(f_ii)]["Y"] == 0)[0]
                    print("training_data X_{} - new/pos/neg - {}/{}/{}".format(self.tags[f_ii], N_data_new, idx_p.shape[0], idx_n.shape[0]))
                    save_data_info[3*f_ii] = idx_p.shape[0]
                    save_data_info[3*f_ii+1] = idx_n.shape[0]
                    save_data_info[3*f_ii+2] = N_data_fail

                    h5f[str(f_ii)]["X"].resize(N_data_new+N_data_fail, axis=0)
                    h5f[str(f_ii)]["Y"].resize(N_data_new+N_data_fail, axis=0)
                    h5f[str(f_ii)]["bpoly"].resize(N_data_new+N_data_fail, axis=0)
                    h5f[str(f_ii)]["len"].resize(N_data_new+N_data_fail, axis=0)
                    h5f[str(f_ii)]["imp"].resize(N_data_new+N_data_fail, axis=0)

                    h5f[str(f_ii)]["X"][:N_data_new, :, :] = copy.deepcopy(self.train_data_new[str(f_ii)]["X"])
                    h5f[str(f_ii)]["X"][N_data_new:N_data_new+N_data_fail, :, :] = copy.deepcopy(self.train_data_fail[str(f_ii)]["X"])
                    h5f[str(f_ii)]["Y"][:N_data_new] = copy.deepcopy(self.train_data_new[str(f_ii)]["Y"])
                    h5f[str(f_ii)]["Y"][N_data_new:N_data_new+N_data_fail] = copy.deepcopy(self.train_data_fail[str(f_ii)]["Y"])
                    h5f[str(f_ii)]["bpoly"][:N_data_new, :] = copy.deepcopy(self.train_data_new[str(f_ii)]["bpoly"])
                    h5f[str(f_ii)]["bpoly"][N_data_new:N_data_new+N_data_fail, :] = copy.deepcopy(self.train_data_fail[str(f_ii)]["bpoly"])
                    h5f[str(f_ii)]["len"][:N_data_new] = copy.deepcopy(self.train_data_new[str(f_ii)]["len"])
                    h5f[str(f_ii)]["len"][N_data_new:N_data_new+N_data_fail] = copy.deepcopy(self.train_data_fail[str(f_ii)]["len"])
                    h5f[str(f_ii)]["imp"][:N_data_new] = copy.deepcopy(self.train_data_new[str(f_ii)]["imp"])
                    h5f[str(f_ii)]["imp"][N_data_new:N_data_new+N_data_fail] = copy.deepcopy(self.train_data_fail[str(f_ii)]["imp"])
                    
                    # h5f[str(f_ii)]["N_pn"][0] += idx_p.shape[0]
                    # h5f[str(f_ii)]["N_pn"][1] += idx_n.shape[0] + N_data_fail
                    h5f["N_data"][f_ii] = N_data+N_data_new+N_data_fail
                h5f.close()
            else:
                h5f = h5py.File(self.training_data_path, 'a')
                for f_ii in range(self.num_fidelity):
                    N_data = int(np.array(h5f["N_data"])[f_ii])
                    N_data_curr = np.array(h5f[str(f_ii)]["Y"]).shape[0]
                    N_data_new = self.train_data_new[str(f_ii)]["X"].shape[0]
                    N_data_fail = self.train_data_fail[str(f_ii)]["X"].shape[0]
                    print("training_data: X_{} - saved/new/fail - {}/{}/{}".format(self.tags[f_ii], N_data, N_data_new, N_data_fail))
                    idx_p = np.where(self.train_data_new[str(f_ii)]["Y"] == 1)[0]
                    idx_n = np.where(self.train_data_new[str(f_ii)]["Y"] == 0)[0]
                    
                    print("training_data X_{} - new/pos/neg - {}/{}/{}".format(self.tags[f_ii], N_data_new, idx_p.shape[0], idx_n.shape[0]))
                    save_data_info[3*f_ii] = idx_p.shape[0]
                    save_data_info[3*f_ii+1] = idx_n.shape[0]
                    save_data_info[3*f_ii+2] = N_data_fail

                    h5f[str(f_ii)]["X"].resize(N_data_curr+N_data_new+N_data_fail, axis=0)
                    h5f[str(f_ii)]["Y"].resize(N_data_curr+N_data_new+N_data_fail, axis=0)
                    h5f[str(f_ii)]["bpoly"].resize(N_data_curr+N_data_new+N_data_fail, axis=0)
                    h5f[str(f_ii)]["len"].resize(N_data_curr+N_data_new+N_data_fail, axis=0)
                    h5f[str(f_ii)]["imp"].resize(N_data_curr+N_data_new+N_data_fail, axis=0)

                    h5f[str(f_ii)]["X"][N_data_curr:N_data_curr+N_data_new, :, :] = copy.deepcopy(self.train_data_new[str(f_ii)]["X"])
                    h5f[str(f_ii)]["X"][N_data_curr+N_data_new:N_data_curr+N_data_new+N_data_fail, :, :] = copy.deepcopy(self.train_data_fail[str(f_ii)]["X"])
                    h5f[str(f_ii)]["Y"][N_data_curr:N_data_curr+N_data_new] = copy.deepcopy(self.train_data_new[str(f_ii)]["Y"])
                    h5f[str(f_ii)]["Y"][N_data_curr+N_data_new:N_data_curr+N_data_new+N_data_fail] = copy.deepcopy(self.train_data_fail[str(f_ii)]["Y"])
                    h5f[str(f_ii)]["bpoly"][N_data_curr:N_data_curr+N_data_new, :] = copy.deepcopy(self.train_data_new[str(f_ii)]["bpoly"])
                    h5f[str(f_ii)]["bpoly"][N_data_curr+N_data_new:N_data_curr+N_data_new+N_data_fail, :] = copy.deepcopy(self.train_data_fail[str(f_ii)]["bpoly"])
                    h5f[str(f_ii)]["len"][N_data_curr:N_data_curr+N_data_new] = copy.deepcopy(self.train_data_new[str(f_ii)]["len"])
                    h5f[str(f_ii)]["len"][N_data_curr+N_data_new:N_data_curr+N_data_new+N_data_fail] = copy.deepcopy(self.train_data_fail[str(f_ii)]["len"])
                    h5f[str(f_ii)]["imp"][N_data_curr:N_data_curr+N_data_new] = copy.deepcopy(self.train_data_new[str(f_ii)]["imp"])
                    h5f[str(f_ii)]["imp"][N_data_curr+N_data_new:N_data_curr+N_data_new+N_data_fail] = copy.deepcopy(self.train_data_fail[str(f_ii)]["imp"])
                    
                    # h5f[str(f_ii)]["N_pn"][0] += idx_p.shape[0]
                    # h5f[str(f_ii)]["N_pn"][1] += idx_n.shape[0] + N_data_fail
                    h5f["N_data"][f_ii] = N_data+N_data_new+N_data_fail
                h5f.close()
        
        return save_data_info
    
    def run_eval_dataset_th(self):
        print("run_eval_dataset_th called")
        self.ip_data = self.dis_model.get_inducing_points()
        train_z = []
        for f_ii in range(self.num_fidelity):
            train_z.append(np2cuda2(self.ip_data[f_ii]))
        dis_model_args = [train_z]
        
        self.eval_dataset_th = self.eval_dataset_th_pool.apipe(
            eval_dataset, self.gp_enc_args, dis_model_args, 
            [self.gp_enc.state_dict(), self.dis_model.state_dict()], 
            self.training_data_path, self.num_fidelity)
        print("run_eval_dataset_th finished")
        return
    
    def update_train_data(self, ep=0):
        est_data_info = np.zeros(2*self.num_fidelity)
        
        # TODO update np2cuda
        est_data_all = []
        print("update_train_data called")
        if self.eval_dataset_th != None:
            est_data_all = self.eval_dataset_th.get()
            print("received eval dataset")
            self.eval_dataset_th_pool.close()
            self.eval_dataset_th_pool.join()
            self.eval_dataset_th_pool.restart()
            for f_ii in range(self.num_fidelity):
                print("est_data")
                print(est_data_all[f_ii].shape)
                est_data_all[f_ii] = np.vstack((
                    est_data_all[f_ii], 
                    self.train_data_new[str(f_ii)]["info"],
                    self.train_data_fail[str(f_ii)]["info"]))
                print(est_data_all[f_ii].shape)
            save_data_info = self.save_train_data(ep=ep)
        elif self.il_type != 0:
            save_data_info = self.save_train_data(ep=ep)
        else:
            save_data_info = self.save_train_data(ep=ep)
            self.load_train_data()
            return est_data_info, save_data_info
        
        num_est_old = int(ep / self.bs_est)
        
        for f_ii in range(self.num_fidelity):
            h5f = h5py.File(self.training_data_path, 'r')
            N_data = int(np.array(h5f["N_data"])[f_ii])
            N_data_batch = self.num_update_ep * self.batch_size
            N_data_tmp = min(N_data, N_data_batch)

            # Balance pos / neg samples
            idx_p = np.where(np.array(h5f[str(f_ii)]["Y"]) == 1)[0]
            idx_n = np.where(np.array(h5f[str(f_ii)]["Y"]) == 0)[0]
            h5f.close()
            num_pn = np.array([idx_p.shape[0], idx_n.shape[0]])
            
            if self.il_type >= 2:
                idx_p_set = []
                idx_n_set = []
                idx_bias = 0
                idx_bias_set = [0]
                for ep_old in range(num_est_old):
                    training_data_path_old = "{}/est_data_old_{}.h5".format(self.log_path, ep_old*self.bs_est)
                    h5f = h5py.File(training_data_path_old, 'r')
                    idx_p_old = np.where(np.array(h5f[str(f_ii)]["Y"]) == 1)[0]
                    idx_n_old = np.where(np.array(h5f[str(f_ii)]["Y"]) == 0)[0]
                    h5f.close()
                    num_pn[0] += idx_p_old.shape[0]
                    num_pn[1] += idx_n_old.shape[0]
                    idx_p_set.append(idx_p_old + idx_bias)
                    idx_n_set.append(idx_n_old + idx_bias)
                    idx_bias += idx_p_old.shape[0] + idx_n_old.shape[0]
                    idx_bias_set.append(idx_bias)
                idx_p_set.append(idx_p + idx_bias)
                idx_n_set.append(idx_n + idx_bias)
                idx_p = np.hstack(idx_p_set)
                idx_n = np.hstack(idx_n_set)
            
            if num_pn[0] < num_pn[1]:
                num_p = min(num_pn[0], int(self.max_bo_data/2.))
                num_n = self.max_bo_data - num_p
            else:
                num_n = min(num_pn[1], int(self.max_bo_data/2.))
                num_p = self.max_bo_data - num_n
            
            # Incremental learning type
            if self.il_type == 0: # Uncertainty boundary
                est_data = est_data_all[f_ii]
                ent = - np.abs(est_data[:,1])/(est_data[:,2] + 1e-9)
                # rew = self.rew_bias + np.clip(np.array(h5f[str(f_ii)]["imp"]), self.rew_min, self.rew_max)
                # ei = rew * est_data[:,0]
                
                idx_p_arr = idx_p[np.argsort(-ent[idx_p]).astype(np.int32)[:num_p]]
                idx_n_arr = idx_n[np.argsort(-ent[idx_n]).astype(np.int32)[:num_n]]
            elif self.il_type == 1: # FIFO
                idx_p_arr = idx_p[-num_p:]
                idx_n_arr = idx_n[-num_n:]
            elif self.il_type == 2: # New data + mem replay
                num_new = self.train_data_new[str(f_ii)]["Y"].shape[0] + self.train_data_fail[str(f_ii)]["Y"].shape[0]
                num_old = N_data - num_new
                # new data
                idx_p_new = idx_p[np.where(idx_p>=num_old)[0]]
                idx_n_new = idx_n[np.where(idx_n>=num_old)[0]]
                
                idx_p_old = idx_p[np.where(idx_p<num_old)[0]]
                idx_n_old = idx_n[np.where(idx_n<num_old)[0]]
                p_idx_p = np.random.permutation(idx_p_old.shape[0])[:num_p-idx_p_new.shape[0]]
                p_idx_n = np.random.permutation(idx_n_old.shape[0])[:num_n-idx_n_new.shape[0]]
                
                idx_p_arr = np.concatenate((idx_p_old[p_idx_p], idx_p_new))
                idx_n_arr = np.concatenate((idx_n_old[p_idx_n], idx_n_new))
            elif self.il_type == 3: # mem replay
                p_idx_p = np.random.permutation(idx_p.shape[0])[:num_p]
                p_idx_n = np.random.permutation(idx_n.shape[0])[:num_n]
                idx_p_arr = idx_p[p_idx_p]
                idx_n_arr = idx_n[p_idx_n]

            rew_idx = np.hstack((idx_p_arr, idx_n_arr))
            # rew_idx = np.argsort(-ent).astype(np.int32)

            # Load dataset
            # data_idx = rew_idx[:N_data_tmp]
            if self.il_type >= 2:
                self.train_data[str(f_ii)]["X"] = np.empty((0, \
                    self.train_data[str(f_ii)]["X"].shape[1], \
                    self.train_data[str(f_ii)]["X"].shape[2]))
                self.train_data[str(f_ii)]["Y"] = np.empty(0)
                self.train_data[str(f_ii)]["bpoly"] = np.empty((0, \
                    self.train_data[str(f_ii)]["bpoly"].shape[1]))
                self.train_data[str(f_ii)]["len"] = np.empty(0)
                self.train_data[str(f_ii)]["info"] = np.empty((0,3))
                self.train_data[str(f_ii)]["imp"] = np.empty(0)
                print(idx_bias_set)
                n_old_data = []
                for ep_old in range(num_est_old):
                    idx_p_t = idx_p_arr[np.where(np.logical_and(idx_p_arr>=idx_bias_set[ep_old], idx_p_arr<idx_bias_set[ep_old+1]))]
                    idx_n_t = idx_n_arr[np.where(np.logical_and(idx_n_arr>=idx_bias_set[ep_old], idx_n_arr<idx_bias_set[ep_old+1]))]
                    idx_p_t -= idx_bias_set[ep_old]
                    idx_n_t -= idx_bias_set[ep_old]
                    data_idx = np.hstack((idx_p_t, idx_n_t))
                    n_old_data.append(data_idx.shape[0])
                    training_data_path_old = "{}/est_data_old_{}.h5".format(self.log_path, ep_old*self.bs_est)
                    h5f = h5py.File(training_data_path_old, 'r')
                    self.train_data[str(f_ii)]["X"] = \
                        np.concatenate((self.train_data[str(f_ii)]["X"], np.array(h5f[str(f_ii)]["X"])[list(data_idx),:,:]))
                    self.train_data[str(f_ii)]["Y"] = \
                        np.concatenate((self.train_data[str(f_ii)]["Y"], np.array(h5f[str(f_ii)]["Y"])[list(data_idx)]))
                    self.train_data[str(f_ii)]["bpoly"] = \
                        np.concatenate((self.train_data[str(f_ii)]["bpoly"], np.array(h5f[str(f_ii)]["bpoly"])[list(data_idx),:]))
                    self.train_data[str(f_ii)]["len"] = \
                        np.concatenate((self.train_data[str(f_ii)]["len"], np.array(h5f[str(f_ii)]["len"])[list(data_idx)]))
                    self.train_data[str(f_ii)]["imp"] = \
                        np.concatenate((self.train_data[str(f_ii)]["imp"], np.array(h5f[str(f_ii)]["imp"])[list(data_idx)]))
                    h5f.close()
                idx_p_t = idx_p_arr[np.where(idx_p_arr>=idx_bias_set[num_est_old])[0]]
                idx_n_t = idx_n_arr[np.where(idx_n_arr>=idx_bias_set[num_est_old])[0]]
                idx_p_t -= idx_bias_set[num_est_old]
                idx_n_t -= idx_bias_set[num_est_old]
                data_idx = np.hstack((idx_p_t, idx_n_t))
                n_old_data.append(data_idx.shape[0])
                h5f = h5py.File(self.training_data_path, 'r')
                self.train_data[str(f_ii)]["X"] = \
                    np.concatenate((self.train_data[str(f_ii)]["X"], np.array(h5f[str(f_ii)]["X"])[list(data_idx),:,:]))
                self.train_data[str(f_ii)]["Y"] = \
                    np.concatenate((self.train_data[str(f_ii)]["Y"], np.array(h5f[str(f_ii)]["Y"])[list(data_idx)]))
                self.train_data[str(f_ii)]["bpoly"] = \
                    np.concatenate((self.train_data[str(f_ii)]["bpoly"], np.array(h5f[str(f_ii)]["bpoly"])[list(data_idx),:]))
                self.train_data[str(f_ii)]["len"] = \
                    np.concatenate((self.train_data[str(f_ii)]["len"], np.array(h5f[str(f_ii)]["len"])[list(data_idx)]))
                self.train_data[str(f_ii)]["imp"] = \
                    np.concatenate((self.train_data[str(f_ii)]["imp"], np.array(h5f[str(f_ii)]["imp"])[list(data_idx)]))
                h5f.close()
                print("n_old_data: {}, {}".format(n_old_data, np.sum(n_old_data)))
            else:
                data_idx = copy.deepcopy(rew_idx)
                h5f = h5py.File(self.training_data_path, 'r')
                self.train_data[str(f_ii)]["X"] = np.array(h5f[str(f_ii)]["X"])[list(data_idx),:,:]
                self.train_data[str(f_ii)]["Y"] = np.array(h5f[str(f_ii)]["Y"])[list(data_idx)]
                self.train_data[str(f_ii)]["bpoly"] = np.array(h5f[str(f_ii)]["bpoly"])[list(data_idx),:]
                self.train_data[str(f_ii)]["len"] = np.array(h5f[str(f_ii)]["len"])[list(data_idx)]
                self.train_data[str(f_ii)]["imp"] = np.array(h5f[str(f_ii)]["imp"])[list(data_idx)]
                h5f.close()

            # Check dataset
            num_p = np.where(self.train_data[str(f_ii)]["Y"] == 1)[0].shape[0]
            num_n = np.where(self.train_data[str(f_ii)]["Y"] == 0)[0].shape[0]
            print("update_train_data - p/n - {}/{}".format(num_p, num_n))
            est_data_info[2*f_ii] = num_p
            est_data_info[2*f_ii+1] = num_n

            if N_data > self.max_bo_data and self.il_type < 2:
                data_save_idx = rew_idx[:self.max_bo_data]
                data_save_idx = np.sort(data_save_idx)
                h5f = h5py.File(self.training_data_path, 'a')
                h5f[str(f_ii)]["X"][:self.max_bo_data, :, :] = h5f[str(f_ii)]["X"][list(data_save_idx), :, :]
                h5f[str(f_ii)]["Y"][:self.max_bo_data] = h5f[str(f_ii)]["Y"][list(data_save_idx)]
                h5f[str(f_ii)]["bpoly"][:self.max_bo_data, :] = h5f[str(f_ii)]["bpoly"][list(data_save_idx), :]
                h5f[str(f_ii)]["len"][:self.max_bo_data] = h5f[str(f_ii)]["len"][list(data_save_idx)]
                h5f[str(f_ii)]["imp"][:self.max_bo_data] = h5f[str(f_ii)]["imp"][list(data_save_idx)]

                h5f[str(f_ii)]["X"].resize(self.max_bo_data, axis=0)
                h5f[str(f_ii)]["Y"].resize(self.max_bo_data, axis=0)
                h5f[str(f_ii)]["bpoly"].resize(self.max_bo_data, axis=0)
                h5f[str(f_ii)]["len"].resize(self.max_bo_data, axis=0)
                h5f[str(f_ii)]["imp"].resize(self.max_bo_data, axis=0)
                h5f["N_data"][f_ii] = self.max_bo_data
                h5f.close()
        
        self.train_dataset = []
        for f_ii in range(self.num_fidelity):
            self.train_dataset.append(TensorDataset(
                np2cuda(self.train_data[str(f_ii)]["X"]), 
                np2cuda(self.train_data[str(f_ii)]["Y"]), 
                np2cuda(self.train_data[str(f_ii)]["len"]), 
                np2cuda(self.train_data[str(f_ii)]["bpoly"])))
        
        return est_data_info, save_data_info
    
    def get_rewards_init(self, data_init, data_opt, data_bpoly, data_len, data_len_i, num_evals=[64,4,1], fidelity=2):
        data_init_np = data_init.cpu().detach().numpy()
        data_opt_np = data_opt.cpu().detach().numpy()
        data_bpoly_np = data_bpoly.cpu().detach().numpy()
        data_len_tmp = data_len.cpu().detach().numpy().astype(np.int32)
        data_len_i_tmp = data_len_i.cpu().detach().numpy().astype(np.int32)
        points_t, data_opt_np_all, denorm_time_init, denorm_time_opt, min_time = self.get_denorm_data(data_init_np, data_opt_np, data_len_tmp, data_len_i_tmp)
        
        mse_error = np.zeros((points_t.shape[0],2))
        mse_error[:,0] = np.linalg.norm(denorm_time_init - denorm_time_opt, axis=1)
        mse_error[:,1] = np.linalg.norm(data_init_np[:,1:,7] - data_opt_np_all[:,1:,7], axis=1)
        for b_ii in range(points_t.shape[0]):
            mse_error[b_ii,:] /= (data_len_tmp[b_ii]-1)
        mse_error_all = mse_error[:,-2] + mse_error[:,-1]
        
        data_info_np = np.zeros((points_t.shape[0], 3))
        data_info_np[:,0] = mse_error_all
        data_info_np[:,1:] = mse_error
        
        traj_time_init = np.sum(denorm_time_init, axis=1)
        traj_time_opt = np.sum(denorm_time_opt, axis=1)
        
        data_imp_np = 1.-traj_time_opt/traj_time_init
        
        f_ii = fidelity-1
        idx_array = list(np.argsort(mse_error_all).astype(np.int32)[:num_evals[f_ii]])
        if f_ii == 2:
            idx_array = []
            points_list = []
            idx_new_list = []
            t_set_list = []
            snap_w_list = []
            for idx_t in range(data_imp_np.shape[0]):
                points_list.append([points_t[idx_t, :data_len_i_tmp[idx_t], :], points_t[idx_t, :data_len_i_tmp[idx_t], :]])
                idx_new_list.append(data_len_i_tmp[idx_t] - data_len_tmp[idx_t])
                t_set_list.append(np.vstack([
                    denorm_time_init[idx_t, :data_len_i_tmp[idx_t]-1], 
                    denorm_time_opt[idx_t, :data_len_i_tmp[idx_t]-1]]))
                snap_w_list.append(np.vstack([
                    np.ones_like(data_init_np[idx_t, 1:data_len_i_tmp[idx_t], 7]), 
                    data_opt_np_all[idx_t, 1:data_len_i_tmp[idx_t], 7]]))
            res_flyable = eval_flyable(points_list, idx_new_list, t_set_list, snap_w_list, flag_wp_update=True)
            idx_array = []
            idx_array_t = list(np.argsort(mse_error_all).astype(np.int32))
            for idx_t in range(data_imp_np.shape[0]):
                if res_flyable[idx_array_t[idx_t]] == 1:
                    idx_array.append(idx_array_t[idx_t])
                if len(idx_array) >= num_evals[f_ii]:
                    break
        
        # Evaluate true rew
        points_list = []
        idx_new_list = []
        t_set_list = []
        snap_w_list = []
        acq_list = []
        for idx_t in idx_array:
            points_list.append([points_t[idx_t, :data_len_i_tmp[idx_t], :], points_t[idx_t, :data_len_i_tmp[idx_t], :]])
            idx_new_list.append(data_len_i_tmp[idx_t] - data_len_tmp[idx_t])
            t_set_list.append(np.vstack([
                denorm_time_init[idx_t, :data_len_i_tmp[idx_t]-1], 
                denorm_time_opt[idx_t, :data_len_i_tmp[idx_t]-1]]))
            snap_w_list.append(np.vstack([
                np.ones_like(data_init_np[idx_t, 1:data_len_i_tmp[idx_t], 7]), 
                data_opt_np_all[idx_t, 1:data_len_i_tmp[idx_t], 7]]))
            acq_list.append(mse_error_all[idx_t])
        
        if "X" in self.train_data_new[str(f_ii)].keys():
            self.train_data_new[str(f_ii)]["X"] = \
                np.concatenate((self.train_data_new[str(f_ii)]["X"], data_opt_np[idx_array, :]))
            self.train_data_new[str(f_ii)]["bpoly"] = \
                np.concatenate((self.train_data_new[str(f_ii)]["bpoly"], data_bpoly_np[idx_array, :]))
            self.train_data_new[str(f_ii)]["len"] = \
                np.concatenate((self.train_data_new[str(f_ii)]["len"], data_len_tmp[idx_array]))
            self.train_data_new[str(f_ii)]["info"] = \
                np.concatenate((self.train_data_new[str(f_ii)]["info"], data_info_np[idx_array, :]))
            self.train_data_new[str(f_ii)]["imp"] = \
                np.concatenate((self.train_data_new[str(f_ii)]["imp"], data_imp_np[idx_array]))
        else:
            self.train_data_new[str(f_ii)]["X"] = data_opt_np[idx_array, :]
            self.train_data_new[str(f_ii)]["Y"] = np.empty(0)
            self.train_data_new[str(f_ii)]["bpoly"] = data_bpoly_np[idx_array, :]
            self.train_data_new[str(f_ii)]["len"] = data_len_tmp[idx_array]
            self.train_data_new[str(f_ii)]["info"] = data_info_np[idx_array, :]
            self.train_data_new[str(f_ii)]["imp"] = data_imp_np[idx_array]
        
        if f_ii == 2:
            self.eval_real_queue["points"].extend(points_list)
            self.eval_real_queue["idx_new"].extend(idx_new_list)
            self.eval_real_queue["t_set"].extend(t_set_list)
            self.eval_real_queue["snap_w"].extend(snap_w_list)
            self.eval_real_queue["acq"].extend(acq_list)
        else:
            if self.flag_eval_zmq and not self.flag_eval_zmq_testonly:
                self.eval_client.req_eval(eval_type=f_ii, data=[points_list, idx_new_list, t_set_list, snap_w_list, self.ep_idx])
            else:
                # import pdb; pdb.set_trace()
                # res_tmp = eval_funcs[-1](points_list, idx_new_list, t_set_list, snap_w_list, self.ep_idx)
                th = self.eval_th_pool.apipe(eval_funcs[f_ii], points_list, idx_new_list, t_set_list, snap_w_list, self.ep_idx)
                self.eval_th_set[f_ii].append(th)
        return
    
    def get_denorm_data(self, data_init_np, data_opt_np, data_len_tmp, data_len_i_tmp):
        ###########################################
        # Denorm time
        points_t = copy.deepcopy(data_init_np[:,:,:4])
        for d_ii in range(3):
            points_t[:,:,d_ii] *= (self.points_scale[d_ii]/2.)
        # points_t[:,:,3] = np.arctan2(data_init_np[:,:,4], data_init_np[:,:,3])
        points_t[:,:,3] = data_init_np[:,:,8]
        
        pos_diff = np.diff(points_t[:,:,:3], axis=1)
        pos_diff = np.linalg.norm(pos_diff[:,:,:3], axis=2)
        for p_ii in range(pos_diff.shape[0]):
            pos_diff[p_ii, data_len_i_tmp[p_ii]-1:] = 0
        avg_time = pos_diff / self.mean_spd
        min_time = pos_diff / self.max_spd
        avg_time_init = np.repeat(np.sum(avg_time, axis=1)[:, np.newaxis], data_init_np.shape[1]-1, axis=1)
        denorm_time_init = data_init_np[:,1:,6] * avg_time_init
        
        ###########################################
        data_opt_np_all = copy.deepcopy(data_init_np)
        for b_ii in range(data_init_np.shape[0]):
            sub_seg_idx = data_len_i_tmp[b_ii] - data_len_tmp[b_ii]
            data_opt_np_all[b_ii,sub_seg_idx+1:data_len_i_tmp[b_ii],6:8] = data_opt_np[b_ii,1:data_len_tmp[b_ii],6:8]
            # Normalize time
            data_opt_np_all[b_ii,sub_seg_idx+1:data_len_i_tmp[b_ii],6] *= np.sum(avg_time[b_ii,sub_seg_idx:]) / np.sum(avg_time[b_ii,:])
            # Normalize snapw
            data_opt_np_all[b_ii,sub_seg_idx+1:data_len_i_tmp[b_ii],7] *= \
                np.sum(data_init_np[b_ii,sub_seg_idx+1:data_len_i_tmp[b_ii],7]) / np.sum(data_opt_np[b_ii,1:data_len_tmp[b_ii],7])
        ###########################################
        denorm_time_opt = data_opt_np_all[:,1:,6] * avg_time_init
        
        return points_t, data_opt_np_all, denorm_time_init, denorm_time_opt, min_time
    
    def get_denorm_data_wp(self, data_init_np, data_opt_np, data_len_tmp, data_len_i_tmp):
        ###########################################
        # Denorm time
        points_t = copy.deepcopy(data_init_np[:,:,:4])
        for d_ii in range(3):
            points_t[:,:,d_ii] *= (self.points_scale[d_ii]/2.)
        points_t[:,:,3] = data_init_np[:,:,8]
        
        pos_diff = np.diff(points_t[:,:,:3], axis=1)
        pos_diff = np.linalg.norm(pos_diff[:,:,:3], axis=2)
        for p_ii in range(pos_diff.shape[0]):
            pos_diff[p_ii, data_len_i_tmp[p_ii]-1:] = 0
        avg_time = pos_diff / self.mean_spd
        min_time = pos_diff / self.max_spd
        avg_time_init = np.repeat(np.sum(avg_time, axis=1)[:, np.newaxis], data_init_np.shape[1]-1, axis=1)
        denorm_time_init = data_init_np[:,1:,6] * avg_time_init
        
        ###########################################
        # Denorm time opt        
        points_opt = copy.deepcopy(points_t)
        for b_ii in range(data_init_np.shape[0]):
            sub_seg_idx = data_len_i_tmp[b_ii] - data_len_tmp[b_ii]
            for d_ii in range(3):
                points_opt[b_ii,sub_seg_idx+1:data_len_i_tmp[b_ii],d_ii] = data_opt_np[b_ii,1:data_len_tmp[b_ii],d_ii] * (self.points_scale[d_ii]/2.)
            points_opt[b_ii,sub_seg_idx+1:data_len_i_tmp[b_ii],3] = data_opt_np[b_ii,1:data_len_tmp[b_ii],8]
        
        pos_diff_t = np.diff(points_opt[:,:,:3], axis=1)
        pos_diff_t = np.linalg.norm(pos_diff_t[:,:,:3], axis=2)
        for p_ii in range(pos_diff_t.shape[0]):
            pos_diff_t[p_ii, data_len_i_tmp[p_ii]-1:] = 0
        avg_time_t = pos_diff_t / self.mean_spd
        min_time_t = pos_diff_t / self.max_spd
        avg_time_opt = np.repeat(np.sum(avg_time_t, axis=1)[:, np.newaxis], data_init_np.shape[1]-1, axis=1)
        
        ###########################################
        data_opt_np_all = copy.deepcopy(data_init_np)
        denorm_time_init_h = copy.deepcopy(denorm_time_init)
        for b_ii in range(data_init_np.shape[0]):
            sub_seg_idx = data_len_i_tmp[b_ii] - data_len_tmp[b_ii]
            data_opt_np_all[b_ii,sub_seg_idx+1:data_len_i_tmp[b_ii],6:8] = data_opt_np[b_ii,1:data_len_tmp[b_ii],6:8]
            data_opt_np_all[b_ii,sub_seg_idx+1:data_len_i_tmp[b_ii],6] *= np.sum(avg_time_t[b_ii,sub_seg_idx:]) / np.sum(avg_time_t[b_ii,:])
            # Normalize snapw
            data_opt_np_all[b_ii,sub_seg_idx+1:data_len_i_tmp[b_ii],7] *= \
                np.sum(data_init_np[b_ii,sub_seg_idx+1:data_len_i_tmp[b_ii],7]) / np.sum(data_opt_np[b_ii,1:data_len_tmp[b_ii],7])
            
            denorm_time_init_h[b_ii,sub_seg_idx:data_len_i_tmp[b_ii]-1] *= avg_time_t[b_ii,sub_seg_idx:data_len_i_tmp[b_ii]-1] / avg_time[b_ii,sub_seg_idx:data_len_i_tmp[b_ii]-1]
        ###########################################
        denorm_time_opt = data_opt_np_all[:,1:,6] * avg_time_opt
        
        return points_t, points_opt, data_opt_np_all, denorm_time_init, denorm_time_init_h, denorm_time_opt, min_time, min_time_t
    
    def get_rewards(self, data_init, data_opt, data_bpoly, data_len, data_len_i, num_evals=[64,4], fidelity=2, flag_eval=True):
        self.dis_model.eval()
        if self.al_type == 3:
            m, v, pm, emb = self.dis_model.predict_proba_full([data_opt, data_len, data_bpoly], fidelity=fidelity)
            est_emb = emb.cpu().detach().numpy()
        else:
            m, v, pm = self.dis_model.predict_proba([data_opt, data_len, data_bpoly], fidelity=fidelity)
        est_data = [pm, m, v]
        data_info_np = np.array([pm[:,1], m, v]).T
        
        data_init_np = data_init.cpu().detach().numpy()
        data_opt_np = data_opt.cpu().detach().numpy()
        data_bpoly_np = data_bpoly.cpu().detach().numpy()
        data_len_tmp = data_len.cpu().detach().numpy().astype(np.int32)
        data_len_i_tmp = data_len_i.cpu().detach().numpy().astype(np.int32)
        points_t, points_opt, data_opt_np_all, denorm_time_init, denorm_time_init_h, denorm_time_opt, min_time, min_time_t = self.get_denorm_data_wp(data_init_np, data_opt_np, data_len_tmp, data_len_i_tmp)
        
        traj_time_init = np.sum(denorm_time_init, axis=1)
        traj_time_init_h = np.sum(denorm_time_init_h, axis=1)
        traj_time_opt = np.sum(denorm_time_opt, axis=1)
        
        data_imp_np = 1.-traj_time_opt/traj_time_init_h
        ###########################################
        r_ii_set = []
        for p_ii in range(denorm_time_opt.shape[0]):
            # print("---")
            # print(denorm_time_opt[p_ii,:data_len_tmp[p_ii]-1]-min_time[p_ii,:data_len_tmp[p_ii]-1])
            # print(min_time[p_ii,:data_len_tmp[p_ii]-1])
            if np.all(denorm_time_opt[p_ii,:data_len_i_tmp[p_ii]-1] > min_time_t[p_ii,:data_len_i_tmp[p_ii]-1]):
                r_ii_set.append(p_ii)
        
        # Reward estimation
        rew_tmp_init = []
        f_ii = fidelity-1
        ei = (self.rew_bias + np.clip(1.-traj_time_opt/traj_time_init_h, None, self.rew_max)) * est_data[0][:,1]
        ent = - np.abs(est_data[1])/(est_data[2] + 1e-9)
        bald = get_bald(est_data[1], est_data[2])
        vratio = 1 - np.max(est_data[0], axis=1)
        rew_tmp = ei

        # Active learning type
        if self.al_type == 0: # Uncertainty boundary
            rew_tmp_init.append(ent)
        elif self.al_type == 1: # BALD
            rew_tmp_init.append(bald)
        elif self.al_type == 2: # Variational ratio
            rew_tmp_init.append(vratio)
        elif self.al_type == 3: # Variational ratio / Coreset
            rew_tmp_init.append(vratio)

        rew_info = dict()
        rew_info["prob"] = est_data[0][:,1]
        rew_info["ei"] = ei
        rew_info["ent"] = ent
        rew_info["vratio"] = vratio
        rew_info["bald"] = bald
        rew_info["ent_m"] = np.abs(est_data[1])
        rew_info["ent_v"] = est_data[2]

        rew = np.zeros_like(rew_tmp)
        for r_ii in r_ii_set:
            rew[r_ii] = rew_tmp[r_ii]
        
        if flag_eval and f_ii == 2:
            idx_array = []
            points_list = []
            idx_new_list = []
            t_set_list = []
            snap_w_list = []
            for idx_t in range(denorm_time_opt.shape[0]):
                points_list.append([points_t[idx_t, :data_len_i_tmp[idx_t], :], points_opt[idx_t, :data_len_i_tmp[idx_t], :]])
                idx_new_list.append(data_len_i_tmp[idx_t] - data_len_tmp[idx_t])
                t_set_list.append(np.vstack([
                    denorm_time_init[idx_t, :data_len_i_tmp[idx_t]-1], 
                    denorm_time_opt[idx_t, :data_len_i_tmp[idx_t]-1]]))
                snap_w_list.append(np.vstack([
                    np.ones_like(data_init_np[idx_t, 1:data_len_i_tmp[idx_t], 7]), 
                    data_opt_np_all[idx_t, 1:data_len_i_tmp[idx_t], 7]]))
            res_flyable = eval_flyable(points_list, idx_new_list, t_set_list, snap_w_list, flag_wp_update=True)
            res_flyable_np = np.array(res_flyable)
            print("res_flyable: {}/{}".format(np.sum(res_flyable_np), res_flyable_np.shape[0]))

        # Evaluate true rew
        if self.al_type == 3:
            N_margin = 10
            idx_array_tmp1 = list(np.argsort(-rew_tmp_init[0]).astype(np.int32)[:num_evals[f_ii]*N_margin])
            chosen = kmeans_pp_centers(list(est_emb[idx_array_tmp1,:]), num_evals[f_ii])
            # print(chosen)
            # chosen = np.sort(np.array(chosen).astype(np.int32))
            idx_array_tmp = list(np.array(idx_array_tmp1)[chosen])
        else:
            if f_ii < 2:
                idx_array_tmp = list(np.argsort(-rew_tmp_init[0]).astype(np.int32)[:num_evals[f_ii]])
            elif f_ii == 2 and flag_eval:
                idx_array_tmp_0 = np.argsort(-rew_tmp_init[0]).astype(np.int32)
                idx_array_tmp = []
                for idx_t in range(idx_array_tmp_0.shape[0]):
                    if res_flyable[idx_array_tmp_0[idx_t]] == 1:
                        idx_array_tmp.append(idx_array_tmp_0[idx_t])
                    if len(idx_array_tmp) >= num_evals[f_ii]:
                        break
        # print("N_ei: {}, N_ent: {}, all: {}".format(N_ei, N_ent, len(idx_array_tmp)))

        # idx_array_tmp = list(ent_idx[:num_evals[f_ii]])
        if flag_eval:
            idx_array = []
            idx_array_fail = []

            points_list = []
            idx_new_list = []
            t_set_list = []
            snap_w_list = []
            acq_list = []
            for idx_t in idx_array_tmp:
                if np.all(denorm_time_opt[idx_t,:data_len_i_tmp[idx_t]-1] > min_time_t[idx_t,:data_len_i_tmp[idx_t]-1]) and \
                    np.all(data_opt_np[idx_t, 1:data_len_tmp[idx_t], 7] > 1e-10):
                    idx_array.append(idx_t)
                    points_list.append([points_t[idx_t, :data_len_i_tmp[idx_t], :], points_opt[idx_t, :data_len_i_tmp[idx_t], :]])
                    idx_new_list.append(data_len_i_tmp[idx_t] - data_len_tmp[idx_t])
                    t_set_list.append(np.vstack([
                        denorm_time_init[idx_t, :data_len_i_tmp[idx_t]-1], 
                        denorm_time_opt[idx_t, :data_len_i_tmp[idx_t]-1]]))
                    snap_w_list.append(np.vstack([
                        np.ones_like(data_init_np[idx_t, 1:data_len_i_tmp[idx_t], 7]), 
                        data_opt_np_all[idx_t, 1:data_len_i_tmp[idx_t], 7]]))
                    acq_list.append(-rew_tmp_init[0][idx_t])
                else:
                    # print("===")
                    # print(data_opt_np[idx_t, 1:data_len_tmp[idx_t], -1])
                    # print(data_opt_np[idx_t, 1:data_len_tmp[idx_t], -1] - 1e-10)
                    # print(denorm_time_opt[idx_t,:data_len_tmp[idx_t]-1]-min_time[idx_t,:data_len_tmp[idx_t]-1])
                    idx_array_fail.append(idx_t)

            # print("train: {}".format(len(idx_array)))
            if len(idx_array) > 0:
                self.train_data_new[str(f_ii)]["X"] = \
                    np.concatenate((self.train_data_new[str(f_ii)]["X"], data_opt_np[idx_array, :]))
                self.train_data_new[str(f_ii)]["bpoly"] = \
                    np.concatenate((self.train_data_new[str(f_ii)]["bpoly"], data_bpoly_np[idx_array, :]))
                self.train_data_new[str(f_ii)]["len"] = \
                    np.concatenate((self.train_data_new[str(f_ii)]["len"], data_len_tmp[idx_array]))
                self.train_data_new[str(f_ii)]["info"] = \
                    np.concatenate((self.train_data_new[str(f_ii)]["info"], data_info_np[idx_array, :]))
                self.train_data_new[str(f_ii)]["imp"] = \
                    np.concatenate((self.train_data_new[str(f_ii)]["imp"], data_imp_np[idx_array]))

                if f_ii == 2:
                    self.eval_real_queue["points"].extend(points_list)
                    self.eval_real_queue["idx_new"].extend(idx_new_list)
                    self.eval_real_queue["t_set"].extend(t_set_list)
                    self.eval_real_queue["snap_w"].extend(snap_w_list)
                    self.eval_real_queue["acq"].extend(acq_list)
                else:
                    if self.flag_eval_zmq and not self.flag_eval_zmq_testonly:
                        self.eval_client.req_eval(eval_type=f_ii, data=[points_list, idx_new_list, t_set_list, snap_w_list, self.ep_idx])
                    else:
                        # import pdb; pdb.set_trace()
                        th = self.eval_th_pool.apipe(eval_funcs[f_ii], points_list, idx_new_list, t_set_list, snap_w_list, self.ep_idx)
                        self.eval_th_set[f_ii].append(th)

            if len(idx_array_fail) > 0 and f_ii < 2:
                self.train_data_fail[str(f_ii)]["X"] = \
                    np.concatenate((self.train_data_fail[str(f_ii)]["X"], data_opt_np[idx_array_fail, :]))
                self.train_data_fail[str(f_ii)]["Y"] = \
                    np.concatenate((self.train_data_fail[str(f_ii)]["Y"], np.zeros_like(data_len_tmp[idx_array_fail])))
                self.train_data_fail[str(f_ii)]["bpoly"] = \
                    np.concatenate((self.train_data_fail[str(f_ii)]["bpoly"], data_bpoly_np[idx_array_fail, :]))
                self.train_data_fail[str(f_ii)]["len"] = \
                    np.concatenate((self.train_data_fail[str(f_ii)]["len"], data_len_tmp[idx_array_fail]))
                self.train_data_fail[str(f_ii)]["info"] = \
                    np.concatenate((self.train_data_fail[str(f_ii)]["info"], data_info_np[idx_array_fail, :]))
                self.train_data_fail[str(f_ii)]["imp"] = \
                    np.concatenate((self.train_data_fail[str(f_ii)]["imp"], data_imp_np[idx_array_fail]))

        return np.array(rew), rew_info
    
    def get_rewards_test(self, data_init, data_opt, data_len, data_len_i, t_new=None, s_new=None):
        data_init_np = data_init.cpu().detach().numpy()
        data_opt_np = data_opt.cpu().detach().numpy()
        data_len_tmp = data_len.cpu().detach().numpy().astype(np.int32)
        data_len_i_tmp = data_len_i.cpu().detach().numpy().astype(np.int32)
        if np.all(t_new != None):
            data_opt_np[:, 1:, 6] = t_new
        if np.all(s_new != None):
            data_opt_np[:, 1:, 7] = s_new
        points_t, data_opt_np_all, denorm_time_init, denorm_time_opt, min_time = self.get_denorm_data(data_init_np, data_opt_np, data_len_tmp, data_len_i_tmp)
        traj_time_init = np.sum(denorm_time_init, axis=1)
        traj_time_opt = np.sum(denorm_time_opt, axis=1)
        
        ###########################################
        # Reward estimation
        rew = np.clip(1.-traj_time_opt/traj_time_init, -1.5, 0.5)
        points_list = []
        idx_new_list = []
        t_set_list = []
        snap_w_list = []

        # Sanity check
        idx_array = []
        idx_array_fail = []
        for idx_t in range(points_t.shape[0]):
            if np.all(denorm_time_opt[idx_t,:data_len_i_tmp[idx_t]-1] > min_time[idx_t,:data_len_i_tmp[idx_t]-1]) and \
                np.all(data_opt_np[idx_t, 1:data_len_tmp[idx_t], 7] > 1e-10):
                idx_array.append(idx_t)
                points_list.append(points_t[idx_t, :data_len_i_tmp[idx_t], :])
                idx_new_list.append(data_len_i_tmp[idx_t] - data_len_tmp[idx_t])
                t_set_list.append(np.vstack([
                    denorm_time_init[idx_t, :data_len_i_tmp[idx_t]-1], 
                    denorm_time_opt[idx_t, :data_len_i_tmp[idx_t]-1]]))
                snap_w_list.append(np.vstack([
                    np.ones_like(data_init_np[idx_t, 1:data_len_i_tmp[idx_t], 7]), 
                    data_opt_np_all[idx_t, 1:data_len_i_tmp[idx_t], 7]]))
            else:
                idx_array_fail.append(idx_t)

        if len(idx_array) > 0:
            self.test_data["rew"] = np.concatenate((self.test_data["rew"], rew[idx_array]))
        if len(idx_array_fail) > 0:
            self.test_data["fail"] = np.concatenate((self.test_data["fail"], rew[idx_array_fail]))
        
        if self.flag_eval_zmq:
            self.eval_client.req_eval(eval_type=-1, data=[points_list, idx_new_list, t_set_list, snap_w_list, self.ep_idx])
        else:
            test_bs = 4
            N_test = int(len(points_list)/test_bs)
            if len(points_list) > N_test * test_bs:
                N_test += 1
            for b_ii in range(N_test):
                idx_i = b_ii*test_bs
                idx_f = min((b_ii+1)*test_bs, len(points_list))
                th = self.test_th_pool.apipe(eval_test, 
                    points_list[idx_i:idx_f], 
                    idx_new_list[idx_i:idx_f], 
                    t_set_list[idx_i:idx_f], 
                    snap_w_list[idx_i:idx_f], self.ep_idx)
                self.test_th_set.append(th)
        
        # res = eval_test(points_list, idx_new_list, t_set_list, snap_w_list, self.ep_idx)
        # self.test_data["res"] = np.concatenate((self.test_data["res"], res[0]))
        # th = self.test_th_pool.apipe(eval_test, points_list, idx_new_list, t_set_list, snap_w_list, self.ep_idx)
        # self.test_th_set.append(th)
        return np.array(rew)
    
    def get_rewards_test_wp(self, data_init, data_opt, data_len, data_len_i, t_new=None, s_new=None, flag_ms=False, flag_ms_h=False):
        data_init_np = data_init.cpu().detach().numpy()
        data_opt_np = data_opt.cpu().detach().numpy()
        data_len_tmp = data_len.cpu().detach().numpy().astype(np.int32)
        data_len_i_tmp = data_len_i.cpu().detach().numpy().astype(np.int32)
        if np.all(t_new != None):
            data_opt_np[:, 1:, 6] = t_new
        if np.all(s_new != None):
            data_opt_np[:, 1:, 7] = s_new
        
        ###########################################
        # Denorm time
        points_t = copy.deepcopy(data_init_np[:,:,:4])
        for d_ii in range(3):
            points_t[:,:,d_ii] *= (self.points_scale[d_ii]/2.)
        points_t[:,:,3] = data_init_np[:,:,8]
        
        pos_diff = np.diff(points_t[:,:,:3], axis=1)
        pos_diff = np.linalg.norm(pos_diff[:,:,:3], axis=2)
        for p_ii in range(pos_diff.shape[0]):
            pos_diff[p_ii, data_len_i_tmp[p_ii]-1:] = 0
        avg_time = pos_diff / self.mean_spd
        min_time = pos_diff / self.max_spd
        avg_time_init = np.repeat(np.sum(avg_time, axis=1)[:, np.newaxis], data_init_np.shape[1]-1, axis=1)
        denorm_time_init = data_init_np[:,1:,6] * avg_time_init
        
        ###########################################
        # Denorm time opt        
        points_opt = copy.deepcopy(points_t)
        for b_ii in range(data_init_np.shape[0]):
            sub_seg_idx = data_len_i_tmp[b_ii] - data_len_tmp[b_ii]
            for d_ii in range(3):
                points_opt[b_ii,sub_seg_idx+1:data_len_i_tmp[b_ii],d_ii] = data_opt_np[b_ii,1:data_len_tmp[b_ii],d_ii] * (self.points_scale[d_ii]/2.)
            points_opt[b_ii,sub_seg_idx+1:data_len_i_tmp[b_ii],3] = data_opt_np[b_ii,1:data_len_tmp[b_ii],8]
        
        pos_diff_t = np.diff(points_opt[:,:,:3], axis=1)
        pos_diff_t = np.linalg.norm(pos_diff_t[:,:,:3], axis=2)
        for p_ii in range(pos_diff_t.shape[0]):
            pos_diff_t[p_ii, data_len_i_tmp[p_ii]-1:] = 0
        avg_time_t = pos_diff_t / self.mean_spd
        if flag_ms:
            min_time_t = copy.deepcopy(min_time)
        else:
            min_time_t = pos_diff_t / self.max_spd
        avg_time_opt = np.repeat(np.sum(avg_time_t, axis=1)[:, np.newaxis], data_init_np.shape[1]-1, axis=1)
        
        ###########################################
        data_opt_np_all = copy.deepcopy(data_init_np)
        for b_ii in range(data_init_np.shape[0]):
            sub_seg_idx = data_len_i_tmp[b_ii] - data_len_tmp[b_ii]
            data_opt_np_all[b_ii,sub_seg_idx+1:data_len_i_tmp[b_ii],6:8] = data_opt_np[b_ii,1:data_len_tmp[b_ii],6:8]
            # Normalize time
            if flag_ms:
                data_opt_np_all[b_ii,sub_seg_idx+1:data_len_i_tmp[b_ii],6] *= np.sum(avg_time[b_ii,sub_seg_idx:]) / np.sum(avg_time[b_ii,:])
                if flag_ms_h:
                    data_opt_np_all[b_ii,sub_seg_idx+1:data_len_i_tmp[b_ii],6] *= avg_time_t[b_ii,sub_seg_idx:data_len_i_tmp[b_ii]-1] / avg_time[b_ii,sub_seg_idx:data_len_i_tmp[b_ii]-1]
            else:
                data_opt_np_all[b_ii,sub_seg_idx+1:data_len_i_tmp[b_ii],6] *= np.sum(avg_time_t[b_ii,sub_seg_idx:]) / np.sum(avg_time_t[b_ii,:])
            # Normalize snapw
            data_opt_np_all[b_ii,sub_seg_idx+1:data_len_i_tmp[b_ii],7] *= \
                np.sum(data_init_np[b_ii,sub_seg_idx+1:data_len_i_tmp[b_ii],7]) / np.sum(data_opt_np[b_ii,1:data_len_tmp[b_ii],7])
        ###########################################
        if flag_ms:
            denorm_time_opt = data_opt_np_all[:,1:,6] * avg_time_init
        else:
            denorm_time_opt = data_opt_np_all[:,1:,6] * avg_time_opt
                
        traj_time_init = np.sum(denorm_time_init, axis=1)
        traj_time_opt = np.sum(denorm_time_opt, axis=1)
        
        ###########################################
        # Reward estimation
        # rew = np.clip(1.-traj_time_opt/traj_time_init, -1.5, 0.5)
        rew = 1.-traj_time_opt/traj_time_init
        points_list = []
        idx_new_list = []
        t_set_list = []
        snap_w_list = []

        # Sanity check
        idx_array = []
        idx_array_fail = []
        for idx_t in range(points_t.shape[0]):
            if np.all(denorm_time_opt[idx_t,:data_len_i_tmp[idx_t]-1] > min_time_t[idx_t,:data_len_i_tmp[idx_t]-1]) and \
                np.all(data_opt_np[idx_t, 1:data_len_tmp[idx_t], 7] > 1e-10):
                idx_array.append(idx_t)
                points_list.append([points_t[idx_t, :data_len_i_tmp[idx_t], :], points_opt[idx_t, :data_len_i_tmp[idx_t], :]])
                idx_new_list.append(data_len_i_tmp[idx_t] - data_len_tmp[idx_t])
                t_set_list.append(np.vstack([
                    denorm_time_init[idx_t, :data_len_i_tmp[idx_t]-1], 
                    denorm_time_opt[idx_t, :data_len_i_tmp[idx_t]-1]]))
                snap_w_list.append(np.vstack([
                    np.ones_like(data_init_np[idx_t, 1:data_len_i_tmp[idx_t], 7]), 
                    data_opt_np_all[idx_t, 1:data_len_i_tmp[idx_t], 7]]))
            else:
                idx_array_fail.append(idx_t)

        if len(idx_array) > 0:
            self.test_data["rew"] = np.concatenate((self.test_data["rew"], rew[idx_array]))
        if len(idx_array_fail) > 0:
            self.test_data["fail"] = np.concatenate((self.test_data["fail"], rew[idx_array_fail]))
        
        if self.flag_eval_zmq:
            self.eval_client.req_eval(eval_type=-2, data=[points_list, idx_new_list, t_set_list, snap_w_list, self.ep_idx])
        else:
            test_bs = 4
            N_test = int(len(points_list)/test_bs)
            if len(points_list) > N_test * test_bs:
                N_test += 1
            for b_ii in range(N_test):
                idx_i = b_ii*test_bs
                idx_f = min((b_ii+1)*test_bs, len(points_list))
                th = self.test_th_pool.apipe(eval_test_wp, 
                    points_list[idx_i:idx_f], 
                    idx_new_list[idx_i:idx_f], 
                    t_set_list[idx_i:idx_f], 
                    snap_w_list[idx_i:idx_f], self.ep_idx)
                self.test_th_set.append(th)

        # res = eval_test(points_list, idx_new_list, t_set_list, snap_w_list, self.ep_idx)
        # self.test_data["res"] = np.concatenate((self.test_data["res"], res[0]))
        # th = self.test_th_pool.apipe(eval_test, points_list, idx_new_list, t_set_list, snap_w_list, self.ep_idx)
        # self.test_th_set.append(th)
        return np.array(rew)
        
    def update_model(self, flag_skip_init=False, N_update=-1, ep=0):
        # Wait until all threads finish
        if self.flag_eval_zmq and not self.flag_eval_zmq_testonly:
            if not flag_skip_init:
                res = self.eval_client.req_join(eval_type=0)
                for f_ii in range(2):
                    self.train_data_new[str(f_ii)]["Y"] = np.concatenate((self.train_data_new[str(f_ii)]["Y"], res[f_ii]))
                    print(res[f_ii].shape)
        else:
            ep_curr = -1
            for f_ii in range(self.num_fidelity):
                for t_ii in range(len(self.eval_th_set[f_ii])):
                    while not self.eval_th_set[f_ii][t_ii].ready():
                        time.sleep(1)
                    res = self.eval_th_set[f_ii][t_ii].get()
                    # print(res)
                    assert res[1] == 0
                    if ep_curr < 0:
                        ep_curr = res[2]
                    assert ep_curr == res[2]
                    self.train_data_new[str(f_ii)]["Y"] = np.concatenate((self.train_data_new[str(f_ii)]["Y"], res[0]))
            self.eval_th_pool.close()
            self.eval_th_pool.join()
            self.eval_th_pool.restart()
            # self.eval_th_pool = ParallelPool(self.max_eval_proc)
            # self.eval_th_pool = ProcessingPool(self.max_eval_proc)
            self.eval_th_set = []
            for f_ii in range(self.num_fidelity):
                self.eval_th_set.append([])
        
        est_data_info = np.zeros(2*self.num_fidelity)
        save_data_info = np.zeros(3*self.num_fidelity)
        # Update dataset
        if self.flag_data_managing and self.flag_initialized:
            est_data_info, save_data_info = self.update_train_data(ep=ep)
        else:
            if not flag_skip_init:
                self.save_train_data(ep=ep)
                return
            self.load_train_data()
            if not self.flag_initialized:
                # if not flag_skip_init:
                #     self.build_inducing_points()
                #     self.create_model()
                self.flag_initialized = True
        self.init_train_data()
        
        train_loader = []
        for f_ii in range(self.num_fidelity):
            train_loader.append(iter(DataLoader(self.train_dataset[f_ii], batch_size=self.batch_size, shuffle=True)))
        
        print('Updating reward model ...')
        self.dis_model.train()
        num_ep = self.num_update_ep
        if N_update > 0:
            num_ep = N_update
        
        acc_all = []
        for f_ii in range(self.num_fidelity):
            acc_all.append(0)
        acc_last = 0
        for ep_ii in range(num_ep):
            for f_ii in range(self.num_fidelity):
                N_batch = len(train_loader[f_ii])
                try:
                    x_batch, y_batch, len_batch, bpoly_batch = next(train_loader[f_ii])
                except StopIteration:
                    train_loader[f_ii] = iter(DataLoader(
                        self.train_dataset[f_ii], batch_size=self.batch_size, shuffle=True))
                    x_batch, y_batch, len_batch, bpoly_batch = next(train_loader[f_ii])
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()
                len_batch = len_batch.cuda()
                bpoly_batch = bpoly_batch.cuda()
                                
                self.optimizer.zero_grad()
                gp_outputs = self.dis_model.forward_train([x_batch, len_batch, bpoly_batch], fidelity=f_ii+1, eval=False)
                
                mll_loss = -self.mll(gp_outputs[-1], y_batch)
                reg_loss = to_var(torch.tensor([0.0], requires_grad=True))
                if len(gp_outputs) >= 2:
                    reg_loss = -self.mll(gp_outputs[0], y_batch) * self.coef_reg
                    if len(gp_outputs) >= 3:
                        reg_loss += -self.mll(gp_outputs[1], y_batch) * self.coef_reg_real
                    # for f_ii_t in range(1,len(gp_outputs)-1):
                    #     reg_loss += -self.mll(gp_outputs[f_ii_t], y_batch) * self.coef_reg
                
                loss = mll_loss + reg_loss
                
                loss.backward(retain_graph=True)
                loss_curr = loss.item()
                # avg_loss += loss_curr
                self.optimizer.step()

                try:
                    if ep_ii == 0 or (ep_ii+1) % self.print_every == 0:
                        m, v, pm = self.dis_model.predict_proba([x_batch, len_batch, bpoly_batch], fidelity=f_ii+1)
                        correct = np.abs(pm[:,1] - y_batch.cpu().detach().numpy()) < 0.5
                        acc = np.sum(correct) / correct.shape[0]
                        print('Ep %d/%d, Eval %s - Loss: %.3f - %.3f, Acc: %.3f' % 
                              (ep_ii+1, num_ep, self.tags[f_ii], mll_loss.item()/self.batch_size*1e3, reg_loss.item()/self.batch_size*1e3, acc))
                        acc_all[f_ii] = acc
                except:
                    print("Error in accuracy estimation")
            # epoch_correct /= N_data
            # print('Epoch %d/%d, Acc: %.3f' % (ep_ii+1, self.num_update_ep, epoch_correct))
            # print('---')
        self.scheduler.step()
        if self.flag_data_managing and not flag_skip_init and self.il_type == 0:
            self.run_eval_dataset_th()
        
        rewest_info = dict()
        rewest_info["acc"] = acc_all[1]
        rewest_info["acc_L"] = acc_all[0]
        rewest_info["est_data"] = est_data_info
        rewest_info["new_data"] = save_data_info
        return rewest_info
    
    def join_rewards_th(self, flag_wp=False):
        if self.flag_eval_zmq:
            if flag_wp:
                res = self.eval_client.req_join(eval_type=-2)
            else:
                res = self.eval_client.req_join(eval_type=-1)
            self.test_data["res"] = np.concatenate((self.test_data["res"], res[0]))
        else:
            # Wait until all threads finish
            ep_curr = -1
            for t_ii in range(len(self.test_th_set)):
                res = None
                while not self.test_th_set[t_ii].ready():
                    time.sleep(1)
                res = self.test_th_set[t_ii].get()
                # print(res)
                assert res[1] == 1
                if ep_curr < 0:
                    ep_curr = res[2]
                assert ep_curr == res[2]
                self.test_data["res"] = np.concatenate((self.test_data["res"], res[0]))
            self.test_th_pool.close()
            self.test_th_pool.join()
            self.test_th_pool.restart()
            self.test_th_set = []
        
        rew = np.concatenate((self.test_data["rew"], self.test_data["fail"]))
        res = np.zeros_like(rew)
        res[:self.test_data["res"].shape[0]] = self.test_data["res"]
        
        idx_pos = np.where(self.test_data["res"] == 1)[0]
        print("pos {} / neg {} / fail {}".format(len(idx_pos), self.test_data["res"].shape[0]-len(idx_pos), self.test_data["fail"].shape[0]))
        
        self.test_data = dict()
        self.test_data["rew"] = np.empty(0)
        self.test_data["res"] = np.empty(0)
        self.test_data["fail"] = np.empty(0)
        
        self.ep_idx += 1
        return res, rew
    
    ################################################################################################
    ## Call these functions for real-world eval
    ################################################################################################
    def save_eval_real_data(self, num_eval=50, ep=0):
        f_ii = 2
        assert self.train_data_new[str(f_ii)]["X"].shape[0] == len(self.eval_real_queue["acq"])
        idx_array = list(np.sort(np.argsort(np.array(self.eval_real_queue["acq"])).astype(np.int32)[:num_eval]))
        TMP_DATA_PATH = "{}/eval_real_ep_{}.pkl".format(self.log_path, ep)
        data_tmp = [
            [self.eval_real_queue["points"][idx] for idx in idx_array],
            [self.eval_real_queue["idx_new"][idx] for idx in idx_array],
            [self.eval_real_queue["t_set"][idx] for idx in idx_array],
            [self.eval_real_queue["snap_w"][idx] for idx in idx_array],
            self.train_data_new[str(f_ii)]["X"][idx_array, :],
            self.train_data_new[str(f_ii)]["bpoly"][idx_array, :],
            self.train_data_new[str(f_ii)]["len"][idx_array],
            self.train_data_new[str(f_ii)]["info"][idx_array, :],
            self.train_data_new[str(f_ii)]["imp"][idx_array],
        ]
        with open(TMP_DATA_PATH, 'wb') as handle:
            pickle.dump(data_tmp, handle, protocol=4)
        
        self.train_data_new[str(f_ii)]["X"] = self.train_data_new[str(f_ii)]["X"][idx_array, :]
        self.train_data_new[str(f_ii)]["Y"] = -np.ones(len(idx_array))
        self.train_data_new[str(f_ii)]["bpoly"] = self.train_data_new[str(f_ii)]["bpoly"][idx_array, :]
        self.train_data_new[str(f_ii)]["len"] = self.train_data_new[str(f_ii)]["len"][idx_array]
        self.train_data_new[str(f_ii)]["info"] = self.train_data_new[str(f_ii)]["info"][idx_array, :]
        self.train_data_new[str(f_ii)]["imp"] = self.train_data_new[str(f_ii)]["imp"][idx_array]
        
        self.reset_real_queue()
        return
    
    # Call it after save_eval_real_data
    def save_train_data_new(self, ep=0):
        # Wait until all threads finish
        if self.flag_eval_zmq and not self.flag_eval_zmq_testonly:
            res = self.eval_client.req_join(eval_type=0)
            for f_ii in range(2):
                self.train_data_new[str(f_ii)]["Y"] = np.concatenate((self.train_data_new[str(f_ii)]["Y"], res[f_ii]))
                print(res[f_ii].shape)
        else:
            ep_curr = -1
            for f_ii in range(self.num_fidelity):
                for t_ii in range(len(self.eval_th_set[f_ii])):
                    while not self.eval_th_set[f_ii][t_ii].ready():
                        time.sleep(1)
                    res = self.eval_th_set[f_ii][t_ii].get()
                    # print(res)
                    assert res[1] == 0
                    if ep_curr < 0:
                        ep_curr = res[2]
                    assert ep_curr == res[2]
                    self.train_data_new[str(f_ii)]["Y"] = np.concatenate((self.train_data_new[str(f_ii)]["Y"], res[0]))
            self.eval_th_pool.close()
            self.eval_th_pool.join()
            self.eval_th_pool.restart()
            # self.eval_th_pool = ParallelPool(self.max_eval_proc)
            # self.eval_th_pool = ProcessingPool(self.max_eval_proc)
            self.eval_th_set = []
            for f_ii in range(self.num_fidelity):
                self.eval_th_set.append([])
        
        TMP_DATA_PATH = "{}/train_data_new_ep_{}.h5".format(self.log_path, ep)
        h5f = h5py.File(TMP_DATA_PATH, 'w')
        grp_new = h5f.create_group("new")
        grp_fail = h5f.create_group("fail")
        for f_ii in range(self.num_fidelity):
            grp_t = grp_new.create_group("{}".format(f_ii))
            grp_t.create_dataset('X', data=self.train_data_new[str(f_ii)]["X"])
            grp_t.create_dataset('Y', data=self.train_data_new[str(f_ii)]["Y"])
            grp_t.create_dataset('bpoly', data=self.train_data_new[str(f_ii)]["bpoly"])
            grp_t.create_dataset('len', data=self.train_data_new[str(f_ii)]["len"])
            grp_t.create_dataset('info', data=self.train_data_new[str(f_ii)]["info"])
            grp_t.create_dataset('imp', data=self.train_data_new[str(f_ii)]["imp"])
            
            grp_t = grp_fail.create_group("{}".format(f_ii))            
            if "X" in self.train_data_fail[str(f_ii)].keys():
                grp_t.create_dataset('X', data=self.train_data_fail[str(f_ii)]["X"])
                grp_t.create_dataset('Y', data=self.train_data_fail[str(f_ii)]["Y"])
                grp_t.create_dataset('bpoly', data=self.train_data_fail[str(f_ii)]["bpoly"])
                grp_t.create_dataset('len', data=self.train_data_fail[str(f_ii)]["len"])
                grp_t.create_dataset('info', data=self.train_data_fail[str(f_ii)]["info"])
                grp_t.create_dataset('imp', data=self.train_data_fail[str(f_ii)]["imp"])
            else:
                grp_t.create_dataset('X', data=np.empty((0, \
                    self.train_data_new[str(f_ii)]["X"].shape[1], \
                    self.train_data_new[str(f_ii)]["X"].shape[2])))
                grp_t.create_dataset('Y', data=np.empty(0))
                grp_t.create_dataset('bpoly', data=np.empty((0, \
                    self.train_data_new[str(f_ii)]["bpoly"].shape[1])))
                grp_t.create_dataset('len', data=np.empty(0))
                grp_t.create_dataset('info', data=np.empty((0,3)))
                grp_t.create_dataset('imp', data=np.empty(0))
        h5f.close()
        
        return
    
    def join_eval_real(self, ep=0):
        f_ii = 2
        TMP_DATA_PATH = "{}/eval_real_ep_{}.pkl".format(self.log_path, ep)
        with open(TMP_DATA_PATH, 'rb') as handle:
            data_tmp = pickle.load(handle)
        
        res = eval_R(data_tmp[0], data_tmp[1], data_tmp[2], data_tmp[3], ep, flag_wp_update=True)
        
        TMP_DATA_PATH2 = "{}/train_data_new_ep_{}.h5".format(self.log_path, ep)
        h5f = h5py.File(TMP_DATA_PATH2, 'a')
        h5f["new"][str(f_ii)]["Y"][:] = res[0]
        h5f.close()
        
        return
    
    def load_and_update_model(self, flag_skip_init=False, N_update=-1, ep=0):
        TMP_DATA_PATH = "{}/train_data_new_ep_{}.h5".format(self.log_path, ep)
        h5f = h5py.File(TMP_DATA_PATH, 'r')
        for f_ii in range(self.num_fidelity):
            self.train_data_new[str(f_ii)]["X"] = np.array(h5f["new"][str(f_ii)]["X"])
            self.train_data_new[str(f_ii)]["Y"] = np.array(h5f["new"][str(f_ii)]["Y"])
            self.train_data_new[str(f_ii)]["bpoly"] = np.array(h5f["new"][str(f_ii)]["bpoly"])
            self.train_data_new[str(f_ii)]["len"] = np.array(h5f["new"][str(f_ii)]["len"])
            self.train_data_new[str(f_ii)]["info"] = np.array(h5f["new"][str(f_ii)]["info"])
            self.train_data_new[str(f_ii)]["imp"] = np.array(h5f["new"][str(f_ii)]["imp"])
            
            self.train_data_fail[str(f_ii)]["X"] = np.array(h5f["fail"][str(f_ii)]["X"])
            self.train_data_fail[str(f_ii)]["Y"] = np.array(h5f["fail"][str(f_ii)]["Y"])
            self.train_data_fail[str(f_ii)]["bpoly"] = np.array(h5f["fail"][str(f_ii)]["bpoly"])
            self.train_data_fail[str(f_ii)]["len"] = np.array(h5f["fail"][str(f_ii)]["len"])
            self.train_data_fail[str(f_ii)]["info"] = np.array(h5f["fail"][str(f_ii)]["info"])
            self.train_data_fail[str(f_ii)]["imp"] = np.array(h5f["fail"][str(f_ii)]["imp"])
        h5f.close()
        
        est_data_info = np.zeros(2*self.num_fidelity)
        save_data_info = np.zeros(3*self.num_fidelity)
        # Update dataset
        if self.flag_data_managing and self.flag_initialized:
            est_data_info, save_data_info = self.update_train_data(ep=ep)
        else:
            if not flag_skip_init:
                self.save_train_data(ep=ep)
                return
            self.load_train_data()
            if not self.flag_initialized:
                # if not flag_skip_init:
                #     self.build_inducing_points()
                #     self.create_model()
                self.flag_initialized = True
        self.init_train_data()
        
        train_loader = []
        for f_ii in range(self.num_fidelity):
            train_loader.append(iter(DataLoader(self.train_dataset[f_ii], batch_size=self.batch_size, shuffle=True)))
        
        print('Updating reward model ...')
        self.dis_model.train()
        num_ep = self.num_update_ep
        if N_update > 0:
            num_ep = N_update
        
        acc_all = []
        for f_ii in range(self.num_fidelity):
            acc_all.append(0)
        acc_last = 0
        for ep_ii in range(num_ep):
            for f_ii in range(self.num_fidelity):
                N_batch = len(train_loader[f_ii])
                try:
                    x_batch, y_batch, len_batch, bpoly_batch = next(train_loader[f_ii])
                except StopIteration:
                    train_loader[f_ii] = iter(DataLoader(
                        self.train_dataset[f_ii], batch_size=self.batch_size, shuffle=True))
                    x_batch, y_batch, len_batch, bpoly_batch = next(train_loader[f_ii])
                
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()
                len_batch = len_batch.cuda()
                bpoly_batch = bpoly_batch.cuda()
                                
                self.optimizer.zero_grad()
                gp_outputs = self.dis_model.forward_train([x_batch, len_batch, bpoly_batch], fidelity=f_ii+1, eval=False)
                mll_loss = -self.mll(gp_outputs[-1], y_batch)
                reg_loss = to_var(torch.tensor([0.0], requires_grad=True))
                if len(gp_outputs) >= 2:
                    reg_loss = -self.mll(gp_outputs[0], y_batch) * self.coef_reg
                    if len(gp_outputs) >= 3:
                        reg_loss += -self.mll(gp_outputs[1], y_batch) * self.coef_reg_real
                    # for f_ii_t in range(1,len(gp_outputs)-1):
                    #     reg_loss += -self.mll(gp_outputs[f_ii_t], y_batch) * self.coef_reg
                
                loss = mll_loss + reg_loss
                
                loss.backward(retain_graph=True)
                loss_curr = loss.item()
                # avg_loss += loss_curr
                self.optimizer.step()

                # if (ep_ii == 0 or (ep_ii+1) % self.print_every == 0):
                #     print("Ep %d/%d, Eval %s, - Loss: %9.4f" % 
                #           (ep_ii+1, num_ep, self.tags[f_ii], loss_curr/self.batch_size))
                
                try:
                    if ep_ii == 0 or (ep_ii+1) % self.print_every == 0:
                        m, v, pm = self.dis_model.predict_proba([x_batch, len_batch, bpoly_batch], fidelity=f_ii+1)
                        correct = np.abs(pm[:,1] - y_batch.cpu().detach().numpy()) < 0.5
                        acc = np.sum(correct) / correct.shape[0]
                        print('Ep %d/%d, Eval %s - Loss: %.3f - %.3f, Acc: %.3f' % 
                              (ep_ii+1, num_ep, self.tags[f_ii], mll_loss.item()/self.batch_size*1e3, reg_loss.item()/self.batch_size*1e3, acc))
                        acc_all[f_ii] = acc
                except:
                    print("Error in accuracy estimation")
            # epoch_correct /= N_data
            # print('Epoch %d/%d, Acc: %.3f' % (ep_ii+1, self.num_update_ep, epoch_correct))
            # print('---')
        self.scheduler.step()
        if self.flag_data_managing and not flag_skip_init and self.il_type == 0:
            self.run_eval_dataset_th()
        
        rewest_info = dict()
        rewest_info["acc"] = acc_all[1]
        rewest_info["acc_L"] = acc_all[0]
        if self.num_fidelity == 3:
            rewest_info["acc_R"] = acc_all[2]
        rewest_info["est_data"] = est_data_info
        rewest_info["new_data"] = save_data_info
        return rewest_info
    
        