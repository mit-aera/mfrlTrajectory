#!/usr/bin/env python
# coding: utf-8

import os, sys, io, random
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

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Normal
from torch.utils.data import DataLoader
from torch.autograd import Variable

sys.path.insert(0, '../')
from mfrl.reward_estimator import *
from mfrl.training_utils import *
from mfrl.network_models import RewEncoder as GPEnc
from mfrl.network_models import WaypointsEncDec
from naiveBayesOpt.models import *
from pyTrajectoryUtils.pyTrajectoryUtils.minSnapTrajectory import *

def load_data_presample_size(epoch=0, tag="sta_am"):
    ep_t = copy.deepcopy(epoch)
    if ep_t >= 50:
        ep_t = ep_t % 50
    ep_a = int(ep_t / 10)
    ep_b = ep_t % 10
    h5f_filedir = "../dataset/mfrl_online/mfrl_presample_batch_{}_{}.h5".format(tag, ep_a)
    h5f = h5py.File(h5f_filedir, 'r')
    N_data = int(h5f["{}".format(ep_b)]["num_batch"][0])
    h5f.close()
    return N_data

def load_data_presample_batch(epoch=0, idx=0, tag="sta"):
    batch_size = 200
    points_scale = np.array([9., 9., 3.])
    ep_t = copy.deepcopy(epoch)
    if ep_t >= 50:
        ep_t = ep_t % 50
    ep_a = int(ep_t / 10)
    ep_b = ep_t % 10
    h5f_filedir = "../dataset/mfrl_online/mfrl_presample_batch_{}_{}.h5".format(tag, ep_a)
    h5f = h5py.File(h5f_filedir, 'r')
    x_batch = torch.tensor(h5f["{}".format(ep_b)]["seq"][idx*batch_size:(idx+1)*batch_size,:,:]).float().cuda()
    len_batch = torch.tensor(h5f["{}".format(ep_b)]["len"][idx*batch_size:(idx+1)*batch_size]).float().cuda()
    bpoly_batch = torch.tensor(h5f["{}".format(ep_b)]["der"][idx*batch_size:(idx+1)*batch_size,:]).float().cuda()
    x_batch_i = torch.tensor(h5f["{}".format(ep_b)]["seqi"][idx*batch_size:(idx+1)*batch_size,:,:]).float().cuda()
    len_batch_i = torch.tensor(h5f["{}".format(ep_b)]["leni"][idx*batch_size:(idx+1)*batch_size]).float()
    h5f.close()
    return x_batch, len_batch, bpoly_batch, x_batch_i, len_batch_i

def expierment_name(args, ts):
    exp_name = str()
    if args.test:
        exp_name += "test_"
    exp_name += "%s" % args.model_name
    if args.flag_robot:
        exp_name += "_robot"
    return exp_name

def get_time_coef(epoch, ppo_time_coef_ep):
    return max(0., 0.8*(1.-1.*epoch/ppo_time_coef_ep))

def get_ent_w(epoch, ent_max_decay, ent_w):
    return max(0., ent_w*(1.-1.*epoch/ent_max_decay))

def main(args):
    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())
    min_snap = MinSnapTrajectory(drone_model="STMCFB", N_POINTS=40)

    MSE = torch.nn.MSELoss(reduction ='sum')
    tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    points_scale = np.array([9.,9.,3.])
    #####################################################################
    if args.flag_robot:
        num_evals = np.array([16, 1, 100])
        num_evals_init = np.array([16, 1, 100])
        num_fidelity = 3
    else:
        num_evals = np.array([16, 1])
        num_evals_init = np.array([16, 1])
        num_fidelity = 2
    #####################################################################
    
    params = dict(
        device=device,
        emb_dim=8,
        emb_hid_dim=args.emb_hid_dim,
        max_seq_len=14,
        min_seq_len=5,
        num_fidelity=num_fidelity,
        hid_dim=args.hidden_size,
        dropout=args.rnn_dropout,
        emb_n_layers=2,
        latent_size=args.latent_size,
        enc_bern_n_layers=3,
        enc_fv_n_layers=2,
        vae_n_layers=2,
        dec_wp_n_layers=3,
        dec_time_n_layers=3,
        dec_snapw_n_layers=3,
    )
    policy = WaypointsEncDec(**params)
    policy_old = WaypointsEncDec(**params)
    gp_enc = GPEnc(**params)
    
    action_std = torch.tensor([args.ppo_action_std, args.ppo_action_std_snap], requires_grad=True, device=torch.device("cuda:0"))
    
    #####################################################################
    h5f_filedir = "../dataset/mfrl_train_fidelity_ratio.h5"
    h5f = h5py.File(h5f_filedir, 'r')
    ALPHA_MARGIN_T = np.mean(np.array(h5f["alpha_diff"]))
    R_ALPHA_MARGIN_T = np.mean(np.array(h5f["R_alpha_diff"]))
    h5f.close()
    #####################################################################
    
    load_dataset = False
    if args.load and not args.load_new_log:
        rew_log_path = args.load_dir
        load_dataset = True
    else:
        rew_log_path = os.path.join(args.logdir, expierment_name(args, ts))
    
    #####################################################################
    
    if torch.cuda.is_available():
        print("Cuda is_available: {}".format(torch.cuda.is_available()))
        policy = policy.cuda()
        policy_old = policy_old.cuda()
        gp_enc = gp_enc.cuda()
    
    flag_load_pretrain = True
    if flag_load_pretrain and not args.load:
        pre_model_name = "mfrl_pretrain"
        if args.flag_robot:
            pre_model_name += "_robot"
        pre_model_epoch = 40000
        print("Loading dataset: {}".format(pre_model_name))
        
        pre_load_data_path = "../logs/mfrl_pretrain/{}".format(pre_model_name)
        PATH_init = "{}/model/ep_{}.pth.tar".format(pre_load_data_path, pre_model_epoch)
        checkpoint_init = torch.load(PATH_init, map_location=torch.device('cuda'))
        policy.load_state_dict(checkpoint_init['model_state_dict'])
        policy_old.load_state_dict(checkpoint_init['model_state_dict'])
    else:
        def init_weights(m):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.normal_(param.data, mean=0, std=0.01)
                else:
                    nn.init.constant_(param.data, 0)
        policy.apply(init_weights)
        policy_old.apply(init_weights)
        gp_enc.apply(init_weights)
    
    #####################################################################        
    rew_estimator = RewardEstimator(
        gp_enc, 
        gp_enc_args = params,
        num_fidelity=num_fidelity,
        latent_size=args.latent_size, 
        min_seq_len=args.min_seq_len, 
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        num_inducing=128,
        log_path=rew_log_path,
        points_scale=points_scale,
        load_dataset=load_dataset,
        load_ep=args.load_model_epoch,
        coef_wp=args.coef_wp,
        coef_kl=args.coef_kl_dis,
        coef_gp=args.coef_gp,
        flag_eval_zmq=args.eval_zmq,
        zmq_server_ip=args.zmq_server_ip,
        al_type=args.rew_al_type, il_type=args.rew_il_type,
        coef_reg=args.coef_reg, coef_reg_real=args.coef_reg_real, 
        bs_est=args.bs_est,
        rew_max=args.rew_max, rew_min=args.rew_min,
        rew_bias=args.rew_bias, flag_eval_zmq_testonly=True
    )
    if flag_load_pretrain and not args.load:
        path_inducing = "{}/model/ep_{}_ip.h5".format(pre_load_data_path, pre_model_epoch)
        rew_estimator.load_inducing_points_pre(path_inducing)
        rew_estimator.create_model()
        PATH_init = "{}/model/ep_{}.pth.tar".format(
            pre_load_data_path, pre_model_epoch)
        checkpoint_init = torch.load(PATH_init, map_location=torch.device('cuda'))
        rew_estimator.gp_enc.load_state_dict(checkpoint_init['enc_state_dict'])
        rew_estimator.dis_model.load_state_dict(checkpoint_init['dis_state_dict'])
    
    #####################################################################
    # optimizer = torch.optim.Adam(list(policy.dec.parameters()), lr=args.learning_rate)
    opt_params = list(policy.parameters()) + [action_std]
    # opt_params = list(policy.parameters())
    optimizer = torch.optim.Adam(opt_params, lr=args.learning_rate)
    # optimizer.add_param_group({"params": action_std})
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.learning_rate_decay)
    
    #####################################################################
    MIN_ITER = 0
    if args.load:
        PATH = "{}/model/ep_{}.pth.tar".format(args.load_dir, args.load_model_epoch)
        checkpoint = torch.load(PATH, map_location=torch.device('cuda'))
        policy.load_state_dict(checkpoint['model_state_dict'])
        policy_old.load_state_dict(checkpoint['model_state_dict'])
        ppo_std = torch.from_numpy(np.array(checkpoint['ppo_action_std'])).cuda()
        action_std.data[0] = ppo_std[0]
        action_std.data[1] = ppo_std[1]
        action_std_old = action_std.clone()
        rew_estimator.gp_enc.load_state_dict(checkpoint['enc_state_dict'])
        rew_estimator.dis_model.load_state_dict(checkpoint['dis_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        rew_estimator.optimizer.load_state_dict(checkpoint['rew_optimizer_state_dict'])
        rew_estimator.scheduler.load_state_dict(checkpoint['rew_scheduler_state_dict'])
        batch_ep_info = np.array(checkpoint['batch_ep_info'])
        MIN_ITER = args.load_model_epoch + 1
        print("load data ep {}, batch_ep_info: {} - {}".format(args.load_model_epoch, batch_ep_info[0][0], batch_ep_info[0][1]))
    #####################################################################

    if args.tensorboard_logging:
        if args.load and not args.load_new_log:
            log_path = args.load_dir
        else:
            log_path = os.path.join(args.logdir, expierment_name(args, ts))
            if not os.path.exists(log_path):
                os.makedirs(log_path)
        model_path = os.path.join(log_path, "model")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        writer = SummaryWriter(log_path)
        writer.add_text("model", str(policy))
        writer.add_text("args", str(args))
        writer.add_text("ts", ts)

    save_model_path = os.path.join(args.save_model_path, ts)
    os.makedirs(save_model_path)

    # with open(os.path.join(save_model_path, 'model_params.json'), 'w') as f:
    #     json.dump(params, f, indent=4)
    #####################################################################
    
    if args.train_step == 0:    
        # Enable/Disable Dropout
        policy.train()
        
        tags = ["sta"]
        b_ii = [[0,0]]
        N_batch_all = [load_data_presample_size(epoch=0, tag=tags[0])]
        
        def get_data():
            x_batch, len_batch, bpoly_batch, x_batch_i, len_batch_i = load_data_presample_batch(epoch=b_ii[0][0], idx=b_ii[0][1], tag=tags[0])
            b_ii[0][1] += 1
            if b_ii[0][1] >= N_batch_all[0]:
                b_ii[0][1] = 0
                b_ii[0][0] += 1
                N_batch_all[0] = load_data_presample_size(epoch=b_ii[0][0], tag=tags[0])
            return x_batch, len_batch, bpoly_batch, x_batch_i, len_batch_i
        #####################################################################
        
        policy.train()
        for minibatch_i in range(args.num_batch):
            x_batch, len_batch, bpoly_batch, x_batch_i, len_batch_i = get_data()
            data_len = len_batch.cpu().detach().numpy().astype(np.int32)
            data_len_i = len_batch_i.cpu().detach().numpy().astype(np.int32)
            
            #############################################################
            # Initialization
            if minibatch_i == 0:
                print("Initializing discriminator")
            
            for f_ii in range(num_fidelity):
                data_init = x_batch_i.clone()
                data_opt = x_batch.clone()
                data_bpoly = bpoly_batch.clone()
                x_batch_t = x_batch.clone()
                if f_ii == 1:
                    data_init[:,:,6] *= ALPHA_MARGIN_T
                    data_opt[:,:,6] *= ALPHA_MARGIN_T
                    x_batch_t[:,:,6] *= ALPHA_MARGIN_T
                    for d_ii in range(4):
                        data_bpoly[:,d_ii:12:4] /= ALPHA_MARGIN_T**(d_ii+1)
                    data_bpoly[:,12] /= ALPHA_MARGIN_T
                    data_bpoly[:,13] /= ALPHA_MARGIN_T**2
                elif f_ii == 2:
                    data_init[:,:,6] *= R_ALPHA_MARGIN_T
                    data_opt[:,:,6] *= R_ALPHA_MARGIN_T
                    x_batch_t[:,:,6] *= R_ALPHA_MARGIN_T
                    for d_ii in range(4):
                        data_bpoly[:,d_ii:12:4] /= R_ALPHA_MARGIN_T**(d_ii+1)
                    data_bpoly[:,12] /= R_ALPHA_MARGIN_T
                    data_bpoly[:,13] /= R_ALPHA_MARGIN_T**2
                x_batch_t = torch.swapaxes(x_batch_t, 0, 1)
                outputs_wp, outputs_time, outputs_snapw, mean, logv = policy.forward(x_batch_t, len_batch, data_bpoly, train=False, fidelity=f_ii+1)
                data_opt[:,:,6] = torch.swapaxes(outputs_time.clone(), 0, 1)
                data_opt[:,:,7] = torch.swapaxes(outputs_snapw.clone(), 0, 1)
                rew_estimator.get_rewards_init(data_init, data_opt, data_bpoly, len_batch, len_batch_i, num_evals=num_evals_init, fidelity=f_ii+1)

            # Get Rewards
            if minibatch_i == 0:
                print("---")
                print("time [output/label]")
                prRed(data_opt[0,1:int(len_batch.data[0]),6])
                prYellow(data_init[0,1:int(len_batch_i.data[0]),6])
                print("---")
                print("snapw [output/label]")
                prRed(data_opt[0,1:int(len_batch.data[0]),7])
                prYellow(data_init[0,1:int(len_batch_i.data[0]),7])
                print("---")
            
            if minibatch_i % args.print_every == 0 or minibatch_i+1 == args.num_batch:
                print("Train Batch %04d/%i" % (minibatch_i, args.num_batch))
        if args.flag_robot:
            rew_estimator.save_eval_real_data(num_eval=num_evals_init[2], ep=0)
        rew_estimator.save_train_data_new(ep=0)
        # else:
        #     rew_estimator.update_model(flag_skip_init=False)
        
        return
    
    elif args.train_step == 1:
        if args.flag_robot:
            rew_estimator.join_eval_real(ep=0)
        rew_estimator.load_and_update_model(flag_skip_init=False)
        rew_estimator.save_inducing_points()
        rew_estimator.copy_train_data_tmp(0)
        
        rew_estimator.load_and_update_model(flag_skip_init=True, N_update=200)
        torch.save({
            'epoch': 0,
            'model_state_dict': policy.state_dict(),
            'ppo_action_std': action_std.detach().cpu().numpy(),
            'enc_state_dict': rew_estimator.gp_enc.state_dict(),
            'dis_state_dict': rew_estimator.dis_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'rew_optimizer_state_dict': rew_estimator.optimizer.state_dict(),
            'rew_scheduler_state_dict': rew_estimator.scheduler.state_dict(),
            'batch_ep_info': np.array([[0,0]]),
            'loss': 0,
            }, os.path.join(model_path, "ep_0.pth.tar"))
        rew_estimator.save_inducing_points()
        rew_estimator.copy_train_data(0)
        return
    
    elif args.train_step == 2: # Run real-world eval
        rew_estimator.join_eval_real(ep=args.load_model_epoch)
        rewest_info = rew_estimator.load_and_update_model(ep=args.load_model_epoch)
        
        writer.add_scalar("Rew/eps_acc", rewest_info["acc"], args.load_model_epoch)
        writer.add_scalar("Rew/eps_acc_L", rewest_info["acc_L"], args.load_model_epoch)
        writer.add_scalar("Rew/eps_acc_R", rewest_info["acc_R"], args.load_model_epoch)
        
        for f_ii in range(num_fidelity):
            num_p = rewest_info["est_data"][2*f_ii]
            num_n = rewest_info["est_data"][2*f_ii+1]
            n_data = num_p + num_n
            data_ratio = -1.
            if n_data > 0:
                data_ratio = num_p / n_data
            writer.add_scalar("Rew/f{}_est_data".format(f_ii), n_data, args.load_model_epoch)
            writer.add_scalar("Rew/f{}_est_ratio".format(f_ii), data_ratio, args.load_model_epoch)

            num_p = rewest_info["new_data"][3*f_ii]
            num_n = rewest_info["new_data"][3*f_ii+1]
            num_f = rewest_info["new_data"][3*f_ii+2]
            n_data = num_p + num_n + num_f
            ratio_set = np.zeros(3)
            if n_data > 0:
                ratio_set[0] = num_p / n_data
                ratio_set[1] = num_n / n_data
                ratio_set[2] = num_f / n_data
            writer.add_scalar("Rew/f{}_p_ratio".format(f_ii), ratio_set[0], args.load_model_epoch)
            writer.add_scalar("Rew/f{}_n_ratio".format(f_ii), ratio_set[1], args.load_model_epoch)
            writer.add_scalar("Rew/f{}_f_ratio".format(f_ii), ratio_set[2], args.load_model_epoch)
        
        torch.save({
            'epoch': args.load_model_epoch,
            'model_state_dict': policy.state_dict(),
            'ppo_action_std': action_std.detach().cpu().numpy(),
            'enc_state_dict': rew_estimator.gp_enc.state_dict(),
            'dis_state_dict': rew_estimator.dis_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'rew_optimizer_state_dict': rew_estimator.optimizer.state_dict(),
            'rew_scheduler_state_dict': rew_estimator.scheduler.state_dict(),
            'batch_ep_info': np.array(batch_ep_info),
            # 'loss': avg_loss,
            }, os.path.join(model_path, "ep_{}.pth.tar".format(args.load_model_epoch)))
        rew_estimator.save_inducing_points()
        rew_estimator.copy_train_data(args.load_model_epoch)
        return
    
    else:
        step = 0
        batch_size = args.batch_size
        MAX_ITER = args.epochs + 1
        if MIN_ITER != 0:
            MAX_ITER = args.epochs + MIN_ITER
        
        flag_debug = True
        procs = ["Train", "Test"]
        tags = ["sta", "test"]
        b_ii = copy.deepcopy(batch_ep_info)
        N_batch_all = [load_data_presample_size(epoch=0, tag=tags[0])]

        def get_data():
            x_batch, len_batch, bpoly_batch, x_batch_i, len_batch_i = load_data_presample_batch(epoch=b_ii[0][0], idx=b_ii[0][1], tag=tags[0])
            data_idx = [copy.deepcopy(b_ii[0][0]), copy.deepcopy(b_ii[0][1]), tags[0]]
            b_ii[0][1] += 1
            if b_ii[0][1] >= N_batch_all[0]:
                b_ii[0][0] += 1
                b_ii[0][1] = 0
                N_batch_all[0] = load_data_presample_size(epoch=b_ii[0][0], tag=tags[0])
            return x_batch, len_batch, bpoly_batch, x_batch_i, len_batch_i, data_idx
        
        for epoch in range(MIN_ITER, MAX_ITER):
            # Enable/Disable Dropout
            policy.train()

            print("=============")
            print("Train Epoch %02d/%i" % (epoch, MAX_ITER-1))
            
            ppo_memory = []
            discounted_rewards_set = []
            rew_mask = []
            avg_rew = []
            for f_ii in range(num_fidelity):
                ppo_memory.append([])
                discounted_rewards_set.append([])
                rew_mask.append([])
                avg_rew.append([])
            data_idx_set = []
            
            for minibatch_i in range(args.num_batch):
                rew_info = []
                for f_ii in range(num_fidelity):
                    if f_ii == 0:
                        x_batch, len_batch, bpoly_batch, x_batch_i, len_batch_i, data_idx = get_data()
                        data_idx_set.append(data_idx)
                    else:
                        data_idx = data_idx_set[minibatch_i]
                        x_batch, len_batch, bpoly_batch, x_batch_i, len_batch_i = load_data_presample_batch(epoch=data_idx[0], idx=data_idx[1], tag=data_idx[2])
                    if f_ii == 1:
                        x_batch[:,:,6] *= ALPHA_MARGIN_T
                        x_batch_i[:,:,6] *= ALPHA_MARGIN_T
                        for d_ii in range(4):
                            bpoly_batch[:,d_ii:12:4] /= ALPHA_MARGIN_T**(d_ii+1)
                        bpoly_batch[:,12] /= ALPHA_MARGIN_T
                        bpoly_batch[:,13] /= ALPHA_MARGIN_T**2
                    elif f_ii == 2:
                        x_batch[:,:,6] *= R_ALPHA_MARGIN_T
                        x_batch_i[:,:,6] *= R_ALPHA_MARGIN_T
                        for d_ii in range(4):
                            bpoly_batch[:,d_ii:12:4] /= R_ALPHA_MARGIN_T**(d_ii+1)
                        bpoly_batch[:,12] /= R_ALPHA_MARGIN_T
                        bpoly_batch[:,13] /= R_ALPHA_MARGIN_T**2
                    
                    data_init = x_batch_i.clone()
                    data_opt_i = x_batch.clone()
                    data_opt = x_batch.clone()
                    data_bpoly = bpoly_batch.clone()
                    data_len = len_batch.cpu().detach().numpy().astype(np.int32)
                    data_len_i = len_batch_i.cpu().detach().numpy().astype(np.int32)
                    x_batch = torch.swapaxes(x_batch, 0, 1)

                    #############################################################

                    # Forward pass
                    hidden, enc_src, mean, logv = policy_old.forward_enc(x_batch, len_batch, bpoly_batch)

                    outputs = to_var(torch.zeros(args.max_seq_len, batch_size, args.waypoints_dim+1, dtype=torch.float))
                    action = to_var(torch.zeros(batch_size, args.max_seq_len-1, 2))
                    action_old = to_var(torch.zeros(batch_size, args.max_seq_len-1, 2))
                    action_log_probs = to_var(torch.zeros(batch_size, args.max_seq_len-1, 2))

                    trg = x_batch[0,:,:8].clone()
                    trg[:,6] = 0
                    outputs[0,:,:6] = trg[:,:6]
                    outputs[0,:,6:] = torch.tensor(float('-inf'))
                    
                    for t in range(args.max_seq_len-1):
                        output, hidden, rnn_output = policy_old.forward_dec_single(trg, hidden, enc_src, len_batch, fidelity=f_ii+1)

                        dist_time = Normal(output[:,-2], to_var(action_std_old[0]))
                        dist_snap = Normal(output[:,-1], to_var(action_std_old[1]))
                        action_t = dist_time.sample()
                        action[:,t,0] = action_t.detach()
                        action_log_probs[:,t,0] = dist_time.log_prob(action_t)

                        action_s = dist_snap.sample()
                        action[:,t,1] = action_s.detach()
                        action_log_probs[:,t,1] = dist_snap.log_prob(action_s)

                        action_concat = torch.hstack((torch.unsqueeze(action_t, -1), torch.unsqueeze(action_s, -1)))
                        action_old[:,t,:] = action_concat.detach()

                        output_t = torch.cat((output[:,:6], action_concat), dim=1)
                        outputs[t+1,:,:] = output
                        trg = output_t
                    
                    for k in range(batch_size):
                        action[k, data_len[k]-1:, :] = torch.tensor(float('-inf'))
                        action_old[k, data_len[k]-1:, :] = 0.
                        action_log_probs[k, data_len[k]-1:, :] = 0.
                        outputs[data_len[k]:, k, :] = 0.

                    outputs_yaw = nn.functional.normalize(outputs[:,:,3:5], p=2.0, dim=2)
                    outputs_wp = torch.cat((outputs[:,:,:3], outputs_yaw, outputs[:,:,5:6]), dim = 2)

                    outputs_time = (torch.exp(action[:,:,0]) / (len_batch-1).unsqueeze(1))
                    outputs_snapw =  F.softmax(action[:,:,1], dim=1)

                    # Get Rewards
                    data_opt[:,1:,6] = outputs_time.clone()
                    data_opt[:,1:,7] = outputs_snapw.clone()

                    if minibatch_i == 0:
                        print("---")
                        print("time [output/label]")
                        prRed(data_opt[0,1:int(len_batch.data[0]),6])
                        prYellow(data_init[0,1:int(len_batch_i.data[0]),6])
                        print("---")
                        print("snapw [output/label]")
                        prRed(data_opt[0,1:int(len_batch.data[0]),7])
                        prYellow(data_init[0,1:int(len_batch_i.data[0]),7])
                        print("---")
                    
                    if f_ii == 2 and epoch % args.eval_real_every != 0:
                        rewards, rew_info_t = rew_estimator.get_rewards(
                            data_init, data_opt, data_bpoly, len_batch, len_batch_i, num_evals, fidelity=f_ii+1, flag_eval=False)
                    else:
                        rewards, rew_info_t = rew_estimator.get_rewards(
                            data_init, data_opt, data_bpoly, len_batch, len_batch_i, num_evals, fidelity=f_ii+1, flag_eval=True)
                    rew_info.append(rew_info_t)
                    avg_rew[f_ii].append(np.mean(rewards))

                    rewards_array = torch.zeros(args.max_seq_len-1, batch_size, dtype=torch.float)
                    discounted_rewards = torch.zeros(args.max_seq_len-1, batch_size, dtype=torch.float)
                    for idx in range(args.max_seq_len-2,-1,-1):
                        for batch_idx in range(batch_size):
                            if data_len[batch_idx] == idx+2:
                                discounted_rewards[idx,batch_idx] = torch.from_numpy(rewards[batch_idx:batch_idx+1])
                                rewards_array[idx,batch_idx] = torch.from_numpy(rewards[batch_idx:batch_idx+1])
                            else:
                                if idx < args.max_seq_len-2:
                                    discounted_rewards[idx,batch_idx] = args.ppo_gamma*discounted_rewards[idx+1,batch_idx]
                    discounted_rewards_set[f_ii].extend(list(discounted_rewards.numpy().T))
                    discounted_rewards = to_var(discounted_rewards)
                    # rewards_array = to_var(rewards_array)

                    mask = np.zeros((batch_size, args.max_seq_len-1), dtype=bool)
                    for t in range(args.max_seq_len-1):
                        for k in range(batch_size):
                            if t < data_len[k]-1:
                                mask[k, t] = 1
                    rew_mask[f_ii].extend(list(mask))

                    ppo_memory_t = []
                    ppo_memory_t.append(action_old.detach()) # action
                    ppo_memory_t.append(action_log_probs.detach())
                    ppo_memory_t.append(discounted_rewards)
                    ppo_memory[f_ii].append(ppo_memory_t)

                    if args.tensorboard_logging:
                        log_idx = epoch * args.num_batch + minibatch_i
                        writer.add_scalar("Train/eps_rew_{}".format(f_ii), avg_rew[f_ii][minibatch_i], log_idx)
                        writer.add_scalar("Train/eps_rew_prob_{}".format(f_ii), np.mean(rew_info[f_ii]["prob"]), log_idx)
                        writer.add_scalar("Train/eps_rew_ei_{}".format(f_ii), np.mean(rew_info[f_ii]["ei"]), log_idx)
                        writer.add_scalar("Train/eps_rew_vratio_{}".format(f_ii), np.mean(rew_info[f_ii]["vratio"]), log_idx)

                        if f_ii == num_fidelity-1 and minibatch_i+1 == args.num_batch and epoch % args.plot_every==0 and epoch > 0:
                            length_i = int(len_batch_i.data[0])
                            points_i = data_init.data[0,:length_i,:3].cpu().numpy()
                            length = int(len_batch.data[0])
                            b_poly_f = bpoly_batch.data[0,:].cpu().numpy()
                            points_f = outputs_wp.data[:length,0,:3].cpu().numpy()
                            t_set = outputs_time.data[1:length,0].cpu().numpy()
                            snap_w = outputs_snapw.data[1:length,0].cpu().numpy()
                            img = plot_output_der(min_snap, points_i, points_f, t_set, snap_w, b_poly_f)
                            writer.add_image("Train/output_wp", img, epoch * args.num_batch + minibatch_i)

                    if minibatch_i % args.print_every == 0 or minibatch_i+1 == args.num_batch:
                        print("Train_%i Batch %04d/%i, rew %9.4f" % (f_ii, minibatch_i, args.num_batch, avg_rew[f_ii][minibatch_i]))
            
            # Estimate variance
            rew_mean_std = []
            for f_ii in range(num_fidelity):
                discounted_rewards_set_t = np.array(discounted_rewards_set[f_ii])
                rew_mask_t = np.array(rew_mask[f_ii])
                rew_mean_std.append(np.zeros((args.max_seq_len-1, 2)))
                for t in range(args.max_seq_len-1):
                    N_active = np.sum(rew_mask_t[:,t])
                    adv = discounted_rewards_set_t[rew_mask_t[:,t], t]
                    if N_active > 0:
                        rew_mean_std[f_ii][t, 0] = adv.mean()
                        rew_mean_std[f_ii][t, 1] = adv.std() + 1e-9
                    else:
                        rew_mean_std[f_ii][t, 0] = 0.
                        rew_mean_std[f_ii][t, 1] = 1.
                rew_mean_std[f_ii] = to_var(torch.from_numpy(rew_mean_std[f_ii]))
            
            # Policy update
            for minibatch_i in range(args.num_batch):
                for f_ii in range(num_fidelity):
                    ppo_memory_t = ppo_memory[f_ii][minibatch_i]
                    data_idx = data_idx_set[minibatch_i]
                    x_batch, len_batch, bpoly_batch, x_batch_i, len_batch_i = load_data_presample_batch(epoch=data_idx[0], idx=data_idx[1], tag=data_idx[2])
                    if f_ii == 1:
                        x_batch[:,:,6] *= ALPHA_MARGIN_T
                        x_batch_i[:,:,6] *= ALPHA_MARGIN_T
                        for d_ii in range(4):
                            bpoly_batch[:,d_ii:12:4] /= ALPHA_MARGIN_T**(d_ii+1)
                        bpoly_batch[:,12] /= ALPHA_MARGIN_T
                        bpoly_batch[:,13] /= ALPHA_MARGIN_T**2
                    elif f_ii == 2:
                        x_batch[:,:,6] *= R_ALPHA_MARGIN_T
                        x_batch_i[:,:,6] *= R_ALPHA_MARGIN_T
                        for d_ii in range(4):
                            bpoly_batch[:,d_ii:12:4] /= R_ALPHA_MARGIN_T**(d_ii+1)
                        bpoly_batch[:,12] /= R_ALPHA_MARGIN_T
                        bpoly_batch[:,13] /= R_ALPHA_MARGIN_T**2
                    data_opt_i = x_batch.clone()
                    data_opt = x_batch.clone()
                    data_bpoly = bpoly_batch.clone()
                    data_len = len_batch.cpu().detach().numpy().astype(np.int32)
                    x_batch = torch.swapaxes(x_batch, 0, 1)

                    action_old = ppo_memory_t[0]
                    action_log_probs_old = ppo_memory_t[1]
                    discounted_rewards = ppo_memory_t[2]

                    #############################################################
                    # Forward pass
                    hidden, enc_src, mean, logv = policy.forward_enc(x_batch, len_batch, bpoly_batch)
                    # hidden = hidden.detach()

                    outputs = to_var(torch.zeros(args.max_seq_len, batch_size, args.waypoints_dim+1, dtype=torch.float))
                    action = to_var(torch.zeros(batch_size, args.max_seq_len-1, 2))
                    action_log_probs = to_var(torch.zeros(batch_size, args.max_seq_len-1, 2))

                    trg = x_batch[0,:,:8].clone()
                    trg[:,6] = 0
                    outputs[0,:,:6] = trg[:,:6]
                    outputs[0,:,6:] = torch.tensor(float('-inf'))
                    
                    for t in range(args.max_seq_len-1):
                        output, hidden, rnn_output = policy.forward_dec_single(trg, hidden, enc_src, len_batch, fidelity=f_ii+1)
                        
                        action[:,t,0] = output[:,-2].detach()
                        action[:,t,1] = output[:,-1].detach()
                        
                        action_log_probs[:,t,0] = - np.log(np.sqrt(2 * np.pi)) - torch.log(action_std[0]) - 0.5 / torch.pow(action_std[0],2) * torch.pow(output[:,-2] - action_old[:,t,0],2)
                        action_log_probs[:,t,1] = - np.log(np.sqrt(2 * np.pi)) - torch.log(action_std[1]) - 0.5 / torch.pow(action_std[1],2) * torch.pow(output[:,-1] - action_old[:,t,1],2)

                        output_t = torch.cat((output[:,:6], action_old[:, t, :]), dim=1)
                        outputs[t+1,:,:] = output
                        trg = output_t
                    
                    for k in range(batch_size):
                        action[k, data_len[k]-1:, :] = torch.tensor(float('-inf'))
                        action_log_probs[k, data_len[k]-1:, :] = 0.
                        outputs[data_len[k]:, k, :] = 0.

                    outputs_yaw = nn.functional.normalize(outputs[:,:,3:5], p=2.0, dim=2)
                    outputs_wp = torch.cat((outputs[:,:,:3], outputs_yaw, outputs[:,:,5:6]), dim = 2)

                    outputs_time = (torch.exp(action[:,:,0]) / (len_batch-1).unsqueeze(1))
                    outputs_snapw =  F.softmax(action[:,:,1], dim=1)

                    # Get Rewards
                    data_opt[:,1:,6] = outputs_time.clone()
                    data_opt[:,1:,7] = outputs_snapw.clone()

                    adv_targ = torch.where(discounted_rewards.T != 0., 
                        (discounted_rewards.T - rew_mean_std[f_ii][:,0].tile((batch_size,1))) / rew_mean_std[f_ii][:,1].tile((batch_size,1)), 0.)
                    ratio = torch.exp(action_log_probs[:,:,0] + action_log_probs[:,:,1] - action_log_probs_old[:,:,0] - action_log_probs_old[:,:,1])
                    surr1 = ratio * adv_targ
                    surr2 = torch.clamp(ratio, 1.0 - args.ppo_clip_param, 1.0 + args.ppo_clip_param) * adv_targ
                    policy_loss = - torch.min(surr1, surr2)
                    action_loss = policy_loss.sum() / batch_size
                    
                    ent_loss = torch.log(action_std).sum()
                    
                    MSE_wp = MSE(outputs_wp, torch.swapaxes(data_opt_i[:,:,:6], 0, 1)) / batch_size
                    if logv is None:
                        KL_loss = to_var(torch.tensor([0.0], requires_grad=True)) / batch_size
                    else:
                        KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp()) / batch_size

                    optimizer.zero_grad()
                    loss = (action_loss - args.ppo_entropy_coef * ent_loss + args.coef_wp * MSE_wp + args.coef_kl * KL_loss)
                    loss.backward()
                    optimizer.step()
                    # avg_loss += loss.item()

                    if minibatch_i % args.print_every == 0 or minibatch_i+1 == args.num_batch:
                        print("Train_%i Batch %04d/%i, Loss %9.4f, wp %9.4f, act %9.4f, ent %9.4f, rew %9.4f, KL %9.4f"
                              % (f_ii, minibatch_i, args.num_batch, loss.item(), 
                                 MSE_wp.item(), action_loss.item(), ent_loss.item(), avg_rew[f_ii][minibatch_i], KL_loss.item()))

                    if args.tensorboard_logging and f_ii == num_fidelity-1:
                        log_idx = epoch * args.num_batch + minibatch_i
                        writer.add_scalar("Train/Loss", loss.item(), log_idx)
                        writer.add_scalar("Train/wp_loss", MSE_wp.item(), log_idx)
                        writer.add_scalar("Train/kl_loss", KL_loss.item(), log_idx)
                        writer.add_scalar("Train/ent_loss", ent_loss.item(), log_idx)
                        writer.add_scalar("Train/action_std_time", action_std.detach().cpu().numpy()[0], log_idx)
                        writer.add_scalar("Train/action_std_snap", action_std.detach().cpu().numpy()[1], log_idx)
                        writer.add_scalar("Train/act_loss", action_loss.item(), log_idx)
            
            action_std_old = action_std.clone()
            policy_old.load_state_dict(policy.state_dict())
            
            if epoch > 0:
                scheduler.step()
            
            if args.flag_robot:
                if epoch % args.eval_real_every == 0:
                    rew_estimator.save_eval_real_data(num_eval=num_evals[2], ep=epoch)
                    rew_estimator.save_train_data_new(ep=epoch)
                else:
                    rewest_info = rew_estimator.update_model(ep=epoch, N_update=200)

                    writer.add_scalar("Rew/eps_acc", rewest_info["acc"], epoch)
                    writer.add_scalar("Rew/eps_acc_L", rewest_info["acc_L"], epoch)
                    num_fidelity_t = 2
                    for f_ii in range(num_fidelity_t):
                        num_p = rewest_info["est_data"][2*f_ii]
                        num_n = rewest_info["est_data"][2*f_ii+1]
                        n_data = num_p + num_n
                        data_ratio = -1.
                        if n_data > 0:
                            data_ratio = num_p / n_data
                        writer.add_scalar("Rew/f{}_est_data".format(f_ii), n_data, epoch)
                        writer.add_scalar("Rew/f{}_est_ratio".format(f_ii), data_ratio, epoch)

                        num_p = rewest_info["new_data"][3*f_ii]
                        num_n = rewest_info["new_data"][3*f_ii+1]
                        num_f = rewest_info["new_data"][3*f_ii+2]
                        n_data = num_p + num_n + num_f
                        ratio_set = np.zeros(3)
                        if n_data > 0:
                            ratio_set[0] = num_p / n_data
                            ratio_set[1] = num_n / n_data
                            ratio_set[2] = num_f / n_data
                        writer.add_scalar("Rew/f{}_p_ratio".format(f_ii), ratio_set[0], epoch)
                        writer.add_scalar("Rew/f{}_n_ratio".format(f_ii), ratio_set[1], epoch)
                        writer.add_scalar("Rew/f{}_f_ratio".format(f_ii), ratio_set[2], epoch)

            if epoch % args.save_every==0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': policy.state_dict(),
                    'ppo_action_std': action_std.detach().cpu().numpy(),
                    'enc_state_dict': rew_estimator.gp_enc.state_dict(),
                    'dis_state_dict': rew_estimator.dis_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'rew_optimizer_state_dict': rew_estimator.optimizer.state_dict(),
                    'rew_scheduler_state_dict': rew_estimator.scheduler.state_dict(),
                    'batch_ep_info': np.array(b_ii),
                    # 'loss': avg_loss,
                    }, os.path.join(model_path, "ep_{}.pth.tar".format(epoch)))
                rew_estimator.save_inducing_points()
                rew_estimator.copy_train_data(epoch)
                if args.flag_robot:
                    if epoch % args.eval_real_every == 0:
                        rew_estimator.copy_train_data_tmp(epoch)

            ###################################################################
            if epoch % args.test_every==0:
                # data_loader = test_loader
                # N_batch = len(test_loader)
                # N_data = N_batch * batch_size
                
                test_ep = int(epoch/args.save_every)
                N_batch = load_data_presample_size(epoch=test_ep, tag=tags[1])
                N_data = N_batch * args.batch_size
                
                # test_set = [0, 0.9, 0.9]
                # test_set = [0, 0.95]
                test_set = [0]

                for t_ii in range(len(test_set)):
                    policy.eval()

                    eps_rew_all = 0
                    eps_acc_all = 0
                    loss_all_wp = 0
                    loss_all_time = 0
                    loss_all_snapw = 0
                    prob = []

                    for minibatch_i in range(N_batch):
                        x_batch, len_batch, bpoly_batch, x_batch_i, len_batch_i = load_data_presample_batch(epoch=test_ep, idx=minibatch_i, tag=tags[1])
                        data_init = x_batch_i.clone()
                        data_opt = x_batch.clone()
                        data_opt_i = x_batch.clone()
                        data_bpoly = bpoly_batch.clone()
                        x_batch = torch.swapaxes(x_batch, 0, 1)
                        outputs_wp, outputs_time, outputs_snapw, mean, logv = policy(x_batch, len_batch, bpoly_batch, train=False, fidelity=2)

                        data_opt[:,:,6] = torch.swapaxes(outputs_time.clone(), 0, 1)
                        data_opt[:,:,7] = torch.swapaxes(outputs_snapw.clone(), 0, 1)

                        # Get Rewards
                        if minibatch_i == 0 and t_ii == 0:
                            print("---")
                            print("time [output/label]")
                            prRed(data_opt[0,1:int(len_batch.data[0]),6])
                            prYellow(data_init[0,1:int(len_batch_i.data[0]),6])
                            print("---")
                            print("snapw [output/label]")
                            prRed(data_opt[0,1:int(len_batch.data[0]),7])
                            prYellow(data_init[0,1:int(len_batch_i.data[0]),7])
                            print("---")

                        if t_ii == 0:
                            rew_estimator.dis_model.eval()
                            m, v, pm = rew_estimator.dis_model.predict_proba([data_opt, len_batch, data_bpoly], fidelity=2)
                            prob.extend(list(pm[:,1]))
                            rew_estimator.get_rewards_test(data_init, data_opt, len_batch, len_batch_i)
                        elif t_ii == 1:
                            t_new, snap_w, alpha_grad, res_prob = rew_estimator.test_get_alpha_policy(
                                policy, data_opt, data_bpoly[:,:], len_batch,
                                risk_th=test_set[t_ii], range_min=0.9, range_max=1.1, N_eval=10, N_step=2, flag_return_prob=True)
                            prob.extend(list(res_prob))
                            rew_estimator.get_rewards_test(data_init, data_opt, len_batch, len_batch_i, t_new=t_new, s_new=snap_w)
                        # loss calculation
                        loss_all_wp += MSE(outputs_wp, torch.swapaxes(data_opt_i[:,:,:6], 0, 1)).item()
                        loss_all_time += MSE(outputs_time, torch.swapaxes(data_opt_i[:,:,6], 0, 1)).item()
                        loss_all_snapw += MSE(outputs_snapw, torch.swapaxes(data_opt_i[:,:,7], 0, 1)).item()

                        if args.tensorboard_logging and (minibatch_i+1 == N_batch) and epoch % args.plot_every==0 and t_ii==0:
                            length_i = int(len_batch_i.data[0])
                            points_i = data_init.data[0,:length_i,:3].cpu().numpy()
                            length = int(len_batch.data[0])
                            b_poly_f = bpoly_batch.data[0,:].cpu().numpy()
                            points_f = outputs_wp.data[:length,0,:3].cpu().numpy()
                            t_set = outputs_time.data[1:length,0].cpu().numpy()
                            snap_w = outputs_snapw.data[1:length,0].cpu().numpy()
                            img = plot_output_der(min_snap, points_i, points_f, t_set, snap_w, b_poly_f)
                            writer.add_image("Test/output_wp", img, epoch * N_batch + minibatch_i)

                    res, rew = rew_estimator.join_rewards_th()
                    prob = np.array(prob)
                    correct = np.abs(prob-res) < 0.5
                    pred_acc = np.sum(correct) / correct.shape[0]
                    
                    idx_pos = np.where(res == 1)[0]
                    if len(idx_pos) == 0:
                        rew_pos = np.zeros(1)
                    else:
                        rew_pos = rew[idx_pos]
                    idx_neg = np.where(res == 0)[0]
                    if len(idx_neg) == 0:
                        rew_neg = np.zeros(1)
                    else:
                        rew_neg = rew[idx_neg]
                    eps_rew_all = (np.sum(rew_pos) + np.sum(rew_neg)) / N_data
                    eps_rew_pos_all = np.sum(rew_pos) / rew_pos.shape[0]
                    eps_rew_neg_all = np.sum(rew_neg) / rew_neg.shape[0]
                    eps_acc_all = len(idx_pos) / N_data

                    loss_all_wp /= N_data
                    loss_all_time /= N_data
                    loss_all_snapw /= N_data

                    if t_ii == 0:
                        prGreen("Test Epoch %02d/%i, rew (all/pos/neg) %6.4f / %6.4f / %6.4f, acc %9.4f, pacc %9.4f, wp %9.4f, time %9.4f, snapw %9.4f"
                              % (epoch, MAX_ITER-1, eps_rew_all, eps_rew_pos_all, eps_rew_neg_all, eps_acc_all, pred_acc, 
                                 loss_all_wp, loss_all_time, loss_all_snapw))
                    elif t_ii == 1:
                        prGreen("Test_%s Epoch %02d/%i, rew (all/pos/neg) %6.4f / %6.4f / %6.4f, acc %9.4f, pacc %9.4f, wp %9.4f, time %9.4f, snapw %9.4f"
                              % (test_set[t_ii], epoch, MAX_ITER-1, eps_rew_all, eps_rew_pos_all, eps_rew_neg_all, eps_acc_all, pred_acc, 
                                 loss_all_wp, loss_all_time, loss_all_snapw))

                    if args.tensorboard_logging:
                        if t_ii == 0:
                            writer.add_scalar("Test/wp_loss", loss_all_wp, epoch)
                            writer.add_scalar("Test/time_loss", loss_all_time, epoch)
                            writer.add_scalar("Test/snapw_loss", loss_all_snapw, epoch)
                            writer.add_scalar("Test/eps_rew", eps_rew_all, epoch)
                            writer.add_scalar("Test/eps_rew_pos", eps_rew_pos_all, epoch)
                            writer.add_scalar("Test/eps_rew_neg", eps_rew_neg_all, epoch)
                            writer.add_scalar("Test/eps_acc", eps_acc_all, epoch)
                            writer.add_scalar("Test/eps_pacc", pred_acc, epoch)
                        else:
                            writer.add_scalar("Test_{}/wp_loss".format(test_set[t_ii]), loss_all_wp, epoch)
                            writer.add_scalar("Test_{}/time_loss".format(test_set[t_ii]), loss_all_time, epoch)
                            writer.add_scalar("Test_{}/snapw_loss".format(test_set[t_ii]), loss_all_snapw, epoch)
                            writer.add_scalar("Test_{}/eps_rew".format(test_set[t_ii]), eps_rew_all, epoch)
                            writer.add_scalar("Test_{}/eps_acc".format(test_set[t_ii]), eps_acc_all, epoch)
                            writer.add_scalar("Test_{}/eps_pacc".format(test_set[t_ii]), pred_acc, epoch)
                            time.sleep(10)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='../dataset')
    parser.add_argument('--dataset', type=str, default='MFRL_dataset')
    parser.add_argument('-mn', '--model_name', type=str, default='mfrl')
    parser.add_argument('--create_data', action='store_true')
    parser.add_argument('--max_seq_len', type=int, default=14)
    parser.add_argument('--min_seq_len', type=int, default=5)
    parser.add_argument('--min_occ', type=int, default=1)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('-si', '--skip_init', action='store_true')
    parser.add_argument('-ts', '--train_step', type=int, default=0)
    parser.add_argument('-ez', '--eval_zmq', action='store_true')
    parser.add_argument('-zip', '--zmq_server_ip', type=str, default="tcp://server-ip:5555")
    parser.add_argument('-robot', dest='flag_robot', action='store_true', help='run real_world experiment')

    parser.add_argument('-ep', '--epochs', type=int, default=100000)
    parser.add_argument('-bs', '--batch_size', type=int, default=200)
    parser.add_argument('-nb', '--num_batch', type=int, default=1000)
    
    parser.add_argument('-rd', '--rnn_dropout', type=float, default=0.0)
    parser.add_argument('-wpd', '--waypoints_dim', type=int, default=7)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-rhs', '--rnn_hidden_size', type=int, default=256)
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')
    parser.add_argument('-ls', '--latent_size', type=int, default=64)
    parser.add_argument('-cgr', '--coef_reg', type=float, default=0.0001)
    parser.add_argument('-rcgr', '--coef_reg_real', type=float, default=0.0001)
    parser.add_argument('-emb_h', '--emb_hid_dim', type=int, default=64)

    # PPO param
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001)
    parser.add_argument('-lrd', '--learning_rate_decay', type=float, default=1.0)
    parser.add_argument('-ppoe', '--ppo_epochs', type=int, default=10)
    parser.add_argument('-ppostd', '--ppo_action_std', type=float, default=0.1)
    parser.add_argument('-ppostds', '--ppo_action_std_snap', type=float, default=0.1)
    parser.add_argument('-ppoclip', '--ppo_clip_param', type=float, default=0.1)
    parser.add_argument('-ppoent', '--ppo_entropy_coef', type=float, default=0.01)
    parser.add_argument('-ppog', '--ppo_gamma', type=float, default=0.9)
    
    parser.add_argument('-cwp', '--coef_wp', type=float, default=1.0)
    parser.add_argument('-ckl', '--coef_kl', type=float, default=0.0001)
    parser.add_argument('-ckld', '--coef_kl_dis', type=float, default=0.01)
    parser.add_argument('-cgp', '--coef_gp', type=float, default=0.1)
    
    # Rew param
    parser.add_argument('-ral', '--rew_al_type', type=int, default=2)
    parser.add_argument('-ril', '--rew_il_type', type=int, default=2)
    parser.add_argument('--bs_est', type=int, default=500)
    parser.add_argument('-rmax', '--rew_max', type=float, default=0.2)
    parser.add_argument('-rmin', '--rew_min', type=float, default=-1.5)
    parser.add_argument('-rbias', '--rew_bias', type=float, default=0.2)
    
    # Load model
    parser.add_argument('-l', '--load', action='store_true')
    parser.add_argument('-ld', '--load_dir', type=str, default='../logs/mfrl/test_data')
    parser.add_argument('-lep', '--load_model_epoch', type=int, default=130)
    parser.add_argument('-lnew', '--load_new_log', action='store_true')
    
    # Load training data
    parser.add_argument('-dmin', '--data_min_dim', type=int, default=5)
    parser.add_argument('-dmax', '--data_max_dim', type=int, default=14)
    parser.add_argument('-smin', '--min_seg_len', type=int, default=3)
    
    parser.add_argument('-r', '--eval_real_every', type=int, default=100)
    parser.add_argument('-v', '--print_every', type=int, default=100)
    parser.add_argument('-p', '--plot_every', type=int, default=20)
    parser.add_argument('-s', '--save_every', type=int, default=5)
    parser.add_argument('--test_every', type=int, default=40)
    parser.add_argument('-tb', '--tensorboard_logging', action='store_true')
    parser.add_argument('-log', '--logdir', type=str, default='../logs/mfrl')
    parser.add_argument('-bin', '--save_model_path', type=str, default='../bin')
    parser.add_argument('-g', dest='flag_switch_gpu', action='store_true', help='switch gpu to gpu 1')

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()
    assert args.rnn_type in ['rnn', 'lstm', 'gru']

    if torch.cuda.is_available():
        if args.flag_switch_gpu:
            torch.cuda.set_device(1)
        else:
            torch.cuda.set_device(0)
    
    main(args)
