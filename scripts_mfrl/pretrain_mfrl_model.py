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

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

sys.path.insert(0, '../')
from mfrl.reward_estimator import *
from mfrl.training_utils import *
from mfrl.network_models import RewEncoder as GPEnc
from mfrl.network_models import WaypointsEncDec
from naiveBayesOpt.models import *
from pyTrajectoryUtils.pyTrajectoryUtils.minSnapTrajectory import *

def expierment_name(args, ts):
    exp_name = str()
    if args.test:
        exp_name += "test_"
    exp_name += "%s" % args.model_name
    if args.flag_robot:
        exp_name += "_robot"
    return exp_name

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

def load_data_presample_batch(epoch=0, idx=0, tag="sta_am"):
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
    der_batch = torch.tensor(h5f["{}".format(ep_b)]["der"][idx*batch_size:(idx+1)*batch_size,:]).float().cuda()
    h5f.close()
    return x_batch, len_batch, der_batch

class Hook():
    def __init__(self, name, module, backward=False):
        self.name = name
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()

def main(args):
    h5f_filedir = "../dataset/mfrl_train_fidelity_ratio.h5"
    h5f = h5py.File(h5f_filedir, 'r')
    alpha_margin = np.mean(np.array(h5f["alpha_diff"]))
    R_alpha_margin = np.mean(np.array(h5f["R_alpha_diff"]))
    h5f.close()
    
    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())
    min_snap = MinSnapTrajectory(drone_model="STMCFB", N_POINTS=40)
    
    MSE = torch.nn.MSELoss(reduction ='sum')
    CE = torch.nn.CrossEntropyLoss(reduction ='sum')
    tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ###########################################################
    points_scale = np.array([9.,9.,3.])
    num_inducing = 128
    alpha_neg = 0.9
    if args.flag_robot:
        num_fidelity = 3
    else:
        num_fidelity = 2
    ###########################################################
    
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
    
    model = WaypointsEncDec(**params)
    gp_enc = GPEnc(**params)

    if torch.cuda.is_available():
        print(torch.cuda.is_available())
        model = model.cuda()
        gp_enc = gp_enc.cuda()
    
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            if "bias" in m.__dict__:
                m.bias.data.fill_(0.01)

    model.apply(init_weights)
    gp_enc.apply(init_weights)
    
    ###########################################################
    MIN_ITER = 0
    if args.load:      
        PATH = "{}/model/ep_{}.pth.tar".format(args.load_dir, args.load_model_epoch)
        checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
        MIN_ITER = np.int(checkpoint['epoch'])
        if args.load_new_log:
            MIN_ITER = 0
        model.load_state_dict(checkpoint['model_state_dict'])
    # ###########################################################
    # opt_params = list(model.parameters_action()) + list(gp_enc.parameters())
    # optimizer = torch.optim.Adam(opt_params, lr=args.learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.995)
    # ###########################################################
    
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
        writer.add_text("model", str(model))
        writer.add_text("args", str(args))
        writer.add_text("ts", ts)

    save_model_path = os.path.join(args.save_model_path, ts)
    os.makedirs(save_model_path)

    # with open(os.path.join(save_model_path, 'model_params.json'), 'w') as f:
    #     json.dump(params, f, indent=4)
    ###########################################################
    
    flag_debug = True
    procs = ["Train", "Train_H", "Train_R", "Test"]
    tags = ["sta", "sim", "real", "test"]
    b_ii = [[0,0],[0,0],[0,0],[0,0]]
    
    N_batch_all = []
    for i in range(len(procs)):
        N_batch_all.append(load_data_presample_size(epoch=0, tag=tags[i]))
        
    def get_data(tag_idx):
        x_batch, len_batch, der_batch = load_data_presample_batch(epoch=b_ii[tag_idx][0], idx=b_ii[tag_idx][1], tag=tags[tag_idx])
        b_ii[tag_idx][1] += 1
        if b_ii[tag_idx][1] >= N_batch_all[tag_idx]:
            b_ii[tag_idx][0] += 1
            b_ii[tag_idx][1] = 0
            N_batch_all[tag_idx] = load_data_presample_size(epoch=b_ii[tag_idx][0], tag=tags[tag_idx])
        return x_batch, len_batch, der_batch
    
    for epoch in range(MIN_ITER, args.epochs+1):        
        # Build inducing points
        if epoch == 0:
            ip_x, ip_len, ip_der = load_data_presample_batch(epoch=0, idx=0, tag=tags[1])
            ip_x = ip_x[:num_inducing]
            ip_len = ip_len[:num_inducing]
            ip_der = ip_der[:num_inducing]
            train_z = []
            for f_ii in range(num_fidelity):
                if args.random_init:
                    perm_idx = torch.randperm(num_inducing)
                    p_idx = perm_idx[int(num_inducing/2):]
                    n_idx = perm_idx[:int(num_inducing/2)]
                    ip_y = torch.ones(num_inducing).cuda()
                    ip_y[n_idx] = 0
                    a_p = torch.rand(int(num_inducing/2))*0.5 + 1.
                    a_n = 1.0 - torch.rand(int(num_inducing/2))*0.5
                    a_p = a_p.cuda()
                    a_n = a_n.cuda()
                    ip_x[p_idx,:,6] *= a_p.unsqueeze(1).repeat(1,ip_x.shape[1])
                    ip_x[n_idx,:,6] *= a_n.unsqueeze(1).repeat(1,ip_x.shape[1])
                else:
                    perm_idx = torch.randperm(num_inducing)[:int(num_inducing/2)]
                    ip_y = torch.ones(num_inducing).cuda()
                    ip_y[perm_idx] = 0
                    ip_x[perm_idx,:,6] *= alpha_neg
                print(ip_x.shape)
                ip_z, _ = gp_enc(torch.swapaxes(ip_x, 0, 1), ip_len, ip_der)
                if f_ii > 0:
                    ip_z = torch.cat([ip_z, ip_y_prev.unsqueeze(1)], axis=1)
                ip_y_prev = ip_y
                train_z.append(np2cuda2(ip_z))
            dis_model = MFCondS2SDeepGPC(train_z, gp_enc).cuda()
            ###########################################################
            # opt_params = list(model.parameters_action()) + list(dis_model.parameters()) + list(gp_enc.parameters())
            opt_params = list(model.parameters())
            optimizer = torch.optim.Adam(opt_params, lr=args.learning_rate)
            opt_params_dec = list(dis_model.parameters())
            optimizer_gp = torch.optim.Adam(opt_params_dec, lr=args.learning_rate_dec)
            # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.995)
            MLL = VariationalELBO(dis_model.likelihood, dis_model, args.batch_size)
            ###########################################################
        
        for p_ii, proc in enumerate(procs):
            if p_ii == 3 and epoch % args.save_every != 0:
                continue
            if not args.flag_robot and p_ii == 2:
                continue
            
            if p_ii == 0 or p_ii == 1:
                model.train()
                dis_model.feature_network.train()
                dis_model.train()
                flag_train=True
                # for param in dis_model.feature_network.enc.parameters():
                #     param.requires_grad = True
            elif p_ii == 2:
                model.train()
                dis_model.feature_network.train()
                dis_model.train()
                flag_train=True
                # for param in dis_model.feature_network.enc.parameters():
                #     param.requires_grad = False
            else:
                model.eval()
                dis_model.feature_network.eval()
                dis_model.eval()
                flag_train=False
            
            if p_ii == 0:
                x_batch, len_batch, der_batch = get_data(tag_idx=0)
                x_batch_gp = x_batch.clone()
                len_batch_gp = len_batch.clone()
                der_batch_gp = der_batch.clone()
                num_fidelity_t = 1
            elif p_ii == 1:
                x_batch, len_batch, der_batch = get_data(tag_idx=0)
                x_batch[:,:,6] *= alpha_margin
                for d_ii in range(4):
                    der_batch[:,d_ii:12:4] /= alpha_margin**(d_ii+1)
                der_batch[:,12] /= alpha_margin
                der_batch[:,13] /= alpha_margin**2
                x_batch_gp, len_batch_gp, der_batch_gp = get_data(tag_idx=1)
                num_fidelity_t = 2
            elif p_ii == 2:
                x_batch, len_batch, der_batch = get_data(tag_idx=0)
                x_batch[:,:,6] *= R_alpha_margin
                for d_ii in range(4):
                    der_batch[:,d_ii:12:4] /= R_alpha_margin**(d_ii+1)
                der_batch[:,12] /= R_alpha_margin
                der_batch[:,13] /= R_alpha_margin**2
                x_batch_gp, len_batch_gp, der_batch_gp = get_data(tag_idx=2)
                num_fidelity_t = 3
            elif p_ii == 3:
                x_batch, len_batch, der_batch = get_data(tag_idx=3)
                x_batch_gp = x_batch.clone()
                len_batch_gp = len_batch.clone()
                der_batch_gp = der_batch.clone()
                num_fidelity_t = 2
            
            if args.random_init:
                perm_idx = torch.randperm(args.batch_size)
                p_idx = perm_idx[int(args.batch_size/2):]
                n_idx = perm_idx[:int(args.batch_size/2)]
                y_batch_gp = torch.ones(args.batch_size).cuda()
                y_batch_gp[n_idx] = 0
                a_p = torch.rand(int(args.batch_size/2))*0.5 + 1.
                a_n = 1.0 - torch.rand(int(args.batch_size/2))*0.5
                a_p = a_p.cuda()
                a_n = a_n.cuda()
                x_batch_gp[p_idx,:,6] *= a_p.unsqueeze(1).repeat(1,ip_x.shape[1])
                x_batch_gp[n_idx,:,6] *= a_n.unsqueeze(1).repeat(1,ip_x.shape[1])
            else:
                perm_idx = torch.randperm(args.batch_size)[:int(args.batch_size/2)]
                y_batch_gp = torch.ones(args.batch_size).cuda()
                y_batch_gp[perm_idx] = 0
                x_batch_gp[perm_idx,:,6] *= alpha_neg
                        
            x_batch = torch.swapaxes(x_batch, 0, 1)
            
            outputs_wp, outputs_time, outputs_snapw, mean, logv = model(x_batch, len_batch, initial=der_batch, train=flag_train, fidelity=num_fidelity_t)
            gp_outputs = dis_model.forward_train([x_batch_gp, len_batch_gp, der_batch_gp[:,:]], fidelity=num_fidelity_t, eval=not flag_train)
                
            # loss calculation
            target = x_batch.clone()
            MSE_wp = MSE(outputs_wp, target[:,:,:6])
            MSE_time = 0
            MSE_snapw = 0
            if logv is None:
                KL_loss = to_var(torch.tensor([0.0], requires_grad=True))
            else:
                KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())

            for idx in range(args.batch_size):
                MSE_time += MSE(outputs_time[1:,idx], target[1:,idx,6]) * (len_batch.data[idx]-1)
                MSE_snapw += MSE(outputs_snapw[1:,idx], target[1:,idx,7]) * (len_batch.data[idx]-1)

            loss = (args.coef_wp * MSE_wp + MSE_time + MSE_snapw + args.coef_kl * KL_loss) / args.batch_size
                
            # GP loss calculation
            gp_loss = -MLL(gp_outputs[-1], y_batch_gp)
            reg_loss = to_var(torch.tensor([0.0], requires_grad=True))
            if num_fidelity_t >= 2:
                reg_loss = -MLL(gp_outputs[0], y_batch_gp) * args.coef_reg
                for f_ii in range(1,num_fidelity_t-1):
                    reg_loss += -MLL(gp_outputs[f_ii], y_batch_gp) * args.coef_reg
                gp_loss += reg_loss

            if p_ii == 0 or p_ii == 1 or p_ii == 2:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                optimizer_gp.zero_grad()
                gp_loss.backward(retain_graph=True)
                optimizer_gp.step()

            if flag_debug and epoch % args.save_every == 0:
                print("======================")
                print("time [output/gp/label]")
                prRed(outputs_time[:,0])
                prYellow(target[:,0,6])
                print("----")
                print("snapw [output/gp/label]")
                prRed(outputs_snapw[:,0])
                prYellow(target[:,0,7])

            if args.tensorboard_logging:
                writer.add_scalar("%s/Loss" % proc.upper(), loss.item(), epoch)
                writer.add_scalar("%s/MSE_wp" % proc.upper(), MSE_wp.item()/args.batch_size, epoch)
                writer.add_scalar("%s/MSE_time" % proc.upper(), MSE_time.item()/args.batch_size, epoch)
                writer.add_scalar("%s/MSE_snapw" % proc.upper(), MSE_snapw.item()/args.batch_size, epoch)
                writer.add_scalar("%s/KL_loss" % proc.upper(), KL_loss.item()/args.batch_size, epoch)
                writer.add_scalar("%s/GP_Loss" % proc.upper(), gp_loss.item(), epoch)

            if epoch % args.print_every == 0:
                m, v, pm = dis_model.predict_proba([x_batch_gp, len_batch_gp, der_batch_gp], fidelity=num_fidelity_t)
                correct = np.abs(pm[:,1] - y_batch_gp.cpu().detach().numpy()) < 0.5
                acc = np.sum(correct) / correct.shape[0]
                print("Epoch %d/%d, %s Loss %9.4f, WP %9.4f, time %9.4f, snapw %9.4f, KL %9.4f, GP %9.4f, Acc %.3f"
                  % (epoch, args.epochs, proc, loss.item(), 
                     MSE_wp.item()/args.batch_size, MSE_time.item()/args.batch_size,
                     MSE_snapw.item()/args.batch_size, KL_loss.item()/args.batch_size,
                     gp_loss.item()/args.batch_size, acc))
            
            if p_ii == 3 and args.tensorboard_logging and (epoch+1) % args.save_every == 0:
                # Add figure in numpy "image" to TensorBoard writer
                length = int(len_batch.data[0])
                points_i = target.data[:length,0,:3].cpu().numpy()
                points_f = outputs_wp.data[:length,0,:3].cpu().numpy()
                der_t = der_batch.data[0].cpu().numpy()
                t_set = outputs_time.data[1:length,0].cpu().numpy()
                snap_w = outputs_snapw.data[1:length,0].cpu().numpy()
                img = plot_output_der(min_snap, points_i, points_f, t_set, snap_w, der_t)
                writer.add_image("%s/output_wp" % proc.upper(), img, epoch * N_batch + minibatch_i)
        

        if epoch%args.save_every==0 and args.tensorboard_logging:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'enc_state_dict': gp_enc.state_dict(),
                'dis_state_dict': dis_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'optimizer_gp_state_dict': optimizer_gp.state_dict(),
                }, os.path.join(model_path, "ep_{}.pth.tar".format(epoch)))
            
            ip_data = dis_model.get_inducing_points()
            h5f = h5py.File(os.path.join(model_path, "ep_{}_ip.h5".format(epoch)), 'w')
            for f_ii in range(num_fidelity):
                h5f.create_dataset("{}".format(f_ii), data=ip_data[f_ii], 
                    maxshape=(None, ip_data[f_ii].shape[1]), chunks=True)
            h5f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='../dataset')
    parser.add_argument('--dataset', type=str, default='MFRL_dataset')
    parser.add_argument('-mn', '--model_name', type=str, default='mfrl_pretrain')
    parser.add_argument('--create_data', action='store_true')
    parser.add_argument('--max_sequence_length', type=int, default=14)
    parser.add_argument('--min_sequence_length', type=int, default=5)
    parser.add_argument('--min_occ', type=int, default=1)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('-ri', '--random_init', action='store_true')
    parser.add_argument('-robot', dest='flag_robot', action='store_true', help='run real_world experiment')

    parser.add_argument('-ep', '--epochs', type=int, default=100000)
    parser.add_argument('-bs', '--batch_size', type=int, default=200)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-lrd', '--learning_rate_dec', type=float, default=0.0001)
    parser.add_argument('-rd', '--rnn_dropout', type=float, default=0.0)
    parser.add_argument('-cwp', '--coef_wp', type=float, default=1.0)
    parser.add_argument('-ckl', '--coef_kl', type=float, default=0.0001)
    parser.add_argument('-ckld', '--coef_kl_dis', type=float, default=0.01)
    parser.add_argument('-cgp', '--coef_gp', type=float, default=0.1)
    parser.add_argument('-an', '--alpha_neg', type=float, default=0.9)

    parser.add_argument('-wpd', '--waypoints_dim', type=int, default=7)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-rhs', '--rnn_hidden_size', type=int, default=256)
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')
    parser.add_argument('-ls', '--latent_size', type=int, default=64)
    parser.add_argument('-cgr', '--coef_reg', type=float, default=0.0001)
    parser.add_argument('-dr', '--data_ratio', type=float, default=1.0)
    parser.add_argument('-emb_h', '--emb_hid_dim', type=int, default=64)

    # Load model
    parser.add_argument('-l', '--load', action='store_true')
    parser.add_argument('-ld', '--load_dir', type=str, default='../logs/directyaw_ms/test')
    parser.add_argument('--load_model_epoch', type=int, default=10000)
    parser.add_argument('-lnew', '--load_new_log', action='store_true')
    
    # Load training data
    parser.add_argument('-dmin', '--data_min_dim', type=int, default=5)
    parser.add_argument('-dmax', '--data_max_dim', type=int, default=14)
    parser.add_argument('-smin', '--min_seg_len', type=int, default=3)
    
    parser.add_argument('-v', '--print_every', type=int, default=100)
    parser.add_argument('-p', '--plot_every', type=int, default=100)
    parser.add_argument('-s', '--save_every', type=int, default=5000)
    parser.add_argument('-tb', '--tensorboard_logging', action='store_true')
    parser.add_argument('-log', '--logdir', type=str, default='../logs/mfrl_pretrain')
    parser.add_argument('-bin', '--save_model_path', type=str, default='../bin')

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']

    main(args)
