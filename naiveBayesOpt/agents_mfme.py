#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys, os, copy, time
import yaml
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from tensorboardX import SummaryWriter
from pyDOE import lhs
from scipy.stats import gmean

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import Linear
import torch.nn as nn
import torch.nn.functional as F

import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel, ScaleKernel, LinearKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import AbstractVariationalGP, GP
from gpytorch.mlls import VariationalELBO, AddedLossTerm
from gpytorch.likelihoods import GaussianLikelihood, BernoulliLikelihood
from gpytorch.models.deep_gps import AbstractDeepGPLayer, AbstractDeepGP, DeepLikelihood

from pyTrajectoryUtils.pyTrajectoryUtils.utils import *
from .models import *
from .trajSampler import TrajSampler_online

class NaiveBayesOpt():
    def __init__(self, *args, **kwargs):
        self.X_L = kwargs.get('X_L', None)
        self.Y_L = kwargs.get('Y_L', None)
        self.N_L = self.X_L.shape[0]
        self.X_H = kwargs.get('X_H', None)
        self.Y_H = kwargs.get('Y_H', None)
        self.N_H = self.X_L.shape[0]
        
        self.lb = kwargs.get('lb', None)
        self.ub = kwargs.get('ub', None)
        self.rand_seed = kwargs.get('rand_seed', None)
        self.C_L = kwargs.get('C_L', None)
        self.C_H = kwargs.get('C_H', None)
        self.sampling_func_L = kwargs.get('sampling_func_L', None)
        self.sampling_func_H = kwargs.get('sampling_func_H', None)
        self.t_set_sim = kwargs.get('t_set_sim', None)
        
        self.delta_L = kwargs.get('delta_L', 0.8)
        self.delta_H = kwargs.get('delta_H', 0.4)
        self.beta = kwargs.get('beta', 0.05)
        self.iter_create_model = kwargs.get('iter_create_model', 200)
        self.N_cand = kwargs.get('N_cand', 1000)

        self.model_prefix = kwargs.get('model_prefix', 'mfbo_test')
        self.writer = SummaryWriter('runs/mfbo/'+self.model_prefix)
        self.gpu_batch_size = kwargs.get('gpu_batch_size', 256)
        self.num_eval_L = kwargs.get('num_eval_L', 1)
        self.num_eval_H = kwargs.get('num_eval_H', 1)
        self.flag_load_model = kwargs.get('flag_load_model', True)
        
        self.model_path = kwargs.get('model_path', './')
                
        np.random.seed(self.rand_seed)
        torch.manual_seed(self.rand_seed)
        
        self.t_dim = self.t_set_sim.shape[0]
        
#         self.smooth_traj_sampler = TrajSampler_online(N_sample=4096, N=self.t_dim, sigma=0.2, cov_mode=1, flag_pytorch=False)
#         self.base_sampler = lambda N_sample: self.smooth_traj_sampler.rsample(N_sample=N_sample)
        self.base_sampler_t = TrajSampler_online(N=self.t_dim, sigma=0.2, flag_load=False, cov_mode=1, flag_pytorch=False)
        self.base_sampler_s= TrajSampler_online(N=self.t_dim, sigma=0.2, flag_load=False, cov_mode=1, flag_pytorch=False)
        
        self.min_time = 1.0
        
        self.alpha_min = np.ones(2*self.t_dim)
        self.min_time_cand = np.ones(self.num_eval_H)
        self.alpha_min_cand = np.ones((self.num_eval_H, 2*self.t_dim))
        self.flag_found_ei_L = False
        self.flag_found_ei_H = False
        
        self.train_x_L = torch.tensor(self.X_L).float().cuda()
        self.train_y_L = torch.tensor(self.Y_L).float().cuda()
        self.train_x_H = torch.tensor(self.X_H).float().cuda()
        self.train_y_H = torch.tensor(self.Y_H).float().cuda()
        train_x = [self.train_x_L, self.train_x_H]
        train_y = [self.train_y_L, self.train_y_H]
        
        self.model = MFDeepGPC(train_x, train_y, num_inducing=128).cuda()
        self.optimizer = torch.optim.Adam([{'params': self.model.parameters()},], lr=0.001)
        
        self.min_loss = -1
        
        # Update min_time
        self.min_time_init = np.sum(self.t_set_sim)
        
        self.rel_snap_init = 1.0
        if np.any(self.Y_H==1):
            self.min_time = np.min(self.func_obj(self.X_H[self.Y_H==1,:]))
            min_time_idx = np.argmin(self.func_obj(self.X_H[self.Y_H==1,:]))
            self.alpha_min = self.lb + self.X_H[self.Y_H==1,:][min_time_idx,:]*(self.ub-self.lb)
            Y_init_t = 1.0*self.sampling_func_H(self.X_H[self.Y_H==1,:][min_time_idx:min_time_idx+1,:])
            self.rel_snap_init = Y_init_t[0,1]
        else:
            self.min_time = 2.0
        prGreen("min_time: {}".format(self.min_time))
        
        return
    
    def sample_data(self, N_sample):
        sample_1 = self.base_sampler_t.rsample(N_sample-np.int(N_sample/2))
        sample_2 = lhs(self.t_dim, np.int(N_sample/2))
        res_t = np.concatenate((sample_1,sample_2),axis=0)
        perm_idx_t = np.random.permutation(N_sample)
        res_t = res_t[perm_idx_t,:]
        
        sample_1 = self.base_sampler_s.rsample(N_sample-np.int(N_sample/2))
        sample_2 = lhs(self.t_dim, np.int(N_sample/2))
        res_s = np.concatenate((sample_1,sample_2),axis=0)
        perm_idx_s = np.random.permutation(N_sample)
        res_s = res_s[perm_idx_s,:]
        
        return np.hstack((res_t,res_s))

    def func_obj(self, x):
        x_denorm = self.lb + x[:,:self.t_dim]*(self.ub-self.lb)
        obj_t = x_denorm.dot(self.t_set_sim) / self.min_time_init
        return obj_t
    
    def load_exp_data(self, \
            filedir='../logs/bo_test/mfbo_test/', \
            filename='exp_data.yaml'):
        
        yamlFile = os.path.join(filedir, filename)
        with open(yamlFile, "r") as input_stream:
            yaml_in = yaml.load(input_stream)
            self.start_iter = np.int(yaml_in["start_iter"])
            self.X_L = np.array(yaml_in["X_L"])
            self.Y_L = np.array(yaml_in["Y_L"])
            self.N_L = self.X_L.shape[0]
            self.X_H = np.array(yaml_in["X_H"])
            self.Y_H = np.array(yaml_in["Y_H"])
            self.N_H = self.X_H.shape[0]
            self.X_cand = np.array(yaml_in["X_cand"])
            self.min_time_array = yaml_in["min_time_array"]
            self.alpha_cand_array = yaml_in["alpha_cand_array"]
            self.fidelity_array = yaml_in["fidelity_array"]
            self.found_ei_array = yaml_in["found_ei_array"]
            self.exp_result_array = yaml_in["exp_result_array"]
            self.rel_snap_array = yaml_in["rel_snap_array"]
            self.alpha_min = np.array(yaml_in["alpha_min"])
            self.min_time = np.float(self.min_time_array[-1])
            self.N_low_fidelity = np.int(yaml_in["N_low_fidelity"])
            
            checkpoint = torch.load(os.path.join(self.model_path, 
                "tmp_gpc_models.pth.tar"), map_location=torch.device('gpu'))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            prGreen("#################################################")
            prGreen("Exp data loaded. start_iter: {}, N_L: {}, N_H: {}"\
                    .format(self.start_iter, self.Y_L.shape[0], self.Y_H.shape[0]))
            prGreen("#################################################")
        return
        
    def save_exp_data(self, \
            filedir='./mfgp_data/', \
            filename='exp_data.yaml'):
        if not os.path.exists(filedir):
            os.makedirs(filedir, exist_ok=True)
        yamlFile = os.path.join(filedir, filename)
        yaml_out = open(yamlFile,"w")
        yaml_out.write("start_iter: {}\n\n".format(self.start_iter))
        
        yaml_out.write("X_L:\n")
        for it in range(self.X_L.shape[0]):
            yaml_out.write("  - [{}]\n".format(', '.join([str(x) for x in self.X_L[it,:]])))
        yaml_out.write("\n")
        yaml_out.write("Y_L: [{}]\n".format(', '.join([str(x) for x in self.Y_L])))
        yaml_out.write("\n")
        
        yaml_out.write("X_H:\n")
        for it in range(self.X_H.shape[0]):
            yaml_out.write("  - [{}]\n".format(', '.join([str(x) for x in self.X_H[it,:]])))
        yaml_out.write("\n")
        yaml_out.write("Y_H: [{}]\n".format(', '.join([str(x) for x in self.Y_H])))
        yaml_out.write("\n")
        
        yaml_out.write("X_cand:\n")
        for it in range(self.X_cand.shape[0]):
            yaml_out.write("  - [{}]\n".format(', '.join([str(x) for x in self.X_cand[it,:]])))
        yaml_out.write("\n")
        
        yaml_out.write("min_time_array: [{}]\n".format(', '.join([str(x) for x in self.min_time_array])))
        yaml_out.write("\n")
        yaml_out.write("alpha_cand_array:\n")
        for it in range(len(self.alpha_cand_array)):
            yaml_out.write("  - [{}]\n".format(', '.join([str(x) for x in self.alpha_cand_array[it]])))
        yaml_out.write("\n")
        yaml_out.write("fidelity_array: [{}]\n".format(', '.join([str(x) for x in self.fidelity_array])))
        yaml_out.write("\n")
        yaml_out.write("found_ei_array: [{}]\n".format(', '.join([str(x) for x in self.found_ei_array])))
        yaml_out.write("\n")
        yaml_out.write("exp_result_array: [{}]\n".format(', '.join([str(x) for x in self.exp_result_array])))
        yaml_out.write("\n")
        yaml_out.write("rel_snap_array: [{}]\n".format(', '.join([str(x) for x in self.rel_snap_array])))
        yaml_out.write("\n")
        yaml_out.write("alpha_min: [{}]\n".format(', '.join([str(x) for x in self.alpha_min])))
        yaml_out.write("\n")
        yaml_out.write("N_low_fidelity: {}\n".format(self.N_low_fidelity))
        yaml_out.write("\n")
        yaml_out.close()
        
        torch.save({
            'epoch': self.start_iter,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.min_loss,
            }, os.path.join(self.model_path, "tmp_models.pth.tar"))
        return
    
    def create_model(self, num_epochs=500):
        self.train_x_L = torch.tensor(self.X_L).float().cuda()
        self.train_y_L = torch.tensor(self.Y_L).float().cuda()
        self.train_dataset_L = TensorDataset(self.train_x_L, self.train_y_L)
        self.train_loader_L = DataLoader(self.train_dataset_L, batch_size=self.gpu_batch_size, shuffle=True)

        self.train_x_H = torch.tensor(self.X_H).float().cuda()
        self.train_y_H = torch.tensor(self.Y_H).float().cuda()
        self.train_dataset_H = TensorDataset(self.train_x_H, self.train_y_H)
        self.train_loader_H = DataLoader(self.train_dataset_H, batch_size=self.gpu_batch_size, shuffle=True)
        
        train_x = [self.train_x_L, self.train_x_H]
        train_y = [self.train_y_L, self.train_y_H]
        
        mll = VariationalELBO(self.model.likelihood, self.model, self.train_x_L.shape[-2]+self.train_x_H.shape[-2])
        
        start_time = time.time()
        N_data = self.X_L.shape[0]
        with gpytorch.settings.fast_computations(log_prob=False, solves=False):
            for i in range(num_epochs):
                avg_loss = 0
                for minibatch_i, (x_batch, y_batch) in enumerate(self.train_loader_L):
                    self.optimizer.zero_grad()
                    output = self.model(x_batch, fidelity=1)
                    loss = -mll(output, y_batch)
                    loss.backward(retain_graph=True)
                    avg_loss += loss.item()/N_data
                    self.optimizer.step()

                for minibatch_i, (x_batch, y_batch) in enumerate(self.train_loader_H):
                    self.optimizer.zero_grad()
                    output = self.model(x_batch, fidelity=2)
                    loss = -mll(output, y_batch)
                    output_L = self.model(x_batch, fidelity=1)
                    loss -= mll(output_L, y_batch)
                    avg_loss += loss.item()/N_data
                    loss.backward(retain_graph=True)
                    self.optimizer.step()
                
                if (i+1)%20 == 0 or i == 0:
                    print('Epoch %d/%d - Loss: %.3f' % (i+1, num_epochs, avg_loss))
                
                if self.min_loss > avg_loss and (i+1) >= 20:
                    print('Early stopped at Epoch %d/%d - Loss: %.3f' % (i+1, num_epochs, avg_loss))
                    break
        
        if self.min_loss < 0:
            self.min_loss = avg_loss
        
        print(" - Time: %.3f" % (time.time() - start_time))
        return
    
    def forward_cand(self):
        self.X_cand = self.sample_data(self.N_cand)
        
        test_x = torch.tensor(self.X_cand).float().cuda()
        test_dataset = TensorDataset(test_x)
        test_loader = DataLoader(test_dataset, batch_size=self.gpu_batch_size, shuffle=False)

        mean_L = np.empty(0)
        var_L = np.empty(0)
        prob_cand_L = np.empty(0)
        prob_cand_L_mean = np.empty(0)
        mean_H = np.empty(0)
        var_H = np.empty(0)
        prob_cand_H = np.empty(0)
        prob_cand_H_mean = np.empty(0)
        
        for minibatch_i, (x_batch,) in enumerate(test_loader):
            p, m, v, pm = self.model.predict_proba_MF(x_batch, fidelity=1, C_L=self.C_L, beta=self.beta, return_all=True)
            mean_L = np.append(mean_L, m)
            var_L = np.append(var_L, v)
            prob_cand_L = np.append(prob_cand_L, p[:,1])
            prob_cand_L_mean = np.append(prob_cand_L_mean, pm[:,1])
        
            p_H, m_H, v_H, pm_H = self.model.predict_proba_MF(x_batch, fidelity=2, C_H=self.C_H, C_L=self.C_L, beta=self.beta, return_all=True)
            mean_H = np.append(mean_H, m_H)
            var_H = np.append(var_H, v_H)
            prob_cand_H = np.append(prob_cand_H, p_H[:,1])
            
        return mean_L, var_L, prob_cand_L, mean_H, var_H, prob_cand_H, prob_cand_L_mean
    
    def compute_next_point_cand(self):
        mean_L, var_L, prob_cand_L, mean_H, var_H, prob_cand_H, prob_cand_L_mean = self.forward_cand()
        min_time_tmp = self.func_obj(self.X_cand)
        
        ent_L = -np.abs(mean_L)/(var_L + 1e-9)*self.C_L
        ei_L = np.multiply((self.min_time-min_time_tmp), prob_cand_L)
        ei_L[prob_cand_L<1-self.delta_L] = 0.
        
        ent_H = -np.abs(mean_H)/(var_H + 1e-9)*self.C_H
        ei_H = np.multiply((self.min_time-min_time_tmp), prob_cand_H)
        ei_H[prob_cand_H<1-self.delta_H] = 0.
        
        ei_L_idx = np.argsort(-ei_L)[:self.num_eval_L]
        ent_L_idx = np.argsort(-ent_L)
        N_ei_L = np.int(np.sum(1.*(ei_L[ei_L_idx]>0.)))
        
        ei_H_idx = np.argsort(-ei_H)[:self.num_eval_H]
        ent_H_idx = np.argsort(-ent_H)
        N_ei_H = np.int(np.sum(1.*(ei_H[ei_H_idx]>0.)))
        
        self.X_next = np.empty((0, self.t_dim*2))
        if N_ei_L > 0:
            self.flag_found_ei_L = True
            prPurple("ei_L: {:.4f}, N_ei_L: {}/{}".format(ei_L[ei_L_idx[0]],N_ei_L,self.num_eval_L))
        else:
            self.flag_found_ei_L = False
            prGreen("ent_L: {:.4f}".format(ent_L[ent_L_idx[0]]))
        
        if N_ei_L > 0 and N_ei_L < self.num_eval_L:
            N_ent_L = self.num_eval_L - N_ei_L
            self.X_next = np.concatenate((self.X_cand[ei_L_idx[:N_ei_L],:], self.X_cand[ent_L_idx[:N_ent_L],:]),axis=0)
        elif N_ei_L >= self.num_eval_L:
            self.X_next = self.X_cand[ei_L_idx[:self.num_eval_L],:]
        else:
            self.X_next = self.X_cand[ent_L_idx[:self.num_eval_L],:]
            self.min_time_cand = min_time_tmp[ent_L_idx[:self.num_eval_L]]
        
        if N_ei_H > 0:
            self.flag_found_ei_H = True
            prPurple("ei_H: {:.4f}, N_ei_H: {}/{}".format(ei_H[ei_H_idx[0]],N_ei_H,self.num_eval_H))
            N_ei_H_t = min(N_ei_H, self.num_eval_H)
            self.X_next = np.concatenate((
                self.X_next, self.X_cand[ei_H_idx[:N_ei_H_t],:]),axis=0)
            self.min_time_cand = min_time_tmp[ei_H_idx[:N_ei_H_t]]
            self.X_next_fidelity = 1
        elif self.N_low_fidelity >= self.MAX_low_fidelity:
            self.flag_found_ei_H = False
            prGreen("ent_H: {:.4f}".format(ent_H[ent_H_idx[0]]))
            self.X_next = np.concatenate((
                self.X_next, self.X_cand[ent_H_idx[:self.num_eval_H],:]),axis=0)
            self.min_time_cand = min_time_tmp[ent_H_idx[:self.num_eval_H]]
            self.X_next_fidelity = 1
        else:
            self.flag_found_ei_H = False
            self.X_next_fidelity = 0
        
        if self.X_next.shape[0] > self.num_eval_L:
            self.alpha_min_cand = copy.deepcopy(self.X_next[self.num_eval_L:,:])
            self.alpha_min_cand = self.lb + (self.ub-self.lb)*self.alpha_min_cand
            print("min time cand: {:.4f}, alpha: {}".format(self.min_time_cand[0], self.alpha_min_cand[0,:]))
        return
    
    def append_next_point(self):
        self.N_low_fidelity += 1
        Y_next_L = 1.0*self.sampling_func_L(self.X_next)
        self.X_L = np.vstack((self.X_L, self.X_next))
        self.Y_L = np.concatenate((self.Y_L, np.array(Y_next_L[:,0])))
        self.N_L = self.X_L.shape[0]
        rel_snap_L = Y_next_L[:,1]
        N_success_L = np.int(np.sum(1.*(Y_next_L[:,0]>0)))
        print("N_L: {}, N_success: {}/{}".format(self.N_L, N_success_L, self.num_eval_L))
        
        if self.X_next.shape[0] > self.num_eval_L:
            X_next_H = self.X_next[self.num_eval_L:,:]
            Y_next_H = 1.0*self.sampling_func_H(X_next_H)
            self.X_H = np.vstack((self.X_H, X_next_H))
            self.Y_H = np.concatenate((self.Y_H, np.array(Y_next_H[:,0])))
            self.N_H = self.X_H.shape[0]
            rel_snap = Y_next_H[:,1]
            N_success_H = np.int(np.sum(1.*(Y_next_H[:,0]>0)))
            print("N_H: {}, N_succuess: {}/{}/{}".format(self.N_H, N_success_H, X_next_H.shape[0], self.num_eval_H))
                    
            # Update min_time
            if np.sum(1.*(Y_next_H[:,0]>0)):
                self.exp_result_array.append(1.0)
                min_time_t = (self.min_time_cand[Y_next_H[:,0]==1])
                min_time_idx = np.argmin(min_time_t)
                min_time_tmp = min_time_t[min_time_idx]
                alpha_min_tmp = (self.alpha_min_cand[Y_next_H[:,0]==1])[min_time_idx]
                if min_time_tmp < self.min_time:
                    pre_min_time = copy.deepcopy(self.min_time)
                    pre_alpha_min = copy.deepcopy(self.alpha_min)
                    self.min_time = min_time_tmp
                    self.alpha_min = alpha_min_tmp
                    prYellow("min time: {:.4f}, alpha: {}".format(self.min_time, self.alpha_min))
                self.rel_snap_array.append((rel_snap[Y_next_H[:,0]==1])[min_time_idx])
                print("rel_snap: {}".format((rel_snap[Y_next_H[:,0]==1])[min_time_idx]))
            else:
                self.exp_result_array.append(0.0)
                self.rel_snap_array.append(rel_snap[0])
                print("rel_snap: {}".format(rel_snap[0]))
        else:
            self.exp_result_array.append(0.0)
            self.rel_snap_array.append(1.0)
        
        print("-------------------------------------------")
        return
    
    def save_result_data(self, filedir, filename_result):
        if not os.path.exists(filedir):
            os.makedirs(filedir, exist_ok=True)
        yamlFile = os.path.join(filedir, filename_result)
        yaml_out = open(yamlFile,"w")
        high_idx = 0
        low_idx = 0
        for it in range(len(self.min_time_array)):
            if self.fidelity_array[it] == 1:
                yaml_out.write("iter{}:\n".format(high_idx))
                high_idx += 1
                low_idx = 0
            else:
                yaml_out.write("iter{}_{}:\n".format(high_idx-1,low_idx))
                low_idx += 1
            yaml_out.write("  found_ei: {}\n".format(self.found_ei_array[it]))
            yaml_out.write("  exp_result: {}\n".format(self.exp_result_array[it]))
            yaml_out.write("  rel_snap: {}\n".format(self.rel_snap_array[it]))
            yaml_out.write("  min_time: {}\n".format(self.min_time_array[it]))
            yaml_out.write("  alpha_min: [{}]\n\n".format(','.join([str(x) for x in self.alpha_min_array[it]])))
#             yaml_out.write("  alpha_cand:\n")
#             for it2 in range(self.alpha_cand_array[it].shape[0]):
#                 yaml_out.write("    - [{}]\n".format(','.join([str(x) for x in self.alpha_cand_array[it][it2,:]])))
        yaml_out.close()
        return

    def active_learning(self, 
            N=15, MAX_low_fidelity=20, \
            filedir='./mfgp_data', \
            filename_result='result.yaml', \
            filename_exp='exp_data.yaml'):
        
        if not hasattr(self, 'start_iter'):
            self.start_iter = 0
            self.min_time_array = []
            self.alpha_min_array = []
            self.alpha_cand_array = []
            self.fidelity_array = []
            self.found_ei_array = []
            self.exp_result_array = []
            self.rel_snap_array = []
            self.min_time_array.append(self.min_time)
            self.alpha_min_array.append(self.alpha_min)
            self.alpha_cand_array.append(self.alpha_min_cand)
            self.exp_result_array.append(1)
            self.rel_snap_array.append(self.rel_snap_init)
            self.fidelity_array.append(1)
            self.found_ei_array.append(1)
            self.writer.add_scalar('/min_time', 1.0, 0)
            self.writer.add_scalar('/num_found_ei', 0, 0)
            self.writer.add_scalar('/num_failure', 0, 0)
            self.writer.add_scalar('/rel_snap', self.rel_snap_init, 0)
        
        self.MAX_low_fidelity = MAX_low_fidelity
        main_iter_start = self.start_iter
        self.min_time = self.min_time_array[-1]
        
        if main_iter_start == N-1:
            self.save_result_data(filedir, filename_result)

        for main_iter in range(main_iter_start, N):
            prGreen("#################################################")
            print('%i / %i' % (main_iter+1,N))
            self.X_next_fidelity = 0
            if not hasattr(self, 'N_low_fidelity'):
                self.N_low_fidelity = 0
            num_found_ei = 0
            num_low_fidelity = self.N_low_fidelity
            while self.X_next_fidelity == 0:
                try:
                    self.create_model(num_epochs=self.iter_create_model)
                    self.compute_next_point_cand()
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print('| WARNING: ran out of memory, retrying batch')
                        for p in self.model.parameters():
                            if p.grad is not None:
                                del p.grad  # free some memory
                        torch.cuda.empty_cache()
                        self.create_model(num_epochs=self.iter_create_model)
                        self.compute_next_point_cand()
                    elif 'cholesky_cuda' in str(e):
                        print('| WARNING: cholesky_cuda')                        
                        if hasattr(self, 'clf'):
                            del self.clf                        
                        if hasattr(self, 'feature_model'):
                            del self.feature_model
                        self.create_model()
                        self.compute_next_point_cand()
                    else:
                        raise e
                self.append_next_point()

                self.min_time_array.append(self.min_time)
                self.alpha_min_array.append(self.alpha_min)
                self.alpha_cand_array.append(self.alpha_min_cand)
                self.fidelity_array.append(self.X_next_fidelity)
                if self.flag_found_ei_L or self.flag_found_ei_H:
                    self.found_ei_array.append(1)
                    num_found_ei += 1
                else:
                    self.found_ei_array.append(0)
                num_low_fidelity += 1
                if self.X_next_fidelity == 0:
                    self.start_iter = main_iter
                    self.N_low_fidelity += 1
                else:
                    self.start_iter = main_iter+1
                    self.N_low_fidelity = 0

                self.save_exp_data(filedir, filename_exp)
                torch.cuda.empty_cache()

            num_failure = 0
            for it in range(len(self.min_time_array)):
                if self.fidelity_array[it] == 1 and self.exp_result_array[it] == 0:
                    num_failure += 1
            self.writer.add_scalar('/min_time', self.min_time, main_iter+1)
            self.writer.add_scalar('/num_low_fidelity', num_low_fidelity, main_iter+1)
            self.writer.add_scalar('/num_found_ei', num_found_ei, main_iter+1)
            self.writer.add_scalar('/num_failure', num_failure, main_iter+1)

#             min_time_idx = 0
#             for it in range(len(self.min_time_array)):
#                 if self.fidelity_array[it] == 1 and self.min_time_array[it] == self.min_time:
#                     min_time_idx = it
#                     break
#             self.writer.add_scalar('/rel_snap', self.rel_snap_array[min_time_idx], main_iter+1)
        
            self.save_result_data(filedir, filename_result)
        return