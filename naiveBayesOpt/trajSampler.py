#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os, sys, time, copy
import yaml, h5py, shutil
import scipy
from os import path
from pyDOE import lhs
import cvxpy as cp
import torch
from gpytorch.distributions import MultivariateNormal

from pyTrajectoryUtils.pyTrajectoryUtils.utils import *

def gaussian_sampler(N_sample=100, N=10, gaussian_mean=0.5, gaussian_var=0.1):
    data_t = np.empty((N_sample,N))
    for i in range(N_sample):
        while True:
            x_t = np.random.normal(loc=gaussian_mean, scale=gaussian_var, size=N)
            if np.all(x_t <= 1.0) and np.all(x_t >= 0.0):
                data_t[i,:] = x_t
                break
    return data_t


class TrajSampler_online():
    def __init__(self, N=10, N_sample=100, x_bound=np.array([0., 1.]), 
                 sigma=50.0, flag_load=False, cov_mode=0, flag_pytorch=True):
        v = np.array([1, -3, 3 ,-1]) # Minimize jerk
        self.N = N        
        if cov_mode == 0:
            A = np.zeros((N+3,N))
            for i in range(N+3):
                for j in range(min(i,N-1),max(i-4,-1),-1):
                    A[i,j] = v[j-i+3]
            R = A.T.dot(A)
            self.cov = np.linalg.inv(R)
        elif cov_mode == 1:
            A = np.zeros((N+3,N))
            for i in range(N+3):
                for j in range(min(i,N-1),max(i-4,-1),-1):
                    A[i,j] = v[j-i+3]
            R = A.T.dot(A)
            X = cp.Variable((N,N), symmetric=True)
            constraints = [X >> np.eye(N)*1e-4]
            constraints += [X[i,i] == 1 for i in range(N)]
            prob = cp.Problem(cp.Minimize(cp.trace(R@X)), constraints)
            prob.solve(solver=cp.CVXOPT)
            self.cov = np.array(X.value)
        
        self.sigma = sigma
        self.x_bound = x_bound
        self.flag_pytorch = flag_pytorch
        if flag_pytorch:
            self.dist = MultivariateNormal(torch.zeros(N),torch.Tensor(self.cov))
        
        self.rand_seed = np.random.get_state()[1][0]
        
    def rsample(self, N_sample=100):
        if self.flag_pytorch:
            x_ret = torch.empty(0,self.N)
        else:
            x_ret = np.empty((0,self.N))
        while x_ret.shape[0] < N_sample:
            N_sample_tmp = np.int(max(self.sigma,1))*N_sample*10
            if self.flag_pytorch:
                x = self.dist.rsample(torch.Size([N_sample_tmp]))
                x_max = torch.min(torch.max(x)/self.sigma,torch.abs(torch.min(x))/self.sigma)
                x_min = -x_max
                accepted = x[(torch.min(x-x_min, axis=1).values>=0.0) & (torch.max(x-x_max, axis=1).values<=0.0)]
                accepted = (accepted-x_min)/(x_max-x_min)*(self.x_bound[1]-self.x_bound[0])+self.x_bound[0]
                x_ret = torch.cat([x_ret,accepted], dim=0)
            else:
                x = np.random.multivariate_normal(np.zeros(self.N), self.cov*self.sigma, size=(N_sample_tmp,))
                x += (self.x_bound[0]+self.x_bound[1])/2
                accepted = x[(np.min(x,axis=1)>=self.x_bound[0]) & (np.max(x,axis=1)<=self.x_bound[1])]
                x_ret = np.concatenate((x_ret, accepted), axis=0)
        x_ret = x_ret[:N_sample, :]
        if self.flag_pytorch:
            x_ret = x_ret.numpy()
        return x_ret

class TrajSampler():
    def __init__(self, N=10, N_sample=100, x_bound=np.array([0., 1.]), 
                 sigma=50.0, flag_load=False, cov_mode=0, flag_pytorch=True):
        v = np.array([1, -3, 3 ,-1]) # Minimize jerk
        self.N = N
        if cov_mode == 0:
            A = np.zeros((N+3,N))
            for i in range(N+3):
                for j in range(min(i,N-1),max(i-4,-1),-1):
                    A[i,j] = v[j-i+3]
            R = A.T.dot(A)
            self.cov = np.linalg.inv(R)
        elif cov_mode == 1:
            A = np.zeros((N+3,N))
            for i in range(N+3):
                for j in range(min(i,N-1),max(i-4,-1),-1):
                    A[i,j] = v[j-i+3]
            R = A.T.dot(A)
            X = cp.Variable((N,N), symmetric=True)
            constraints = [X >> np.eye(N)*1e-4]
            constraints += [X[i,i] == 1 for i in range(N)]
            prob = cp.Problem(cp.Minimize(cp.trace(R@X)), constraints)
            prob.solve()
            self.cov = np.array(X.value)
        
        self.sigma = sigma
        self.x_bound = x_bound
        self.flag_pytorch = flag_pytorch
        if flag_pytorch:
            self.dist = MultivariateNormal(torch.zeros(N),torch.Tensor(self.cov))
        
        self.rand_seed = np.random.get_state()[1][0]
        self.flag_load = False
        if flag_load:
            self.batch_size = 20
            self.num_batch = 500
            self.read_idx = 0
            self.cached_data = np.empty((0,self.N))
            self.N_sample_init = N_sample
            str_sigma = np.int(self.sigma)
            if self.sigma < 1:
                str_sigma = '0p'+str(np.int(self.sigma*10))
            self.check_data(N_sample=self.N_sample_init, \
                            fileprefix='sig_{}_dim_{}'.format(str_sigma,np.int(self.N)))
        self.flag_load = flag_load
        
    def check_data(self, N_sample=100, filedir='./mfgp_data/pre_smooth_traj', fileprefix='sig_50_dim_10'):
        seed = self.rand_seed
        filedir = '/home/eris/Workspace/trajectoryLearning/mfgp_data/pre_smooth_traj'
        filename = fileprefix+'_seed_{}_NS_{}_NB_{}.h5'.format(seed, N_sample, self.batch_size)
        data_path = os.path.join(filedir,filename)
        print("Check data path: {}".format(data_path))
        flag_sample = False
        gen_start_idx = 0
        if not os.path.exists(data_path):
            flag_sample = True
        else:
            h5f = h5py.File(data_path, 'r')
            for i in range(self.num_batch):
                if str(i) not in h5f.keys():
                    flag_sample = True
                    break
                else:
                    gen_start_idx = i
            h5f.close()
        if flag_sample:
            print("Generating data...")
            for i in range(gen_start_idx, self.num_batch):
                print("{}/{}".format(i+1,self.num_batch))
                h5f = h5py.File(data_path, 'a')
                data_to_save = np.empty((0,self.N))
                for j in range(self.batch_size):
                    data_to_save = np.append(data_to_save,self.rsample(N_sample),axis=0)
                if str(i) in h5f.keys():
                    data = h5f[str(i)]
                    data[...] = data_to_save
                else:
                    h5f.create_dataset(str(i), data=data_to_save)
                h5f.close()
                shutil.copyfile(data_path, data_path+'.bak')
    
    def load_data(self, idx=0, N_sample=100, filedir='./mfgp_data/pre_smooth_traj', fileprefix='sig_50_dim_10'):
        filedir = '/home/daedalus/hdd/pre_smooth_traj'
        batch_idx = np.int(idx/self.batch_size)
        item_idx = idx % self.batch_size
        batch_idx = batch_idx % 50
        if item_idx == 0:
            seed = self.rand_seed
            filename = fileprefix+'_seed_{}_NS_{}_NB_{}.h5'.format(seed, N_sample, self.batch_size)
            data_path = os.path.join(filedir,filename)
            h5f = h5py.File(data_path, 'r')
            self.cached_data = np.array(h5f[str(batch_idx)][:])
            h5f.close()
        return self.cached_data[item_idx*N_sample:(item_idx+1)*N_sample,:]
        
    def rsample(self, N_sample=100):
        if self.flag_load and N_sample % self.N_sample_init == 0:
            str_sigma = np.int(self.sigma)
            if self.sigma < 1:
                str_sigma = '0p'+str(np.int(self.sigma*10))
            x_ret = np.empty((0,self.N))
            for i in range(np.int(N_sample/self.N_sample_init)):
                x_ret_tmp = self.load_data( \
                    idx=self.read_idx, \
                    N_sample=self.N_sample_init, \
                    fileprefix='sig_{}_dim_{}'.format(str_sigma,np.int(self.N)))
                x_ret = np.append(x_ret, x_ret_tmp, axis=0)
                self.read_idx += 1
            return x_ret
        
        if self.flag_pytorch:
            x_ret = torch.empty(0,self.N)
        else:
            x_ret = np.empty((0,self.N))
        while x_ret.shape[0] < N_sample:
            N_sample_tmp = np.int(max(self.sigma,1))*N_sample*10
            if self.flag_pytorch:
                x = self.dist.rsample(torch.Size([N_sample_tmp]))
                x_max = torch.min(torch.max(x)/self.sigma,torch.abs(torch.min(x))/self.sigma)
                x_min = -x_max
                accepted = x[(torch.min(x-x_min, axis=1).values>=0.0) & (torch.max(x-x_max, axis=1).values<=0.0)]
                accepted = (accepted-x_min)/(x_max-x_min)*(self.x_bound[1]-self.x_bound[0])+self.x_bound[0]
                x_ret = torch.cat([x_ret,accepted], dim=0)
            else:
                x = np.random.multivariate_normal(np.zeros(self.N), self.cov*self.sigma, size=(N_sample_tmp,))
                x += (self.x_bound[0]+self.x_bound[1])/2
                accepted = x[(np.min(x,axis=1)>=self.x_bound[0]) & (np.max(x,axis=1)<=self.x_bound[1])]
                x_ret = np.concatenate((x_ret, accepted), axis=0)
        x_ret = x_ret[:N_sample, :]
        if self.flag_pytorch:
            x_ret = x_ret.numpy()
        return x_ret
