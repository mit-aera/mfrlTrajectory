#!/usr/bin/env python
# coding: utf-8

import os, sys
import copy
import numpy as np
import pandas as pd
import scipy
from scipy import optimize
from pyDOE import lhs
import h5py
import yaml
import matplotlib.pyplot as plt
from itertools import cycle
from mpl_toolkits.mplot3d import Axes3D

from .quadModel import QuadModel
from .trajectorySimulation import TrajectorySimulation
from .utils import *

class MinSnapTrajectory(BaseTrajFunc):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if 'cfg_path_sim' in kwargs:
            cfg_path_sim = kwargs['cfg_path_sim']
        else:
            cfg_path_sim = None
        
        if 'drone_model' in kwargs:
            drone_model = kwargs['drone_model']
        else:
            drone_model = None
        
        self._quadModel = QuadModel(cfg_path=cfg_path_sim, drone_model=drone_model)
        
        self.sim = TrajectorySimulation(*args, **kwargs)
        
        self.cache_sys_mat_t = dict()
        self.cache_sys_mat_t["list"] = []
        return
    
    ###############################################################################    
    def plot_ws(self, t_set, d_ordered, d_ordered_yaw, direct_yaw=False):
        flag_loop = self.check_flag_loop(t_set,d_ordered)
        N_POLY = t_set.shape[0]
        
        status = np.zeros((self.N_POINTS*N_POLY,18))

        t_array = np.zeros(self.N_POINTS*N_POLY)
        for i in range(N_POLY):
            for j in range(self.N_POINTS):
                t_array[i*self.N_POINTS+j] = t_array[i*self.N_POINTS+j-1] + t_set[i]/self.N_POINTS
         
        for der in range(5):
            if flag_loop:
                V_t = self.generate_sampling_matrix_loop(t_set, N=self.N_POINTS, der=der)
            else:
                V_t = self.generate_sampling_matrix(t_set, N=self.N_POINTS, der=der, endpoint=True)
            status[:,3*der:3*(der+1)] = V_t.dot(d_ordered)
        
        if np.all(d_ordered_yaw != None):
            status_yaw_xy = np.zeros((self.N_POINTS*N_POLY,3,2))
            for der in range(3):
                if flag_loop:
                    V_t = self.generate_sampling_matrix_loop_yaw(t_set, N=self.N_POINTS, der=der)
                else:
                    V_t = self.generate_sampling_matrix_yaw(t_set, N=self.N_POINTS, der=der, endpoint=True)
                status_yaw_xy[:,der,:] = V_t.dot(d_ordered_yaw)
            if not direct_yaw:
                status[:,15:] = self.get_yaw_der(status_yaw_xy)
            else:
                status[:,15:] = status_yaw_xy[:,:,0]
        ws, ctrl = self._quadModel.getWs_vector(status)

        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(111)
        for i in range(4):
            ax.plot(t_array, ws[:,i], '-', label='ms {}'.format(i))
        ax.legend()
        ax.grid()
        plt.show()
        plt.pause(0.1)
        return
    
    ###############################################################################
    def snap_obj(self, x, b, v=None, b_ext_init=None, b_ext_end=None, 
                 deg_init_min=0, deg_init_max=4, deg_end_min=0, deg_end_max=0, kt=0):
        flag_const_b = True
        for i in range(1,b.shape[0]):
            if np.any(b[i,:] != b[0,:]):
                flag_const_b = False
        if flag_const_b:
            res = 0
            d_ordered_res =  np.zeros((b.shape[0]*self.N_DER,b.shape[1]))
            for i in range(b.shape[0]):
                d_ordered_res[i*self.N_DER,:] = b[i,:]
            return res, d_ordered_res
        
        flag_loop = self.check_flag_loop_points(x,b)
        N_POLY = x.shape[0]

        # v contains derivative equality constraints 
        # dim are: (waypoints, dimensions, derivatives (0: vel, 1: acc etc.))
        # if constraint for some wp and dir is not active set to np.nan
        if not np.any(v != None):
            v = np.zeros((b.shape[0], b.shape[1], self.N_DER-1))
            v[:] = np.nan
        else:
            assert v.shape == (b.shape[0], b.shape[1], self.N_DER-1)

        # give priority to deg_init if used to set derivatives to zero
        for i in range(deg_init_min, deg_init_max):
            v[0, :, i] = 0.
        if not flag_loop:
            for i in range(deg_end_min, deg_end_max):
                v[-1, :, i] = 0.

        # print('v = {0}'.format(v))
        
        if flag_loop:
            P = self.generate_perm_matrix(x.shape[0]-1, self.N_DER)
            A_sys = self.generate_sampling_matrix_loop(x, N=self.N_POINTS, der=self.MAX_SYS_DEG)
        else:
            P = self.generate_perm_matrix(x.shape[0], self.N_DER)
            A_sys = self.generate_sampling_matrix(x, N=self.N_POINTS, der=self.MAX_SYS_DEG, endpoint=True)
        D_tw = self.generate_weight_matrix(x, self.N_POINTS)
        
        A_sys_t = A_sys.dot(P.T)[:,:]
        R = A_sys_t.T.dot(D_tw).dot(A_sys_t)

        # print('R.shape[0] = {0}'.format(R.shape[0]))
        # print('b.shape[0] = {0}'.format( b.shape[0]))
        # print('N_DER = {0}'.format(self.N_DER))

        d_tmp = np.empty([b.shape[0]*self.N_DER,0])

        for dim in range(0,b.shape[1]):

            b_dim = b[:,dim].flatten()

            # b fixed indices
            b_idx = np.where(~np.isnan(b_dim))[0]

            # b free indices
            bf_idx = np.where(np.isnan(b_dim))[0]

            v_dim = v[:,dim,:].flatten()

            # fixed derivatives
            v_idx = np.where(~np.isnan(v_dim))[0]

            # free derivatives
            vf_idx = np.where(np.isnan(v_dim))[0]

            # free variables
            R_idx = np.concatenate((bf_idx, vf_idx+b.shape[0]))
            # fixed variables
            Rc_idx =np.concatenate((b_idx, v_idx+b.shape[0]))

            # print('R_idx = {0}'.format(R_idx))       
            Rpp = R[np.ix_(R_idx,R_idx)]
            if Rpp.shape[0] > 0:
                Rpp_inv = self.get_matrix_inv(Rpp)
            else:
                Rpp_inv = Rpp

            d_dim = -Rpp_inv.dot(R[np.ix_(Rc_idx,R_idx)].T).dot(np.concatenate((b_dim[b_idx], v_dim[v_idx])))
            # print('b:')
            # print(b)

            for idx in b_idx:
                d_dim = np.insert(d_dim, idx, b_dim[idx], axis = 0)
                
            for idx in v_idx:
                d_dim = np.insert(d_dim, b.shape[0] + idx, v_dim[idx], axis = 0)

            # print('d_p = {0}'.format(d_p.shape))
            # print('d_tmp = {0}'.format(d_tmp.shape))
            # print('d_dim = {0}'.format(d_dim[np.newaxis].T.shape))
            d_tmp = np.append(d_tmp, d_dim[np.newaxis].T, axis=1)

        # print('N_DER: {0}'.format(self.N_DER))
        # print('shape d_tmp: {0}'.format(d_tmp.shape))
        # print('shape b: {0}'.format(b.shape))
        # print('shape R: {0}'.format(R.shape))

        res = np.trace(d_tmp.T.dot(R).dot(d_tmp)) + kt*np.sum(x)
        d_ordered = P.T.dot(d_tmp)

        # print('d_ordered: {}'.format(d_ordered))
        
        if res < 0:
            res = 1e10
        return res, d_ordered
    
    def acc_obj(self, x, b, v=None, b_ext_init=None, b_ext_end=None, 
                 deg_init_min=0, deg_init_max=2, deg_end_min=0, deg_end_max=0):
        flag_const_b = True
        for i in range(1,b.shape[0]):
            if np.any(b[i,:] != b[0,:]):
                flag_const_b = False
        if flag_const_b:
            res = 0
            d_ordered_res =  np.zeros((b.shape[0]*self.N_DER_YAW,b.shape[1]))
            for i in range(b.shape[0]):
                d_ordered_res[i*self.N_DER_YAW,:] = b[i,:]
            return res, d_ordered_res
        
        flag_loop = self.check_flag_loop_points(x,b)
        N_POLY = x.shape[0]

        # v contains derivative equality constraints 
        # dim are: (waypoints, dimensions, derivatives (0: vel, 1: acc etc.))
        # if constraint for some wp and dir is not active set to np.nan
        if not np.any(v != None):
            v = np.zeros((b.shape[0], b.shape[1], self.N_DER_YAW-1))
            v[:] = np.nan
        else:
            assert v.shape == (b.shape[0], b.shape[1], self.N_DER_YAW-1)

        # give priority to deg_init if used to set derivatives to zero
        for i in range(deg_init_min, deg_init_max):
            v[0, :, i] = 0.
        if not flag_loop:
            for i in range(deg_end_min, deg_end_max):
                v[-1, :, i] = 0.
        
        if flag_loop:
            P = self.generate_perm_matrix(x.shape[0]-1, self.N_DER_YAW)
            A_sys = self.generate_sampling_matrix_loop_yaw(x, N=self.N_POINTS, der=self.MAX_SYS_DEG_YAW)
        else:
            P = self.generate_perm_matrix(x.shape[0], self.N_DER_YAW)
            A_sys = self.generate_sampling_matrix_yaw(x, N=self.N_POINTS, der=self.MAX_SYS_DEG_YAW, endpoint=True)
        D_tw = self.generate_weight_matrix(x, self.N_POINTS)

        A_sys_t = A_sys.dot(P.T)[:,:]
        R = A_sys_t.T.dot(D_tw).dot(A_sys_t)
        
        d_tmp = np.empty([b.shape[0]*self.N_DER_YAW,0])

        for dim in range(0,b.shape[1]):

            b_dim = b[:,dim].flatten()

            # b fixed indices
            b_idx = np.where(~np.isnan(b_dim))[0]

            # b free indices
            bf_idx = np.where(np.isnan(b_dim))[0]

            v_dim = v[:,dim,:].flatten()

            # fixed derivatives
            v_idx = np.where(~np.isnan(v_dim))[0]

            # free derivatives
            vf_idx = np.where(np.isnan(v_dim))[0]

            # free variables
            R_idx = np.concatenate((bf_idx, vf_idx+b.shape[0]))
            # fixed variables
            Rc_idx =np.concatenate((b_idx, v_idx+b.shape[0]))

            # print('R_idx = {0}'.format(R_idx))       
            Rpp = R[np.ix_(R_idx,R_idx)]
            if Rpp.shape[0] > 0:
                Rpp_inv = self.get_matrix_inv(Rpp)
            else:
                Rpp_inv = Rpp

            d_dim = -Rpp_inv.dot(R[np.ix_(Rc_idx,R_idx)].T).dot(np.concatenate((b_dim[b_idx], v_dim[v_idx])))
            # print('b:')
            # print(b)

            for idx in b_idx:
                d_dim = np.insert(d_dim, idx, b_dim[idx], axis = 0)
                
            for idx in v_idx:
                d_dim = np.insert(d_dim, b.shape[0] + idx, v_dim[idx], axis = 0)

            # print('d_p = {0}'.format(d_p.shape))
            # print('d_tmp = {0}'.format(d_tmp.shape))
            # print('d_dim = {0}'.format(d_dim[np.newaxis].T.shape))
            d_tmp = np.append(d_tmp, d_dim[np.newaxis].T, axis=1)

        # print('N_DER_YAW: {0}'.format(self.N_DER_YAW))
        # print('shape d_tmp: {0}'.format(d_tmp.shape))
        # print('shape b: {0}'.format(b.shape))
        # print('shape R: {0}'.format(R.shape))

        res = np.trace(d_tmp.T.dot(R).dot(d_tmp))
        d_ordered = P.T.dot(d_tmp)
        
        if res < 0:
            res = 1e10
        return res, d_ordered
    
    ###############################################################################    
    def get_min_snap_traj(self, \
        points, alpha_scale=1.0, flag_loop=False, yaw_mode=0, \
        deg_init_min=0, deg_init_max=4, \
        deg_end_min=0, deg_end_max=0, \
        deg_init_yaw_min=0, deg_init_yaw_max=2, \
        deg_end_yaw_min=0, deg_end_yaw_max=0, \
        mu=1.0, kt=0, kt2=0, \
        points_mean=1, \
        t_set_init=None, \
        flag_rand_init=False, flag_numpy_opt=False, flag_scipy_opt=False, \
        flag_t_set_normalize=False, flag_opt_alpha=False):
                
        # pos_obj = lambda x, b: self.snap_obj(x,b,
        #     deg_init_min=deg_init_min,deg_init_max=deg_init_max,
        #     deg_end_min=deg_end_min,deg_end_max=deg_end_max)
        # yaw_obj = lambda x, b: self.acc_obj(x,b,
        #     deg_init_min=deg_init_yaw_min,deg_init_max=deg_init_yaw_max,
        #     deg_end_min=deg_end_yaw_min,deg_end_max=deg_end_yaw_max)
        
        if deg_end_max == 0 and deg_end_yaw_max == 0:
            flag_nonstop = True
        else:
            flag_nonstop = False
        def pos_obj(x, b):
            b_t = np.zeros((b.shape[0], 4))
            b_t[:,:3] = np.copy(b[:,:3])
            return self.obj_func_acc(points=b_t, t_set=x, flag_loop=False, flag_nonstop=flag_nonstop)
        def yaw_obj(x, b):
            b_t = np.zeros((b.shape[0], 4))
            b_t[:,3] = np.copy(b[:,0])            
            return self.obj_func_acc(points=b_t, t_set=x, flag_loop=False, \
                                    flag_yaw=True, flag_direct_yaw=True, flag_nonstop=flag_nonstop)
        
        if flag_loop:
            N_POLY = points.shape[0]
        else:
            N_POLY = points.shape[0]-1
        
        if np.all(t_set_init != None):
            t_set = np.copy(t_set_init)
        else:
            t_set = np.linalg.norm(np.diff(points[:,:3], axis=0),axis=1)*2/points_mean
        
        if flag_loop:
            t_set = np.append(t_set,np.linalg.norm(points[-1,:3]-points[0,:3])*2/points_mean)
        b = (copy.deepcopy(points[:,:3]))/points_mean
        
        MAX_ITER = 10
        lr = 0.5*N_POLY
        dt = 1e-3
        
        if yaw_mode == 0 or yaw_mode == 1:
            b_yaw = np.zeros((points.shape[0],2))
            b_yaw[:,0] = 1
        elif yaw_mode == 2:
            if points.shape[1] != 4:
                print("Wrong points format. Append yaw column")
            b_yaw = np.zeros((points.shape[0],2))
            b_yaw[:,0] = np.cos(points[:,-1])
            b_yaw[:,1] = np.sin(points[:,-1])
        elif yaw_mode == 3:
            b_yaw = np.zeros((points.shape[0],1))
            b_yaw[:,0] = points[:,-1]
        else:
            raise("Wrong yaw_mode")
        
        def f_obj(x):
            x_t = x*1.0
            f0, d_ordered = pos_obj(x=x_t, b=b)
            if yaw_mode == 0:
                b_yaw = np.zeros((points.shape[0],2))
                b_yaw[:,0] = 1
            elif yaw_mode == 1:
                b_yaw = self.get_yaw_forward(x_t, d_ordered)
            elif yaw_mode == 2:
                if points.shape[1] != 4:
                    print("Wrong points format. Append yaw column")
                b_yaw = np.zeros((points.shape[0],2))
                b_yaw[:,0] = np.cos(points[:,-1])
                b_yaw[:,1] = np.sin(points[:,-1])
            elif yaw_mode == 3:
                b_yaw = np.zeros((points.shape[0],1))
                b_yaw[:,0] = points[:,-1]
            else:
                raise("Wrong yaw_mode")
            if yaw_mode > 0:
                f0_yaw, _ = yaw_obj(x=x_t, b=b_yaw)
            else:
                f0_yaw = 0
            f0 += mu*f0_yaw + kt*np.sum(x_t) + kt2*np.sum(x_t)**2
            return f0
        
        if flag_rand_init:
            # Random init
            N_rand = 100
            alpha_set_tmp = lhs(t_set.shape[0], N_rand)*1.8+0.1
            min_t_set = t_set
            min_f_obj = -1
            for i in range(N_rand):
                t_set_tmp = np.multiply(t_set, alpha_set_tmp[i,:])
                t_set_tmp *= np.sum(t_set)/np.sum(t_set_tmp)
                f_obj_tmp = f_obj(t_set_tmp)
                if (min_f_obj == -1 or min_f_obj > f_obj_tmp) and f_obj_tmp > 0:
                    min_f_obj = f_obj_tmp
                    min_t_set = t_set_tmp
            t_set = min_t_set

        if flag_numpy_opt:
            # Optimizae time ratio
            for t in range(MAX_ITER):
                grad = np.zeros(N_POLY)

                f0, d_ordered = pos_obj(x=t_set, b=b)
                if yaw_mode == 1:
                    b_yaw = self.get_yaw_forward(t_set, d_ordered)
                elif yaw_mode == 2:
                    if points.shape[1] != 4:
                        print("Wrong points format. Append yaw column")
                    b_yaw[:,0] = np.cos(points[:,-1])
                    b_yaw[:,1] = np.sin(points[:,-1])
                elif yaw_mode == 3:
                    b_yaw = np.zeros((points.shape[0],1))
                    b_yaw[:,0] = points[:,-1]
                if yaw_mode > 0:
                    f0_yaw, _ = yaw_obj(x=t_set, b=b_yaw)
                else:
                    f0_yaw = 0
                f0 += mu*f0_yaw + kt*np.sum(t_set) + kt2*np.sum(t_set)**2

                for i in range(N_POLY):
                    t_set_tmp = copy.deepcopy(t_set)
                    t_set_tmp[i] += dt

                    f1, _ = pos_obj(x=t_set_tmp, b=b)
                    if yaw_mode > 0:
                        f1_yaw, _ = yaw_obj(x=t_set_tmp, b=b_yaw)
                    else:
                        f1_yaw = 0
                    f1 += mu*f1_yaw + kt*np.sum(t_set_tmp) + kt2*np.sum(t_set_tmp)**2
                    grad[i] = (f1-f0)/dt

                err = np.mean(np.abs(grad))
                grad /= max(np.linalg.norm(grad),1e-3)

                t_set_tmp = t_set-lr*grad

                if np.any(t_set_tmp < 0.0):
                    lr *= 0.1
                    continue

                f_tmp, d_ordered = pos_obj(x=t_set_tmp*np.sum(t_set)/np.sum(t_set_tmp), b=b)
                if f0 > 0:
                    f_ratio = f_tmp/f0
                else:
                    raise("Wrong overall snaps")
                    f_ratio = 0
                
                if lr < 1e-20 and f_ratio < 1e-2:
                    break

                t_set -= lr*grad

                if err < 1e-3 and f_ratio < 1e-2:
                    break
        
        if flag_scipy_opt:
            bounds = []
            for i in range(t_set.shape[0]):
                bounds.append((0.01, 100.0))
            res_x, res_f, res_d = scipy.optimize.fmin_l_bfgs_b(\
                f_obj, x0=t_set, bounds=bounds, \
                approx_grad=True, epsilon=1e-4, maxiter=MAX_ITER, \
                iprint=1, disp=False)
            t_set = np.array(res_x)
        
        x_t = t_set*1.0*np.shape(t_set)[0]/np.sum(t_set)
        rel_snap, _ = pos_obj(x=x_t, b=b)

        if flag_t_set_normalize:
            t_set /= np.sum(t_set)
            t_set *= 10.
        
        _, d_ordered = pos_obj(t_set, b=b)
        d_ordered *= points_mean
        
        if yaw_mode == 1:
            b_yaw = self.get_yaw_forward(t_set, d_ordered)
            
        if yaw_mode == 0:
            d_ordered_yaw = None
        else:
            _, d_ordered_yaw = yaw_obj(x=t_set, b=b_yaw)
#         _, d_ordered_yaw = yaw_obj(x=t_set, b=b_yaw)
        
        if flag_opt_alpha:
            if yaw_mode == 3:
                return self.optimize_alpha_acc(points, t_set, d_ordered, d_ordered_yaw, 
                   alpha_scale, sanity_check_t=None, precision=0.0001, direct_yaw=True)
            else:
                return self.optimize_alpha_acc(points, t_set, d_ordered, d_ordered_yaw, 
                   alpha_scale, sanity_check_t=None, precision=0.0001, direct_yaw=False)
        else:
            return t_set, d_ordered, d_ordered_yaw
    
    def get_status(self, t_set, d_ordered, d_ordered_yaw=None, freq=200):
        flag_loop = self.check_flag_loop(t_set, d_ordered)
        dt = 1./freq
        total_time = np.sum(t_set)
        cum_time = np.zeros(t_set.shape[0]+1)
        cum_time[1:] = np.cumsum(t_set)
        cum_time[0] = 0
        
        N = int(np.floor(total_time/dt))
        poly_idx = 0
        
        t_array = total_time*np.array(range(N))/N
        status = np.zeros((N,20))
        status[:,0] = t_array
        status[:,1] = 1
        
        T2_mat = np.diag(self.generate_basis(t_set[poly_idx],self.N_DER-1,0))
        der0 = T2_mat.dot(d_ordered[poly_idx*self.N_DER:(poly_idx+1)*self.N_DER,:])
        der1 = T2_mat.dot(d_ordered[(poly_idx+1)*self.N_DER:(poly_idx+2)*self.N_DER,:])
        
        if np.all(d_ordered_yaw != None):
            T2_mat_yaw = np.diag(self.generate_basis(t_set[poly_idx],self.N_DER_YAW-1,0))
            der0_yaw = T2_mat_yaw.dot(d_ordered_yaw[poly_idx*self.N_DER_YAW:(poly_idx+1)*self.N_DER_YAW,:])
            der1_yaw = T2_mat_yaw.dot(d_ordered_yaw[(poly_idx+1)*self.N_DER_YAW:(poly_idx+2)*self.N_DER_YAW,:])
        
        for i in range(N):
            if t_array[i] > cum_time[poly_idx+1]:
                poly_idx += 1
                T2_mat = np.diag(self.generate_basis(t_set[poly_idx],self.N_DER-1,0))
                der0 = T2_mat.dot(d_ordered[poly_idx*self.N_DER:(poly_idx+1)*self.N_DER,:])
                if flag_loop:
                    poly_idx_next = (poly_idx+1)%(t_set.shape[0])
                else:
                    poly_idx_next = poly_idx+1
                
                der1 = T2_mat.dot(d_ordered[poly_idx_next*self.N_DER:(poly_idx_next+1)*self.N_DER,:])
                if np.all(d_ordered_yaw != None):
                    T2_mat_yaw = np.diag(self.generate_basis(t_set[poly_idx],self.N_DER_YAW-1,0))
                    der0_yaw = T2_mat_yaw.dot(d_ordered_yaw[poly_idx*self.N_DER_YAW:(poly_idx+1)*self.N_DER_YAW,:])
                    der1_yaw = T2_mat_yaw.dot(d_ordered_yaw[poly_idx_next*self.N_DER_YAW:(poly_idx_next+1)*self.N_DER_YAW,:])
            
            t_step = (t_array[i] - cum_time[poly_idx])/t_set[poly_idx]
            
            for der in range(5):
                v0, v1 = self.generate_single_point_matrix(t_step, der=der)
                status[i,2+3*der:2+3*(der+1)] = (v0.dot(der0)+v1.dot(der1))/(t_set[poly_idx]**der)
            
            if np.all(d_ordered_yaw != None):
                status_yaw_xy = np.zeros((1,3,2))
                for der in range(3):
                    v0, v1 = self.generate_single_point_matrix_yaw(t_step, der=der)
                    status_yaw_xy[:,der,:] = (v0.dot(der0_yaw)+v1.dot(der1_yaw))/(t_set[poly_idx]**der)
                status[i,17:] = self.get_yaw_der(status_yaw_xy)[0,:]
        
        if flag_loop:
            status = np.append(status,status[0:1,:],axis=0)
            status[-1,0] = total_time
        return status
    
    def get_alpha_matrix(self, alpha, N_wp):
        T_alpha = np.diag(self.generate_basis(1./alpha,self.N_DER-1,0))
        T_alpha_all = np.zeros((self.N_DER*N_wp,self.N_DER*N_wp))
        for i in range(N_wp):
            T_alpha_all[i*self.N_DER:(i+1)*self.N_DER,i*self.N_DER:(i+1)*self.N_DER] = T_alpha
        return T_alpha_all

    def get_alpha_matrix_yaw(self, alpha, N_wp):
        T_alpha = np.diag(self.generate_basis(1./alpha,self.N_DER_YAW-1,0))
        T_alpha_all = np.zeros((self.N_DER_YAW*N_wp,self.N_DER_YAW*N_wp))
        for i in range(N_wp):
            T_alpha_all[i*self.N_DER_YAW:(i+1)*self.N_DER_YAW,i*self.N_DER_YAW:(i+1)*self.N_DER_YAW] = T_alpha
        return T_alpha_all
    
    def sanity_check(self, t_set, d_ordered, d_ordered_yaw=None, flag_parallel=False, direct_yaw=False):
        flag_loop = self.check_flag_loop(t_set,d_ordered)
        N_POLY = t_set.shape[0]
        
        status = np.zeros((self.N_POINTS*N_POLY,18))

        for der in range(5):
            if flag_loop:
                V_t = self.generate_sampling_matrix_loop(t_set, N=self.N_POINTS, der=der)
            else:
                V_t = self.generate_sampling_matrix(t_set, N=self.N_POINTS, der=der, endpoint=True)
            status[:,3*der:3*(der+1)] = V_t.dot(d_ordered)
        
        if np.all(d_ordered_yaw != None):
            status_yaw_xy = np.zeros((self.N_POINTS*N_POLY,3,2))
            for der in range(3):
                if flag_loop:
                    V_t = self.generate_sampling_matrix_loop_yaw(t_set, N=self.N_POINTS, der=der)
                else:
                    V_t = self.generate_sampling_matrix_yaw(t_set, N=self.N_POINTS, der=der, endpoint=True)
                status_yaw_xy[:,der,:d_ordered_yaw.shape[1]] = V_t.dot(d_ordered_yaw)
            if direct_yaw:
                status[:,15:] = status_yaw_xy[:,:,0]
            else:
                status[:,15:] = self.get_yaw_der(status_yaw_xy)

        if flag_parallel:
            ws, ctrl = self._quadModel.getWs_vector(status)
            if np.any(ws < self._quadModel.w_min) or np.any(ws > self._quadModel.w_max):
                return False
        else:
            for j in range(self.N_POINTS*N_POLY):
                ws, ctrl = self._quadModel.getWs(status[j,:])
                if np.any(ws < self._quadModel.w_min) or np.any(ws > self._quadModel.w_max):
                    return False

        return True
    
    def optimize_alpha(self, points, t_set, d_ordered, d_ordered_yaw, alpha_scale=1.0, sanity_check_t=None, flag_return_alpha=False):
        if sanity_check_t == None:
            sanity_check_t = self.sanity_check

        # Optimizae alpha
        alpha = 2.0
        dalpha = 0.1
        alpha_tmp = alpha
        t_set_ret = copy.deepcopy(t_set)
        d_ordered_ret = copy.deepcopy(d_ordered)
        N_wp = int(d_ordered.shape[0]/self.N_DER)
        
        if np.all(d_ordered_yaw != None):
            d_ordered_yaw_ret = copy.deepcopy(d_ordered_yaw)
        else:
            d_ordered_yaw_ret = None
        
        while True:
            t_set_opt = t_set * alpha
            d_ordered_opt = self.get_alpha_matrix(alpha,N_wp).dot(d_ordered)
            if np.all(d_ordered_yaw != None):
                d_ordered_yaw_opt = self.get_alpha_matrix_yaw(alpha,N_wp).dot(d_ordered_yaw)
            else:
                d_ordered_yaw_opt = None
            
            if not sanity_check_t(t_set_opt, d_ordered_opt, d_ordered_yaw_opt):
                alpha += 1.0
            else:
                break
            
        while True:
            alpha_tmp = alpha - dalpha
            t_set_opt = t_set * alpha_tmp
            d_ordered_opt = self.get_alpha_matrix(alpha_tmp,N_wp).dot(d_ordered)
            if np.all(d_ordered_yaw != None):
                d_ordered_yaw_opt = self.get_alpha_matrix_yaw(alpha_tmp,N_wp).dot(d_ordered_yaw)
            else:
                d_ordered_yaw_opt = None
            
            if not sanity_check_t(t_set_opt, d_ordered_opt, d_ordered_yaw_opt):
                dalpha *= 0.1
            else:
                alpha = alpha_tmp
                t_set_ret = t_set_opt
                d_ordered_ret = d_ordered_opt
                d_ordered_yaw_ret = d_ordered_yaw_opt
            
            if dalpha < 1e-5 or alpha < 1e-5:
#                 print("Optimize alpha: {}".format(alpha))
                break
        
        t_set = t_set_ret * alpha_scale
        d_ordered = self.get_alpha_matrix(alpha_scale,N_wp).dot(d_ordered_ret)
        if np.all(d_ordered_yaw != None):
            d_ordered_yaw = self.get_alpha_matrix_yaw(alpha_scale,N_wp).dot(d_ordered_yaw_ret)
        else:
            d_ordered_yaw = None
        
        if flag_return_alpha:
            return t_set, d_ordered, d_ordered_yaw, alpha
        else:
            return t_set, d_ordered, d_ordered_yaw

    def optimize_alpha_acc(self, \
        points, t_set, d_ordered, d_ordered_yaw, alpha_scale=1.0, \
        sanity_check_t=None, flag_return_alpha=False, precision=0.001, direct_yaw=False):
        
        if sanity_check_t == None:
            sanity_check_t = lambda t_set, d_ordered, d_ordered_yaw=None: \
                self.sanity_check(t_set, d_ordered, d_ordered_yaw, flag_parallel=True, direct_yaw=direct_yaw)
            
        t_set_ret = copy.deepcopy(t_set)
        d_ordered_ret = copy.deepcopy(d_ordered)
        N_wp = int(d_ordered.shape[0]/self.N_DER)
        
        if np.all(d_ordered_yaw != None):
            d_ordered_yaw_ret = copy.deepcopy(d_ordered_yaw)
        else:
            d_ordered_yaw_ret = None
        

        if sanity_check_t(t_set, d_ordered, d_ordered_yaw):
            # Optimizae alpha
            alpha_high = 1.0
            alpha_low = 0.0
        else:
            alpha_high = 2.0
            alpha_low = 1.0
            while True:
                t_set_opt = t_set * alpha_high
                d_ordered_opt = self.get_alpha_matrix(alpha_high, N_wp).dot(d_ordered)
                if np.all(d_ordered_yaw != None):
                    d_ordered_yaw_opt = self.get_alpha_matrix_yaw(alpha_high ,N_wp).dot(d_ordered_yaw)
                else:
                    d_ordered_yaw_opt = None

                if not sanity_check_t(t_set_opt, d_ordered_opt, d_ordered_yaw_opt):
                    alpha_low = alpha_high
                    alpha_high *= 2.0
                else:
                    break

        while True:
            alpha_mid = (alpha_high + alpha_low)/2.0
            
            t_set_opt = t_set * alpha_mid
            d_ordered_opt = self.get_alpha_matrix(alpha_mid, N_wp).dot(d_ordered)
            if np.all(d_ordered_yaw != None):
                d_ordered_yaw_opt = self.get_alpha_matrix_yaw(alpha_mid, N_wp).dot(d_ordered_yaw)
            else:
                d_ordered_yaw_opt = None
            
            if not sanity_check_t(t_set_opt, d_ordered_opt, d_ordered_yaw_opt):
                alpha_low = alpha_mid
            else:
                alpha_high = alpha_mid
                t_set_ret = t_set_opt
                d_ordered_ret = d_ordered_opt
                d_ordered_yaw_ret = d_ordered_yaw_opt
            
            if alpha_high-alpha_low < precision:
                alpha = alpha_high
                break
        
        t_set = t_set_ret * alpha_scale
        d_ordered = self.get_alpha_matrix(alpha_scale,N_wp).dot(d_ordered_ret)
        if np.all(d_ordered_yaw != None):
            d_ordered_yaw = self.get_alpha_matrix_yaw(alpha_scale,N_wp).dot(d_ordered_yaw_ret)
        else:
            d_ordered_yaw = None
        
        if flag_return_alpha:
            return t_set, d_ordered, d_ordered_yaw, alpha
        else:
            return t_set, d_ordered, d_ordered_yaw

    def optimize_alpha_acc_clip(self, \
        points, t_set, d_ordered, d_ordered_yaw, alpha_scale=1.0, range_min=0.5, range_max=1.5, \
        flag_return_alpha=False, precision=0.001):
        
        N_points_t = self.N_POINTS
        flag_loop = self.check_flag_loop(t_set, d_ordered)
        N_POLY = t_set.shape[0]
        status = np.zeros((N_points_t*N_POLY,18))
        for der in range(5):
            if flag_loop:
                V_t = self.generate_sampling_matrix_loop(t_set, N=N_points_t, der=der)
            else:
                V_t = self.generate_sampling_matrix(t_set, N=N_points_t, der=der, endpoint=True)
            status[:,3*der:3*(der+1)] = V_t.dot(d_ordered)
        if np.all(d_ordered_yaw != None):
            status_yaw_xy = np.zeros((N_points_t*N_POLY,3,2))
            for der in range(3):
                if flag_loop:
                    V_t = self.generate_sampling_matrix_loop_yaw(t_set, N=N_points_t, der=der)
                else:
                    V_t = self.generate_sampling_matrix_yaw(t_set, N=N_points_t, der=der, endpoint=True)
                status_yaw_xy[:,der,:] = V_t.dot(d_ordered_yaw)

        def sanity_check_t(alpha):
            status_t = copy.deepcopy(status)
            for der in range(1,5):
                status_t[:,3*der:3*(der+1)] /= (alpha**der)
            
            if np.all(d_ordered_yaw != None):
                status_yaw_xy_t = copy.deepcopy(status_yaw_xy)
                for der in range(1,3):
                    status_yaw_xy_t[idx_set[idx]:idx_set[idx+1],der,:] = status_yaw_xy[:,der,:] / (alpha**der)
                # Needs to optimize
                status_t[:,15:] = self.get_yaw_der(status_yaw_xy_t)
            
            ws, ctrl = self._quadModel.getWs_vector(status_t)
            if np.any(ws < self._quadModel.w_min) or np.any(ws > self._quadModel.w_max):
                return False
            return True
        
        def return_val(alpha):
            t_set_ret = t_set * alpha
            d_ordered_ret = self.get_alpha_matrix(alpha,N_wp).dot(d_ordered)
            if np.all(d_ordered_yaw != None):
                d_ordered_yaw_ret = self.get_alpha_matrix_yaw(alpha,N_wp).dot(d_ordered_yaw)
            else:
                d_ordered_yaw_ret = None

            if flag_return_alpha:
                return t_set_ret, d_ordered_ret, d_ordered_yaw_ret, alpha
            else:
                return t_set_ret, d_ordered_ret, d_ordered_yaw_ret

        
        t_set_ret = copy.deepcopy(t_set)
        d_ordered_ret = copy.deepcopy(d_ordered)
        N_wp = int(d_ordered.shape[0]/self.N_DER)
        
        if np.all(d_ordered_yaw != None):
            d_ordered_yaw_ret = copy.deepcopy(d_ordered_yaw)
        else:
            d_ordered_yaw_ret = None
        
        if sanity_check_t(1.0):
            if sanity_check_t(range_min):
                return return_val(range_min)
            alpha_high = 1.0
            alpha_low = range_min
        else:
            if not sanity_check_t(range_max):
                return return_val(range_max)
            alpha_high = range_max
            alpha_low = 1.0
            while True:
                if not sanity_check_t(alpha_high):
                    alpha_low = alpha_high
                    alpha_high *= 2.0
                else:
                    break

        while True:
            alpha_mid = (alpha_high + alpha_low)/2.0
            if not sanity_check_t(alpha_mid):
                alpha_low = alpha_mid
            else:
                alpha_high = alpha_mid
            
            if alpha_high-alpha_low < precision:
                alpha = alpha_high
                break
        
        return return_val(alpha)
    
    def update_traj(self, \
        points, t_set, alpha_set, snap_w=None, \
        yaw_mode=0, \
        deg_init_min=0, deg_init_max=4, \
        deg_end_min=0, deg_end_max=0, \
        deg_init_yaw_min=0, deg_init_yaw_max=2, \
        deg_end_yaw_min=0, deg_end_yaw_max=0, \
        mu=1.0, kt=0, bernstein=None, end_der=None, return_snap=False):

        # points_mean = np.mean(np.abs(points[:,:3]))
        # points_mean_global = np.mean(points[:,:3],axis=0)

        if points.shape[0] == t_set.shape[0]:
            flag_loop = True
        else:
            flag_loop = False

        t_set_new = np.multiply(t_set, alpha_set)

        snap_b_ext_init = np.zeros((4,3))
        yaw_b_ext_init = None

        if np.all(bernstein != None):
            x_b_coeff = bernstein[:10].reshape(-1,1)
            x_b_poly = BPoly(x_b_coeff,[0,1])
            y_b_coeff = bernstein[10:20].reshape(-1,1)
            y_b_poly = BPoly(y_b_coeff,[0,1])
            for i in range(4):
                x_b_poly = BPoly.derivative(x_b_poly)
                snap_b_ext_init[i,0] = x_b_poly(1)
                y_b_poly = BPoly.derivative(y_b_poly)
                snap_b_ext_init[i,1] = y_b_poly(1)
        elif np.all(end_der != None):
            snap_b_ext_init[:,0] = end_der[0:4]
            snap_b_ext_init[:,1] = end_der[4:8]
            snap_b_ext_init[:,2] = end_der[8:12]
            if yaw_mode == 3:
                yaw_b_ext_init = np.zeros((2,1))
                yaw_b_ext_init[:,0] = end_der[12:14]
                # yaw_b_ext_init = end_der[12:14]

        if deg_end_max == 0 and deg_end_yaw_max == 0:
            flag_nonstop = True
        else:
            flag_nonstop = False
        
        if points.shape[1] == 3:
            points = np.hstack((points, np.ones((points.shape[0],1))))
        
        res, d_ordered = self.obj_func_acc(points=points, t_set=t_set_new, flag_loop=False, snap_w=snap_w, b_ext_init=snap_b_ext_init, flag_nonstop=flag_nonstop)
        res_yaw, d_ordered_yaw = self.obj_func_acc(points=points, t_set=t_set_new, flag_loop=False, snap_w=snap_w, \
                                flag_yaw=True, flag_direct_yaw=True, b_ext_init=yaw_b_ext_init, flag_nonstop=flag_nonstop)
        if not return_snap:
            return t_set_new, d_ordered, d_ordered_yaw
        else:
            return t_set_new, d_ordered, d_ordered_yaw, res + mu*res_yaw
    
    def obj_func_acc(self, points, t_set, flag_loop, snap_w=None, flag_yaw=False, flag_direct_yaw=False, 
                     b_ext_init=None, b_ext_end=None, flag_nonstop=False):
        if np.any(snap_w == None):
            snap_w = np.ones_like(t_set)
        else:
            snap_w *= t_set.shape[0]/np.sum(snap_w)
        
        if not flag_yaw:
            # points_mean = np.mean(np.abs(points[:,:3]))
            # points_mean_global = np.mean(points[:,:3],axis=0)
            # b = (copy.deepcopy(points[:,:3])-points_mean_global)/points_mean
            b = copy.deepcopy(points[:,:3])
            deg_init_min=0
            deg_init_max=4
            deg_end_min=0
            deg_end_max=4
            N_DER_t = self.N_DER
            SYS_DEG = self.MAX_SYS_DEG
        else:
            if not flag_direct_yaw:
                b = np.zeros((points.shape[0],2))
                b[:,0] = np.cos(points[:,3])
                b[:,1] = np.sin(points[:,3])
            else:
                b = np.zeros((points.shape[0],1))
                b[:,0] = points[:,3]
            deg_init_min=0
            deg_init_max=2
            deg_end_min=0
            deg_end_max=2
            N_DER_t = self.N_DER_YAW
            SYS_DEG = self.MAX_SYS_DEG_YAW
        if flag_nonstop:
            deg_end_max = 0
        
        flag_const_b = True
        for i in range(1,b.shape[0]):
            if np.any(b[i,:] != b[0,:]):
                flag_const_b = False
        if flag_const_b:
            res = 0
            d_ordered_res =  np.zeros((b.shape[0]*N_DER_t,b.shape[1]))
            for i in range(b.shape[0]):
                d_ordered_res[i*N_DER_t,:] = b[i,:]
            return res, d_ordered_res
        
        N_POLY = t_set.shape[0]
        if np.any(b_ext_init == None):
            b_ext_init = np.zeros((deg_init_max-deg_init_min,b.shape[1]))
        if np.any(b_ext_end == None):
            b_ext_end = np.zeros((deg_end_max-deg_end_min,b.shape[1]))

        if flag_loop:
            P = self.generate_perm_matrix(N_POLY-1, N_DER_t)
            if not flag_yaw:
                A_sys = self.generate_sampling_matrix_loop(t_set, N=self.N_POINTS, der=SYS_DEG)
            else:
                A_sys = self.generate_sampling_matrix_loop_yaw(t_set, N=self.N_POINTS, der=SYS_DEG)
            A_sys_t = A_sys.dot(P.T)
            D_tw = self.generate_weight_matrix(np.multiply(t_set,snap_w), self.N_POINTS)
            R = A_sys_t.T.dot(D_tw).dot(A_sys_t)
        else:
            P = self.generate_perm_matrix(N_POLY, N_DER_t)
            R = np.zeros((N_DER_t*(N_POLY+1),N_DER_t*(N_POLY+1)))
            for i in range(N_POLY):
                if not flag_yaw:
                    T2_t = np.array([1/t_set[i]**4,1/t_set[i]**3,1/t_set[i]**2,1/t_set[i]**1,1.])
                else:
                    T2_t = np.array([1/t_set[i]**2,1/t_set[i]**1,1.])
                T2_t = np.concatenate((T2_t,T2_t))
                T2_mat = np.diag(T2_t)
                if not flag_yaw:
                    if i == N_POLY-1:
                        R_t = T2_mat.dot(self.v_mat_sanity_end[SYS_DEG]).dot(T2_mat)*t_set[i]/self.N_POINTS
                    else:
                        R_t = T2_mat.dot(self.v_mat_sanity[SYS_DEG]).dot(T2_mat)*t_set[i]/self.N_POINTS
                else:
                    if i == N_POLY-1:
                        R_t = T2_mat.dot(self.v_mat_sanity_yaw_end[SYS_DEG]).dot(T2_mat)*t_set[i]/self.N_POINTS
                    else:
                        R_t = T2_mat.dot(self.v_mat_sanity_yaw[SYS_DEG]).dot(T2_mat)*t_set[i]/self.N_POINTS
                
                R[i*N_DER_t:(i+2)*N_DER_t,i*N_DER_t:(i+2)*N_DER_t] += R_t * snap_w[i]
            R = P.dot(R).dot(P.T)

        if flag_loop:
            R_idx = np.concatenate((np.arange(b.shape[0],b.shape[0]+deg_init_min),
                                    np.arange(b.shape[0]+deg_init_max,R.shape[0])), axis=0)
            b_idx = np.concatenate((np.arange(b.shape[0]),
                                    np.arange(b.shape[0]+deg_init_min,b.shape[0]+deg_init_max)), axis=0)
        else:
            R_idx = np.concatenate((np.arange(b.shape[0],b.shape[0]+deg_init_min),
                                    np.arange(b.shape[0]+deg_init_max,R.shape[0]-N_DER_t+1+deg_end_min),
                                    np.arange(R.shape[0]-N_DER_t+1+deg_end_max,R.shape[0])), axis=0)
            b_idx = np.concatenate((np.arange(b.shape[0]),
                                    np.arange(b.shape[0]+deg_init_min,b.shape[0]+deg_init_max),
                                    np.arange(R.shape[0]-N_DER_t+1+deg_end_min,R.shape[0]-N_DER_t+1+deg_end_max),
                                   ), axis=0)
        Rpp = R[np.ix_(R_idx,R_idx)]
        try:
            Rpp_inv = self.get_matrix_inv(Rpp)
        except RuntimeError as e:
            Rpp_inv = np.eye(Rpp.shape[0])
        
        if flag_loop:
            d_p = -Rpp_inv.dot(R[np.ix_(b_idx,R_idx)].T).dot(np.concatenate((b,b_ext_init),axis=0))
        else:
            d_p = -Rpp_inv.dot(R[np.ix_(b_idx,R_idx)].T).dot(np.concatenate((b,b_ext_init,b_ext_end),axis=0))
        
        if flag_loop:
            d_tmp = np.concatenate((
                b,
                d_p[:deg_init_min,:],
                b_ext_init,
                d_p[deg_init_min:,:]),axis=0)
        else:
            d_tmp = np.concatenate((
                b,
                d_p[:deg_init_min,:],
                b_ext_init,
                d_p[deg_init_min:d_p.shape[0]-N_DER_t+1+deg_end_max,:],
                b_ext_end,
                d_p[d_p.shape[0]-N_DER_t+1+deg_end_max:,:]),axis=0)
        
        res = np.trace(d_tmp.T.dot(R).dot(d_tmp))
        d_ordered = P.T.dot(d_tmp)
        
        if res < 0:
            res = 1e16
            
#         if not flag_yaw:
#             d_ordered *= points_mean
        return res, d_ordered
    
    def get_status_acc(self, t_set, d_ordered, flag_loop, flag_yaw=False):
        N_POLY = t_set.shape[0]
        
        if not flag_yaw:
            N_DER_t = self.N_DER
            SYS_DEG = self.MAX_SYS_DEG
            POS_DIM = d_ordered.shape[1]
            v0_sanity_t = self.v0_sanity
            v1_sanity_t = self.v1_sanity
            v0_sanity_end_t = self.v0_sanity_end
            v1_sanity_end_t = self.v1_sanity_end
        else:
            N_DER_t = self.N_DER_YAW
            SYS_DEG = self.MAX_SYS_DEG_YAW
            POS_DIM = d_ordered.shape[1]
            v0_sanity_t = self.v0_sanity_yaw
            v1_sanity_t = self.v1_sanity_yaw
            v0_sanity_end_t = self.v0_sanity_yaw_end
            v1_sanity_end_t = self.v1_sanity_yaw_end
        status = np.zeros((self.N_POINTS*N_POLY,(SYS_DEG+1)*POS_DIM))
            
        if flag_loop:
            for i in range(N_POLY):
                if not flag_yaw:
                    T2_mat = np.diag(np.array([1,t_set[i],t_set[i]**2,t_set[i]**3,t_set[i]**4]))
                else:
                    T2_mat = np.diag(np.array([1,t_set[i],t_set[i]**2]))
                for der in range(SYS_DEG+1):
                    if i == N_POLY-1:
                        der_0 = d_ordered[i*N_DER_t:(i+1)*N_DER_t,:]
                        der_1 = d_ordered[:N_DER_t,:]
                    else:
                        der_0 = d_ordered[i*N_DER_t:(i+1)*N_DER_t,:]
                        der_1 = d_ordered[(i+1)*N_DER_t:(i+2)*N_DER_t,:]
                    v0_d = v0_sanity_t[der].dot(T2_mat).dot(der_0)
                    v1_d = v1_sanity_t[der].dot(T2_mat).dot(der_1)
                        
                    status[i*self.N_POINTS:(i+1)*self.N_POINTS,POS_DIM*der:POS_DIM*(der+1)] = v0_d+v1_d
                    T2_mat /= t_set[i]
        else:
            for i in range(N_POLY):
                if not flag_yaw:
                    T2_mat = np.diag(np.array([1,t_set[i],t_set[i]**2,t_set[i]**3,t_set[i]**4]))
                else:
                    T2_mat = np.diag(np.array([1,t_set[i],t_set[i]**2]))
                for der in range(SYS_DEG+1):
                    if i == N_POLY-1:
                        v0_d = v0_sanity_end_t[der].dot(T2_mat).dot(d_ordered[i*N_DER_t:(i+1)*N_DER_t,:])
                        v1_d = v1_sanity_end_t[der].dot(T2_mat).dot(d_ordered[(i+1)*N_DER_t:(i+2)*N_DER_t,:])
                    else:
                        v0_d = v0_sanity_t[der].dot(T2_mat).dot(d_ordered[i*N_DER_t:(i+1)*N_DER_t,:])
                        v1_d = v1_sanity_t[der].dot(T2_mat).dot(d_ordered[(i+1)*N_DER_t:(i+2)*N_DER_t,:])
                    status[i*self.N_POINTS:(i+1)*self.N_POINTS,POS_DIM*der:POS_DIM*(der+1)] = v0_d+v1_d
                    T2_mat /= t_set[i]
        
        return status

    def optimize_alpha_acc_yaw(self, \
         points_list, t_set_list, \
         snap_w_list=[], \
         range_min=0.5, range_max=1.5, opt_N_eval=11, opt_N_step=2, direct_yaw=False, flag_nonstop=False):
#         start_time = time.time()
        batch_size = len(points_list)
        
        status_list = []
        for idx_t in range(batch_size):
            points = points_list[idx_t]
            t_set = t_set_list[idx_t]
            if len(snap_w_list) > 0:
                snap_w = snap_w_list[idx_t]
            else:
                snap_w = np.ones_like(t_set)
            flag_loop = self.check_flag_loop_points(t_set, points)
            N_POLY = t_set.shape[0]
            res, d_ordered = self.obj_func_acc(points, t_set, flag_loop, snap_w=snap_w, flag_nonstop=flag_nonstop)
            
            try:
                res, d_ordered = self.obj_func_acc(points, t_set, flag_loop, snap_w=snap_w, flag_nonstop=flag_nonstop)
            except RuntimeError as e:
                status = np.zeros((self.N_POINTS*N_POLY,18))
                return status
            res_yaw, d_ordered_yaw = self.obj_func_acc(points, t_set, flag_loop, snap_w=snap_w, flag_yaw=True, flag_direct_yaw=direct_yaw, flag_nonstop=flag_nonstop)
            
            # generate status
            status = np.zeros((self.N_POINTS*N_POLY,18))
            status[:,:15] = self.get_status_acc(t_set, d_ordered, flag_loop)
            status_yaw = self.get_status_acc(t_set, d_ordered_yaw, flag_loop, flag_yaw=True)
            status_yaw = np.array(np.split(status_yaw,3,axis=1))
            status_yaw = np.swapaxes(status_yaw,0,1)
            if not direct_yaw:
                status[:,15:] = self.get_yaw_der(status_yaw)
            else:
                status[:,15:] = status_yaw[:,:,0]
            
            status_list.append(np.array(status))
        
#         print("status_list %9.4f sec" % (time.time() - start_time))

#         start_time = time.time()
        
        def sanity_check_t(alpha_set_list, res_ii_idx_inv):
            batch_size_t = len(alpha_set_list)
            status_set = []
            len_status_set = np.zeros(batch_size_t)
            len_status_set_idx = np.zeros(batch_size_t)
            for i, alpha_set in enumerate(alpha_set_list):
                r_ii = int(res_ii_idx_inv[i])
                for alpha in alpha_set:
                    status_t = status_list[r_ii].copy()
                    mat_alpha = np.array([1,alpha,alpha**2,alpha**3,alpha**4,1])
                    mat_alpha = np.repeat(mat_alpha,3)
                    mat_alpha[-2] = alpha
                    mat_alpha[-1] = alpha**2
                    status_t = status_t / mat_alpha
                    status_set.append(status_t)
                len_status = status_list[r_ii].shape[0]
                len_status_set[i] = len_status
                len_status_set_idx[i] = len_status*len(alpha_set)
            status_set = np.concatenate(status_set, axis=0)
            idx_set = np.cumsum(len_status_set_idx)
            idx_set = np.concatenate((np.zeros(1),idx_set)).astype(np.int32)
            failure_idx_set = np.ones(batch_size_t)*-1.
            
            ws, ctrl = self._quadModel.getWs_vector(status_set)
            if np.all(status_set == 0.):
                ws = np.ones_like(ws) * -1.

            for i in range(batch_size_t):
                max_idx_h_t = np.where(np.any(ws[idx_set[i]:idx_set[i+1],:] > self._quadModel.w_max, axis=1))[0]
                max_idx_l_t = np.where(np.any(ws[idx_set[i]:idx_set[i+1],:] < self._quadModel.w_min, axis=1))[0]
                max_idx_h = -1
                if max_idx_h_t.shape[0] > 0:
                    max_idx_h = np.max(max_idx_h_t)
                max_idx_l = -1
                if max_idx_l_t.shape[0] > 0:
                    max_idx_l = np.max(max_idx_l_t)
                max_idx = max(max_idx_h, max_idx_l)
                if max_idx == -1:
                    failure_idx = -1
                else:
                    failure_idx = int(max_idx/len_status_set[i])
                failure_idx_set[i] = failure_idx
                
            return failure_idx_set
        
        N_eval = opt_N_eval
        N_step = opt_N_step
        
        alpha_low_list = np.ones(batch_size)*range_min
        alpha_high_list = np.ones(batch_size)*range_max
        flag_eval = np.ones(batch_size)
        res_alpha = np.ones(batch_size)
        for step in range(N_step):
            alpha_set_list = []
            res_ii_idx = np.zeros(batch_size)
            res_ii_idx_inv = []
            res_ii_idx_t = 0
            for b_ii in range(batch_size):
                if flag_eval[b_ii] == 1:
                    if step == 0:
                        alpha_set_list.append(list(np.linspace(alpha_low_list[b_ii],alpha_high_list[b_ii],N_eval)))
                    else:
                        alpha_set_list.append(list(np.linspace(alpha_low_list[b_ii],alpha_high_list[b_ii],N_eval))[1:-1])
                    res_ii_idx[b_ii] = res_ii_idx_t
                    res_ii_idx_inv.append(b_ii)
                    res_ii_idx_t += 1
            res_idx = sanity_check_t(alpha_set_list, res_ii_idx_inv)
            
            for b_ii in range(batch_size):
                if flag_eval[b_ii] == 1:
                    r_ii = int(res_ii_idx[b_ii])
                    if res_idx[r_ii] == -1 and step == 0:
                        res_alpha[b_ii] = alpha_low_list[b_ii]
                        flag_eval[b_ii] = 0
                    elif res_idx[r_ii] == -1 and step > 0:
                        alpha_high_list[b_ii] = alpha_set_list[r_ii][0]
                    elif res_idx[r_ii] == len(alpha_set_list[r_ii])-1 and step == 0:
                        res_alpha[b_ii] = alpha_high_list[b_ii]
                        flag_eval[b_ii] = 0
                    elif res_idx[r_ii] == len(alpha_set_list[r_ii])-1 and step > 0:
                        alpha_low_list[b_ii] = alpha_set_list[r_ii][-1]
                    else:
                        alpha_low_list[b_ii] = alpha_set_list[r_ii][int(res_idx[r_ii])]
                        alpha_high_list[b_ii] = alpha_set_list[r_ii][int(res_idx[r_ii])+1]
            
            if np.sum(flag_eval) == 0:
                break
        
        for b_ii in range(batch_size):
            if flag_eval[b_ii] == 1:
                res_alpha[b_ii] = alpha_high_list[b_ii]
#         print("alpha_opt %9.4f sec" % (time.time() - start_time))
        return res_alpha

    def optimize_alpha_acc_yaw_sim(self, \
         points_list, t_set_list, snap_w_list=[], \
         range_min=0.5, range_max=1.5, opt_N_eval=11, opt_N_step=2, direct_yaw=False, flag_robust=False, flag_nonstop=False):
        if flag_robust:
            N_trial = 5
        else:
            N_trial = 1
        
        def eval_sim(args):
            t_set = args[0]
            d_ordered = args[1]
            d_ordered_yaw = args[2]
            debug_array = self.sim.run_simulation_from_der(
                t_set, d_ordered, d_ordered_yaw, N_trial=N_trial, 
                max_pos_err=0.2, min_pos_err=0.1, 
                max_yaw_err=15., min_yaw_err=5., 
                freq_ctrl=200, freq_sim=400, flag_debug=False, direct_yaw=direct_yaw)
            for i in range(N_trial):
                failure_idx = debug_array[i]["failure_idx"]
                if failure_idx != -1:
                    return False
            return True
        
        def sanity_check_t(alpha_set_list, res_ii_idx_inv):
            batch_size_t = len(alpha_set_list)
            N_alpha = len(alpha_set_list[0])
            failure_idx_set = np.ones(batch_size_t)*-1.
            
            alpha_flag = np.ones(batch_size_t)
            for al_ii in range(N_alpha-1,-1,-1):
                al_ii_idx_inv = []
                data_list = []
                for i, alpha_set in enumerate(alpha_set_list):
                    r_ii = int(res_ii_idx_inv[i])
                    if alpha_flag[i] == 1:
                        t_set_tmp = t_set_list[r_ii]*alpha_set[al_ii]
                        
                        points = points_list[r_ii]
                        t_set_tmp = t_set_list[r_ii]*alpha_set[al_ii]
                        if len(snap_w_list) > 0:
                            snap_w = snap_w_list[r_ii]
                        else:
                            snap_w = np.ones_like(t_set_tmp)
                        flag_loop = self.check_flag_loop_points(t_set_tmp, points)
                        res, d_ordered_tmp = self.obj_func_acc(points, t_set_tmp, flag_loop, snap_w=snap_w, flag_nonstop=flag_nonstop)
                        res_yaw, d_ordered_yaw_tmp = self.obj_func_acc(points, t_set_tmp, flag_loop, snap_w=snap_w, flag_yaw=True, flag_direct_yaw=direct_yaw, flag_nonstop=flag_nonstop)
                        data_list.append((t_set_tmp, d_ordered_tmp, d_ordered_yaw_tmp))
                        al_ii_idx_inv.append(i)
                 
                res_eval = np.array(parmap(eval_sim, data_list))
                
                for i in range(res_eval.shape[0]):
                    b_ii = al_ii_idx_inv[i]
                    if not res_eval[i]:
                        alpha_flag[b_ii] = 0
                        failure_idx_set[b_ii] = al_ii
                
                if np.all(alpha_flag == 0):
                    break
            
            return failure_idx_set
        
        batch_size = len(points_list)
        
        N_eval = opt_N_eval
        N_step = opt_N_step
        
        alpha_low_list = np.ones(batch_size)*range_min
        alpha_high_list = np.ones(batch_size)*range_max
        flag_eval = np.ones(batch_size)
        res_alpha = np.ones(batch_size)
        for step in range(N_step):
            alpha_set_list = []
            res_ii_idx = np.zeros(batch_size)
            res_ii_idx_inv = []
            res_ii_idx_t = 0
            for b_ii in range(batch_size):
                if flag_eval[b_ii] == 1:
                    if step == 0:
                        alpha_set_list.append(list(np.linspace(alpha_low_list[b_ii],alpha_high_list[b_ii],N_eval)))
                    else:
                        alpha_set_list.append(list(np.linspace(alpha_low_list[b_ii],alpha_high_list[b_ii],N_eval))[1:-1])
                    res_ii_idx[b_ii] = res_ii_idx_t
                    res_ii_idx_inv.append(b_ii)
                    res_ii_idx_t += 1
            res_idx = sanity_check_t(alpha_set_list, res_ii_idx_inv)
            
            for b_ii in range(batch_size):
                if flag_eval[b_ii] == 1:
                    r_ii = int(res_ii_idx[b_ii])
                    if res_idx[r_ii] == -1 and step == 0:
                        res_alpha[b_ii] = alpha_low_list[b_ii]
                        flag_eval[b_ii] = 0
                    elif res_idx[r_ii] == -1 and step > 0:
                        alpha_high_list[b_ii] = alpha_set_list[r_ii][0]
                    elif res_idx[r_ii] == len(alpha_set_list[r_ii])-1 and step == 0:
                        res_alpha[b_ii] = alpha_high_list[b_ii]
                        flag_eval[b_ii] = 0
                    elif res_idx[r_ii] == len(alpha_set_list[r_ii])-1 and step > 0:
                        alpha_low_list[b_ii] = alpha_set_list[r_ii][-1]
                    else:
                        alpha_low_list[b_ii] = alpha_set_list[r_ii][int(res_idx[r_ii])]
                        alpha_high_list[b_ii] = alpha_set_list[r_ii][int(res_idx[r_ii])+1]
            
            if np.sum(flag_eval) == 0:
                break
        
        for b_ii in range(batch_size):
            if flag_eval[b_ii] == 1:
                res_alpha[b_ii] = alpha_high_list[b_ii]
#         print("alpha_opt %9.4f sec" % (time.time() - start_time))
        return res_alpha

    def sanity_check_acc_yaw(self, \
         points_list, t_set_list, snap_w_list=[], direct_yaw=False, flag_sta=True):
        # if flag_sta:
        #     print("low")
        # else:
        #     print("high")
        
        def eval_sim(args):
            t_set = args[0]
            d_ordered = args[1]
            d_ordered_yaw = args[2]
            if np.sum(t_set) > 300.:
                return False
            debug_array = self.sim.run_simulation_from_der(
                t_set, d_ordered, d_ordered_yaw, N_trial=1, 
                max_pos_err=0.2, min_pos_err=0.1, 
                max_yaw_err=15., min_yaw_err=5., 
                freq_ctrl=200, freq_sim=400, flag_debug=False, direct_yaw=direct_yaw)
            failure_idx = debug_array[0]["failure_idx"]
            if failure_idx != -1:
                return False
            else:
                return True
        
        data_list = []
        for r_ii in range(len(points_list)):
            points = points_list[r_ii]
            t_set = t_set_list[r_ii]
            if len(snap_w_list) > 0:
                snap_w = snap_w_list[r_ii]
            else:
                snap_w = np.ones_like(t_set)
                
            flag_loop = self.check_flag_loop_points(t_set, points)
            res, d_ordered = self.obj_func_acc(points, t_set, flag_loop, snap_w=snap_w)
            res_yaw, d_ordered_yaw = self.obj_func_acc(points, t_set, flag_loop, snap_w=snap_w, flag_yaw=True, flag_direct_yaw=direct_yaw)
            if flag_sta:
                # generate status
                N_POLY = t_set.shape[0]
                status = np.zeros((self.N_POINTS*N_POLY,18))
                status[:,:15] = self.get_status_acc(t_set, d_ordered, flag_loop)
                status_yaw = self.get_status_acc(t_set, d_ordered_yaw, flag_loop, flag_yaw=True)
                status_yaw = np.array(np.split(status_yaw,3,axis=1))
                status_yaw = np.swapaxes(status_yaw,0,1)
                if not direct_yaw:
                    status[:,15:] = self.get_yaw_der(status_yaw)
                else:
                    status[:,15:] = status_yaw[:,:,0]
                # data_list.append(np.array(status))
                data_list.append(np.array(status))
            else:
                data_list.append((t_set, d_ordered, d_ordered_yaw))

        if flag_sta:
            batch_size_t = len(data_list)
            len_status_set = np.zeros(batch_size_t)
            for r_ii in range(batch_size_t):
                len_status_set[r_ii] = data_list[r_ii].shape[0]
            idx_set = np.cumsum(len_status_set)
            idx_set = np.concatenate((np.zeros(1),idx_set))
            idx_set = idx_set.astype(np.int)
            
            # status_set = np.concatenate(data_list, axis=0)
            # ws, ctrl = self._quadModel.getWs_vector_cupy(status_set)
            status_set = np.concatenate(data_list, axis=0)
            res_eval = np.zeros(batch_size_t)
            try:
                ws, ctrl = self._quadModel.getWs_vector(status_set)
                if np.all(status_set == 0.):
                    ws = np.ones_like(ws) * -1.

                for i in range(batch_size_t):
                    # if np.all(ws[idx_set[i]:idx_set[i+1],:] <= self._quadModel.w_max) and \
                    #     np.all(ws[idx_set[i]:idx_set[i+1],:] >= self._quadModel.w_min):
                    if np.all(ws[idx_set[i]:idx_set[i+1],:] <= self._quadModel.w_max) and \
                        np.all(ws[idx_set[i]:idx_set[i+1],:] >= self._quadModel.w_min):
                        res_eval[i] = 1
            except:
                print("Error in sanity check sta")
        else:
            if len(data_list) >= 100:
                try:
                    res_eval = np.array(parmap(eval_sim, data_list))
                except:
                    print("Error in sanity check sim")
            else:
                res_eval = np.zeros(len(data_list))
                try:
                    for i in range(len(data_list)):
                        res_eval[i] = eval_sim(data_list[i])
                except:
                    print("Error in sanity check sim")
        
        return res_eval
    
    def sanity_check_acc_yaw_online(self, \
         points_list, idx_new_list, t_set_list, snap_w_list=[], direct_yaw=True, flag_sta=True, flag_wp_update=False):
        # if flag_sta:
        #     print("low")
        # else:
        #     print("high")
        
        def eval_sim(args):
            t_set = args[0]
            d_ordered = args[1]
            d_ordered_yaw = args[2]
            if np.sum(t_set) > 300.:
                return False
            debug_array = self.sim.run_simulation_from_der(
                t_set, d_ordered, d_ordered_yaw, N_trial=1, 
                max_pos_err=0.2, min_pos_err=0.1, 
                max_yaw_err=15., min_yaw_err=5., 
                freq_ctrl=200, freq_sim=400, flag_debug=False, direct_yaw=direct_yaw)
            failure_idx = debug_array[0]["failure_idx"]
            if failure_idx != -1:
                return False
            else:
                return True
        
        data_list = []
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
            flag_loop = self.check_flag_loop_points(t_set_i, points)
            res, d_ordered = self.obj_func_acc(points, t_set_i, flag_loop, snap_w=snap_w_i)
            res_yaw, d_ordered_yaw = self.obj_func_acc(points, t_set_i, flag_loop, snap_w=snap_w_i, flag_yaw=True, flag_direct_yaw=direct_yaw)
            d_init = d_ordered[5*idx_new_list[r_ii]+1:5*(idx_new_list[r_ii]+1),:]
            d_yaw_init = d_ordered_yaw[3*idx_new_list[r_ii]+1:3*(idx_new_list[r_ii]+1),0:1]
            points_opt = points_new[idx_new_list[r_ii]:,:]
            t_set_opt = t_set_list[r_ii][1,idx_new_list[r_ii]:]
            snap_w_opt = snap_w_f[idx_new_list[r_ii]:]
            
            res, d_ordered_opt = self.obj_func_acc(points_opt, t_set_opt, flag_loop, snap_w=snap_w_opt, b_ext_init=d_init)
            res_yaw, d_ordered_yaw_opt = self.obj_func_acc(points_opt, t_set_opt, flag_loop, snap_w=snap_w_opt, \
                                                       flag_yaw=True, flag_direct_yaw=direct_yaw, b_ext_init=d_yaw_init)
            
            # np.set_printoptions(precision=3)
            # print(points)
            # print(t_set_i)
            # print(snap_w_i)
            # import pdb; pdb.set_trace()
            # try:
            #     res, d_ordered_opt = self.obj_func_acc(points_opt, t_set_opt, flag_loop, snap_w=snap_w_opt, b_ext_init=None)
            # except:
            #     np.set_printoptions(precision=3)
            #     print(d_ordered.shape)
            #     print(d_ordered)
            #     print(d_init)
            #     print(idx_new_list[r_ii])
            #     print(t_set_i)
            #     print(t_set_opt)
            #     print(snap_w_opt)
            #     import pdb; pdb.set_trace()
            # try:
            #     res_yaw, d_ordered_yaw_opt = self.obj_func_acc(points_opt, t_set_opt, flag_loop, snap_w=snap_w_opt, \
            #                                            flag_yaw=True, flag_direct_yaw=direct_yaw, b_ext_init=None)
            # except:
            #     np.set_printoptions(precision=3)
            #     print(d_ordered_yaw.shape)
            #     print(d_ordered_yaw)
            #     print(d_yaw_init)
            #     print(idx_new_list[r_ii])
            #     print(t_set_i)
            #     print(t_set_opt)
            #     print(snap_w_opt)
            #     import pdb; pdb.set_trace()
            
            # t_set = t_set_i
            # if np.max(np.abs(d_ordered[5*idx_new_list[r_ii]:,:] - d_ordered_opt)) > 1e-3:
            #     np.set_printoptions(precision=3)
            #     print(t_set_i)
            #     print(t_set_opt)
            #     print(snap_w_i)
            #     print(snap_w_opt)
            #     print(d_ordered[:,:])
            #     print(d_ordered_opt)
            #     print(np.abs(d_ordered[5*idx_new_list[r_ii]:,:] - d_ordered_opt))
            #     import pdb; pdb.set_trace()
            t_set = t_set_list[r_ii][1,:]
            d_ordered[5*idx_new_list[r_ii]:,:] = d_ordered_opt
            d_ordered_yaw[3*idx_new_list[r_ii]:,:] = d_ordered_yaw_opt
            ##########################################
            
            if flag_sta:
                # generate status
                N_POLY = t_set.shape[0]
                status = np.zeros((self.N_POINTS*N_POLY,18))
                status[:,:15] = self.get_status_acc(t_set, d_ordered, flag_loop)
                status_yaw = self.get_status_acc(t_set, d_ordered_yaw, flag_loop, flag_yaw=True)
                status_yaw = np.array(np.split(status_yaw,3,axis=1))
                status_yaw = np.swapaxes(status_yaw,0,1)
                if not direct_yaw:
                    status[:,15:] = self.get_yaw_der(status_yaw)
                else:
                    status[:,15:] = status_yaw[:,:,0]
                # data_list.append(np.array(status))
                data_list.append(np.array(status))
            else:
                data_list.append((t_set, d_ordered, d_ordered_yaw))

        if flag_sta:
            batch_size_t = len(data_list)
            len_status_set = np.zeros(batch_size_t)
            for r_ii in range(batch_size_t):
                len_status_set[r_ii] = data_list[r_ii].shape[0]
            idx_set = np.cumsum(len_status_set)
            idx_set = np.concatenate((np.zeros(1),idx_set))
            idx_set = idx_set.astype(np.int)
            
            # status_set = np.concatenate(data_list, axis=0)
            # ws, ctrl = self._quadModel.getWs_vector_cupy(status_set)
            status_set = np.concatenate(data_list, axis=0)
            res_eval = np.zeros(batch_size_t)
            try:
                ws, ctrl = self._quadModel.getWs_vector(status_set)
                if np.all(status_set == 0.):
                    ws = np.ones_like(ws) * -1.

                for i in range(batch_size_t):
                    # if np.all(ws[idx_set[i]:idx_set[i+1],:] <= self._quadModel.w_max) and \
                    #     np.all(ws[idx_set[i]:idx_set[i+1],:] >= self._quadModel.w_min):
                    if np.all(ws[idx_set[i]:idx_set[i+1],:] <= self._quadModel.w_max) and \
                        np.all(ws[idx_set[i]:idx_set[i+1],:] >= self._quadModel.w_min):
                        res_eval[i] = 1
                    # else:
                    #     print(ws[idx_set[i]:idx_set[i+1],:])
                    #     print(np.min(ws[idx_set[i]:idx_set[i+1],:]))
                    #     print(np.max(ws[idx_set[i]:idx_set[i+1],:]))
                    #     import pdb; pdb.set_trace()
            except:
                print("Error in sanity check sta")
        else:
            if len(data_list) >= 100:
                try:
                    res_eval = np.array(parmap(eval_sim, data_list))
                except:
                    print("Error in sanity check sim")
            else:
                res_eval = np.zeros(len(data_list))
                try:
                    for i in range(len(data_list)):
                        res_eval[i] = eval_sim(data_list[i])
                except:
                    print("Error in sanity check sim")
        
        return res_eval
    
    def optimize_alpha_acc_sim(self, \
         points_list, t_set_list, \
         range_min=0.5, range_max=1.5, opt_N_eval=11, opt_N_step=2):
        
        def snap_obj_tmp(args):
            points = args[0]
            t_set = args[1]
            points_mean = np.mean(np.abs(points[:,:3]))
            points_mean_global = np.mean(points[:,:3],axis=0)
            b = (copy.deepcopy(points[:,:3])-points_mean_global)/points_mean

            flag_loop = self.check_flag_loop_points(t_set,b)
            N_POLY = t_set.shape[0]
            deg_init_min=0
            deg_init_max=4
            deg_end_min=0
            deg_end_max=4
            b_ext_init = np.zeros((deg_init_max-deg_init_min,b.shape[1]))
            b_ext_end = np.zeros((deg_end_max-deg_end_min,b.shape[1]))

            if flag_loop:
                P = self.generate_perm_matrix(N_POLY-1, self.N_DER)
                A_sys = self.generate_sampling_matrix_loop(t_set, N=self.N_POINTS, der=self.MAX_SYS_DEG)
                A_sys_t = A_sys.dot(P.T)
                D_tw = self.generate_weight_matrix(t_set, self.N_POINTS)
                R = A_sys_t.T.dot(D_tw).dot(A_sys_t)
            else:
                P = self.generate_perm_matrix(N_POLY, self.N_DER)
                R = np.zeros((self.N_DER*(N_POLY+1),self.N_DER*(N_POLY+1)))
                for i in range(N_POLY):
                    T2_t = np.array([1/t_set[i]**4,1/t_set[i]**3,1/t_set[i]**2,1/t_set[i]**1,1.])
                    T2_t = np.concatenate((T2_t,T2_t))
                    T2_mat = np.diag(T2_t)
                    if i == N_POLY-1:
                        R_t = T2_mat.dot(self.v_mat_sanity_end[self.MAX_SYS_DEG]).dot(T2_mat)*t_set[i]/self.N_POINTS
                    else:
                        R_t = T2_mat.dot(self.v_mat_sanity[self.MAX_SYS_DEG]).dot(T2_mat)*t_set[i]/self.N_POINTS
                    R[i*self.N_DER:(i+2)*self.N_DER,i*self.N_DER:(i+2)*self.N_DER] += R_t
                R = P.dot(R).dot(P.T)
            
            if flag_loop:
                R_idx = np.concatenate((np.arange(b.shape[0],b.shape[0]+deg_init_min),
                                        np.arange(b.shape[0]+deg_init_max,R.shape[0])), axis=0)
            else:
                R_idx = np.concatenate((np.arange(b.shape[0],b.shape[0]+deg_init_min),
                                        np.arange(b.shape[0]+deg_init_max,R.shape[0]-self.N_DER+1+deg_end_min),
                                        np.arange(R.shape[0]-self.N_DER+1+deg_end_max,R.shape[0])), axis=0)
            Rpp = R[np.ix_(R_idx,R_idx)]
            Rpp_inv = self.get_matrix_inv(Rpp)

            d_p = -Rpp_inv.dot(R[np.ix_(np.arange(b.shape[0]),R_idx)].T).dot(b)
            if flag_loop:
                d_tmp = np.concatenate((
                    b,
                    d_p[:deg_init_min,:],
                    b_ext_init,
                    d_p[deg_init_min:,:]),axis=0)
            else:
                d_tmp = np.concatenate((
                    b,
                    d_p[:deg_init_min,:],
                    b_ext_init,
                    d_p[deg_init_min:d_p.shape[0]-self.N_DER+1+deg_end_max,:],
                    b_ext_end,
                    d_p[d_p.shape[0]-self.N_DER+1+deg_end_max:,:]),axis=0)

            d_ordered = P.T.dot(d_tmp)
            d_ordered_yaw = None
            d_ordered *= points_mean
            
            return d_ordered
        
#         start_time = time.time()
        
        def eval_sim(args):
            t_set = args[0]
            d_ordered = args[1]
            debug_array = self.sim.run_simulation_from_der(
                t_set, d_ordered, None, N_trial=1, 
                max_pos_err=0.2, min_pos_err=0.1, 
                max_yaw_err=120., min_yaw_err=60., 
                freq_ctrl=200, freq_sim=400, flag_debug=False)
            failure_idx = debug_array[0]["failure_idx"]
            if failure_idx != -1:
                return False
            else:
                return True
        
        def sanity_check_t(alpha_set_list, res_ii_idx_inv):
            batch_size_t = len(alpha_set_list)
            N_alpha = len(alpha_set_list[0])
            failure_idx_set = np.ones(batch_size_t)*-1.
            
            alpha_flag = np.ones(batch_size_t)
            for al_ii in range(N_alpha-1,-1,-1):
                al_ii_idx_inv = []
                data_list = []
                for i, alpha_set in enumerate(alpha_set_list):
                    r_ii = int(res_ii_idx_inv[i])
                    if alpha_flag[i] == 1:
                        t_set_tmp = t_set_list[r_ii]*alpha_set[al_ii]
                        data_list.append((t_set_tmp, snap_obj_tmp((points_list[r_ii], t_set_tmp))))
                        al_ii_idx_inv.append(i)
                res_eval = np.array(parmap(eval_sim, data_list))
#                 res_eval = []
#                 for i in range(len(data_list)):
#                     res_eval.append(eval_sim(data_list[i]))
                
                for i in range(res_eval.shape[0]):
                    b_ii = al_ii_idx_inv[i]
                    if not res_eval[i]:
                        alpha_flag[b_ii] = 0
                        failure_idx_set[b_ii] = al_ii
                
                if np.all(alpha_flag == 0):
                    break
            
            return failure_idx_set
        
        batch_size = len(points_list)
        
        N_eval = opt_N_eval
        N_step = opt_N_step
        
        alpha_low_list = np.ones(batch_size)*range_min
        alpha_high_list = np.ones(batch_size)*range_max
        flag_eval = np.ones(batch_size)
        res_alpha = np.ones(batch_size)
        for step in range(N_step):
            alpha_set_list = []
            res_ii_idx = np.zeros(batch_size)
            res_ii_idx_inv = []
            res_ii_idx_t = 0
            for b_ii in range(batch_size):
                if flag_eval[b_ii] == 1:
                    if step == 0:
                        alpha_set_list.append(list(np.linspace(alpha_low_list[b_ii],alpha_high_list[b_ii],N_eval)))
                    else:
                        alpha_set_list.append(list(np.linspace(alpha_low_list[b_ii],alpha_high_list[b_ii],N_eval))[1:-1])
                    res_ii_idx[b_ii] = res_ii_idx_t
                    res_ii_idx_inv.append(b_ii)
                    res_ii_idx_t += 1
            res_idx = sanity_check_t(alpha_set_list, res_ii_idx_inv)
            
            for b_ii in range(batch_size):
                if flag_eval[b_ii] == 1:
                    r_ii = int(res_ii_idx[b_ii])
                    if res_idx[r_ii] == -1 and step == 0:
                        res_alpha[b_ii] = alpha_low_list[b_ii]
                        flag_eval[b_ii] = 0
                    elif res_idx[r_ii] == -1 and step > 0:
                        alpha_high_list[b_ii] = alpha_set_list[r_ii][0]
                    elif res_idx[r_ii] == len(alpha_set_list[r_ii])-1 and step == 0:
                        res_alpha[b_ii] = alpha_high_list[b_ii]
                        flag_eval[b_ii] = 0
                    elif res_idx[r_ii] == len(alpha_set_list[r_ii])-1 and step > 0:
                        alpha_low_list[b_ii] = alpha_set_list[r_ii][-1]
                    else:
                        alpha_low_list[b_ii] = alpha_set_list[r_ii][int(res_idx[r_ii])]
                        alpha_high_list[b_ii] = alpha_set_list[r_ii][int(res_idx[r_ii])+1]
            
            if np.sum(flag_eval) == 0:
                break
        
        for b_ii in range(batch_size):
            if flag_eval[b_ii] == 1:
                res_alpha[b_ii] = alpha_high_list[b_ii]
#         print("alpha_opt %9.4f sec" % (time.time() - start_time))
        return res_alpha

    def optimize_alpha_acc_clip3(self, \
         points_list, t_set_list, \
         snap_w_list, \
         range_min=0.5, range_max=1.5, opt_N_eval=11, opt_N_step=2):
        
        def snap_obj_tmp(args):
            points = args[0]
            t_set = args[1]
            snap_w = args[2]
            flag_loop = self.check_flag_loop_points(t_set, points)
            N_POLY = t_set.shape[0]
            try:
                res, d_ordered = self.obj_func_acc(points, t_set, flag_loop, snap_w=snap_w)
            except RuntimeError as e:
                status = np.ones((self.N_POINTS*N_POLY,18)) * -1.
                return status
            # generate status
            status = np.zeros((self.N_POINTS*N_POLY,18))
            status[:,:15] = self.get_status_acc(t_set, d_ordered, flag_loop)

            return status
        
#         start_time = time.time()
        batch_size = len(points_list)
        
        status_list = []
        for idx_t in range(batch_size):
#             if len(snap_w_list) > 0:
#                 snap_w = snap_w_list[idx_t]
#             else:
#                 snap_w = np.ones_like(t_set)
            status_list.append(np.array(snap_obj_tmp((points_list[idx_t], t_set_list[idx_t], snap_w_list[idx_t]))))
#         print("status_list %9.4f sec" % (time.time() - start_time))

#         start_time = time.time()
        
        def sanity_check_t(alpha_set_list, res_ii_idx_inv):
            batch_size_t = len(alpha_set_list)
            status_set = []
            len_status_set = np.zeros(batch_size_t)
            len_status_set_idx = np.zeros(batch_size_t)
            for i, alpha_set in enumerate(alpha_set_list):
                r_ii = int(res_ii_idx_inv[i])
                for alpha in alpha_set:
                    status_t = status_list[r_ii].copy()
                    mat_alpha = np.array([1,alpha,alpha**2,alpha**3,alpha**4,1])
                    mat_alpha = np.repeat(mat_alpha,3)
                    status_t = status_t / mat_alpha
                    status_set.append(status_t)
                len_status = status_list[r_ii].shape[0]
                len_status_set[i] = len_status
                len_status_set_idx[i] = len_status*len(alpha_set)
            status_set = np.concatenate(status_set, axis=0)
            idx_set = np.cumsum(len_status_set_idx)
            idx_set = np.concatenate((np.zeros(1),idx_set))
            failure_idx_set = np.ones(batch_size_t)*-1.
            
            ws, ctrl = self._quadModel.getWs_vector(status_set)

            for i in range(batch_size_t):
                max_idx_h_t = np.where(np.any(ws[idx_set[i]:idx_set[i+1],:] > self._quadModel.w_max, axis=1))[0]
                max_idx_l_t = np.where(np.any(ws[idx_set[i]:idx_set[i+1],:] < self._quadModel.w_min, axis=1))[0]
                max_idx_h = -1
                if max_idx_h_t.shape[0] > 0:
                    max_idx_h = np.max(max_idx_h_t)
                max_idx_l = -1
                if max_idx_l_t.shape[0] > 0:
                    max_idx_l = np.max(max_idx_l_t)
                max_idx = max(max_idx_h, max_idx_l)
                if max_idx == -1:
                    failure_idx = -1
                else:
                    failure_idx = int(max_idx/len_status_set[i])
                failure_idx_set[i] = failure_idx
                
            return failure_idx_set
        
        N_eval = opt_N_eval
        N_step = opt_N_step
        
        alpha_low_list = np.ones(batch_size)*range_min
        alpha_high_list = np.ones(batch_size)*range_max
        flag_eval = np.ones(batch_size)
        res_alpha = np.ones(batch_size)
        for step in range(N_step):
            alpha_set_list = []
            res_ii_idx = np.zeros(batch_size)
            res_ii_idx_inv = []
            res_ii_idx_t = 0
            for b_ii in range(batch_size):
                if flag_eval[b_ii] == 1:
                    if step == 0:
                        alpha_set_list.append(list(np.linspace(alpha_low_list[b_ii],alpha_high_list[b_ii],N_eval)))
                    else:
                        alpha_set_list.append(list(np.linspace(alpha_low_list[b_ii],alpha_high_list[b_ii],N_eval))[1:-1])
                    res_ii_idx[b_ii] = res_ii_idx_t
                    res_ii_idx_inv.append(b_ii)
                    res_ii_idx_t += 1
            res_idx = sanity_check_t(alpha_set_list, res_ii_idx_inv)
            
            for b_ii in range(batch_size):
                if flag_eval[b_ii] == 1:
                    r_ii = int(res_ii_idx[b_ii])
                    if res_idx[r_ii] == -1 and step == 0:
                        res_alpha[b_ii] = alpha_low_list[b_ii]
                        flag_eval[b_ii] = 0
                    elif res_idx[r_ii] == -1 and step > 0:
                        alpha_high_list[b_ii] = alpha_set_list[r_ii][0]
                    elif res_idx[r_ii] == len(alpha_set_list[r_ii])-1 and step == 0:
                        res_alpha[b_ii] = alpha_high_list[b_ii]
                        flag_eval[b_ii] = 0
                    elif res_idx[r_ii] == len(alpha_set_list[r_ii])-1 and step > 0:
                        alpha_low_list[b_ii] = alpha_set_list[r_ii][-1]
                    else:
                        alpha_low_list[b_ii] = alpha_set_list[r_ii][int(res_idx[r_ii])]
                        alpha_high_list[b_ii] = alpha_set_list[r_ii][int(res_idx[r_ii])+1]
            
            if np.sum(flag_eval) == 0:
                break
        
        for b_ii in range(batch_size):
            if flag_eval[b_ii] == 1:
                res_alpha[b_ii] = alpha_high_list[b_ii]
#         print("alpha_opt %9.4f sec" % (time.time() - start_time))
        return res_alpha
    
    def optimize_alpha_acc_clip2(self, \
        points, t_set, d_ordered, d_ordered_yaw, \
        alpha_scale=1.0, range_min=0.5, range_max=1.5, \
        flag_return_alpha=False, precision=0.001):
        
        N_points_t = self.N_POINTS
        flag_loop = self.check_flag_loop_points(t_set, d_ordered)
        N_POLY = t_set.shape[0]
        status = np.zeros((N_points_t*N_POLY,18))
        for der in range(5):
            if flag_loop:
                V_t = self.generate_sampling_matrix_loop(t_set, N=N_points_t, der=der)
            else:
                V_t = self.generate_sampling_matrix(t_set, N=N_points_t, der=der, endpoint=True)
            status[:,3*der:3*(der+1)] = V_t.dot(d_ordered)
        if np.all(d_ordered_yaw != None):
            status_yaw_xy = np.zeros((N_points_t*N_POLY,3,2))
            for der in range(3):
                if flag_loop:
                    V_t = self.generate_sampling_matrix_loop_yaw(t_set, N=N_points_t, der=der)
                else:
                    V_t = self.generate_sampling_matrix_yaw(t_set, N=N_points_t, der=der, endpoint=True)
                status_yaw_xy[:,der,:] = V_t.dot(d_ordered_yaw)

        def sanity_check_t(alpha_set):
            status_set = []
            for alpha in alpha_set:
                status_t = copy.deepcopy(status)
                for der in range(1,5):
                    status_t[:,3*der:3*(der+1)] /= (alpha**der)
                status_set.append(status_t)
            len_status = status.shape[0]
            status_set = np.concatenate(status_set, axis=0)
            
            ws, ctrl = self._quadModel.getWs_vector(status_set)
            min_idx_h_t = np.where(np.any(ws > self._quadModel.w_max, axis=1))[0]
            min_idx_l_t = np.where(np.any(ws < self._quadModel.w_min, axis=1))[0]
            
            min_idx_h = -1
            if min_idx_h_t.shape[0] > 0:
                min_idx_h = np.max(min_idx_h_t)
            min_idx_l = -1
            if min_idx_l_t.shape[0] > 0:
                min_idx_l = np.max(min_idx_l_t)
            min_idx = max(min_idx_h, min_idx_l)
            if min_idx == -1:
                failure_idx = -1
            else:
                failure_idx = int(min_idx/len_status)
            return failure_idx
        
        def return_val(alpha):
            t_set_ret = t_set * alpha
            d_ordered_ret = self.get_alpha_matrix(alpha,N_wp).dot(d_ordered)
            if np.all(d_ordered_yaw != None):
                d_ordered_yaw_ret = self.get_alpha_matrix_yaw(alpha,N_wp).dot(d_ordered_yaw)
            else:
                d_ordered_yaw_ret = None

            if flag_return_alpha:
                return t_set_ret, d_ordered_ret, d_ordered_yaw_ret, alpha
            else:
                return t_set_ret, d_ordered_ret, d_ordered_yaw_ret

        
        t_set_ret = copy.deepcopy(t_set)
        d_ordered_ret = copy.deepcopy(d_ordered)
        N_wp = int(d_ordered.shape[0]/self.N_DER)
        
        if np.all(d_ordered_yaw != None):
            d_ordered_yaw_ret = copy.deepcopy(d_ordered_yaw)
        else:
            d_ordered_yaw_ret = None
        
        N_eval = 10
        N_step = 2
        alpha_low = range_min
        alpha_high = range_max
        for step in range(N_step):
            alpha_set = list(np.linspace(alpha_low,alpha_high,N_eval))
            res_idx = sanity_check_t(alpha_set)
            if res_idx == -1:
                return return_val(alpha_low)
            if res_idx == len(alpha_set)-1:
                return return_val(alpha_high)
            alpha_low = alpha_set[res_idx]
            alpha_high = alpha_set[res_idx+1]
        return return_val(alpha_high)

    
if __name__ == "__main__":
    poly = MinSnapTraj(MAX_POLY_DEG = 9, MAX_SYS_DEG = 4, N_POINTS = 40)
    
    
