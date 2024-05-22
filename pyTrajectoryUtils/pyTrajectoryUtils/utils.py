#!/usr/bin/env python
# coding: utf-8

import os, sys, copy, yaml
from os import path
import numpy as np
import pandas as pd
from scipy.special import factorial
from scipy.special import comb, perm
from multiprocessing import Pool, Pipe, TimeoutError, Process
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter
from parfor import pmap

def prRed(prt): print("\033[91m {}\033[00m" .format(prt))
def prGreen(prt): print("\033[92m {}\033[00m" .format(prt))
def prYellow(prt): print("\033[93m {}\033[00m" .format(prt))
def prLightPurple(prt): print("\033[94m {}\033[00m" .format(prt))
def prPurple(prt): print("\033[95m {}\033[00m" .format(prt))
def prCyan(prt): print("\033[96m {}\033[00m" .format(prt))
def prLightGray(prt): print("\033[97m {}\033[00m" .format(prt))
def prBlack(prt): print("\033[98m {}\033[00m" .format(prt))

def spawn(f):
    def fun(pipe,x):
        pipe.send(f(x))
        pipe.close()
    return fun

def parmap(f,X):
    pipe=[Pipe() for x in X]
    proc=[Process(target=spawn(f),args=(c,x)) for x,(p,c) in zip(X,pipe)]
    [p.start() for p in proc]
    [p.join() for p in proc]
    return [p.recv() for (p,c) in pipe]

# def parmap(f,X):
#     return pmap(f, X, bar=False)

def ned2enu(x):
    P = np.zeros((3,3))
    P[0,1] = 1
    P[1,0] = 1
    P[2,2] = -1
    if len(x.shape) == 1 and x.shape[0] >= 3:
        return np.append(x[:3].dot(P), x[3:])
    elif len(x.shape) == 2 and x.shape[1] >= 3:
        return np.append(x[:,:3].dot(P), x[:,3:], axis=1)
    else:
        raise ValueError
        
def ne2en(x):
    P = np.zeros((2,2))
    P[0,1] = 1
    P[1,0] = 1
    if len(x.shape) == 1 and x.shape[0] == 2:
        return x.dot(P)
    elif len(x.shape) == 2 and x.shape[1] == 2:
        return x.dot(P)
    else:
        raise ValueError

def quat2Euler(q):
    # w x y z
    roll = np.arctan2(2*(q[0]*q[1]+q[2]*q[3]), (1.-2.*(np.power(q[1],2)+np.power(q[2],2))))
    sinp = 2.*(q[0]*q[2]-q[3]*q[1])
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi/2, sinp)
    else:
        pitch = np.arcsin(sinp)
    yaw = np.arctan2(2.*(q[0]*q[3]+q[1]*q[2]), (1.-2.*(np.power(q[2],2)+np.power(q[3],2))))
    return roll, pitch, yaw

def Euler2quat(att):
    cr = np.cos(att[0]/2)
    sr = np.sin(att[0]/2)
    cp = np.cos(att[1]/2)
    sp = np.sin(att[1]/2)
    cy = np.cos(att[2]/2)
    sy = np.sin(att[2]/2)
    
    q = np.zeros(4)
    q[0] = cy * cp * cr + sy * sp * sr
    q[1] = sr * cp * cy - cr * sp * sy
    q[2] = cr * sp * cy + sr * cp * sy
    q[3] = cr * cp * sy - sr * sp * cy
    
    return q

def get_waypoints(name_yaml, name_traj, flag_t_set=False):
    if path.exists(name_yaml):
        file_path = name_yaml
    else:
        curr_path = os.path.dirname(os.path.abspath(__file__))
        file_path = curr_path+"/../waypoints/{}.yaml".format(name_yaml)
    with open(file_path, 'r') as stream:
        try:
            dict_ = yaml.safe_load(stream)
            points = np.array(dict_['{}'.format(str(name_traj))]['points'])
            if 't_set' in dict_['{}'.format(str(name_traj))] and flag_t_set:
                t_set = np.array(dict_['{}'.format(str(name_traj))]['t_set'])
                return points, t_set
            else:
                if flag_t_set:
                    return points, None
                else:
                    return points
        except yaml.YAMLError as exc:
            print(exc)
    return

def mod2pi(yaw):
    yawm = yaw - 2. * np.pi * np.floor(yaw / 2. / np.pi);
    return yawm

def get_closed_turn_direct_yaw(points_yaw):
    if len(points_yaw.shape) == 1:
        points_yaw_i = mod2pi(copy.deepcopy(points_yaw))
    else:
        points_yaw_i = mod2pi(copy.deepcopy(points_yaw[:,0]))
    
    points_yaw_t = copy.deepcopy(points_yaw_i)
    
    n_twopi = 0
    for i in range(points_yaw_i.shape[0]-1):
        if points_yaw_i[i] > points_yaw_i[i+1]:
            dyaw = points_yaw_i[i] - points_yaw_i[i+1]
            if dyaw > 2 * np.pi - dyaw:
                n_twopi += 1
            points_yaw_t[i+1] = points_yaw_i[i+1] + n_twopi * 2 * np.pi
        else:
            dyaw = points_yaw_i[i+1] - points_yaw_i[i]
            if dyaw > 2 * np.pi - dyaw:
                n_twopi -= 1
            points_yaw_t[i+1] = points_yaw_i[i+1] + n_twopi * 2 * np.pi
    return points_yaw_t

def get_closed_turn_direct_yaw_online(points_yaw):
    if len(points_yaw.shape) == 1:
        points_yaw_i = copy.deepcopy(points_yaw)
    else:
        points_yaw_i = copy.deepcopy(points_yaw[:,0])
    
    points_yaw_t = copy.deepcopy(points_yaw_i)
    
    n_twopi = 0
    for i in range(points_yaw_i.shape[0]-1):
        if points_yaw_i[i] > points_yaw_i[i+1]:
            dyaw = points_yaw_i[i] - points_yaw_i[i+1]
            if dyaw > 2 * np.pi - dyaw:
                n_twopi += 1
            points_yaw_t[i+1] = points_yaw_i[i+1] + n_twopi * 2 * np.pi
        else:
            dyaw = points_yaw_i[i+1] - points_yaw_i[i]
            if dyaw > 2 * np.pi - dyaw:
                n_twopi -= 1
            points_yaw_t[i+1] = points_yaw_i[i+1] + n_twopi * 2 * np.pi
    return points_yaw_t

# Hermite Iterpolation for symmetric end points derivatives
#  - Phillips, G. M. (1973). Explicit forms for certain Hermite approximations. BIT Numerical Mathematics, 13(2), 177-180.
# Hermite Iterpolation for general end points derivatives
#  - Mennig, J., Auerbach, T., & Hälg, W. (1983). Two point Hermite approximations for the solution of linear initial value and boundary value problems. Computer methods in applied mechanics and engineering, 39(2), 199-224.
class BaseTrajFunc(object):
    def __init__(self, *args, **kwargs):
        if "MAX_POLY_DEG" in kwargs:
            MAX_POLY_DEG = kwargs["MAX_POLY_DEG"]
        else:
            MAX_POLY_DEG = 9
        
        if "MAX_SYS_DEG" in kwargs:
            MAX_SYS_DEG = kwargs["MAX_SYS_DEG"]
        else:
            MAX_SYS_DEG = 4
        
        if "MAX_POLY_DEG_YAW" in kwargs:
            MAX_POLY_DEG_YAW = kwargs["MAX_POLY_DEG_YAW"]
        else:
            MAX_POLY_DEG_YAW = 5
        
        if "MAX_SYS_DEG_YAW" in kwargs:
            MAX_SYS_DEG_YAW = kwargs["MAX_SYS_DEG_YAW"]
        else:
            MAX_SYS_DEG_YAW = 2
        
        if "N_POINTS" in kwargs:
            N_POINTS = kwargs["N_POINTS"]
        else:
            N_POINTS = 200
                
        self.MAX_POLY_DEG = MAX_POLY_DEG
        self.MAX_SYS_DEG = MAX_SYS_DEG # SNAP
        self.MAX_POLY_DEG_YAW = MAX_POLY_DEG_YAW
        self.MAX_SYS_DEG_YAW = MAX_SYS_DEG_YAW # YawAcc
        self.N_POINTS = N_POINTS # Number of collocation points
        self.N_DER = self.MAX_POLY_DEG-self.MAX_SYS_DEG
        self.N_DER_YAW = self.MAX_POLY_DEG_YAW-self.MAX_SYS_DEG_YAW
        
        self.v0_mat, self.v1_mat = self.generate_interpolation_matrix_coeff()
        self.v0_sanity = np.zeros((MAX_SYS_DEG+1, N_POINTS, MAX_POLY_DEG-MAX_SYS_DEG))
        self.v1_sanity = np.zeros((MAX_SYS_DEG+1, N_POINTS, MAX_POLY_DEG-MAX_SYS_DEG))
        self.v_mat_sanity = np.zeros((MAX_SYS_DEG+1, self.N_DER*2, self.N_DER*2))
        for der in range(MAX_SYS_DEG+1):
            self.v0_sanity[der,:,:], self.v1_sanity[der,:,:], \
            self.v_mat_sanity[der,:,:] = self.generate_single_sampling_matrix(N=N_POINTS, der=der, endpoint=False)
            
        self.v0_sanity_end = np.zeros((MAX_SYS_DEG+1, N_POINTS, MAX_POLY_DEG-MAX_SYS_DEG))
        self.v1_sanity_end = np.zeros((MAX_SYS_DEG+1, N_POINTS, MAX_POLY_DEG-MAX_SYS_DEG))
        self.v_mat_sanity_end = np.zeros((MAX_SYS_DEG+1, self.N_DER*2, self.N_DER*2))
        for der in range(MAX_SYS_DEG+1):
            self.v0_sanity_end[der,:,:], self.v1_sanity_end[der,:,:], \
            self.v_mat_sanity_end[der,:,:] = self.generate_single_sampling_matrix(N=N_POINTS, der=der, endpoint=True)
        
        self.v0_mat_yaw, self.v1_mat_yaw = self.generate_interpolation_matrix_coeff(m=MAX_POLY_DEG_YAW+1,p=MAX_SYS_DEG_YAW+1)
        self.v0_sanity_yaw = np.zeros((MAX_SYS_DEG_YAW+1, N_POINTS, MAX_POLY_DEG_YAW-MAX_SYS_DEG_YAW))
        self.v1_sanity_yaw = np.zeros((MAX_SYS_DEG_YAW+1, N_POINTS, MAX_POLY_DEG_YAW-MAX_SYS_DEG_YAW))
        self.v_mat_sanity_yaw = np.zeros((MAX_SYS_DEG_YAW+1, self.N_DER_YAW*2, self.N_DER_YAW*2))
        for der in range(MAX_SYS_DEG_YAW+1):
            self.v0_sanity_yaw[der,:,:], self.v1_sanity_yaw[der,:,:], \
            self.v_mat_sanity_yaw[der,:,:] = self.generate_single_sampling_matrix_yaw(N=N_POINTS, der=der, endpoint=False)

        self.v0_sanity_yaw_end = np.zeros((MAX_SYS_DEG_YAW+1, N_POINTS, MAX_POLY_DEG_YAW-MAX_SYS_DEG_YAW))
        self.v1_sanity_yaw_end = np.zeros((MAX_SYS_DEG_YAW+1, N_POINTS, MAX_POLY_DEG_YAW-MAX_SYS_DEG_YAW))
        self.v_mat_sanity_yaw_end = np.zeros((MAX_SYS_DEG_YAW+1, self.N_DER_YAW*2, self.N_DER_YAW*2))
        for der in range(MAX_SYS_DEG_YAW+1):
            self.v0_sanity_yaw_end[der,:,:], self.v1_sanity_yaw_end[der,:,:], \
            self.v_mat_sanity_yaw_end[der,:,:] = self.generate_single_sampling_matrix_yaw(N=N_POINTS, der=der, endpoint=True)
            
#         self.cache_sampling_mat = dict()
#         self.cache_sampling_mat["list"] = [[],[],[],[],[]]
#         self.cache_sampling_mat_loop = dict()
#         self.cache_sampling_mat_loop["list"] = [[],[],[],[],[]]
#         self.cache_sampling_mat_yaw = dict()
#         self.cache_sampling_mat_yaw["list"] = [[],[],[]]
#         self.cache_sampling_mat_yaw_loop = dict()
#         self.cache_sampling_mat_yaw_loop["list"] = [[],[],[]]
        self.cache_perm_mat = dict()
        self.cache_perm_mat["list"] = []
            
    ###############################################################################
    def generate_basis(self, s, POLY_DEG=-1, der=0):
        if POLY_DEG == -1:
            POLY_DEG = self.MAX_POLY_DEG
        basis = np.zeros(POLY_DEG+1)
        for i in range(basis.shape[0]):
            if i >= der:
                basis[i] = s**(i-der)*perm(i,der)
        return basis
    
    def generate_interpolation_matrix_coeff(self, m=-1, p=-1):
        if m == -1:
            m = self.MAX_POLY_DEG + 1
        if p == -1:
            p = self.MAX_SYS_DEG + 1
        
        v0 = np.zeros((p,p,m))
        v1 = np.zeros((p,p,m))
        
        for der in range(p):
            for j in range(0,p):
                for k in range(0,p-j):
                    for i in range(0,m-p+1):
                        if k+i+j >= der:
                            v0[der,j,k+i+j-der] += (-1)**i*comb(m-p+k-1,k)*comb(m-p,i)*perm(k+i+j,der)/factorial(j)

            for j in range(0,m-p):
                for k in range(0,m-p-j):
                    for i in range(0,k+j+1):
                        if p+i >= der:
                            v1[der,j,p+i-der] += (-1)**i*comb(p+k-1,k)*comb(k+j,i)*perm(p+i,der)*(-1)**j/factorial(j)
        return v0, v1
    
    def generate_single_point_matrix(self, x, der=1):
        m = self.MAX_POLY_DEG + 1
        p = self.MAX_SYS_DEG + 1
                
        basis = self.generate_basis(x, POLY_DEG=m-1, der=0)
        v0 = self.v0_mat[der,:,:].dot(basis)
        v1 = self.v1_mat[der,:,:].dot(basis)
        v0[np.where(np.abs(v0)<1e-10)] = 0
        v1[np.where(np.abs(v1)<1e-10)] = 0
        return v0, v1
      
    def generate_single_sampling_matrix(self, N=20, der=1, endpoint=False):
        m = self.MAX_POLY_DEG + 1
        p = self.MAX_SYS_DEG + 1
        
        v0 = np.zeros((N,m-p))
        v1 = np.zeros((N,m-p))
        
        if endpoint == True:
            N_max = N-1
        else:
            N_max = N
        
        for idx_v in range(N):
            x_t = 1.0*idx_v/N_max
            v0[idx_v,:], v1[idx_v,:] = self.generate_single_point_matrix(x_t, der)
        
        return v0, v1, np.block([[v0.T.dot(v0), v0.T.dot(v1)],[v1.T.dot(v0), v1.T.dot(v1)]])
    
    def generate_sampling_matrix(self, t_set, N=20, der=1, endpoint=False):
        N_POLY = t_set.shape[0]
        
#         if N == self.N_POINTS and N_POLY in self.cache_sampling_mat["list"][der]:
#             V = np.array(self.cache_sampling_mat["{}_{}".format(N_POLY, der)])
#             return V
        
        if N != self.N_POINTS or der > self.MAX_SYS_DEG:
            v0, v1 = self.generate_single_sampling_matrix(N=N, der=der, endpoint=False)
            if endpoint:
                v0_end, v1_end = self.generate_single_sampling_matrix(N=N, der=der, endpoint=True)
        else:
            v0, v1 = self.v0_sanity[der,:,:], self.v1_sanity[der,:,:]
            if endpoint:
                v0_end, v1_end = self.v0_sanity_end[der,:,:], self.v1_sanity_end[der,:,:]

        
        V = np.zeros((N*N_POLY,self.N_DER*(N_POLY+1)))
        for i in range(N_POLY):
            T2_mat = np.diag(self.generate_basis(t_set[i],self.N_DER-1,0))/(t_set[i]**der)
            if endpoint and i==(N_POLY-1):
                V[i*N:(i+1)*N,i*self.N_DER:(i+1)*self.N_DER] = v0_end.dot(T2_mat)
                V[i*N:(i+1)*N,(i+1)*self.N_DER:(i+2)*self.N_DER] = v1_end.dot(T2_mat)
            else:
                V[i*N:(i+1)*N,i*self.N_DER:(i+1)*self.N_DER] = v0.dot(T2_mat)
                V[i*N:(i+1)*N,(i+1)*self.N_DER:(i+2)*self.N_DER] = v1.dot(T2_mat)
        
#         if N == self.N_POINTS:
#             self.cache_sampling_mat["{}_{}".format(N_POLY, der)] = copy.deepcopy(V)
#             self.cache_sampling_mat["list"][der].append(N_POLY)

        return V
    
    def generate_sampling_matrix_loop(self, t_set, N=20, der=1):
        N_POLY = t_set.shape[0]-1
        
#         if N == self.N_POINTS and N_POLY in self.cache_sampling_mat_loop["list"][der]:
#             V = np.array(self.cache_sampling_mat_loop["{}_{}".format(N_POLY, der)])
#             return V
        
        if N != self.N_POINTS or der > self.MAX_SYS_DEG:
            v0, v1 = self.generate_single_sampling_matrix(N=N, der=der, endpoint=False)
        else:
            v0, v1 = self.v0_sanity[der,:,:], self.v1_sanity[der,:,:]
        
        V = np.zeros((N*(N_POLY+1),self.N_DER*(N_POLY+1)))
        for i in range(N_POLY):
            T2_mat = np.diag(self.generate_basis(t_set[i],self.N_DER-1,0))/(t_set[i]**der)
            V[i*N:(i+1)*N,i*self.N_DER:(i+1)*self.N_DER] = v0.dot(T2_mat)
            V[i*N:(i+1)*N,(i+1)*self.N_DER:(i+2)*self.N_DER] = v1.dot(T2_mat)
        
        T2_mat = np.diag(self.generate_basis(t_set[N_POLY],self.N_DER-1,0))/(t_set[N_POLY]**der)
        V[N_POLY*N:(N_POLY+1)*N,N_POLY*self.N_DER:(N_POLY+1)*self.N_DER] = v0.dot(T2_mat)
        V[N_POLY*N:(N_POLY+1)*N,:self.N_DER] = v1.dot(T2_mat)

#         if N == self.N_POINTS:
#             self.cache_sampling_mat_loop["{}_{}".format(N_POLY, der)] = copy.deepcopy(V)
#             self.cache_sampling_mat_loop["list"][der].append(N_POLY)
        
        return V
    
    def generate_single_point_matrix_yaw(self, x, der=1):
        m = self.MAX_POLY_DEG_YAW + 1
        p = self.MAX_SYS_DEG_YAW + 1
                
        basis = self.generate_basis(x, POLY_DEG=m-1, der=0)
        v0 = self.v0_mat_yaw[der,:,:].dot(basis)
        v1 = self.v1_mat_yaw[der,:,:].dot(basis)
        return v0, v1
      
    def generate_single_sampling_matrix_yaw(self, N=20, der=1, endpoint=False):
        m = self.MAX_POLY_DEG_YAW + 1
        p = self.MAX_SYS_DEG_YAW + 1
        
        v0 = np.zeros((N,m-p))
        v1 = np.zeros((N,m-p))
        
        if endpoint == True:
            N_max = N-1
        else:
            N_max = N
        
        for idx_v in range(N):
            x_t = 1.0*idx_v/N_max
            v0[idx_v,:], v1[idx_v,:] = self.generate_single_point_matrix_yaw(x_t, der)
        
        return v0, v1, np.block([[v0.T.dot(v0), v0.T.dot(v1)],[v1.T.dot(v0), v1.T.dot(v1)]])
    
    def generate_sampling_matrix_yaw(self, t_set, N=20, der=1, endpoint=False):
        N_POLY = t_set.shape[0]
        
#         if N == self.N_POINTS and N_POLY in self.cache_sampling_mat_yaw["list"][der]:
#             V = np.array(self.cache_sampling_mat_yaw["{}_{}".format(N_POLY, der)])
#             return V
        
        if N != self.N_POINTS or der > self.MAX_SYS_DEG_YAW:
            v0, v1 = self.generate_single_sampling_matrix_yaw(N=N, der=der, endpoint=False)
            if endpoint:
                v0_end, v1_end = self.generate_single_sampling_matrix_yaw(N=N, der=der, endpoint=True)
        else:
            v0, v1 = self.v0_sanity_yaw[der,:,:], self.v1_sanity_yaw[der,:,:]
            if endpoint:
                v0_end, v1_end = self.v0_sanity_yaw_end[der,:,:], self.v1_sanity_yaw_end[der,:,:]

        V = np.zeros((N*N_POLY,self.N_DER_YAW*(N_POLY+1)))
        for i in range(N_POLY):
            T2_mat = np.diag(self.generate_basis(t_set[i],self.N_DER_YAW-1,0))/(t_set[i]**der)
            if endpoint and i==(N_POLY-1):
                V[i*N:(i+1)*N,i*self.N_DER_YAW:(i+1)*self.N_DER_YAW] = v0_end.dot(T2_mat)
                V[i*N:(i+1)*N,(i+1)*self.N_DER_YAW:(i+2)*self.N_DER_YAW] = v1_end.dot(T2_mat)
            else:
                V[i*N:(i+1)*N,i*self.N_DER_YAW:(i+1)*self.N_DER_YAW] = v0.dot(T2_mat)
                V[i*N:(i+1)*N,(i+1)*self.N_DER_YAW:(i+2)*self.N_DER_YAW] = v1.dot(T2_mat)

#         if N == self.N_POINTS:
#             self.cache_sampling_mat_yaw["{}_{}".format(N_POLY, der)] = copy.deepcopy(V)
#             self.cache_sampling_mat_yaw["list"][der].append(N_POLY)

        return V
    
    def generate_sampling_matrix_loop_yaw(self, t_set, N=20, der=1):
        N_POLY = t_set.shape[0]-1
        
#         if N == self.N_POINTS and N_POLY in self.cache_sampling_mat_yaw_loop["list"][der]:
#             V = np.array(self.cache_sampling_mat_yaw_loop["{}_{}".format(N_POLY, der)])
#             return V
        
        if N != self.N_POINTS or der > self.MAX_SYS_DEG_YAW:
            v0, v1 = self.generate_single_sampling_matrix_yaw(N=N, der=der, endpoint=False)
        else:
            v0, v1 = self.v0_sanity_yaw[der,:,:], self.v1_sanity_yaw[der,:,:]
        N_POLY = t_set.shape[0]-1
        
        V = np.zeros((N*(N_POLY+1),self.N_DER_YAW*(N_POLY+1)))
        for i in range(N_POLY):
            T2_mat = np.diag(self.generate_basis(t_set[i],self.N_DER_YAW-1,0))/(t_set[i]**der)
            V[i*N:(i+1)*N,i*self.N_DER_YAW:(i+1)*self.N_DER_YAW] = v0.dot(T2_mat)
            V[i*N:(i+1)*N,(i+1)*self.N_DER_YAW:(i+2)*self.N_DER_YAW] = v1.dot(T2_mat)
        
        T2_mat = np.diag(self.generate_basis(t_set[N_POLY],self.N_DER_YAW-1,0))/(t_set[N_POLY]**der)
        V[N_POLY*N:(N_POLY+1)*N,N_POLY*self.N_DER_YAW:(N_POLY+1)*self.N_DER_YAW] = v0.dot(T2_mat)
        V[N_POLY*N:(N_POLY+1)*N,:self.N_DER_YAW] = v1.dot(T2_mat)

#         if N == self.N_POINTS:
#             self.cache_sampling_mat_yaw_loop["{}_{}".format(N_POLY, der)] = copy.deepcopy(V)
#             self.cache_sampling_mat_yaw_loop["list"][der].append(N_POLY)

        return V
    
    def generate_weight_matrix(self, t_set, N_POINTS):
#         N_POLY = t_set.shape[0]
#         tw = np.ones(N_POLY*N_POINTS)
#         for i in range(N_POLY):
#             tw[i*N_POINTS:(i+1)*N_POINTS] = t_set[i]/N_POINTS
        tw = np.repeat(t_set,N_POINTS)/N_POINTS

        return np.diag(tw)
    
    def generate_perm_matrix(self, N_POLY, N_der):
        if N_der == self.N_DER and N_POLY in self.cache_perm_mat["list"]:
            return np.array(self.cache_perm_mat["{}".format(N_POLY)])
        
        # Generate permutation matrix
        P = np.zeros(((N_POLY+1)*N_der,(N_POLY+1)*N_der), dtype='float64')
        for i in range(N_POLY+1):
            P[i,i*N_der]=1
            P[N_POLY+1+i*(N_der-1):N_POLY+1+(i+1)*(N_der-1),1+i*N_der:(i+1)*N_der] = np.diag(np.ones(N_der-1))

        if N_der == self.N_DER:
            self.cache_perm_mat["{}".format(N_POLY)] = copy.deepcopy(P)
            self.cache_perm_mat["list"].append(N_POLY)
        
        return P
    
    def get_matrix_inv(self, M):
        if np.linalg.cond(M) < 1/np.finfo(M.dtype).eps:
            M_inv = np.linalg.inv(M)
        else:
            U, s, V = np.linalg.svd(M, full_matrices=True)
            M_inv = np.matmul(V.T,np.matmul(np.diag(1/s),U.T))
        
        return M_inv
    
    def get_yaw_der(self, yaw_xy):
        yaw = np.arctan2(yaw_xy[:,0,1],yaw_xy[:,0,0])
        dyaw = -np.multiply(yaw_xy[:,0,1],yaw_xy[:,1,0]) \
                +np.multiply(yaw_xy[:,0,0],yaw_xy[:,1,1])
        
        yaw_xy_norm2 = np.linalg.norm(yaw_xy[:,0,:], axis=1)**2
        dyaw /= yaw_xy_norm2
        
        ddyaw = (-np.multiply(yaw_xy[:,0,1],yaw_xy[:,2,0]) \
                 +np.multiply(yaw_xy[:,0,0],yaw_xy[:,2,1]) \
                 -2*np.multiply(
                         np.multiply(yaw_xy[:,0,0],yaw_xy[:,1,0]) \
                         +np.multiply(yaw_xy[:,0,1],yaw_xy[:,1,1]), \
                    dyaw)) \
                 / yaw_xy_norm2
    
        yaw_ret = np.zeros((yaw.shape[0],3))
        yaw_ret[:,0] = yaw
        yaw_ret[:,1] = dyaw
        yaw_ret[:,2] = ddyaw
        
        return yaw_ret
    
    def check_flag_loop(self, t_set, d_ordered):
        if d_ordered.shape[0] == t_set.shape[0]*self.N_DER and t_set.shape[0] != 1:
            return True
        elif d_ordered.shape[0] == self.N_DER and t_set.shape[0] == 1:
            return True
        else:
            return False

    def check_flag_loop_points(self, t_set, points):
        if points.shape[0] == t_set.shape[0]:
            return True
        else:
            return False

    def der_to_poly(self, t_set, d_ordered, d_ordered_yaw=None):
        flag_loop = self.check_flag_loop(t_set, d_ordered)
        N_poly = t_set.shape[0]
        poly_coeff = []
        poly_coeff_yaw = []
        
        for i in range(N_poly):
            poly_tmp = np.zeros((self.MAX_POLY_DEG+1,d_ordered.shape[1]))
            T_array0 = np.zeros((self.MAX_SYS_DEG+1,self.MAX_POLY_DEG+1))
            T_array = np.zeros((self.MAX_SYS_DEG+1,self.MAX_POLY_DEG+1))
            for der in range(self.MAX_SYS_DEG+1):
                T_array0[der,:] = self.generate_basis(0.0,self.MAX_POLY_DEG,der)
                T_array[der,:] = self.generate_basis(t_set[i],self.MAX_POLY_DEG,der)
            if np.all(d_ordered_yaw != None):
                poly_tmp_yaw = np.zeros((self.MAX_POLY_DEG_YAW+1,d_ordered_yaw.shape[1]))
                T_array0_yaw = np.zeros((self.MAX_SYS_DEG_YAW+1,self.MAX_POLY_DEG_YAW+1))
                T_array_yaw = np.zeros((self.MAX_SYS_DEG_YAW+1,self.MAX_POLY_DEG_YAW+1))
                for der in range(self.MAX_SYS_DEG_YAW+1):
                    T_array0_yaw[der,:] = self.generate_basis(0.0,self.MAX_POLY_DEG_YAW,der)
                    T_array_yaw[der,:] = self.generate_basis(t_set[i],self.MAX_POLY_DEG_YAW,der)
            
            der0 = d_ordered[i*self.N_DER:(i+1)*self.N_DER,:]
            if flag_loop:
                poly_idx_next = (i+1)%(t_set.shape[0])
            else:
                poly_idx_next = i+1
            der1 = d_ordered[poly_idx_next*self.N_DER:(poly_idx_next+1)*self.N_DER,:]
            
            poly_tmp[:self.N_DER,:] = np.linalg.inv(T_array0[:,:self.N_DER]).dot(der0)
            poly_tmp[self.N_DER:,:] = np.linalg.inv(T_array[:,self.N_DER:]).dot(der1-T_array[:,:self.N_DER].dot(poly_tmp[:self.N_DER,:]))
            
            poly_coeff.append(poly_tmp)
        
            if np.all(d_ordered_yaw != None):
                der0_yaw = d_ordered_yaw[i*self.N_DER_YAW:(i+1)*self.N_DER_YAW,:]
                der1_yaw = d_ordered_yaw[poly_idx_next*self.N_DER_YAW:(poly_idx_next+1)*self.N_DER_YAW,:]
                poly_tmp_yaw[:self.N_DER_YAW,:] = np.linalg.inv(T_array0_yaw[:,:self.N_DER_YAW]).dot(der0_yaw)
                poly_tmp_yaw[self.N_DER_YAW:,:] = np.linalg.inv(T_array_yaw[:,self.N_DER_YAW:]).dot(der1_yaw-T_array_yaw[:,:self.N_DER_YAW].dot(poly_tmp_yaw[:self.N_DER_YAW,:]))
                poly_coeff_yaw.append(poly_tmp_yaw)
            
        poly_coeff = np.array(poly_coeff)
        poly_coeff_yaw = np.array(poly_coeff_yaw)
        
        return poly_coeff, poly_coeff_yaw

class BaseTrajFunc2(object):
    def __init__(self, *args, **kwargs):
        if "MAX_POLY_DEG" in kwargs:
            MAX_POLY_DEG = kwargs["MAX_POLY_DEG"]
        else:
            MAX_POLY_DEG = 9
        
        if "MAX_SYS_DEG" in kwargs:
            MAX_SYS_DEG = kwargs["MAX_SYS_DEG"]
        else:
            MAX_SYS_DEG = 4
        
        if "MAX_POLY_DEG_YAW" in kwargs:
            MAX_POLY_DEG_YAW = kwargs["MAX_POLY_DEG_YAW"]
        else:
            MAX_POLY_DEG_YAW = 5
        
        if "MAX_SYS_DEG_YAW" in kwargs:
            MAX_SYS_DEG_YAW = kwargs["MAX_SYS_DEG_YAW"]
        else:
            MAX_SYS_DEG_YAW = 2
        
        if "N_POINTS" in kwargs:
            N_POINTS = kwargs["N_POINTS"]
        else:
            N_POINTS = 200
                
        self.MAX_POLY_DEG = MAX_POLY_DEG
        self.MAX_SYS_DEG = MAX_SYS_DEG # SNAP
        self.MAX_POLY_DEG_YAW = MAX_POLY_DEG_YAW
        self.MAX_SYS_DEG_YAW = MAX_SYS_DEG_YAW # YawAcc
        self.N_POINTS = N_POINTS # Number of collocation points
        self.N_DER = self.MAX_POLY_DEG-self.MAX_SYS_DEG
        self.N_DER_YAW = self.MAX_POLY_DEG_YAW-self.MAX_SYS_DEG_YAW
        
        self.v0_mat, self.v1_mat = self.generate_interpolation_matrix_coeff()
        self.v0_sanity = np.zeros((MAX_SYS_DEG+1, N_POINTS, MAX_POLY_DEG-MAX_SYS_DEG))
        self.v1_sanity = np.zeros((MAX_SYS_DEG+1, N_POINTS, MAX_POLY_DEG-MAX_SYS_DEG))
        for der in range(MAX_SYS_DEG+1):
            self.v0_sanity[der,:,:], self.v1_sanity[der,:,:] = self.generate_single_sampling_matrix(N=N_POINTS, der=der, endpoint=False)
            
        self.v0_sanity_end = np.zeros((MAX_SYS_DEG+1, N_POINTS, MAX_POLY_DEG-MAX_SYS_DEG))
        self.v1_sanity_end = np.zeros((MAX_SYS_DEG+1, N_POINTS, MAX_POLY_DEG-MAX_SYS_DEG))
        for der in range(MAX_SYS_DEG+1):
            self.v0_sanity_end[der,:,:], self.v1_sanity_end[der,:,:] = self.generate_single_sampling_matrix(N=N_POINTS, der=der, endpoint=True)
        
        self.v0_mat_yaw, self.v1_mat_yaw = self.generate_interpolation_matrix_coeff(m=MAX_POLY_DEG_YAW+1,p=MAX_SYS_DEG_YAW+1)
        self.v0_sanity_yaw = np.zeros((MAX_SYS_DEG_YAW+1, N_POINTS, MAX_POLY_DEG_YAW-MAX_SYS_DEG_YAW))
        self.v1_sanity_yaw = np.zeros((MAX_SYS_DEG_YAW+1, N_POINTS, MAX_POLY_DEG_YAW-MAX_SYS_DEG_YAW))
        for der in range(MAX_SYS_DEG_YAW+1):
            self.v0_sanity_yaw[der,:,:], self.v1_sanity_yaw[der,:,:] = self.generate_single_sampling_matrix_yaw(N=N_POINTS, der=der, endpoint=False)

        self.v0_sanity_yaw_end = np.zeros((MAX_SYS_DEG_YAW+1, N_POINTS, MAX_POLY_DEG_YAW-MAX_SYS_DEG_YAW))
        self.v1_sanity_yaw_end = np.zeros((MAX_SYS_DEG_YAW+1, N_POINTS, MAX_POLY_DEG_YAW-MAX_SYS_DEG_YAW))
        for der in range(MAX_SYS_DEG_YAW+1):
            self.v0_sanity_yaw_end[der,:,:], self.v1_sanity_yaw_end[der,:,:] = self.generate_single_sampling_matrix_yaw(N=N_POINTS, der=der, endpoint=True)
        
        self.cache_perm_mat = dict()
        self.cache_perm_mat["list"] = []

        self.direct_yaw_ref = False
            
    ###############################################################################
    def generate_basis(self, s, POLY_DEG=-1, der=0):
        if POLY_DEG == -1:
            POLY_DEG = self.MAX_POLY_DEG
        basis = np.zeros(POLY_DEG+1)
        for i in range(basis.shape[0]):
            if i >= der:
                basis[i] = s**(i-der)*perm(i,der)
        return basis
    
    def generate_interpolation_matrix_coeff(self, m=-1, p=-1):
        if m == -1:
            m = self.MAX_POLY_DEG + 1
        if p == -1:
            p = self.MAX_SYS_DEG + 1
        
        v0 = np.zeros((p,p,m))
        v1 = np.zeros((p,p,m))
        
        for der in range(p):
            for j in range(0,p):
                for k in range(0,p-j):
                    for i in range(0,m-p+1):
                        if k+i+j >= der:
                            v0[der,j,k+i+j-der] += (-1)**i*comb(m-p+k-1,k)*comb(m-p,i)*perm(k+i+j,der)/factorial(j)

            for j in range(0,m-p):
                for k in range(0,m-p-j):
                    for i in range(0,k+j+1):
                        if p+i >= der:
                            v1[der,j,p+i-der] += (-1)**i*comb(p+k-1,k)*comb(k+j,i)*perm(p+i,der)*(-1)**j/factorial(j)
        return v0, v1
    
    def generate_single_point_matrix(self, x, der=1):
        m = self.MAX_POLY_DEG + 1
        p = self.MAX_SYS_DEG + 1
                
        basis = self.generate_basis(x, POLY_DEG=m-1, der=0)
        v0 = self.v0_mat[der,:,:].dot(basis)
        v1 = self.v1_mat[der,:,:].dot(basis)
        v0[np.where(np.abs(v0)<1e-10)] = 0
        v1[np.where(np.abs(v1)<1e-10)] = 0
        return v0, v1
      
    def generate_single_sampling_matrix(self, N=20, der=1, endpoint=False):
        m = self.MAX_POLY_DEG + 1
        p = self.MAX_SYS_DEG + 1
        
        v0 = np.zeros((N,m-p))
        v1 = np.zeros((N,m-p))
        
        if endpoint == True:
            N_max = N-1
        else:
            N_max = N
        
        for idx_v in range(N):
            x_t = 1.0*idx_v/N_max
            v0[idx_v,:], v1[idx_v,:] = self.generate_single_point_matrix(x_t, der)
        
        return v0, v1
    
    def generate_sampling_matrix(self, t_set, N=20, der=1, endpoint=False):
        N_POLY = t_set.shape[0]
        
        if N != self.N_POINTS or der > self.MAX_SYS_DEG:
            v0, v1 = self.generate_single_sampling_matrix(N=N, der=der, endpoint=False)
            if endpoint:
                v0_end, v1_end = self.generate_single_sampling_matrix(N=N, der=der, endpoint=True)
        else:
            v0, v1 = self.v0_sanity[der,:,:], self.v1_sanity[der,:,:]
            if endpoint:
                v0_end, v1_end = self.v0_sanity_end[der,:,:], self.v1_sanity_end[der,:,:]

        
        V = np.zeros((N*N_POLY,self.N_DER*(N_POLY+1)))
        for i in range(N_POLY):
            T2_mat = np.diag(self.generate_basis(t_set[i],self.N_DER-1,0))/(t_set[i]**der)
            if endpoint and i==(N_POLY-1):
                V[i*N:(i+1)*N,i*self.N_DER:(i+1)*self.N_DER] = v0_end.dot(T2_mat)
                V[i*N:(i+1)*N,(i+1)*self.N_DER:(i+2)*self.N_DER] = v1_end.dot(T2_mat)
            else:
                V[i*N:(i+1)*N,i*self.N_DER:(i+1)*self.N_DER] = v0.dot(T2_mat)
                V[i*N:(i+1)*N,(i+1)*self.N_DER:(i+2)*self.N_DER] = v1.dot(T2_mat)
        
        return V
    
    def generate_sampling_matrix_loop(self, t_set, N=20, der=1):
        N_POLY = t_set.shape[0]-1
        
        if N != self.N_POINTS or der > self.MAX_SYS_DEG:
            v0, v1 = self.generate_single_sampling_matrix(N=N, der=der, endpoint=False)
        else:
            v0, v1 = self.v0_sanity[der,:,:], self.v1_sanity[der,:,:]
        
        V = np.zeros((N*(N_POLY+1),self.N_DER*(N_POLY+1)))
        for i in range(N_POLY):
            T2_mat = np.diag(self.generate_basis(t_set[i],self.N_DER-1,0))/(t_set[i]**der)
            V[i*N:(i+1)*N,i*self.N_DER:(i+1)*self.N_DER] = v0.dot(T2_mat)
            V[i*N:(i+1)*N,(i+1)*self.N_DER:(i+2)*self.N_DER] = v1.dot(T2_mat)
        
        T2_mat = np.diag(self.generate_basis(t_set[N_POLY],self.N_DER-1,0))/(t_set[N_POLY]**der)
        V[N_POLY*N:(N_POLY+1)*N,N_POLY*self.N_DER:(N_POLY+1)*self.N_DER] = v0.dot(T2_mat)
        V[N_POLY*N:(N_POLY+1)*N,:self.N_DER] = v1.dot(T2_mat)
        
        return V
    
    def generate_single_point_matrix_yaw(self, x, der=1):
        m = self.MAX_POLY_DEG_YAW + 1
        p = self.MAX_SYS_DEG_YAW + 1
                
        basis = self.generate_basis(x, POLY_DEG=m-1, der=0)
        v0 = self.v0_mat_yaw[der,:,:].dot(basis)
        v1 = self.v1_mat_yaw[der,:,:].dot(basis)
        return v0, v1
      
    def generate_single_sampling_matrix_yaw(self, N=20, der=1, endpoint=False):
        m = self.MAX_POLY_DEG_YAW + 1
        p = self.MAX_SYS_DEG_YAW + 1
        
        v0 = np.zeros((N,m-p))
        v1 = np.zeros((N,m-p))
        
        if endpoint == True:
            N_max = N-1
        else:
            N_max = N
        
        for idx_v in range(N):
            x_t = 1.0*idx_v/N_max
            v0[idx_v,:], v1[idx_v,:] = self.generate_single_point_matrix_yaw(x_t, der)
        
        return v0, v1
    
    def generate_sampling_matrix_yaw(self, t_set, N=20, der=1, endpoint=False):
        N_POLY = t_set.shape[0]
        
        if N != self.N_POINTS or der > self.MAX_SYS_DEG_YAW:
            v0, v1 = self.generate_single_sampling_matrix_yaw(N=N, der=der, endpoint=False)
            if endpoint:
                v0_end, v1_end = self.generate_single_sampling_matrix_yaw(N=N, der=der, endpoint=True)
        else:
            v0, v1 = self.v0_sanity_yaw[der,:,:], self.v1_sanity_yaw[der,:,:]
            if endpoint:
                v0_end, v1_end = self.v0_sanity_yaw_end[der,:,:], self.v1_sanity_yaw_end[der,:,:]

        V = np.zeros((N*N_POLY,self.N_DER_YAW*(N_POLY+1)))
        for i in range(N_POLY):
            T2_mat = np.diag(self.generate_basis(t_set[i],self.N_DER_YAW-1,0))/(t_set[i]**der)
            if endpoint and i==(N_POLY-1):
                V[i*N:(i+1)*N,i*self.N_DER_YAW:(i+1)*self.N_DER_YAW] = v0_end.dot(T2_mat)
                V[i*N:(i+1)*N,(i+1)*self.N_DER_YAW:(i+2)*self.N_DER_YAW] = v1_end.dot(T2_mat)
            else:
                V[i*N:(i+1)*N,i*self.N_DER_YAW:(i+1)*self.N_DER_YAW] = v0.dot(T2_mat)
                V[i*N:(i+1)*N,(i+1)*self.N_DER_YAW:(i+2)*self.N_DER_YAW] = v1.dot(T2_mat)

        return V
    
    def generate_sampling_matrix_loop_yaw(self, t_set, N=20, der=1):
        N_POLY = t_set.shape[0]-1
        
        if N != self.N_POINTS or der > self.MAX_SYS_DEG_YAW:
            v0, v1 = self.generate_single_sampling_matrix_yaw(N=N, der=der, endpoint=False)
        else:
            v0, v1 = self.v0_sanity_yaw[der,:,:], self.v1_sanity_yaw[der,:,:]
        N_POLY = t_set.shape[0]-1
        
        V = np.zeros((N*(N_POLY+1),self.N_DER_YAW*(N_POLY+1)))
        for i in range(N_POLY):
            T2_mat = np.diag(self.generate_basis(t_set[i],self.N_DER_YAW-1,0))/(t_set[i]**der)
            V[i*N:(i+1)*N,i*self.N_DER_YAW:(i+1)*self.N_DER_YAW] = v0.dot(T2_mat)
            V[i*N:(i+1)*N,(i+1)*self.N_DER_YAW:(i+2)*self.N_DER_YAW] = v1.dot(T2_mat)
        
        T2_mat = np.diag(self.generate_basis(t_set[N_POLY],self.N_DER_YAW-1,0))/(t_set[N_POLY]**der)
        V[N_POLY*N:(N_POLY+1)*N,N_POLY*self.N_DER_YAW:(N_POLY+1)*self.N_DER_YAW] = v0.dot(T2_mat)
        V[N_POLY*N:(N_POLY+1)*N,:self.N_DER_YAW] = v1.dot(T2_mat)
        
        return V
    
    def generate_weight_matrix(self, t_set, N_POINTS):
        tw = np.repeat(t_set,N_POINTS)/N_POINTS
        return np.diag(tw)
    
    def generate_perm_matrix(self, N_POLY, N_der):
        if N_der == self.N_DER and N_POLY in self.cache_perm_mat["list"]:
            return np.array(self.cache_perm_mat["{}".format(N_POLY)])
        
        # Generate permutation matrix
        P = np.zeros(((N_POLY+1)*N_der,(N_POLY+1)*N_der), dtype='float64')
        for i in range(N_POLY+1):
            P[i,i*N_der]=1
            P[N_POLY+1+i*(N_der-1):N_POLY+1+(i+1)*(N_der-1),1+i*N_der:(i+1)*N_der] = np.diag(np.ones(N_der-1))

        if N_der == self.N_DER:
            self.cache_perm_mat["{}".format(N_POLY)] = copy.deepcopy(P)
            self.cache_perm_mat["list"].append(N_POLY)
        
        return P
    
    def get_matrix_inv(self, M):
        if np.linalg.cond(M) < 1/np.finfo(M.dtype).eps:
            M_inv = np.linalg.inv(M)
        else:
            U, s, V = np.linalg.svd(M, full_matrices=True)
            M_inv = np.matmul(V.T,np.matmul(np.diag(1/s),U.T))
        
        return M_inv
    
    def get_yaw_der(self, yaw_xy):

        if self.direct_yaw_ref:
            yaw = yaw_xy[:,0,0]
            dyaw = yaw_xy[:,1,0]
            ddyaw = yaw_xy[:,2,0]
        else:
            yaw = np.arctan2(yaw_xy[:,0,1],yaw_xy[:,0,0])
            dyaw = -np.multiply(yaw_xy[:,0,1],yaw_xy[:,1,0]) \
                    +np.multiply(yaw_xy[:,0,0],yaw_xy[:,1,1])
        
            yaw_xy_norm2 = np.linalg.norm(yaw_xy[:,0,:], axis=1)**2
            dyaw /= yaw_xy_norm2
        
            ddyaw = (-np.multiply(yaw_xy[:,0,1],yaw_xy[:,2,0]) \
                     +np.multiply(yaw_xy[:,0,0],yaw_xy[:,2,1]) \
                     -2*np.multiply(
                             np.multiply(yaw_xy[:,0,0],yaw_xy[:,1,0]) \
                             +np.multiply(yaw_xy[:,0,1],yaw_xy[:,1,1]), \
                        dyaw)) \
                     / yaw_xy_norm2

        yaw_ret = np.zeros((yaw.shape[0],3))
        yaw_ret[:,0] = yaw
        yaw_ret[:,1] = dyaw
        yaw_ret[:,2] = ddyaw
        
        # print('yaw_ret: {}'.format(yaw_ret))

        return yaw_ret
    
    def check_flag_loop(self, t_set, d_ordered):
        if d_ordered.shape[0] == t_set.shape[0]*self.N_DER and t_set.shape[0] != 1:
            return True
        elif d_ordered.shape[0] == self.N_DER and t_set.shape[0] == 1:
            return True
        else:
            return False

    def check_flag_loop_points(self, t_set, points):
        if points.shape[0] == t_set.shape[0]:
            return True
        else:
            return False
    
    def der_to_poly(self, t_set, d_ordered, d_ordered_yaw=None):
        flag_loop = self.check_flag_loop(t_set, d_ordered)
        N_poly = t_set.shape[0]
        poly_coeff = []
        poly_coeff_yaw = []
        
        for i in range(N_poly):
            poly_tmp = np.zeros((self.MAX_POLY_DEG+1,d_ordered.shape[1]))
            T_array0 = np.zeros((self.MAX_SYS_DEG+1,self.MAX_POLY_DEG+1))
            T_array = np.zeros((self.MAX_SYS_DEG+1,self.MAX_POLY_DEG+1))
            for der in range(self.MAX_SYS_DEG+1):
                T_array0[der,:] = self.generate_basis(0.0,self.MAX_POLY_DEG,der)
                T_array[der,:] = self.generate_basis(t_set[i],self.MAX_POLY_DEG,der)
            if np.all(d_ordered_yaw != None):
                poly_tmp_yaw = np.zeros((self.MAX_POLY_DEG_YAW+1,d_ordered_yaw.shape[1]))
                T_array0_yaw = np.zeros((self.MAX_SYS_DEG_YAW+1,self.MAX_POLY_DEG_YAW+1))
                T_array_yaw = np.zeros((self.MAX_SYS_DEG_YAW+1,self.MAX_POLY_DEG_YAW+1))
                for der in range(self.MAX_SYS_DEG_YAW+1):
                    T_array0_yaw[der,:] = self.generate_basis(0.0,self.MAX_POLY_DEG_YAW,der)
                    T_array_yaw[der,:] = self.generate_basis(t_set[i],self.MAX_POLY_DEG_YAW,der)
            
            der0 = d_ordered[i*self.N_DER:(i+1)*self.N_DER,:]
            if flag_loop:
                poly_idx_next = (i+1)%(t_set.shape[0])
            else:
                poly_idx_next = i+1
            der1 = d_ordered[poly_idx_next*self.N_DER:(poly_idx_next+1)*self.N_DER,:]
            
            poly_tmp[:self.N_DER,:] = np.linalg.inv(T_array0[:,:self.N_DER]).dot(der0)
            poly_tmp[self.N_DER:,:] = np.linalg.inv(T_array[:,self.N_DER:]).dot(der1-T_array[:,:self.N_DER].dot(poly_tmp[:self.N_DER,:]))
            
            poly_coeff.append(poly_tmp)
        
            if np.all(d_ordered_yaw != None):
                der0_yaw = d_ordered_yaw[i*self.N_DER_YAW:(i+1)*self.N_DER_YAW,:]
                der1_yaw = d_ordered_yaw[poly_idx_next*self.N_DER_YAW:(poly_idx_next+1)*self.N_DER_YAW,:]
                poly_tmp_yaw[:self.N_DER_YAW,:] = np.linalg.inv(T_array0_yaw[:,:self.N_DER_YAW]).dot(der0_yaw)
                poly_tmp_yaw[self.N_DER_YAW:,:] = np.linalg.inv(T_array_yaw[:,self.N_DER_YAW:]).dot(der1_yaw-T_array_yaw[:,:self.N_DER_YAW].dot(poly_tmp_yaw[:self.N_DER_YAW,:]))
                poly_coeff_yaw.append(poly_tmp_yaw)
            
        poly_coeff = np.array(poly_coeff)
        poly_coeff_yaw = np.array(poly_coeff_yaw)
        
        return poly_coeff, poly_coeff_yaw

class TrajectoryTools(BaseTrajFunc):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_trajectory_der(self, t_set, d_ordered, d_ordered_yaw=None, freq=200):
        flag_loop = self.check_flag_loop(t_set,d_ordered)
        dt = 1./freq
        total_time = np.sum(t_set)
        cum_time = np.zeros(t_set.shape[0]+1)
        cum_time[1:] = np.cumsum(t_set)
        cum_time[0] = 0
        
        N = int(np.floor(total_time/dt))
        poly_idx = 0
        
        t_array = total_time*np.array(range(N+1))/N
        status = np.zeros((N+1,20))
        status[:,0] = t_array
        status[:,1] = 1
        
        T2_mat = np.diag(self.generate_basis(t_set[poly_idx],self.N_DER-1,0))
        der0 = T2_mat.dot(d_ordered[poly_idx*self.N_DER:(poly_idx+1)*self.N_DER,:])
        der1 = T2_mat.dot(d_ordered[(poly_idx+1)*self.N_DER:(poly_idx+2)*self.N_DER,:])
                        
        if np.all(d_ordered_yaw != None):
            T2_mat_yaw = np.diag(self.generate_basis(t_set[poly_idx],self.N_DER_YAW-1,0))
            der0_yaw = T2_mat_yaw.dot(d_ordered_yaw[poly_idx*self.N_DER_YAW:(poly_idx+1)*self.N_DER_YAW,:])
            der1_yaw = T2_mat_yaw.dot(d_ordered_yaw[(poly_idx+1)*self.N_DER_YAW:(poly_idx+2)*self.N_DER_YAW,:])
        
        for i in range(N+1):
            if t_array[i] > cum_time[poly_idx+1] + 1e-6:
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
    
    ###############################################################################
    def save_trajectory_csv(self, t_set, d_ordered, d_ordered_yaw=None, \
                            traj_dir='./trajectory', traj_name="test", freq=200):
        flag_loop = self.check_flag_loop(t_set,d_ordered)
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
        if not os.path.exists(traj_dir):
            os.makedirs(traj_dir)
        csvFile = os.path.join(traj_dir,"{}.csv".format(traj_name))
        pd.DataFrame(status).to_csv(csvFile, index=False)
        
        return
    
    def save_pose_csv(self, drone_model, t_set, d_ordered, d_ordered_yaw=None, \
            traj_dir='./trajectory', traj_name="test", freq=200, flag_quad=False):
        if not flag_quad:
            status = self.get_trajectory_der(t_set, d_ordered, d_ordered_yaw=d_ordered_yaw, freq=freq)
            status = np.delete(status, 1, 1)

            u, state = drone_model.getWs_vectorcalc(status[:,1:])
            pose = np.concatenate((status[:,0:1], state[:,0:3], state[:,6:10], u[:,2:4]),axis=1)
        else:
            status = self.get_trajectory_der(t_set, d_ordered, d_ordered_yaw=d_ordered_yaw, freq=freq)
            u, state =drone_model.getWs_vector(status)
            att = np.concatenate((status[:,2:5],state["roll"], state["pitch"], status[:,17:18]),axis=1)
            quat = np.zeros((att.shape[0], 4))
            for i in range(att.shape[1]):
                quat[i,:] = Euler2quat(att[i,:])
            pose = np.concatenate((status[:,0:1], status[:,2:5], quat, u[:,2:4]),axis=1)

        if not os.path.exists(traj_dir):
            os.makedirs(traj_dir)

        csvFile = os.path.join(traj_dir,"{}.csv".format(traj_name))
        pd.DataFrame(pose).to_csv(csvFile, index=False, header=False)
        
        return
    
    def save_trajectory_hdf5(self, t_set, d_ordered, d_ordered_yaw=None, \
                             traj_dir='./trajectory', traj_name="test"):
      
        if not os.path.exists(traj_dir):
            os.makedirs(traj_dir)
        hdf5File = os.path.join(traj_dir,'{}.h5'.format(traj_name))
        h5f = h5py.File(hdf5File, 'w')
        h5f.create_dataset('t_set', data=t_set)
        h5f.create_dataset('d_ordered', data=d_ordered)
        if np.all(d_ordered_yaw != None):
            h5f.create_dataset('d_ordered_yaw', data=d_ordered_yaw)
        h5f.close()
        return
    
    def save_trajectory_yaml(self, t_set, d_ordered, d_ordered_yaw=None, \
                         traj_dir='./trajectory', traj_name="test"):
        
        poly_coeff, poly_coeff_yaw = self.der_to_poly(t_set, d_ordered, d_ordered_yaw)
        
#         print("\nContinuity checking")
#         print(poly_coeff.shape)
        for poly_ii in range(poly_coeff.shape[0]-1):
            for der_ii in range(5):
                basis_curr = self.generate_basis(t_set[poly_ii],self.MAX_POLY_DEG, der_ii)
                basis_next = self.generate_basis(0.0,self.MAX_POLY_DEG, der_ii)
                val_curr = basis_curr.dot(poly_coeff[poly_ii, :, :])
                val_next = basis_next.dot(poly_coeff[poly_ii+1, :, :])
                if np.any((np.abs(val_curr - val_next) > 1e-3)):
                    print("[ERROR] poly_ii: {}, der_ii: {}".format(poly_ii, der_ii))
                    print(val_curr)
                    print(val_next)

        if not os.path.exists(traj_dir):
            os.makedirs(traj_dir)
        dim_prefix = ["x","y","z","yaw_c","yaw_s"]
        yamlFile = os.path.join(traj_dir,'{}.yaml'.format(traj_name))
        yaml_out = open(yamlFile,"w")
        yaml_out.write("coeff:\n")
        for dim_ii in range(poly_coeff.shape[2]):
            yaml_out.write("  {}:\n".format(dim_prefix[dim_ii]))
            for poly_idx in range(poly_coeff.shape[0]):
                yaml_out.write("    - [{}]\n".format(','.join([str(x) for x in poly_coeff[poly_idx,:,dim_ii]])))
            yaml_out.write("\n")
        
        yaw = np.zeros((self.MAX_POLY_DEG_YAW+1,2))
        yaw[0,0] = 1.0
        for dim_ii in range(3,5):
            yaml_out.write("  {}:\n".format(dim_prefix[dim_ii]))
            
            for poly_idx in range(poly_coeff.shape[0]):
                if np.all(d_ordered_yaw != None):
                    yaw = poly_coeff_yaw[poly_idx,:,:]
                yaml_out.write("    - [{}]\n".format(','.join([str(x) for x in yaw[:,dim_ii-3]])))
            yaml_out.write("\n")
        yaml_out.write("dt: [{}]\n".format(','.join([str(x) for x in t_set])))
        yaml_out.close()
        return
    
    ###############################################################################
    def plot_trajectory(self, \
        t_set, d_ordered, d_ordered_yaw=None, \
        flag_save=False, save_dir='', save_idx='test'):
        
        d_ordered_t = ned2enu(d_ordered)
        if np.all(d_ordered_yaw != None):
            d_ordered_yaw_t = ne2en(d_ordered_yaw)
        flag_loop = self.check_flag_loop(t_set,d_ordered_t)
        N_POLY = t_set.shape[0]
        
        if flag_loop:
            V_t = self.generate_sampling_matrix_loop(t_set, N=self.N_POINTS, der=0)
        else:
            V_t = self.generate_sampling_matrix(t_set, N=self.N_POINTS, der=0, endpoint=True)
        val = V_t.dot(d_ordered_t)
        val_mean = (np.max(val, axis=0)+np.min(val, axis=0))/2
        val -= val_mean
        
        if np.all(d_ordered_yaw != None):
            if flag_loop:
                V_t_yaw = self.generate_sampling_matrix_loop_yaw(t_set, N=self.N_POINTS, der=0)
            else:
                V_t_yaw = self.generate_sampling_matrix_yaw(t_set, N=self.N_POINTS, der=0, endpoint=True)
            val_yaw_xy = V_t_yaw.dot(d_ordered_yaw_t)
            val_yaw = np.arctan2(val_yaw_xy[:,1],val_yaw_xy[:,0])
        
        plot_lim = np.max(np.abs(val))*1.2
        fig = plt.figure(figsize=(20,20))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(d_ordered_t[::(self.MAX_SYS_DEG+1),0]-val_mean[0], \
                d_ordered_t[::(self.MAX_SYS_DEG+1),1]-val_mean[1], \
                d_ordered_t[::(self.MAX_SYS_DEG+1),2]-val_mean[2], \
                label='waypoints', color="steelblue", zorder=2)
        ax.plot(val[:,0], val[:,1], val[:,2], label='trajectory', color="orange")
        if np.all(d_ordered_yaw != None):
            q_ratio = 10
            q_skip = 0
            q = ax.quiver(val[q_skip::q_ratio,0], val[q_skip::q_ratio,1], val[q_skip::q_ratio,2], \
                  np.cos(val_yaw)[q_skip::q_ratio], np.sin(val_yaw)[q_skip::q_ratio], \
                  np.zeros(val[q_skip::q_ratio,0].shape[0]), \
                  length=plot_lim*0.02, arrow_length_ratio=0.2, color="darkorange")
        
        max_lim_x = max(max(np.max(val[:,0]),-np.min(val[:,0]))*1.15,1.0)
        max_lim_y = max(max(np.max(val[:,1]),-np.min(val[:,1]))*1.15,1.0)
        max_lim_z = max(max(np.max(val[:,2]),-np.min(val[:,2]))*1.15,1.0)
        ax.set_xlim(-max_lim_x, max_lim_x)
        ax.set_ylim(-max_lim_y, max_lim_y)
        ax.set_zlim(-max_lim_z, max_lim_z)
        ax.legend()
        if flag_save:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig('{}/{}_trajectory_minsnap.png'.format(save_dir,save_idx))
            plt.close()

        t_array = np.zeros(self.N_POINTS*N_POLY)
        for i in range(N_POLY):
            for j in range(self.N_POINTS):
                t_array[i*self.N_POINTS+j] = t_array[i*self.N_POINTS+j-1] + t_set[i]/self.N_POINTS
        
        for der in range(1,5):
            if flag_loop:
                V_t = self.generate_sampling_matrix_loop(t_set, N=self.N_POINTS, der=der)
            else:
                V_t = self.generate_sampling_matrix(t_set, N=self.N_POINTS, der=der, endpoint=True)
            
            val = V_t.dot(d_ordered_t)
            val_norm = np.linalg.norm(val, axis=1, ord=2)
            end_points = np.zeros(N_POLY+1)
            for i in range(N_POLY):
                end_points[i] = val_norm[i*self.N_POINTS]
            end_points[-1] = val_norm[-1]
            
            fig = plt.figure(figsize=(7,7))
            ax = fig.add_subplot(111)
            ax.plot(t_array, val_norm, '-', label='der {} profile'.format(der))
            ax.legend()
            ax.scatter(np.append(np.array([0]),np.cumsum(t_set)), end_points, c='red')
            if flag_save:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                plt.savefig('{}/{}_der_{}.png'.format(save_dir,save_idx,der))
                plt.close()
        
        if np.all(d_ordered_yaw != None):
            end_points = np.zeros(N_POLY+1)
            for i in range(N_POLY):
                end_points[i] = val_yaw[i*self.N_POINTS]
            end_points[-1] = val_yaw[-1]
            
            fig = plt.figure(figsize=(7,7))
            ax = fig.add_subplot(111)
            ax.plot(t_array, val_yaw, '-', label='yaw profile')
            ax.legend()
            ax.scatter(np.append(np.array([0]),np.cumsum(t_set)), end_points, c='red')
            if flag_save:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                plt.savefig('{}/{}_yaw.png'.format(save_dir,save_idx,der))
                plt.close()
            
            for der in range(1,3):
                if flag_loop:
                    V_t = self.generate_sampling_matrix_loop_yaw(t_set, N=self.N_POINTS, der=der)
                else:
                    V_t = self.generate_sampling_matrix_yaw(t_set, N=self.N_POINTS, der=der, endpoint=True)
                val = V_t.dot(d_ordered_yaw_t)

                end_points = np.zeros(N_POLY+1)
                for i in range(N_POLY):
                    end_points[i] = val[i*self.N_POINTS,0]
                end_points[-1] = val[-1,0]
                    
                fig = plt.figure(figsize=(7,7))
                ax = fig.add_subplot(111)
                ax.plot(t_array, val[:,0], '-', label='cos_yaw der {} profile'.format(der))
                ax.plot(t_array, val[:,1], '-', label='sin_yaw der {} profile'.format(der))
                ax.legend()
                ax.scatter(np.append(np.array([0]),np.cumsum(t_set)), end_points, c='red')
                if flag_save:
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    plt.savefig('{}/{}_yaw_der_{}.png'.format(save_dir,save_idx,der))
                    plt.close()
        if not flag_save:
            plt.show()
        
        return
    
    ###############################################################################
    def _get_sample_pos_data(self, t_set, d_ordered, d_ordered_yaw=None, direct_yaw=False):
        flag_loop = self.check_flag_loop(t_set,d_ordered)
        N_POLY = t_set.shape[0]
        if flag_loop:
            V_t = self.generate_sampling_matrix_loop(t_set, N=self.N_POINTS, der=0)
            if np.all(d_ordered_yaw != None):
                V_t_yaw = self.generate_sampling_matrix_loop_yaw(t_set, N=self.N_POINTS, der=0)
        else:
            V_t = self.generate_sampling_matrix(t_set, N=self.N_POINTS, der=0, endpoint=True)
            if np.all(d_ordered_yaw != None):
                V_t_yaw = self.generate_sampling_matrix_yaw(t_set, N=self.N_POINTS, der=0)
        val = V_t.dot(d_ordered)
        if np.all(d_ordered_yaw != None):
            val_yaw_xy = V_t_yaw.dot(d_ordered_yaw)
            if not direct_yaw:
                val_yaw = np.arctan2(val_yaw_xy[:,1],val_yaw_xy[:,0])
            else:
                val_yaw = val_yaw_xy[:,0]
            return val, val_yaw
        return val, None
        
    def plot_trajectory_2D_single(self, ax, t_set, d_ordered, d_ordered_yaw=None, direct_yaw=False):
        d_ordered_t = ned2enu(d_ordered)
        if np.all(d_ordered_yaw != None):
            if not direct_yaw:
                d_ordered_yaw_t = ne2en(d_ordered_yaw)
            else:
                d_ordered_yaw_t = -d_ordered_yaw
                n_yaw = int(d_ordered_yaw.shape[0]/self.N_DER_YAW)
                for i in range(n_yaw):
                    d_ordered_yaw_t[self.N_DER_YAW*i,:] += np.pi/2.
        flag_loop = self.check_flag_loop(t_set, d_ordered_t)
        N_POLY = t_set.shape[0]
        val, val_yaw = self._get_sample_pos_data(t_set, d_ordered_t, d_ordered_yaw_t, direct_yaw)
        val_mean = (np.max(val, axis=0)+np.min(val, axis=0))/2
        val -= val_mean
        
        max_lim_x = max(np.max(val[:,0]),-np.min(val[:,0]))*1.15
        max_lim_y = max(np.max(val[:,1]),-np.min(val[:,1]))*1.15
        ax.set_xlim(-max_lim_x, max_lim_x)
        ax.set_ylim(-max_lim_y, max_lim_y)
        
        x_tick = max(np.floor(max_lim_x/3.), 1.0)
        y_tick = max(np.floor(max_lim_y/3.), 1.0)
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(x_tick))
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(y_tick))
        ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.0f'))
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.0f'))

        if flag_loop:
            end_points = np.zeros((N_POLY, d_ordered_t.shape[1]))
            end_points_yaw = np.zeros((N_POLY, 2))
            for i in range(N_POLY):
                end_points[i] = val[i*self.N_POINTS,:]
                if np.all(d_ordered_yaw != None):
                    end_points_yaw[i,0] = np.cos(val_yaw[i*self.N_POINTS])
                    end_points_yaw[i,1] = np.sin(val_yaw[i*self.N_POINTS])
        else:
            end_points = np.zeros((N_POLY+1, d_ordered_t.shape[1]))
            end_points_yaw = np.zeros((N_POLY+1, 2))
            for i in range(N_POLY):
                end_points[i] = val[i*self.N_POINTS,:]
                if np.all(d_ordered_yaw != None):
                    end_points_yaw[i,0] = np.cos(val_yaw[i*self.N_POINTS])
                    end_points_yaw[i,1] = np.sin(val_yaw[i*self.N_POINTS])
            end_points[-1] = val[-1,:]
            if np.all(d_ordered_yaw != None):
                end_points_yaw[-1,0] = np.cos(val_yaw[-1])
                end_points_yaw[-1,1] = np.sin(val_yaw[-1])
        
        if (np.max(val[:,2])-np.min(val[:,2])) > 1e-2:
            max_alt = max(np.max(val[:,1]),-np.min(val[:,1]))*1.05
            z_norm = (val[:,2])/max_alt + 0.5
        else:
            max_alt = 0.5
            z_norm = np.ones(val.shape[0])/2.0
        
        CMAP='viridis'
        _cmap = plt.get_cmap(CMAP)
        ax.set_prop_cycle(color=[_cmap(z_norm[i]) for i in range(z_norm.shape[0])])
        cbar = plt.colorbar(cm.ScalarMappable(norm=mpl.colors.Normalize(-max_alt,max_alt), cmap=CMAP), \
                            ax=ax, use_gridspec=True)
        cbar.ax.set_ylabel("Altitude [m]", rotation=90, va="bottom", labelpad=10, fontsize='12')
        if max_alt == 0.5:
            cbar.ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
        else:
            z_tick = max(np.floor(max_alt)/2., 0.5)
            cbar.ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(z_tick))
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        cbar.ax.tick_params(labelsize='12')
        
        for i in range(z_norm.shape[0]-1):
            ax.plot(val[i:(i+2),0], val[i:(i+2),1], linewidth=4, zorder=1)
        
        if np.all(d_ordered_yaw != None):
            ax.quiver(end_points[1:,0], end_points[1:,1], \
                      end_points_yaw[1:,0], end_points_yaw[1:,1], \
                      color="red", zorder=2, units='height', \
                      scale_units='height', scale=14., \
                      headwidth=6.0, headlength=8.0, headaxislength=8.0)
            ax.quiver(end_points[0:1,0], end_points[0:1,1], \
                      end_points_yaw[0:1,0], end_points_yaw[0:1,1], \
                      color="green", zorder=2, units='height', \
                      scale_units='height', scale=14., \
                      headwidth=6.0, headlength=8.0, headaxislength=8.0)
        else:
            ax.scatter(end_points[1:,0], end_points[1:,1], c='red', zorder=2)
            ax.scatter(end_points[0,0], end_points[0,1], c='green', zorder=2)
        
        ax.grid()
        return
        
    def get_max_speed(self, t_set, d_ordered, flag_print=False):
        flag_loop = self.check_flag_loop(t_set,d_ordered)
        if flag_loop:
            V_t = self.generate_sampling_matrix_loop(t_set, N=self.N_POINTS, der=1)
        else:
            V_t = self.generate_sampling_matrix(t_set, N=self.N_POINTS, der=1, endpoint=True)
        val = V_t.dot(d_ordered)
        val_norm = np.linalg.norm(val, axis=1, ord=2)
        max_speed = np.max(val_norm)
        if flag_print:
            prRed("max_speed = {} m/s".format(max_speed))
        return max_speed
    
    ###############################################################################
    def plot_trajectory_animation(self, \
        t_set, d_ordered, d_ordered_yaw=None, \
        flag_save=False, save_dir='', save_file='test'):
        
        d_ordered_t = ned2enu(d_ordered)
        if np.all(d_ordered_yaw != None):
            d_ordered_yaw_t = ne2en(d_ordered_yaw)
        flag_loop = self.check_flag_loop(t_set, d_ordered_t)
        N_POLY = t_set.shape[0]
        val, val_yaw = self._get_sample_pos_data(t_set, d_ordered_t, d_ordered_yaw_t)
        val_mean = (np.max(val, axis=0)+np.min(val, axis=0))/2
        val -= val_mean
        
        if flag_loop:
            end_points = np.zeros((N_POLY, d_ordered.shape[1]))
            for i in range(N_POLY):
                end_points[i] = val[i*self.N_POINTS,:]
            end_points_yaw = np.zeros((N_POLY, 2))
            if np.all(d_ordered_yaw != None):
                for i in range(N_POLY):
                    end_points_yaw[i,0] = np.cos(val_yaw[i*self.N_POINTS])
                    end_points_yaw[i,1] = np.sin(val_yaw[i*self.N_POINTS])
        else:
            end_points = np.zeros((N_POLY+1, d_ordered.shape[1]))
            for i in range(N_POLY):
                end_points[i] = val[i*self.N_POINTS,:]
            end_points[-1] = val[-1,:]
            end_points_yaw = np.zeros((N_POLY+1, 2))
            if np.all(d_ordered_yaw != None):
                for i in range(N_POLY):
                    end_points_yaw[i,0] = np.cos(val_yaw[i*self.N_POINTS])
                    end_points_yaw[i,1] = np.sin(val_yaw[i*self.N_POINTS])
                end_points_yaw[-1,0] = np.cos(val_yaw[-1])
                end_points_yaw[-1,1] = np.sin(val_yaw[-1])
        
        # Create a figure and a 3D Axes
        fig = plt.figure(figsize=(10,10))
        ax = Axes3D(fig)

        def init():
            ax.plot(val[:,0], val[:,1], val[:,2], color="midnightblue", linewidth=4)
            
            if np.all(d_ordered_yaw != None):
                ax.quiver(end_points[1:,0], end_points[1:,1], end_points[1:,2], \
                          end_points_yaw[1:,0], end_points_yaw[1:,1], np.zeros(int(end_points[1:,:].shape[0])), \
                          color="red", zorder=2, length=0.4, arrow_length_ratio=0.4)
                ax.quiver(end_points[0:1,0], end_points[0:1,1], end_points[0:1,2], \
                          end_points_yaw[0:1,0], end_points_yaw[0:1,1], np.zeros(int(end_points[0:1,:].shape[0])), \
                          color="green", zorder=2, length=0.4, arrow_length_ratio=0.4)
            else:
                ax.scatter(end_points[1:,0], end_points[1:,1], end_points[1:,2], \
                           c='red', zorder=2)
                ax.scatter(end_points[0:1,0], end_points[0:1,1], end_points[0:1,2], \
                           c='green', zorder=2)
            
            max_lim_x = max(max(np.max(val[:,0]),-np.min(val[:,0]))*1.15,1.0)
            max_lim_y = max(max(np.max(val[:,1]),-np.min(val[:,1]))*1.15,1.0)
            max_lim_z = max(max(np.max(val[:,2]),-np.min(val[:,2]))*1.15,1.0)
            ax.set_xlim(-max_lim_x, max_lim_x)
            ax.set_ylim(-max_lim_y, max_lim_y)
            ax.set_zlim(-max_lim_z, max_lim_z)
            
            ax.tick_params(axis='both', which='major', labelsize='18')
            ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(2.0))
            ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(2.0))
            ax.zaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))

            return fig,

        def animate(i):
            ax.view_init(elev=30., azim=i)
            return fig,

        # Animate
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=1440, interval=20, blit=True)
        # Save
        if flag_save:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            anim.save('{}/{}.mp4'.format(save_dir,save_file), fps=30, extra_args=['-vcodec', 'libx264'])
        
