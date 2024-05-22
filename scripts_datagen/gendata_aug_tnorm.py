#!/usr/bin/env python
# coding: utf-8

import os, sys, io, random
import json, argparse
import time, yaml, copy
import numpy as np
from multiprocessing import cpu_count
from collections import OrderedDict, defaultdict

sys.path.insert(0, '../')
from pyTrajectoryUtils.pyTrajectoryUtils.utils import *
from pyTrajectoryUtils.pyTrajectoryUtils.minSnapTrajectory import *

poly = MinSnapTrajectory(drone_model="STMCFB", N_POINTS=40)

def flip_aug(x, y, yaw, mode=0):
    if mode == 0:
        return x, y, yaw
    elif mode == 1:
        return -y, x, yaw + np.pi/2.
    elif mode == 2:
        return -x, -y, yaw + np.pi
    elif mode == 3:
        return y, -x, yaw - np.pi/2.
    elif mode == 4:
        return y, x, -yaw + np.pi/2.
    elif mode == 5:
        return -x, y, -yaw + np.pi
    elif mode == 6:
        return -y, -x, -yaw - np.pi/2.
    elif mode == 7:
        return x, -y, -yaw
    return

def flip_aug_der(x, y, yaw, mode=0):
    if mode == 0:
        return x, y, yaw
    elif mode == 1:
        return -y, x, yaw
    elif mode == 2:
        return -x, -y, yaw
    elif mode == 3:
        return y, -x, yaw
    elif mode == 4:
        return y, x, -yaw
    elif mode == 5:
        return -x, y, -yaw
    elif mode == 6:
        return -y, -x, -yaw
    elif mode == 7:
        return x, -y, -yaw
    return

def main(args):
    yaw_ratio = 0.
    
    suffix = ""
    if args.test:
        mfrl_datapath = "mfrl_test"
    else:
        mfrl_datapath = "mfrl_train"
    
    h5f_filedir = "../dataset/{}{}.h5".format(mfrl_datapath, suffix)
    if args.aug:
        suffix += "_aug"
    h5f_write = "../dataset/{}{}_tnorm.h5".format(mfrl_datapath, suffix)
    
    print(h5f_write)
    
    n_aug = 1
    if args.aug:
        n_aug = 8
    
    tags = ['sta', 'sim', 'real']
    tags_enable = [True, True, True]
    for p_ii in range(args.data_min_dim, args.data_max_dim+1):
        for tag_idx, tag in enumerate(tags):
            if not tags_enable[tag_idx]:
                continue
            h5f = h5py.File(h5f_filedir, 'r')
            if args.test:
                if tag == 'real':
                    rand_idx = np.array(h5f["{}".format(p_ii)]["rand_idx"]).astype(np.int32)
                    points_arr = np.array(h5f['{}'.format(p_ii)]["points"][:])[rand_idx,:,:]
                else:
                    points_arr = np.array(h5f['{}'.format(p_ii)]["points"])
            else:
                points_arr = np.array(h5f['{}'.format(p_ii)]["points_{}".format(tag)])
            alpha_set = np.array(h5f['{}'.format(p_ii)]["alpha_{}".format(tag)])
            h5f.close()
            print("Tag {}, Dim: {} - N_data: {}".format(tag, p_ii, points_arr.shape[0]))

            # points_diff = p.diff(points_arr[:,:,:4], axis=1)
            # if np.any(np.abs(points_diff[:,:,3]) > np.pi):
            #     print("Wrong yaw")
            # pos_diff = np.linalg.norm(points_diff[:,:,:3], axis=2)
            # yaw_diff = np.abs(points_diff[:,:,3])
            # avg_time = (pos_diff + yaw_ratio * yaw_diff) / mean_spd
            # avg_time = np.repeat(np.sum(avg_time, axis=1)[:, np.newaxis], points_arr.shape[1]-1, axis=1)
            
            der_arr = np.zeros((points_arr.shape[0], p_ii, 14))
            for b_ii in range(points_arr.shape[0]):
                points_t = points_arr[b_ii][:,:4]
                t_set = points_arr[b_ii][1:,5] * alpha_set[b_ii]
                
                res, d_ordered = poly.obj_func_acc(points_t, t_set, flag_loop=False, snap_w=np.ones_like(t_set))
                res_yaw, d_ordered_yaw = poly.obj_func_acc(points_t, t_set, flag_loop=False, snap_w=np.ones_like(t_set), flag_yaw=True, flag_direct_yaw=True)
                der_t = np.zeros((p_ii, 14))
                for p_jj in range(p_ii):
                    der_arr[b_ii,p_jj,:4] = d_ordered[1+5*p_jj:5+5*p_jj,0]
                    der_arr[b_ii,p_jj,4:8] = d_ordered[1+5*p_jj:5+5*p_jj,1]
                    der_arr[b_ii,p_jj,8:12] = d_ordered[1+5*p_jj:5+5*p_jj,2]
                    der_arr[b_ii,p_jj,12:14] = d_ordered_yaw[1+3*p_jj:3+3*p_jj,0] 
        
            o_data_all = np.zeros((n_aug*points_arr.shape[0], points_arr.shape[1], points_arr.shape[2]+2))
            o_der_all = np.zeros((n_aug*der_arr.shape[0], der_arr.shape[1], der_arr.shape[2]))
            for t_ii in range(n_aug): # flip trajectory
                o_data = np.zeros((points_arr.shape[0], points_arr.shape[1], points_arr.shape[2]+2))
                x, y, yaw = flip_aug(points_arr[:,:,0], points_arr[:,:,1], points_arr[:,:,3], mode=t_ii)
                o_data[:,:,0] = x
                o_data[:,:,1] = y
                o_data[:,:,2] = points_arr[:,:,2]
                o_data[:,:,3] = np.cos(yaw)
                o_data[:,:,4] = np.sin(yaw)
                o_data[:,:,5] = points_arr[:,:,4]
                label_aug = np.repeat(alpha_set[:, np.newaxis], points_arr.shape[1]-1, axis=1)
                o_data[:,1:,6] = points_arr[:,1:,5] * label_aug
                snapw_norm = np.repeat(np.sum(points_arr[:,:,6], axis=1)[:, np.newaxis], points_arr.shape[1], axis=1)
                o_data[:,:,7] = points_arr[:,:,6] / snapw_norm
                o_data[:,:,8] = yaw

                b_x, b_y, b_yaw = flip_aug_der(der_arr[:,:,:4], der_arr[:,:,4:8], der_arr[:,:,12:14], mode=t_ii)
                o_der = copy.deepcopy(der_arr)
                o_der[:,:,:4] = b_x
                o_der[:,:,4:8] = b_y
                o_der[:,:,12:14] = b_yaw

                o_data_all[t_ii*points_arr.shape[0]:(t_ii+1)*points_arr.shape[0], :, :] = o_data
                o_der_all[t_ii*o_der.shape[0]:(t_ii+1)*o_der.shape[0], :, :] = o_der

            h5f = h5py.File(h5f_write, 'a')
            if "{}".format(tag) in h5f.keys():
                grp = h5f["{}".format(tag)]
            else:                
                grp = h5f.create_group("{}".format(tag))
            grp1 = grp.create_group("{}".format(p_ii))
            grp1.create_dataset('points', data=np.array(o_data_all))
            grp1.create_dataset('der', data=np.array(o_der_all))
            h5f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', action='store_true')
    parser.add_argument('-a', '--aug', action='store_true')
    
    # Load training data
    parser.add_argument('-dmin', '--data_min_dim', type=int, default=5)
    parser.add_argument('-dmax', '--data_max_dim', type=int, default=14)

    args = parser.parse_args()

    main(args)
