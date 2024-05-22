#!/usr/bin/env python
# coding: utf-8

import os, sys, io, random, shutil
import json, argparse
import time, yaml, copy
import numpy as np
from multiprocessing import cpu_count
from collections import OrderedDict, defaultdict

sys.path.insert(0, '../')
from pyTrajectoryUtils.pyTrajectoryUtils.minSnapTrajectory import *
from pyTrajectoryUtils.pyTrajectoryUtils.utils import *

def main(args):
    min_seq_len=5
    max_seq_len=14
    min_seg_len=3
    mean_spd = 4.
    points_scale = np.array([9.,9.,3.])
    
    N_min_epoch = [0, 0, 0, 0, 0]
    N_max_epoch = [50, 50, 50, 50, 50]
    
    h5f_dir_set = [
        "../dataset/mfrl_train_aug_tnorm.h5",
        "../dataset/mfrl_train_aug_tnorm.h5",
        "../dataset/mfrl_train_aug_tnorm.h5",
        "../dataset/mfrl_test_tnorm.h5",
        "../dataset/mfrl_test_tnorm.h5",
    ]
    h5f_tag = [
        "sta",
        "sim",
        "real",
        "sim",
        "real",
    ]
    
    procs = ["sta", "sim", "real", "test", "test_real"]
    for proc_ii, proc in enumerate(procs):
        
        if N_max_epoch[proc_ii] > N_min_epoch[proc_ii]:
            print("{} - reading data...".format(proc))
            points_array_set = dict()
            der_array_set = dict()
            print(h5f_dir_set[proc_ii])
            h5f = h5py.File(h5f_dir_set[proc_ii], 'r')
            for i in range(min_seq_len, max_seq_len+1):
                points_array_set["{}".format(i)] = np.array(h5f[h5f_tag[proc_ii]]["{}".format(i)]["points"])[:]
                der_array_set["{}".format(i)] = np.array(h5f[h5f_tag[proc_ii]]["{}".format(i)]["der"])[:]
            h5f.close()
        
        for epoch in range(N_min_epoch[proc_ii], N_max_epoch[proc_ii]):
            print("{} - epoch: {}/{}".format(proc, epoch, N_max_epoch[proc_ii]))
            train_seq = []
            train_len = []
            train_der = []
            train_seq_init = []
            train_len_init = []
            for i in range(min_seq_len, max_seq_len+1):
                points_array = copy.deepcopy(points_array_set["{}".format(i)])
                points_array_denorm = copy.deepcopy(points_array_set["{}".format(i)])
                der_array = copy.deepcopy(der_array_set["{}".format(i)])

                # Normalize
                for dim_ii in range(3):
                    points_array[:,:,dim_ii] /= (points_scale[dim_ii]/2.)
                    der_array[:,:,dim_ii*4:(dim_ii+1)*4] /= (points_scale[dim_ii]/2.)

                n_seg = 0
                for j in range(0, i-min_seg_len+1):
                    sub_seg_len = i-j
                    limit_scale = 1
                    if proc_ii == 4 or proc_ii == 5:
                        limit_scale = 10
                    limit = (points_array.shape[0] * limit_scale) // (max_seq_len - sub_seg_len + 1)
                    perm_idx = np.random.permutation(points_array.shape[0] * limit_scale)[:limit]
                    for p_ii_t in perm_idx:
                        p_ii = p_ii_t % points_array.shape[0]
                        points_opt = copy.deepcopy(points_array[p_ii, j:, :9])
                        # points_opt[0,6] = 0
                        points_opt[0,7] = 0
                        pos_diff = np.linalg.norm(np.diff(points_array_denorm[p_ii, j:, :3], axis=0), axis=1)
                        avg_time = np.sum(pos_diff) / mean_spd
                        points_opt[:,6] /= avg_time # Normalize time
                        points_opt[1:,7] /= np.sum(points_opt[1:,7]) # Normalize snapw
                        tmp_wp = np.pad(points_opt, ((0,max_seq_len-sub_seg_len), (0,0)), 'constant', constant_values=0)
                        tmp_der = der_array[p_ii, j, :]

                        points_init = copy.deepcopy(points_array[p_ii, :, :9])
                        pos_diff_init = np.linalg.norm(np.diff(points_array_denorm[p_ii, :, :3], axis=0), axis=1)
                        avg_time_init = np.sum(pos_diff_init) / mean_spd
                        points_init[1:,6] /= avg_time_init # Normalize time
                        points_init[1:,7] /= np.sum(points_init[1:,7]) # Normalize snapw
                        tmp_wp_init = np.pad(points_init, ((0,max_seq_len-i), (0,0)), 'constant', constant_values=0)

                        train_seq.append(tmp_wp)
                        train_len.append(sub_seg_len)
                        train_der.append(tmp_der)
                        train_seq_init.append(tmp_wp_init)
                        train_len_init.append(i)

            print(" - N_data : {}".format(len(train_seq)))
            
            ep_a = int(epoch / 10)
            ep_b = epoch % 10
            h5f_write = "../dataset/mfrl_online/mfrl_presample_batch_{}_{}.h5".format(proc, ep_a)
            h5f_write_bc = "../dataset/mfrl_online/mfrl_presample_batch_{}_backup.h5".format(proc)
            if epoch == 0:
                h5f = h5py.File(h5f_write, 'w')
            else:
                h5f = h5py.File(h5f_write, 'a')
            
            grp = h5f.create_group("{}".format(ep_b))
            
            N_batch = int(len(train_seq)/args.batch_size)
            batch_idx_all = np.random.permutation(N_batch*args.batch_size)
            chunck_size = 200
            if args.batch_size < chunck_size:
                chunck_size = args.batch_size
            grp.create_dataset('num_batch', data=np.array([N_batch]))
            grp.create_dataset('seq', data=np.array(train_seq)[batch_idx_all, :, :], chunks=(chunck_size,max_seq_len,9), dtype='f', compression="gzip")
            grp.create_dataset('len', data=np.array(train_len)[batch_idx_all])
            grp.create_dataset('der', data=np.array(train_der)[batch_idx_all, :], chunks=(chunck_size,14), dtype='f', compression="gzip")
            grp.create_dataset('seqi', data=np.array(train_seq_init)[batch_idx_all, :, :], chunks=(chunck_size,max_seq_len,9), dtype='f', compression="gzip")
            grp.create_dataset('leni', data=np.array(train_len_init)[batch_idx_all])
            h5f.close()
            
            if epoch % 10 == 0:
                shutil.copyfile(h5f_write, h5f_write_bc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Load training data
    parser.add_argument('-dmin', '--data_min_dim', type=int, default=5)
    parser.add_argument('-dmax', '--data_max_dim', type=int, default=14)
    parser.add_argument('-bs', '--batch_size', type=int, default=200)

    args = parser.parse_args()

    main(args)
