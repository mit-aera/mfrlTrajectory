#!/usr/bin/env python
# coding: utf-8

import os, sys, io, random, shutil, glob
import logging
import json
import time
import torch
import argparse
import numpy as np
import yaml, copy, h5py
from multiprocessing import cpu_count
from collections import OrderedDict, defaultdict
from datetime import datetime

import zmq
import pickle
import zlib
import subprocess

from mfrl.training_utils import *
# from threading import Thread, Lock
# from multiprocessing import Pool, Pipe, TimeoutError, Process
from pathos.pp import ParallelPool
from pathos.multiprocessing import ProcessingPool

def get_timestamp():
    now = datetime.now() # current date and time
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    return date_time

def send_zipped_pickle(socket, obj, flags=0, protocol=4):
    """pickle an object, and zip the pickle before sending it"""
    p = pickle.dumps(obj, protocol)
    z = zlib.compress(p)
    return socket.send(z, flags=flags)

def recv_zipped_pickle(socket, flags=0, protocol=4):
    """inverse of send_zipped_pickle"""
    z = socket.recv(flags)
    p = zlib.decompress(z)
    return pickle.loads(p)


class RewardEstimatorServer():
    def __init__(self, tcp_port, flag_online=False, tmp_dir="./tmp", ntasks=50):
        self.num_fidelity = 2
        self.TCP_PORT = tcp_port
        self.flag_online = flag_online

        self.eval_th_set = []
        for f_ii in range(self.num_fidelity):
            self.eval_th_set.append([])        
        self.test_th_set = []
        
        self.eval_proc = []
        for f_ii in range(self.num_fidelity):
            self.eval_proc.append(None)
        self.test_proc = None
        
        self.num_task = ntasks
        
        self.test_bs = 4
        
        self.curr_path = os.path.dirname(os.path.abspath(__file__))
        
        self.tmp_dir = tmp_dir
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        
        files = glob.glob("{}/*".format(self.tmp_dir))
        for f in files:
            os.remove(f)
        
        self.curr_job = []
        self.curr_process = []
        self.last_job = [0, 0]
        return
    
    def run_job(self, eval_type, b_idx, ntasks=-1):
        tag = "{}_{}".format(eval_type, b_idx)
        if ntasks == -1:
            ntask_t = self.num_task
        else:
            ntask_t = ntasks
        
        suffix = ""
        suffix += "-t {} -d {} -i {}".format(eval_type, self.tmp_dir, b_idx)
        if self.flag_online:
            suffix += " -o 1"
        else:
            suffix += " -o 0"
        
        s_out = os.path.join(self.tmp_dir, "slurm_{}_out.log".format(tag))
        s_err = os.path.join(self.tmp_dir, "slurm_{}_err.log".format(tag))
        # command = 'srun --job-name="{}" --nodes 3 --ntasks={} --cpus-per-task=1 --time 01:00:00 python3 {}/reward_estimator_mpislurm_node.py {}'.format(tag, N_task, self.curr_path, suffix)
        command = 'sbatch --job-name="{}" --ntasks={} --output={} --error={} --export=ALL,num_batch={},curr_path={} {}/run_mpislurm.sh {}'.format(tag, ntask_t, s_out, s_err, ntask_t, self.curr_path, self.curr_path, suffix)
        print("exec command:")
        print(command)
        # process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        output = subprocess.check_output(command.split())
        job_id = output.split()[-1].decode('utf-8')
        return job_id
    
    def check_job(self, job_id):
        flag_running = False
        try:
            res = subprocess.check_output('squeue -o %t --job {}'.format(job_id).split())
            status = res.split()[-1].decode('utf-8')
            if status == "R" or status == "PD":
                flag_running = True
        except:
            flag_running = False
        return flag_running
    
    def cancel_job(self, eval_type):
        tag = "eval_{}_{}".format(eval_type)
        command = 'scancel --job-name="{}"'.format(tag)        
        return
    
    def run(self):
        context = zmq.Context()
        server = context.socket(zmq.REP)
        server.bind("tcp://*:{}".format(self.TCP_PORT))

        while True:
            # request = server.recv()
            request = recv_zipped_pickle(server)
            data_send = [np.random.random(4)]
            if request[1] == 0: # start thread
                date_time = get_timestamp()
                print("[{}] Received eval request, eval_type: {}".format(date_time, request[2]))
                if self.flag_online:
                    points_list = request[3][0]
                    idx_new_list = request[3][1]
                    t_set_list = request[3][2]
                    snap_w_list = request[3][3]
                    ep_idx = request[3][4]
                    if request[2] <= -1: # eval_test, eval_test_wp
                        N_test = int(len(points_list)/self.test_bs)
                        if len(points_list) > N_test * self.test_bs:
                            N_test += 1
                        if len(self.test_th_set) == 0:
                            rcode_base = 0
                        elif (self.test_th_set[-1]+1) % self.num_task == 0:
                            rcode_base = self.test_th_set[-1]+1
                        else:
                            rcode_base = (int((self.test_th_set[-1]+1)/self.num_task) + 1) * self.num_task
                        for b_ii in range(N_test):
                            idx_i = b_ii*self.test_bs
                            idx_f = min((b_ii+1)*self.test_bs, len(points_list))
                            rcode = rcode_base + b_ii
                            data_tmp = [
                                points_list[idx_i:idx_f], 
                                idx_new_list[idx_i:idx_f], 
                                t_set_list[idx_i:idx_f], 
                                snap_w_list[idx_i:idx_f], ep_idx]
                            data_tmp_path = "{}/eval_{}_{}.pkl".format(self.tmp_dir, request[2], rcode)
                            with open(data_tmp_path, 'wb') as handle:
                                pickle.dump(data_tmp, handle, protocol=4)
                            self.test_th_set.append(rcode)
                        process = self.run_job(eval_type=request[2], b_idx=rcode_base, ntasks=rcode+1-rcode_base)
                        self.curr_process.append(process)
                        self.curr_job.append([request[2], rcode_base, rcode+1-rcode_base])
                    else: # eval_L / eval_H
                        data_tmp = [points_list, idx_new_list, t_set_list, snap_w_list, ep_idx]
                        rcode = len(self.eval_th_set[request[2]])
                        data_tmp_path = "{}/eval_{}_{}.pkl".format(self.tmp_dir, request[2], rcode)
                        with open(data_tmp_path, 'wb') as handle:
                            pickle.dump(data_tmp, handle, protocol=4)
                        self.eval_th_set[request[2]].append(rcode)
                        if (rcode+1) % self.num_task == 0:
                            process = self.run_job(eval_type=request[2], b_idx=rcode-self.num_task+1)
                            self.curr_process.append(process)
                            self.curr_job.append([request[2], rcode-self.num_task+1, -1])
                            self.last_job[request[2]] = rcode-self.num_task+1
                            # self.eval_proc[request[2]] = process
                else:
                    points_list = request[3][0]
                    t_set_list = request[3][1]
                    snap_w_list = request[3][2]
                    ep_idx = request[3][3]
                    if request[2] == -1: # eval_test
                        N_test = int(len(points_list)/self.test_bs)
                        if len(points_list) > N_test * self.test_bs:
                            N_test += 1
                        if len(self.test_th_set) == 0:
                            rcode_base = 0
                        elif (self.test_th_set[-1]+1) % self.num_task == 0:
                            rcode_base = self.test_th_set[-1]+1
                        else:
                            rcode_base = (int((self.test_th_set[-1]+1)/self.num_task) + 1) * self.num_task
                        for b_ii in range(N_test):
                            idx_i = b_ii*self.test_bs
                            idx_f = min((b_ii+1)*self.test_bs, len(points_list))
                            rcode = rcode_base + b_ii
                            data_tmp = [ 
                                points_list[idx_i:idx_f], 
                                t_set_list[idx_i:idx_f], 
                                snap_w_list[idx_i:idx_f], ep_idx]
                            data_tmp_path = "{}/eval_{}_{}.pkl".format(self.tmp_dir, request[2], rcode)
                            with open(data_tmp_path, 'wb') as handle:
                                pickle.dump(data_tmp, handle, protocol=4)
                            self.test_th_set.append(rcode)
                        process = self.run_job(eval_type=request[2], b_idx=rcode_base, ntasks=rcode+1-rcode_base)
                    else: # eval_L / eval_H                        
                        data_tmp = [points_list, t_set_list, snap_w_list, ep_idx]
                        rcode = len(self.eval_th_set[request[2]])
                        data_tmp_path = "{}/eval_{}_{}.pkl".format(self.tmp_dir, request[2], rcode)
                        with open(data_tmp_path, 'wb') as handle:
                            pickle.dump(data_tmp, handle, protocol=4)
                        self.eval_th_set[request[2]].append(rcode)
                        if (rcode+1) % self.num_task == 0:
                            process = self.run_job(eval_type=request[2], b_idx=rcode-self.num_task+1)
                            # self.eval_proc[request[2]] = process
                data_send = [request[0]]
            elif request[1] == 1: # join thread
                date_time = get_timestamp()
                print("[{}] Received join request, eval_type: {}".format(date_time, request[2]))
                res_all = []
                flag_ready = True
                if request[2] <= -1: # eval_test
                    res_all = [np.empty(0)]
                    for t_ii in range(len(self.test_th_set)):
                        if self.test_th_set[t_ii] % self.num_task == 0:
                            data_tmp_path = "{}/eval_{}_{}_res.pkl".format(
                                self.tmp_dir, request[2], self.test_th_set[t_ii])
                            if not os.path.exists(data_tmp_path):
                                print("[Not ready] {}".format(data_tmp_path))
                                flag_ready = False
                                break
                    if flag_ready:
                        for t_ii in range(len(self.test_th_set)):
                            if self.test_th_set[t_ii] % self.num_task == 0:
                                data_tmp_path = "{}/eval_{}_{}_res.pkl".format(
                                    self.tmp_dir, request[2], self.test_th_set[t_ii])
                                with open(data_tmp_path, 'rb') as handle:
                                    res = pickle.load(handle)
                                for y_ii in range(len(res)):
                                    # print(res)
                                    assert res[y_ii][1] == 1
                                    ep_curr = res[y_ii][2]
                                    res_all[0] = np.concatenate((res_all[0], np.array(res[y_ii][0])))
                        # self.test_proc.terminate()
                        # self.cancel_job(request[2])
                        self.test_th_set = []
                        # files = glob.glob("{}/*".format(self.tmp_dir))
                        # for f in files:
                        #     os.remove(f)
                    
                    flag_proc_running = False
                    finished_proc = []
                    for job_id in self.curr_process:
                        if self.check_job(job_id):
                            flag_proc_running = True
                        else:
                            finished_proc.append(job_id)
                    for job_id in finished_proc:
                        self.curr_process.remove(job_id)
                        
                    if flag_proc_running:
                        print("Slurm jobs not finished")
                    
                    if not flag_proc_running and not flag_ready:
                        print("Resubmitting jobs")
                        for t_ii in range(len(self.test_th_set)):
                            if self.test_th_set[t_ii] % self.num_task == 0:
                                data_tmp_path = "{}/eval_{}_{}_res.pkl".format(
                                    self.tmp_dir, request[2], self.test_th_set[t_ii])
                                if not os.path.exists(data_tmp_path):
                                    for j_ii in range(len(self.curr_job)):
                                        if self.curr_job[j_ii][0] == request[2] and self.curr_job[j_ii][1] == self.test_th_set[t_ii]:
                                            process = self.run_job(eval_type=self.curr_job[j_ii][0], b_idx=self.curr_job[j_ii][1], ntasks=self.curr_job[j_ii][2])
                                            self.curr_process.append(process)
                
                else: # eval_L / eval_H
                    for f_ii in range(self.num_fidelity):
                        if self.last_job[f_ii] + self.num_task < len(self.eval_th_set[f_ii]):
                            rcode_tmp = int(len(self.eval_th_set[f_ii])/self.num_task) * self.num_task
                            process = self.run_job(eval_type=f_ii, b_idx=rcode_tmp, ntasks=len(self.eval_th_set[f_ii])-rcode_tmp)
                            self.curr_job.append([f_ii, rcode_tmp, len(self.eval_th_set[f_ii])-rcode_tmp])
                            self.last_job[f_ii] = rcode_tmp
                            self.curr_process.append(process)
                    
                    res_all = []
                    for f_ii in range(self.num_fidelity):
                        res_all.append(np.empty(0))
                    for f_ii in range(self.num_fidelity):
                        for t_ii in range(len(self.eval_th_set[f_ii])):
                            if self.eval_th_set[f_ii][t_ii] % self.num_task == 0:
                                data_tmp_path = "{}/eval_{}_{}_res.pkl".format(
                                    self.tmp_dir, f_ii, self.eval_th_set[f_ii][t_ii])
                                if not os.path.exists(data_tmp_path):
                                    flag_ready = False
                                    break
                        if not flag_ready:
                            break
                    
                    if flag_ready:
                        for f_ii in range(self.num_fidelity):
                            for t_ii in range(len(self.eval_th_set[f_ii])):
                                if self.eval_th_set[f_ii][t_ii] % self.num_task == 0:
                                    data_tmp_path = "{}/eval_{}_{}_res.pkl".format(
                                        self.tmp_dir, f_ii, self.eval_th_set[f_ii][t_ii])
                                    with open(data_tmp_path, 'rb') as handle:
                                        res = pickle.load(handle)
                                    for y_ii in range(len(res)):
                                        ep_curr = res[y_ii][2]
                                        res_all[f_ii] = np.concatenate((res_all[f_ii], np.array(res[y_ii][0])))
                                        # print("f{} - res {}".format(f_ii, len(res[y_ii][0])))
                        # for f_ii in range(self.num_fidelity):
                            # self.eval_proc[f_ii].terminate()
                            # self.cancel_job(f_ii)
                        self.eval_th_set = []
                        for f_ii in range(self.num_fidelity):
                            self.eval_th_set.append([])
                    
                    flag_proc_running = False
                    finished_proc = []
                    for job_id in self.curr_process:
                        if self.check_job(job_id):
                            flag_proc_running = True
                        else:
                            finished_proc.append(job_id)
                    for job_id in finished_proc:
                        self.curr_process.remove(job_id)
                        
                    if flag_proc_running:
                        print("Slurm jobs not finished")
                    
                    if not flag_proc_running and not flag_ready:
                        print("Resubmitting jobs")
                        for f_ii in range(self.num_fidelity):
                            for t_ii in range(len(self.eval_th_set[f_ii])):
                                if self.eval_th_set[f_ii][t_ii] % self.num_task == 0:
                                    data_tmp_path = "{}/eval_{}_{}_res.pkl".format(
                                        self.tmp_dir, f_ii, self.eval_th_set[f_ii][t_ii])
                                    if not os.path.exists(data_tmp_path):
                                        for j_ii in range(len(self.curr_job)):
                                            if self.curr_job[j_ii][0] == f_ii and self.curr_job[j_ii][1] == self.eval_th_set[f_ii][t_ii]:
                                                process = self.run_job(eval_type=self.curr_job[j_ii][0], 
                                                                       b_idx=self.curr_job[j_ii][1], ntasks=self.curr_job[j_ii][2])
                                                self.curr_process.append(process)
                data_send = [request[0], res_all, flag_ready]
            
            # logging.info("Normal request (%s)", request)
            # time.sleep(0.5)  # Do some heavy work
            # server.send(data_send)
            send_zipped_pickle(server, data_send)
            
            if request[1] == 1 and data_send[2] == True:
                files = glob.glob("{}/*".format(self.tmp_dir))
                try:
                    for f in files:
                        os.remove(f)
                except:
                    print("failed to remove {}".format(self.tmp.dir))
                
                self.curr_process = []
                self.curr_job = []
        
        return
