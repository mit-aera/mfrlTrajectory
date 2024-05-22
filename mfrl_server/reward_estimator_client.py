#!/usr/bin/env python
# coding: utf-8

import os, sys, io, random, shutil
import logging
import json
import time
import argparse
import numpy as np
import yaml, copy, h5py
from collections import OrderedDict, defaultdict

from pyTrajectoryUtils.pyTrajectoryUtils.minSnapTrajectory import *

from datetime import datetime

import zmq
import pickle
import zlib

from naiveBayesOpt.models import *
from mfrl.training_utils import *
# from threading import Thread, Lock
# from multiprocessing import Pool, Pipe, TimeoutError, Process
from pathos.pp import ParallelPool
from pathos.multiprocessing import ProcessingPool

min_snap = MinSnapTrajectory(drone_model="CFB", N_POINTS=40)

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

def eval_offline_L(points_list, t_set_list, snap_w_list, ep):
    res = min_snap.sanity_check_acc_yaw( \
        points_list, t_set_list, snap_w_list, direct_yaw=True, flag_sta=True)
    return [res, 0, ep]
def eval_offline_H(points_list, t_set_list, snap_w_list, ep):
    res = min_snap.sanity_check_acc_yaw( \
        points_list, t_set_list, snap_w_list, direct_yaw=True, flag_sta=False)
    return [res, 0, ep]
def eval_offline_test(points_list, t_set_list, snap_w_list, ep):
    res = min_snap.sanity_check_acc_yaw( \
        points_list, t_set_list, snap_w_list, direct_yaw=True, flag_sta=False)
    return [res, 1, ep]
eval_funcs_offline = [eval_offline_L, eval_offline_H]

def eval_online_L(points_list, idx_new_list, t_set_list, snap_w_list, ep):
    res = min_snap.sanity_check_acc_yaw_online( \
        points_list, idx_new_list, t_set_list, snap_w_list, direct_yaw=True, flag_sta=True)
    # res = np.random.randint(2, size=len(points_list))
    return [res, 0, ep]
def eval_online_H(points_list, idx_new_list, t_set_list, snap_w_list, ep):
    res = min_snap.sanity_check_acc_yaw_online( \
        points_list, idx_new_list, t_set_list, snap_w_list, direct_yaw=True, flag_sta=False)
    return [res, 0, ep]
def eval_online_test(points_list, idx_new_list, t_set_list, snap_w_list, ep):
    res = min_snap.sanity_check_acc_yaw_online( \
        points_list, idx_new_list, t_set_list, snap_w_list, direct_yaw=True, flag_sta=False)
    return [res, 1, ep]
def eval_online_test_wp(points_list, idx_new_list, t_set_list, snap_w_list, ep):
    res = min_snap.sanity_check_acc_yaw_online( \
        points_list, idx_new_list, t_set_list, snap_w_list, direct_yaw=True, flag_sta=False, flag_wp_update=True)
    return [res, 1, ep]
eval_funcs_online = [eval_online_L, eval_online_H]

class RewardEstimatorClient():
    def __init__(self, 
            sever_ip,
            req_timeout = 250000,
            req_timeout_test = 250000,
            req_timeout_join = 25000,
            req_retries = -1):
        self.REQUEST_TIMEOUT = req_timeout
        self.REQUEST_TIMEOUT_TEST = req_timeout_test
        self.REQUEST_TIMEOUT_JOIN = req_timeout_join
        self.REQUEST_RETRIES = req_retries
        self.SERVER_ENDPOINT = sever_ip
        return
    
    # eval type: 0: L, 1: H, -1: test
    def req_eval(self, eval_type, data):
        context = zmq.Context()
        client = context.socket(zmq.REQ)
        client.connect(self.SERVER_ENDPOINT)
        data_send = [np.random.random(5), 0, eval_type, data] # 0: start thread
        send_zipped_pickle(client, data_send)

        retries_left = self.REQUEST_RETRIES
        req_timeout = self.REQUEST_TIMEOUT
        if eval_type <= -1:
            req_timeout = self.REQUEST_TIMEOUT_TEST
        reply = None
        while True:
            if (client.poll(req_timeout) & zmq.POLLIN) != 0:
                reply = recv_zipped_pickle(client)
                if np.all(reply[0] == data_send[0]):
                    # logging.info("Server replied OK (%s)", reply[0][0])
                    retries_left = self.REQUEST_RETRIES
                    break
                else:
                    # logging.error("Malformed reply from server: %s", reply)
                    continue
            
            if retries_left >= 0:
                retries_left -= 1
            logging.warning("No response from server")
            # Socket is confused. Close and remove it.
            client.setsockopt(zmq.LINGER, 0)
            client.close()
            if retries_left == 0:
                sys.exit()
            
            # logging.info("Reconnecting to server…")
            # Create new connection
            client = context.socket(zmq.REQ)
            client.connect(self.SERVER_ENDPOINT)
            # logging.info("Resending (%s)", request)
            send_zipped_pickle(client, data_send)
        return
    
    def req_join(self, eval_type):
        context = zmq.Context()
        client = context.socket(zmq.REQ)
        client.connect(self.SERVER_ENDPOINT)
        data_send = [np.random.random(4), 1, eval_type] # 1: join thread
        send_zipped_pickle(client, data_send)

        retries_left = self.REQUEST_RETRIES
        reply = None
        while True:
            while True:
                if (client.poll(self.REQUEST_TIMEOUT_JOIN) & zmq.POLLIN) != 0:
                    reply = recv_zipped_pickle(client)
                    if np.all(reply[0] == data_send[0]):
                        retries_left = self.REQUEST_RETRIES
                        break
                    else:
                        continue

                if retries_left >= 0:
                    retries_left -= 1
                # Socket is confused. Close and remove it.
                client.setsockopt(zmq.LINGER, 0)
                client.close()
                if retries_left == 0:
                    sys.exit()

                # Create new connection
                client = context.socket(zmq.REQ)
                client.connect(self.SERVER_ENDPOINT)
                send_zipped_pickle(client, data_send)
            if reply[2]: # flag ready
                break
            else:
                date_time = get_timestamp()
                print("[{}] eval res is not ready, eval_type: {}".format(date_time, eval_type))
                time.sleep(1)                
        return reply[1]
