#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import time
import argparse

from .reward_estimator_server import RewardEstimatorServer as RES_slurm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', type=int, default=5555)
    parser.add_argument('-o', '--flag_online', action='store_true')
    parser.add_argument('-d', '--tmp_dir', type=str, default='/mnt/home/tmp')
    args = parser.parse_args()
    
    print("tmp_dir: {}".format(args.tmp_dir))
    print("port: {}".format(args.port))
    zmq_server = RES_slurm(tcp_port=args.port, flag_online=True, tmp_dir=args.tmp_dir, ntasks=20)
    zmq_server.run()
