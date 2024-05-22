#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os, sys, time, copy
import yaml, h5py, shutil
from os import path

def get_min_time_array(filedir, filename, MAX_ITER = 50):
    yamlFile = os.path.join(filedir, filename)
    min_time_array = []
    with open(yamlFile, "r") as input_stream:
        yaml_in = yaml.safe_load(input_stream)
        for i in range(MAX_ITER):
            if 'iter{}'.format(i) in yaml_in:
                min_time_array.append(np.float(yaml_in['iter{}'.format(i)]['min_time']))
    return min_time_array

def check_result_data(filedir, filename, MAX_ITER = 50):
    yamlFile = os.path.join(filedir, filename)
    if not os.path.exists(yamlFile):
        return False
    data = get_min_time_array(filedir, filename, MAX_ITER)
    if len(data) is not MAX_ITER:
        return False
    else:
        return True
