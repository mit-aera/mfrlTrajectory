#!/usr/bin/env python
# coding: utf-8

import os, sys, io, random
import json
import time
import torch
import argparse
import numpy as np

def unwrap(x):
    return (x+np.pi)%(2*np.pi)-np.pi

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