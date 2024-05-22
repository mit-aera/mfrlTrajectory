#!/usr/bin/env python
# coding: utf-8

import os, sys, time, copy, yaml
from scipy.special import factorial, comb, perm
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import h5py

from .utils import *
from pyMulticopterDynamicsSim import multicopterDynamicsSim as uav_dyns
from pyTrajectoryTrackingController import trajectoryTrackingController as ndi

plt.switch_backend('agg')

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

def unwrap(x):
    return (x+np.pi)%(2*np.pi)-np.pi

class TrajectorySimulation(BaseTrajFunc):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if 'cfg_path' in kwargs:
            cfg_path = kwargs['cfg_path']
        else:
            curr_path = os.path.dirname(os.path.abspath(__file__))
            cfg_path = curr_path+"/../config/multicopter_sim.yaml"
        
        if 'drone_model' in kwargs:
            drone_model = kwargs['drone_model']
        else:
            drone_model = "default"
        
        self.uav_sim = uav_dyns(cfg_path, drone_model)
        self.uav_sim.resetMotorSpeeds()
        self.controller = ndi(cfg_path, drone_model)
        
        with open(cfg_path, 'r') as stream:
            try:
                cfg = yaml.safe_load(stream)
                thrustCoef = np.double(cfg['motor_model']['thrust_coefficient'])
                mass = np.double(cfg['uav_model'][drone_model]['vehicle_mass'])
                self.gravity = np.double(cfg['simulation']['gravity'])
            except yaml.YAMLError as exc:
                print(exc)
        self.w_sta = np.sqrt(mass*self.gravity/thrustCoef/4)
        
        return
    
    def simulation_core(self, 
        status_ref, N_trial = 1,
        max_pos_err=5.0, min_pos_err=0.5, 
        max_yaw_err=15.0, min_yaw_err=5.0, 
        freq_ctrl=200, freq_sim=2000, traj_ref_path=None, flag_debug=True, flag_no_yawflip=False):
        
        max_yaw_err = np.cos(max_yaw_err*np.pi/180.0)
        min_yaw_err = np.cos(min_yaw_err*np.pi/180.0)
        
        max_time = 100
        dt_micro_ctrl = int(1e6/freq_ctrl)
        freq_sim_update = int(freq_sim/freq_ctrl)
        N = min(status_ref.shape[0], max_time*freq_ctrl)

        debug_array = [dict() for i in range(N_trial)]

        for trial in range(N_trial):
            if flag_debug:
                print("Flight #{}".format(trial))
            pos_err = 0
            
            traj_ref = status_ref[0,:]
            pos = status_ref[0,2:5]
            vel = status_ref[0,5:8]
            acc = status_ref[0,8:11] - np.array([0,0,self.gravity])
            att = np.zeros(3)
            att[2] = status_ref[0,17:18]
            att_q = Euler2quat(att)
            angV = np.zeros(3)
            angA = np.zeros(3)
            ms = np.ones(4)*self.w_sta
            ma = np.zeros(4)
            curr_time = 0

            self.uav_sim.setVehicleState(pos, vel, att_q, angV, ms)
            imu_state_t = self.uav_sim.getIMUMeasurement()
            raw_acc = imu_state_t["acc"]
            raw_gyro = imu_state_t["gyro"]
            self.controller.resetFilter(curr_time, raw_acc, raw_gyro, ms)
            acc = raw_acc

            pos_array = np.zeros((N,3))
            vel_array = np.zeros((N,3))
            acc_array = np.zeros((N,3))
            att_array = np.zeros((N,3))
            att_q_array = np.zeros((N,4))
            raw_acc_array = np.zeros((N,3))
            raw_gyro_array = np.zeros((N,3))
            filtered_acc_array = np.zeros((N,3))
            filtered_gyro_array = np.zeros((N,3))
            ms_array = np.zeros((N,4))
            ms_c_array = np.zeros((N,4))
            time_array = np.zeros(N)
            pos_err_array = np.zeros(N)
            yaw_err_array = np.zeros(N)
            failure_idx = -1
            failure_start_idx = -1
            failure_end_idx = -1

            for it in range(N):
                curr_time = int(1.0*(it+1)/freq_ctrl*1e6)

                traj_ref = status_ref[it,2:]
                ms_c = self.controller.computeControlInputs(1.0/freq_ctrl, traj_ref, pos, vel, acc, att_q, angV, angA, ms, ma)

                for it_sim in range(freq_sim_update):
                    self.uav_sim.proceedState(1.0/freq_sim, ms_c)
                uav_state_t = self.uav_sim.getVehicleState()
                imu_state_t = self.uav_sim.getIMUMeasurement()

                raw_acc = imu_state_t["acc"]
                raw_gyro = imu_state_t["gyro"]
                raw_ms = uav_state_t["motorSpeed"]
                filtered_state_t = self.controller.getFilteredState(curr_time,raw_acc,raw_gyro,raw_ms)

                pos = uav_state_t["position"]
                vel = uav_state_t["velocity"]
                acc = filtered_state_t["acceleration"]
                att_q = uav_state_t["attitude"]
                att = quat2Euler(att_q)

                angV_f = uav_state_t["angularVelocity"]
                angV = filtered_state_t["angularVelocity"]
                angA = filtered_state_t["angularAcceleration"]
                ms = filtered_state_t["motorSpeed"]
                ma = filtered_state_t["motorAcceleration"]

                time_array[it] = 1.0*(it+1)/freq_ctrl
                pos_array[it,:] = pos
                vel_array[it,:] = vel
                acc_array[it,:] = acc
                att_array[it,:] = att
                att_q_array[it,:] = att_q
                raw_acc_array[it,:] = raw_acc
                raw_gyro_array[it,:] = raw_gyro
                filtered_acc_array[it,:] = acc
                filtered_gyro_array[it,:] = angV_f
                ms_array[it,:] = ms
                ms_c_array[it,:] = ms_c
                
                if it < N-1:
                    pos_err = np.linalg.norm(status_ref[it+1,2:5]-pos)
                    if not flag_no_yawflip:
                        yaw_err = max(np.cos(status_ref[it+1,17]-att[2]), np.cos(np.pi-status_ref[it+1,17]+att[2]))
                    else:
                        yaw_err = np.cos(status_ref[it+1,17]-att[2])
                    pos_err_array[it+1] = pos_err
                    yaw_err_array[it+1] = np.arccos(yaw_err)
                
                if (pos_err > min_pos_err or yaw_err < min_yaw_err) and failure_start_idx == -1:
                    failure_start_idx = it
                
                if (pos_err < min_pos_err and yaw_err > min_yaw_err) and failure_start_idx != -1:
                    failure_start_idx = -1

                if pos_err > max_pos_err or yaw_err < max_yaw_err:
                    if flag_debug:
                        print("Failed. Progress: {}%, Ref total time: {}".format(100.0*it/N, status_ref[-1,0]))
                    failure_idx = it
                    break
            
            debug_array[trial]["ref"] = status_ref
            debug_array[trial]["time"] = time_array
            debug_array[trial]["pos"] = pos_array
            debug_array[trial]["vel"] = vel_array
            debug_array[trial]["acc"] = acc_array
            debug_array[trial]["att"] = att_array
            debug_array[trial]["att_q"] = att_q_array
            debug_array[trial]["acc_raw"] = raw_acc_array
            debug_array[trial]["acc_lpf"] = filtered_acc_array
            debug_array[trial]["gyro_raw"] = raw_gyro_array
            debug_array[trial]["gyro_lpf"] = filtered_gyro_array
            debug_array[trial]["ms"] = ms_array
            debug_array[trial]["ms_c"] = ms_c_array
            debug_array[trial]["failure_idx"] = failure_idx
            debug_array[trial]["failure_start_idx"] = failure_start_idx
            debug_array[trial]["failure_end_idx"] = failure_end_idx
            debug_array[trial]["pos_err"] = pos_err_array
            debug_array[trial]["yaw_err"] = yaw_err_array
            
            if flag_debug:
                if pos_err < max_pos_err and yaw_err > max_yaw_err:
                    if traj_ref_path is not None:
                        print("Succeed. Total time: {}.\n - Traj path: {}".format(time_array[-1], traj_ref_path))
                    else:
                        print("Succeed. Total time: {}.".format(time_array[-1]))        
                print("Position error: {}/{}, Yaw error: {}/{}". \
                      format(str(round(max(pos_err_array),2)),max_pos_err, \
                             str(round(max(yaw_err_array),2)),str(round(np.arccos(max_yaw_err),2))))
        return debug_array

    def run_simulation(self, 
        traj_ref_path='trajectory/test.csv', N_trial=1, 
        max_pos_err=5.0, min_pos_err=0.5, 
        max_yaw_err=15.0, min_yaw_err=5.0, 
        freq_ctrl=200, freq_sim=2000, flag_debug=False):
        
        df = pd.read_csv(traj_ref_path, sep=',', header=None)
        status_ref = df.values[1:,:]
        debug_array = self.simulation_core(
            status_ref, N_trial=N_trial, 
            max_pos_err=max_pos_err, min_pos_err=min_pos_err, 
            max_yaw_err=max_yaw_err, min_yaw_err=min_yaw_err, 
            freq_ctrl=freq_ctrl, freq_sim=freq_sim, traj_ref_path=traj_ref_path, flag_debug=flag_debug)
        
        return debug_array
    
    def run_simulation_from_der(self, 
        t_set, d_ordered, d_ordered_yaw=None,
        N_trial=1, max_pos_err=5.0, min_pos_err=0.5, 
        max_yaw_err=15.0, min_yaw_err=5.0, 
        freq_ctrl=200, freq_sim=2000, flag_debug=False, direct_yaw=False, flag_no_yawflip=False):
    
        flag_loop = self.check_flag_loop(t_set,d_ordered)
                
        dt = 1./freq_ctrl
        total_time = np.sum(t_set)
        cum_time = np.zeros(t_set.shape[0]+1)
        cum_time[1:] = np.cumsum(t_set)
        cum_time[0] = 0
        
        N = int(np.floor(total_time/dt))
        poly_idx = 0
        
        t_array = total_time*np.array(range(N))/N
        status_ref = np.zeros((N,20))
        status_ref[:,0] = t_array
        status_ref[:,1] = 1
        
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
                status_ref[i,2+3*der:2+3*(der+1)] = (v0.dot(der0)+v1.dot(der1))/(t_set[poly_idx]**der)
            
            if np.all(d_ordered_yaw != None):
                status_yaw_xy = np.zeros((1,3,d_ordered_yaw.shape[1]))
                for der in range(3):
                    v0, v1 = self.generate_single_point_matrix_yaw(t_step, der=der)
                    status_yaw_xy[:,der,:] = (v0.dot(der0_yaw)+v1.dot(der1_yaw))/(t_set[poly_idx]**der)
                
                if d_ordered_yaw.shape[1] == 2:
                    status_ref[i,17:] = self.get_yaw_der(status_yaw_xy)[0,:]
                elif d_ordered_yaw.shape[1]:
                    status_ref[i,17:] = status_yaw_xy[0,:,0]
        
        debug_array = self.simulation_core(
            status_ref, N_trial=N_trial, 
            max_pos_err=max_pos_err, min_pos_err=min_pos_err, 
            max_yaw_err=max_yaw_err, min_yaw_err=min_yaw_err, 
            freq_ctrl=freq_ctrl, freq_sim=freq_sim, flag_debug=flag_debug, flag_no_yawflip=flag_no_yawflip)
        
        return debug_array

    def plot_result(self, debug_value, save_dir="trajectory/result", 
                    flag_save=False, save_idx="0", t_set=None, d_ordered=None):
        fontsize_t = 7
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        status_ref = debug_value["ref"]
        time_array = debug_value["time"]
        pos_array = debug_value["pos"]
        vel_array = debug_value["vel"]
        acc_array = debug_value["acc"]
        att_array = debug_value["att"]
        raw_acc_array = debug_value["acc_raw"]
        filtered_acc_array = debug_value["acc_lpf"]
        raw_gyro_array = debug_value["gyro_raw"]
        filtered_gyro_array = debug_value["gyro_lpf"]
        ms_array = debug_value["ms"]
        ms_c_array = debug_value["ms_c"]
        failure_idx = debug_value["failure_idx"]
        failure_start_idx = debug_value["failure_start_idx"]
        t_fail = debug_value["time"][debug_value["failure_start_idx"]]
        
        pos_array_ref_t = ned2enu(status_ref[:,2:5])
        pos_array_t = ned2enu(pos_array)
        val_mean = (np.max(pos_array_ref_t, axis=0)+np.min(pos_array_ref_t, axis=0))/2
        pos_array_ref_t -= val_mean
        max_lim_x = max(max(np.max(pos_array_ref_t[:,0]),-np.min(pos_array_ref_t[:,0]))*1.15,1.0)
        max_lim_y = max(max(np.max(pos_array_ref_t[:,1]),-np.min(pos_array_ref_t[:,1]))*1.15,1.0)
        max_lim_z = max(max(np.max(pos_array_ref_t[:,2]),-np.min(pos_array_ref_t[:,2]))*1.15,1.0)
        pos_array_ref_t += val_mean
        
        if np.all(t_set != None) and np.all(d_ordered != None) and failure_idx != -1:
            poly_idx_fail = -1
            while t_fail > 0:
                if poly_idx_fail == t_set.shape[0]-1:
                    break
                t_fail_tmp = t_fail - t_set[poly_idx_fail+1]
                if t_fail_tmp < 0:
                    poly_idx_fail += 1
                    break
                t_fail = t_fail_tmp
                poly_idx_fail += 1
            print("t_fail: {}".format(t_fail))
            print("poly_idx_fail: {}".format(poly_idx_fail))
        
        plt.ioff()
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(pos_array_t[:failure_idx,0], pos_array_t[:failure_idx,1], pos_array_t[:failure_idx,2], 
                label='result', color="steelblue", zorder=2)
        ax.plot(pos_array_ref_t[:,0], pos_array_ref_t[:,1], pos_array_ref_t[:,2], label='reference', color="orange")
        
        if failure_start_idx != -1:
            ax.scatter(pos_array_t[failure_start_idx,0], 
                       pos_array_t[failure_start_idx,1], 
                       pos_array_t[failure_start_idx,2], 
                       marker='o', c='b', alpha=1.0, s=100.0)
            ax.scatter(pos_array_ref_t[failure_start_idx,0], 
                       pos_array_ref_t[failure_start_idx,1], 
                       pos_array_ref_t[failure_start_idx,2], 
                       marker='o', c='b', alpha=1.0, s=100.0, label='failure_start')
        
        if failure_idx != -1:
            ax.scatter(pos_array_t[failure_idx,0], 
                       pos_array_t[failure_idx,1], 
                       pos_array_t[failure_idx,2], 
                       marker='o', c='r', alpha=1.0, s=100.0)
            ax.scatter(pos_array_ref_t[failure_idx,0], 
                       pos_array_ref_t[failure_idx,1], 
                       pos_array_ref_t[failure_idx,2], 
                       marker='o', c='r', alpha=1.0, s=100.0, label='failure_end')
            
            if np.all(t_set != None) and np.all(d_ordered != None) and poly_idx_fail != -1:
                d_ordered_t = ned2enu(d_ordered)
                ax.scatter(d_ordered_t[poly_idx_fail*5,0], 
                           d_ordered_t[poly_idx_fail*5,1], 
                           d_ordered_t[poly_idx_fail*5,2], 
                           marker='o', c='g', alpha=1.0, s=100.0)
                ax.scatter(d_ordered_t[(poly_idx_fail+1)*5,0], 
                           d_ordered_t[(poly_idx_fail+1)*5,1], 
                           d_ordered_t[(poly_idx_fail+1)*5,2], 
                           marker='X', c='g', alpha=1.0, s=100.0)
        
        ax.set_xlim(-max_lim_x+val_mean[0], max_lim_x+val_mean[0])
        ax.set_ylim(-max_lim_y+val_mean[1], max_lim_y+val_mean[1])
        ax.set_zlim(-max_lim_z+val_mean[2], max_lim_z+val_mean[2])
        ax.legend(fontsize=fontsize_t)
        if flag_save:
            plt.savefig('{}/{}_trajectory.png'.format(save_dir,save_idx))
            plt.close()
        
        plt.ioff()
        fig = plt.figure(figsize=(6,4))
        ax = fig.add_subplot(111)
        ax.plot(pos_array_ref_t[:,0], pos_array_ref_t[:,1], label='reference', color="orange", linewidth=3, zorder=1)
        ax.plot(pos_array_t[:failure_idx,0], pos_array_t[:failure_idx,1], label='result', color="steelblue", linewidth=3, zorder=2)
        if failure_start_idx != -1:
            ax.scatter(pos_array_t[failure_start_idx,0], 
                       pos_array_t[failure_start_idx,1],
                       marker='o', c='b', alpha=1.0, s=100.0)
            ax.scatter(pos_array_ref_t[failure_start_idx,0], 
                       pos_array_ref_t[failure_start_idx,1],
                       marker='o', c='b', alpha=1.0, s=100.0, label='failure_start')
        
        if failure_idx != -1:
            ax.scatter(pos_array_t[failure_idx,0], 
                       pos_array_t[failure_idx,1],
                       marker='o', c='r', alpha=1.0, s=100.0)
            ax.scatter(pos_array_ref_t[failure_idx,0], 
                       pos_array_ref_t[failure_idx,1],
                       marker='o', c='r', alpha=1.0, s=100.0, label='failure_end')
            
            if np.all(t_set != None) and np.all(d_ordered != None) and poly_idx_fail != -1:
                d_ordered_t = ned2enu(d_ordered)
                ax.scatter(d_ordered_t[poly_idx_fail*5,0], 
                           d_ordered_t[poly_idx_fail*5,1],
                           marker='o', c='g', alpha=1.0, s=100.0)
                ax.scatter(d_ordered_t[(poly_idx_fail+1)*5,0], 
                           d_ordered_t[(poly_idx_fail+1)*5,1],
                           marker='X', c='g', alpha=1.0, s=100.0)
        ax.set_xlim(-max_lim_x+val_mean[0], max_lim_x+val_mean[0])
        ax.set_ylim(-max_lim_y+val_mean[1], max_lim_y+val_mean[1])
        ax.legend(fontsize=fontsize_t)
        ax.grid()
        if flag_save:
            plt.savefig('{}/{}_trajectory_2D.png'.format(save_dir,save_idx))
            plt.close()
            
        tau_v = status_ref[:,8:11] - [0,0,self.gravity]
        tau = -np.linalg.norm(tau_v,axis=1)
        bz = tau_v/tau[:,np.newaxis]
        yaw_ref = status_ref[:,17]
        roll_ref = np.arcsin(np.einsum('ij,ji->i', bz, 
                      np.array([np.sin(yaw_ref),-np.cos(yaw_ref),np.zeros_like(yaw_ref)])))
        pitch_ref = np.arctan(np.einsum('ij,ji->i', bz, 
                       np.array([np.cos(yaw_ref)/bz[:,2],np.sin(yaw_ref)/bz[:,2],np.zeros_like(yaw_ref)])))
        att_ref = np.array([roll_ref, pitch_ref, yaw_ref]).T
        att_ref = unwrap(att_ref)
        att_array = unwrap(att_array)

        def plot_state(time_array, state, state_ref, label_txt='vel', dim=3):
            start_idx = 0
            if failure_idx >= 0:
                end_idx = min(status_ref.shape[0], state.shape[0], time_array.shape[0], failure_idx)
            else:
                end_idx = min(status_ref.shape[0], state.shape[0], time_array.shape[0])
            time_array_t = time_array[start_idx:end_idx]

            plt.ioff()
            fig = plt.figure(figsize=(6,3))
            ax = fig.add_subplot(111)
            for i in range(dim):
                ax.plot(time_array_t, state[start_idx:end_idx,i], '-', label='{} dim {}'.format(label_txt,i))
                ax.plot(time_array_t, state_ref[start_idx:end_idx,i], '-', label='{} ref dim {}'.format(label_txt,i))
                ax.legend(fontsize=fontsize_t)
            ax.grid()
            if flag_save:
                plt.savefig('{}/{}_{}.png'.format(save_dir,save_idx,label_txt))
                plt.close()

        plot_state(time_array, pos_array, status_ref[:,2:5], label_txt='pos', dim=3)
        plot_state(time_array, vel_array, status_ref[:,5:8], label_txt='vel')
        plot_state(time_array, acc_array+np.array([0,0,9.81]), status_ref[:,8:11], label_txt='acc')
        plot_state(time_array, att_array, att_ref, label_txt='att')
        plot_state(time_array, raw_acc_array, filtered_acc_array, label_txt='imu_acc')
        plot_state(time_array, raw_gyro_array, filtered_gyro_array, label_txt='gyro')
        plot_state(time_array, ms_array[:,0:1], ms_c_array[:,0:1], label_txt='ms0', dim=1)
        plot_state(time_array, ms_array[:,1:2], ms_c_array[:,1:2], label_txt='ms1', dim=1)
        plot_state(time_array, ms_array[:,2:3], ms_c_array[:,2:3], label_txt='ms2', dim=1)
        plot_state(time_array, ms_array[:,3:4], ms_c_array[:,3:4], label_txt='ms3', dim=1)
        
        if not flag_save:
            plt.show()

        idx = 0
        t_prev = 0
        for t in time_array:
            if t < t_prev:
                break
            idx += 1
            t_prev = t
        
        # Save IMU and Camera data
        df = pd.DataFrame({
            "timestamp":time_array[:idx],
            "position.x":pos_array[:idx,0],
            "position.y":pos_array[:idx,1],
            "position.z":pos_array[:idx,2],
            "velocity.x":vel_array[:idx,0],
            "velocity.y":vel_array[:idx,1],
            "velocity.z":vel_array[:idx,2],
            "acceleration.x":acc_array[:idx,0],
            "acceleration.y":acc_array[:idx,1],
            "acceleration.z":acc_array[:idx,2],
            "attitude.x":att_array[:idx,0],
            "attitude.y":att_array[:idx,1],
            "attitude.z":att_array[:idx,2]})
        df.to_csv(os.path.join(save_dir,"log_{}.csv".format(save_idx)), sep=',', index=False)
        
        return