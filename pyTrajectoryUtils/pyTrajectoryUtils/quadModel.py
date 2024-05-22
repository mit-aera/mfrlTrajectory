#!/usr/bin/env python
# coding: utf-8
# Solve differential flatness and check feasibility of control command
# Use NED coordinate

import os, sys, time, copy, yaml
import numpy as np
# import cupy as cp

class QuadModel:
    def __init__(self, cfg_path=None, drone_model=None):
        if cfg_path == None:
            curr_path = os.path.dirname(os.path.abspath(__file__))
            cfg_path = curr_path+"/../config/multicopter_sim.yaml"
        if drone_model == None:
            drone_model="default"
        
        with open(cfg_path, 'r') as stream:
            try:
                cfg = yaml.safe_load(stream)
                self.thrustCoef = np.double(cfg['motor_model']['thrust_coefficient'])
                self.torqueCoef = np.double(cfg['motor_model']['torque_coefficient'])
                self.armLength = np.double(cfg['motor_model']['moment_arm'])
                self.mass = np.double(cfg['uav_model'][drone_model]['vehicle_mass'])
                self.Ixx = np.double(cfg['uav_model'][drone_model]['vehicle_inertia_xx'])
                self.Iyy = np.double(cfg['uav_model'][drone_model]['vehicle_inertia_yy'])
                self.Izz = np.double(cfg['uav_model'][drone_model]['vehicle_inertia_zz'])
                self.w_max = np.double(cfg['motor_model']['max_prop_speed'])
                self.w_min = np.double(cfg['motor_model']['min_prop_speed'])
                self.gravity = np.double(cfg['simulation']['gravity'])
                self.dragCoef = np.double(cfg['motor_model']['drag_coefficient'])
                self.w_sta = np.sqrt(self.mass*self.gravity/self.thrustCoef/4.0)
            except yaml.YAMLError as exc:
                print(exc)
        
        lt = self.armLength*self.thrustCoef
        k0 = self.torqueCoef
        k1 = self.thrustCoef
        self.G1 = np.array([[lt,-lt,-lt,lt],\
                           [lt,lt,-lt,-lt],\
                           [-k0,k0,-k0,k0],\
                           [-k1,-k1,-k1,-k1]])

        self.J = np.diag(np.array([self.Ixx,self.Iyy,self.Izz]))
        self.G1_inv = np.linalg.inv(self.G1)
        
        return

    def getWs(self, status):
        pos = np.array(status[0:3])
        vel = np.array(status[3:6])
        acc = np.array(status[6:9])
        jer = np.array(status[9:12])
        sna = np.array(status[12:15])
        yaw = status[15]
        dyaw = status[16]
        ddyaw = status[17]

        # Total thrust
        tau_v = acc - np.array([0,0,self.gravity])
        tau = -np.linalg.norm(tau_v)
        bz = tau_v/tau
        Thrust = self.mass*tau

        # roll & pitch
        roll = np.arcsin(np.dot(bz,[np.sin(yaw),-np.cos(yaw),0]))
        pitch = np.arctan(np.dot(bz,[np.cos(yaw),np.sin(yaw),0])/bz[2])
        bx = np.array([np.cos(yaw)*np.cos(pitch),np.sin(yaw)*np.cos(pitch),-np.sin(pitch)])
        by = np.array([-np.sin(yaw)*np.cos(roll)+np.cos(yaw)*np.sin(pitch)*np.sin(roll),\
              np.cos(yaw)*np.cos(roll)+np.sin(yaw)*np.sin(pitch)*np.sin(roll),\
              np.cos(pitch)*np.sin(roll)])

        # dzhi & Omega
        dzhi = np.dot(np.array([-1*by,bx/np.cos(roll),np.zeros(3)]),jer)/tau \
                +np.array([np.sin(pitch),-np.cos(pitch)*np.tan(roll),1])*dyaw
        S_inv = np.array([[1,0,-np.sin(pitch)],\
                          [0,np.cos(roll),np.cos(pitch)*np.sin(roll)],\
                          [0,-np.sin(roll),np.cos(pitch)*np.cos(roll)]])
        Omega = np.dot(S_inv,dzhi)

        C_inv = np.array([-1*by/tau,bx/np.cos(roll)/tau,bz])
        d = np.array([np.cos(yaw)*np.sin(roll)-np.cos(roll)*np.sin(yaw)*np.sin(pitch),\
                     np.sin(yaw)*np.sin(roll)+np.cos(roll)*np.cos(yaw)*np.sin(pitch),0])*tau
        dtau = np.dot(bz,jer-dyaw*d)

        # ddzhi & dOmega
        dS = np.array([[0,np.cos(roll)*np.tan(pitch),-np.sin(roll)*np.tan(pitch)],\
                    [0,-np.sin(roll),-np.cos(roll)],\
                    [0,np.cos(roll)/np.cos(pitch),-np.sin(roll)/np.cos(pitch)]])*dzhi[0]\
            +np.array([[0,np.sin(roll)/np.cos(pitch)/np.cos(pitch),np.cos(roll)/np.cos(pitch)/np.cos(pitch)],\
                    [0,0,0],\
                    [0,np.sin(roll)*np.tan(pitch)/np.cos(pitch),np.cos(roll)*np.tan(pitch)/np.cos(pitch)]])*dzhi[1]

        e = 2*dtau*np.dot(np.array([-1*by,bx,np.zeros(3)]).T,Omega)\
            +tau*np.dot(np.array([bx,by,bz]).T,np.array([Omega[0]*Omega[2],Omega[1]*Omega[2],-Omega[0]*Omega[0]-Omega[1]*Omega[1]]))\
            -tau*np.dot(np.array([-1*by,bx,np.zeros(3)]).T,np.dot(S_inv,np.dot(dS,Omega)))

        ddzhi = np.dot(C_inv,sna-ddyaw*d-e)
        ddzhi[2] = ddyaw

        dOmega = -np.dot(S_inv,np.dot(dS,Omega))+np.dot(S_inv,ddzhi)
        Mu = np.dot(self.J,dOmega) + np.cross(Omega,np.dot(self.J,Omega))
        MT = np.zeros(4)
        MT[:3] = Mu
        MT[3] = Thrust

        Ws2 = np.dot(self.G1_inv,MT)
        # Ws2 = np.clip(Ws2, np.power(self.w_min,2), np.power(self.w_max,2))
        # Ws = np.sqrt(Ws2)
        Ws = np.copysign(np.sqrt(np.abs(Ws2)),Ws2)
        state = {
            'roll':roll,
            'pitch':pitch,
            'dzhi':dzhi,
            'ut':MT
        }

        return Ws, state

    def getWs_vector(self, status):
        pos = np.array(status[:,0:3])
        vel = np.array(status[:,3:6])
        acc = np.array(status[:,6:9])
        jer = np.array(status[:,9:12])
        sna = np.array(status[:,12:15])
        yaw = np.array(status[:,15:16])
        dyaw = np.array(status[:,16:17])
        ddyaw = np.array(status[:,17:18])

        # Total thrust
        tau_v = acc - np.array([0,0,self.gravity])
        tau = -np.linalg.norm(tau_v,axis=1)[:,np.newaxis]
        bz = tau_v/tau
        Thrust = self.mass*tau
        
        # roll & pitch
        roll = np.arcsin(np.einsum('ij,ij->i', bz, 
                   np.concatenate((
                   np.sin(yaw),
                   -np.cos(yaw),
                   np.zeros_like(yaw)),axis=1)))[:,np.newaxis]
        pitch = np.arctan(np.einsum('ij,ij->i', bz, 
                    np.concatenate((
                    np.cos(yaw)/bz[:,2:3],
                    np.sin(yaw)/bz[:,2:3],
                    np.zeros_like(yaw)),axis=1)))[:,np.newaxis]       
        bx = np.concatenate((
            np.cos(yaw)*np.cos(pitch),
            np.sin(yaw)*np.cos(pitch),
            -np.sin(pitch)),axis=1)
        by = np.concatenate((
            -np.sin(yaw)*np.cos(roll)+np.cos(yaw)*np.sin(pitch)*np.sin(roll),
              np.cos(yaw)*np.cos(roll)+np.sin(yaw)*np.sin(pitch)*np.sin(roll),
              np.cos(pitch)*np.sin(roll)),axis=1)

        # dzhi & Omega
        dzhi = np.einsum('ijk,ij->ik',
                   np.concatenate((
                       -by[:,:,np.newaxis],
                       (bx/np.cos(roll))[:,:,np.newaxis],
                       np.zeros_like(by[:,:,np.newaxis])),axis=2),jer)/tau \
                +np.concatenate((np.sin(pitch),-np.cos(pitch)*np.tan(roll),np.ones_like(pitch)),axis=1)*dyaw
        S_inv = np.swapaxes(np.concatenate((
                np.concatenate((np.ones_like(pitch),np.zeros_like(pitch),-np.sin(pitch)),axis=1)[:,:,np.newaxis],
                np.concatenate((np.zeros_like(roll),np.cos(roll),np.cos(pitch)*np.sin(roll)),axis=1)[:,:,np.newaxis],
                np.concatenate((np.zeros_like(roll),-np.sin(roll),np.cos(pitch)*np.cos(roll)),axis=1)[:,:,np.newaxis]),axis=2),1,2)
        Omega = np.einsum('ijk,ik->ij',S_inv,dzhi)

        C_inv = np.swapaxes(np.concatenate((
                (-1*by/tau)[:,:,np.newaxis],
                (bx/np.cos(roll)/tau)[:,:,np.newaxis],
                bz[:,:,np.newaxis]),axis=2),1,2)
        d = np.concatenate((
            np.cos(yaw)*np.sin(roll)-np.cos(roll)*np.sin(yaw)*np.sin(pitch),
            np.sin(yaw)*np.sin(roll)+np.cos(roll)*np.cos(yaw)*np.sin(pitch),
            np.zeros_like(yaw)),axis=1)*tau
        dtau = np.einsum('ij,ij->i',bz,jer-d*dyaw)[:,np.newaxis]

        # ddzhi & dOmega
        dS = np.swapaxes(np.concatenate((
             np.concatenate((np.zeros_like(roll),np.cos(roll)*np.tan(pitch),-np.sin(roll)*np.tan(pitch)),axis=1)[:,:,np.newaxis],
             np.concatenate((np.zeros_like(roll),-np.sin(roll),-np.cos(roll)),axis=1)[:,:,np.newaxis],
             np.concatenate((np.zeros_like(roll),np.cos(roll)/np.cos(pitch),-np.sin(roll)/np.cos(pitch)),axis=1)[:,:,np.newaxis]
             ),axis=2),1,2)*(dzhi[:,0])[:,np.newaxis,np.newaxis] \
            +np.swapaxes(np.concatenate((
             np.concatenate((
                 np.zeros_like(roll),
                 np.sin(roll)/np.cos(pitch)/np.cos(pitch),
                 np.cos(roll)/np.cos(pitch)/np.cos(pitch)),axis=1)[:,:,np.newaxis],
             np.concatenate((np.zeros_like(roll),np.zeros_like(roll),np.zeros_like(roll)),axis=1)[:,:,np.newaxis],
             np.concatenate((
                 np.zeros_like(roll),
                 np.sin(roll)*np.tan(pitch)/np.cos(pitch),
                 np.cos(roll)*np.tan(pitch)/np.cos(pitch)),axis=1)[:,:,np.newaxis]
             ),axis=2),1,2)*(dzhi[:,1])[:,np.newaxis,np.newaxis]

        e = 2*dtau*np.einsum('ijk,ik->ij',
                np.concatenate((-by[:,:,np.newaxis],bx[:,:,np.newaxis],np.zeros_like(by[:,:,np.newaxis])),axis=2),Omega) \
            +tau*np.einsum('ijk,ik->ij',
                np.concatenate((bx[:,:,np.newaxis],by[:,:,np.newaxis],bz[:,:,np.newaxis]),axis=2),
                np.concatenate(((Omega[:,0]*Omega[:,2])[:,np.newaxis],
                               (Omega[:,1]*Omega[:,2])[:,np.newaxis],
                               (-Omega[:,0]*Omega[:,0]-Omega[:,1]*Omega[:,1])[:,np.newaxis]),axis=1)) \
            -tau*np.einsum('ijk,ik->ij',
                np.concatenate((-by[:,:,np.newaxis],bx[:,:,np.newaxis],np.zeros_like(by[:,:,np.newaxis])),axis=2),
                np.einsum('ijk,ik->ij',S_inv,np.einsum('ijk,ik->ij',dS,Omega)))

        ddzhi = np.einsum('ijk,ik->ij',C_inv,sna-ddyaw*d-e)
        ddzhi[:,2:] = ddyaw

        dOmega = -np.einsum('ijk,ik->ij',S_inv,np.einsum('ijk,ik->ij',dS,Omega)) \
                 +np.einsum('ijk,ik->ij',S_inv,ddzhi)
        
        I = np.einsum('ijk,ik->ij',np.repeat(self.J[np.newaxis,:,:],Omega.shape[0],0),Omega)
        Mu = np.einsum('ijk,ik->ij',np.repeat(self.J[np.newaxis,:,:],dOmega.shape[0],0),dOmega) \
            +np.concatenate((
                (Omega[:,1]*I[:,2]-Omega[:,2]*I[:,1])[:,np.newaxis],
                (Omega[:,2]*I[:,0]-Omega[:,0]*I[:,2])[:,np.newaxis],
                (Omega[:,0]*I[:,1]-Omega[:,1]*I[:,0])[:,np.newaxis]),axis=1)
        MT = np.zeros((Omega.shape[0],4))
        MT[:,:3] = Mu
        MT[:,3:] = Thrust

        Ws2 = np.einsum('ijk,ik->ij',np.repeat(self.G1_inv[np.newaxis,:,:],MT.shape[0],0),MT)
        Ws = np.copysign(np.sqrt(np.abs(Ws2)),Ws2)
        state = {
            'roll':roll,
            'pitch':pitch,
            'dzhi':dzhi,
            'ut':MT
        }

        return Ws, state

#     def getWs_vector_cupy(self, status):
#         pos = cp.array(status[:,0:3])
#         vel = cp.array(status[:,3:6])
#         acc = cp.array(status[:,6:9])
#         jer = cp.array(status[:,9:12])
#         sna = cp.array(status[:,12:15])
#         yaw = cp.array(status[:,15:16])
#         dyaw = cp.array(status[:,16:17])
#         ddyaw = cp.array(status[:,17:18])

#         # Total thrust
#         tau_v = acc - cp.array([0,0,self.gravity])
#         tau = -cp.linalg.norm(tau_v,axis=1)[:,cp.newaxis]
#         bz = tau_v/tau
#         Thrust = self.mass*tau
        
#         # roll & pitch
#         roll = cp.arcsin(cp.einsum('ij,ij->i', bz, 
#                    cp.concatenate((
#                    cp.sin(yaw),
#                    -cp.cos(yaw),
#                    cp.zeros_like(yaw)),axis=1)))[:,cp.newaxis]
#         pitch = cp.arctan(cp.einsum('ij,ij->i', bz, 
#                     cp.concatenate((
#                     cp.cos(yaw)/bz[:,2:3],
#                     cp.sin(yaw)/bz[:,2:3],
#                     cp.zeros_like(yaw)),axis=1)))[:,cp.newaxis]       
#         bx = cp.concatenate((
#             cp.cos(yaw)*cp.cos(pitch),
#             cp.sin(yaw)*cp.cos(pitch),
#             -cp.sin(pitch)),axis=1)
#         by = cp.concatenate((
#             -cp.sin(yaw)*cp.cos(roll)+cp.cos(yaw)*cp.sin(pitch)*cp.sin(roll),
#               cp.cos(yaw)*cp.cos(roll)+cp.sin(yaw)*cp.sin(pitch)*cp.sin(roll),
#               cp.cos(pitch)*cp.sin(roll)),axis=1)

#         # dzhi & Omega
#         dzhi = cp.einsum('ijk,ij->ik',
#                    cp.concatenate((
#                        -by[:,:,cp.newaxis],
#                        (bx/cp.cos(roll))[:,:,cp.newaxis],
#                        cp.zeros_like(by[:,:,cp.newaxis])),axis=2),jer)/tau \
#                 +cp.concatenate((cp.sin(pitch),-cp.cos(pitch)*cp.tan(roll),cp.ones_like(pitch)),axis=1)*dyaw
#         S_inv = cp.swapaxes(cp.concatenate((
#                 cp.concatenate((cp.ones_like(pitch),cp.zeros_like(pitch),-cp.sin(pitch)),axis=1)[:,:,cp.newaxis],
#                 cp.concatenate((cp.zeros_like(roll),cp.cos(roll),cp.cos(pitch)*cp.sin(roll)),axis=1)[:,:,cp.newaxis],
#                 cp.concatenate((cp.zeros_like(roll),-cp.sin(roll),cp.cos(pitch)*cp.cos(roll)),axis=1)[:,:,cp.newaxis]),axis=2),1,2)
#         Omega = cp.einsum('ijk,ik->ij',S_inv,dzhi)

#         C_inv = cp.swapaxes(cp.concatenate((
#                 (-1*by/tau)[:,:,cp.newaxis],
#                 (bx/cp.cos(roll)/tau)[:,:,cp.newaxis],
#                 bz[:,:,cp.newaxis]),axis=2),1,2)
#         d = cp.concatenate((
#             cp.cos(yaw)*cp.sin(roll)-cp.cos(roll)*cp.sin(yaw)*cp.sin(pitch),
#             cp.sin(yaw)*cp.sin(roll)+cp.cos(roll)*cp.cos(yaw)*cp.sin(pitch),
#             cp.zeros_like(yaw)),axis=1)*tau
#         dtau = cp.einsum('ij,ij->i',bz,jer-d*dyaw)[:,cp.newaxis]

#         # ddzhi & dOmega
#         dS = cp.swapaxes(cp.concatenate((
#              cp.concatenate((cp.zeros_like(roll),cp.cos(roll)*cp.tan(pitch),-cp.sin(roll)*cp.tan(pitch)),axis=1)[:,:,cp.newaxis],
#              cp.concatenate((cp.zeros_like(roll),-cp.sin(roll),-cp.cos(roll)),axis=1)[:,:,cp.newaxis],
#              cp.concatenate((cp.zeros_like(roll),cp.cos(roll)/cp.cos(pitch),-cp.sin(roll)/cp.cos(pitch)),axis=1)[:,:,cp.newaxis]
#              ),axis=2),1,2)*(dzhi[:,0])[:,cp.newaxis,cp.newaxis] \
#             +cp.swapaxes(cp.concatenate((
#              cp.concatenate((
#                  cp.zeros_like(roll),
#                  cp.sin(roll)/cp.cos(pitch)/cp.cos(pitch),
#                  cp.cos(roll)/cp.cos(pitch)/cp.cos(pitch)),axis=1)[:,:,cp.newaxis],
#              cp.concatenate((cp.zeros_like(roll),cp.zeros_like(roll),cp.zeros_like(roll)),axis=1)[:,:,cp.newaxis],
#              cp.concatenate((
#                  cp.zeros_like(roll),
#                  cp.sin(roll)*cp.tan(pitch)/cp.cos(pitch),
#                  cp.cos(roll)*cp.tan(pitch)/cp.cos(pitch)),axis=1)[:,:,cp.newaxis]
#              ),axis=2),1,2)*(dzhi[:,1])[:,cp.newaxis,cp.newaxis]

#         e = 2*dtau*cp.einsum('ijk,ik->ij',
#                 cp.concatenate((-by[:,:,cp.newaxis],bx[:,:,cp.newaxis],cp.zeros_like(by[:,:,cp.newaxis])),axis=2),Omega) \
#             +tau*cp.einsum('ijk,ik->ij',
#                 cp.concatenate((bx[:,:,cp.newaxis],by[:,:,cp.newaxis],bz[:,:,cp.newaxis]),axis=2),
#                 cp.concatenate(((Omega[:,0]*Omega[:,2])[:,cp.newaxis],
#                                (Omega[:,1]*Omega[:,2])[:,cp.newaxis],
#                                (-Omega[:,0]*Omega[:,0]-Omega[:,1]*Omega[:,1])[:,cp.newaxis]),axis=1)) \
#             -tau*cp.einsum('ijk,ik->ij',
#                 cp.concatenate((-by[:,:,cp.newaxis],bx[:,:,cp.newaxis],cp.zeros_like(by[:,:,cp.newaxis])),axis=2),
#                 cp.einsum('ijk,ik->ij',S_inv,cp.einsum('ijk,ik->ij',dS,Omega)))

#         ddzhi = cp.einsum('ijk,ik->ij',C_inv,sna-ddyaw*d-e)
#         ddzhi[:,2:] = ddyaw

#         dOmega = -cp.einsum('ijk,ik->ij',S_inv,cp.einsum('ijk,ik->ij',dS,Omega)) \
#                  +cp.einsum('ijk,ik->ij',S_inv,ddzhi)
        
#         I = cp.einsum('ijk,ik->ij',cp.repeat(self.J[cp.newaxis,:,:],Omega.shape[0],0),Omega)
#         Mu = cp.einsum('ijk,ik->ij',cp.repeat(self.J[cp.newaxis,:,:],dOmega.shape[0],0),dOmega) \
#             +cp.concatenate((
#                 (Omega[:,1]*I[:,2]-Omega[:,2]*I[:,1])[:,cp.newaxis],
#                 (Omega[:,2]*I[:,0]-Omega[:,0]*I[:,2])[:,cp.newaxis],
#                 (Omega[:,0]*I[:,1]-Omega[:,1]*I[:,0])[:,cp.newaxis]),axis=1)
#         MT = cp.zeros((Omega.shape[0],4))
#         MT[:,:3] = Mu
#         MT[:,3:] = Thrust

#         Ws2 = cp.einsum('ijk,ik->ij',cp.repeat(self.G1_inv[cp.newaxis,:,:],MT.shape[0],0),MT)
#         Ws = cp.copysign(cp.sqrt(cp.abs(Ws2)),Ws2)
#         state = {
#             'roll':roll,
#             'pitch':pitch,
#             'dzhi':dzhi,
#             'ut':MT
#         }

#         return Ws, state
    
    def getWs_vector_drag(self, status):
        pos = np.array(status[:,0:3])
        vel = np.array(status[:,3:6])
        acc = np.array(status[:,6:9])
        jer = np.array(status[:,9:12])
        sna = np.array(status[:,12:15])
        yaw = np.array(status[:,15:16])
        dyaw = np.array(status[:,16:17])
        ddyaw = np.array(status[:,17:18])

        # Total thrust
        tau_v = acc - np.array([0,0,self.gravity]) - self.dragCoef*np.linalg.norm(vel, axis=1, keepdims=True)*vel
        tau = -np.linalg.norm(tau_v,axis=1)[:,np.newaxis]
        bz = tau_v/tau
        Thrust = self.mass*tau
        
        # roll & pitch
        roll = np.arcsin(np.einsum('ij,ij->i', bz, 
                   np.concatenate((
                   np.sin(yaw),
                   -np.cos(yaw),
                   np.zeros_like(yaw)),axis=1)))[:,np.newaxis]
        pitch = np.arctan(np.einsum('ij,ij->i', bz, 
                    np.concatenate((
                    np.cos(yaw)/bz[:,2:3],
                    np.sin(yaw)/bz[:,2:3],
                    np.zeros_like(yaw)),axis=1)))[:,np.newaxis]       
        bx = np.concatenate((
            np.cos(yaw)*np.cos(pitch),
            np.sin(yaw)*np.cos(pitch),
            -np.sin(pitch)),axis=1)
        by = np.concatenate((
            -np.sin(yaw)*np.cos(roll)+np.cos(yaw)*np.sin(pitch)*np.sin(roll),
              np.cos(yaw)*np.cos(roll)+np.sin(yaw)*np.sin(pitch)*np.sin(roll),
              np.cos(pitch)*np.sin(roll)),axis=1)

        # dzhi & Omega
        dzhi = np.einsum('ijk,ij->ik',
                   np.concatenate((
                       -by[:,:,np.newaxis],
                       (bx/np.cos(roll))[:,:,np.newaxis],
                       np.zeros_like(by[:,:,np.newaxis])),axis=2),jer)/tau \
                +np.concatenate((np.sin(pitch),-np.cos(pitch)*np.tan(roll),np.ones_like(pitch)),axis=1)*dyaw
        S_inv = np.swapaxes(np.concatenate((
                np.concatenate((np.ones_like(pitch),np.zeros_like(pitch),-np.sin(pitch)),axis=1)[:,:,np.newaxis],
                np.concatenate((np.zeros_like(roll),np.cos(roll),np.cos(pitch)*np.sin(roll)),axis=1)[:,:,np.newaxis],
                np.concatenate((np.zeros_like(roll),-np.sin(roll),np.cos(pitch)*np.cos(roll)),axis=1)[:,:,np.newaxis]),axis=2),1,2)
        Omega = np.einsum('ijk,ik->ij',S_inv,dzhi)

        C_inv = np.swapaxes(np.concatenate((
                (-1*by/tau)[:,:,np.newaxis],
                (bx/np.cos(roll)/tau)[:,:,np.newaxis],
                bz[:,:,np.newaxis]),axis=2),1,2)
        d = np.concatenate((
            np.cos(yaw)*np.sin(roll)-np.cos(roll)*np.sin(yaw)*np.sin(pitch),
            np.sin(yaw)*np.sin(roll)+np.cos(roll)*np.cos(yaw)*np.sin(pitch),
            np.zeros_like(yaw)),axis=1)*tau
        dtau = np.einsum('ij,ij->i',bz,jer-d*dyaw)[:,np.newaxis]

        # ddzhi & dOmega
        dS = np.swapaxes(np.concatenate((
             np.concatenate((np.zeros_like(roll),np.cos(roll)*np.tan(pitch),-np.sin(roll)*np.tan(pitch)),axis=1)[:,:,np.newaxis],
             np.concatenate((np.zeros_like(roll),-np.sin(roll),-np.cos(roll)),axis=1)[:,:,np.newaxis],
             np.concatenate((np.zeros_like(roll),np.cos(roll)/np.cos(pitch),-np.sin(roll)/np.cos(pitch)),axis=1)[:,:,np.newaxis]
             ),axis=2),1,2)*(dzhi[:,0])[:,np.newaxis,np.newaxis] \
            +np.swapaxes(np.concatenate((
             np.concatenate((
                 np.zeros_like(roll),
                 np.sin(roll)/np.cos(pitch)/np.cos(pitch),
                 np.cos(roll)/np.cos(pitch)/np.cos(pitch)),axis=1)[:,:,np.newaxis],
             np.concatenate((np.zeros_like(roll),np.zeros_like(roll),np.zeros_like(roll)),axis=1)[:,:,np.newaxis],
             np.concatenate((
                 np.zeros_like(roll),
                 np.sin(roll)*np.tan(pitch)/np.cos(pitch),
                 np.cos(roll)*np.tan(pitch)/np.cos(pitch)),axis=1)[:,:,np.newaxis]
             ),axis=2),1,2)*(dzhi[:,1])[:,np.newaxis,np.newaxis]

        e = 2*dtau*np.einsum('ijk,ik->ij',
                np.concatenate((-by[:,:,np.newaxis],bx[:,:,np.newaxis],np.zeros_like(by[:,:,np.newaxis])),axis=2),Omega) \
            +tau*np.einsum('ijk,ik->ij',
                np.concatenate((bx[:,:,np.newaxis],by[:,:,np.newaxis],bz[:,:,np.newaxis]),axis=2),
                np.concatenate(((Omega[:,0]*Omega[:,2])[:,np.newaxis],
                               (Omega[:,1]*Omega[:,2])[:,np.newaxis],
                               (-Omega[:,0]*Omega[:,0]-Omega[:,1]*Omega[:,1])[:,np.newaxis]),axis=1)) \
            -tau*np.einsum('ijk,ik->ij',
                np.concatenate((-by[:,:,np.newaxis],bx[:,:,np.newaxis],np.zeros_like(by[:,:,np.newaxis])),axis=2),
                np.einsum('ijk,ik->ij',S_inv,np.einsum('ijk,ik->ij',dS,Omega)))

        ddzhi = np.einsum('ijk,ik->ij',C_inv,sna-ddyaw*d-e)
        ddzhi[:,2:] = ddyaw

        dOmega = -np.einsum('ijk,ik->ij',S_inv,np.einsum('ijk,ik->ij',dS,Omega)) \
                 +np.einsum('ijk,ik->ij',S_inv,ddzhi)
        
        I = np.einsum('ijk,ik->ij',np.repeat(self.J[np.newaxis,:,:],Omega.shape[0],0),Omega)
        Mu = np.einsum('ijk,ik->ij',np.repeat(self.J[np.newaxis,:,:],dOmega.shape[0],0),dOmega) \
            +np.concatenate((
                (Omega[:,1]*I[:,2]-Omega[:,2]*I[:,1])[:,np.newaxis],
                (Omega[:,2]*I[:,0]-Omega[:,0]*I[:,2])[:,np.newaxis],
                (Omega[:,0]*I[:,1]-Omega[:,1]*I[:,0])[:,np.newaxis]),axis=1)
        MT = np.zeros((Omega.shape[0],4))
        MT[:,:3] = Mu
        MT[:,3:] = Thrust

        Ws2 = np.einsum('ijk,ik->ij',np.repeat(self.G1_inv[np.newaxis,:,:],MT.shape[0],0),MT)
        Ws = np.copysign(np.sqrt(np.abs(Ws2)),Ws2)
        state = {
            'roll':roll,
            'pitch':pitch,
            'dzhi':dzhi,
            'ut':MT
        }

        return Ws, state

    def controlInput_check(self, ws, unused):
        del unused
        if np.any(ws < self.w_min) or np.any(ws > self.w_max):
            return False
        else:
            return True
        
#         def debug(status):
#             roll_ref = np.zeros((status.shape[0],1))
#             pitch_ref = np.zeros((status.shape[0],1))
#             bx_ref = np.zeros((status.shape[0],3))
#             by_ref = np.zeros((status.shape[0],3))
#             Omega_ref = np.zeros((status.shape[0],3))
#             dtau_ref = np.zeros((status.shape[0],3))
#             dS_ref = np.zeros((status.shape[0],3,3))
#             e_ref = np.zeros((status.shape[0],3))
#             ddzhi_ref = np.zeros((status.shape[0],3))
#             dOmega_ref = np.zeros((status.shape[0],3))
#             I_ref = np.zeros((status.shape[0],3))
#             Mu_ref = np.zeros((status.shape[0],3))
            
#             for i in range(status.shape[0]):
#                 pos = np.array(status[i,0:3])
#                 vel = np.array(status[i,3:6])
#                 acc = np.array(status[i,6:9])
#                 jer = np.array(status[i,9:12])
#                 sna = np.array(status[i,12:15])
#                 yaw = status[i,15]
#                 dyaw = status[i,16]
#                 ddyaw = status[i,17]
                
#                 # Total thrust
#                 tau_v = acc - [0,0,self.gravity]
#                 tau = -np.linalg.norm(tau_v)
#                 bz = tau_v/tau
#                 Thrust = self.mass*tau

#                 # roll & pitch
#                 roll = np.arcsin(np.dot(bz,[np.sin(yaw),-np.cos(yaw),0]))
#                 pitch = np.arctan(np.dot(bz,[np.cos(yaw),np.sin(yaw),0])/bz[2])
#                 bx = np.array([np.cos(yaw)*np.cos(pitch),np.sin(yaw)*np.cos(pitch),-np.sin(pitch)])
#                 by = np.array([-np.sin(yaw)*np.cos(roll)+np.cos(yaw)*np.sin(pitch)*np.sin(roll),\
#                       np.cos(yaw)*np.cos(roll)+np.sin(yaw)*np.sin(pitch)*np.sin(roll),\
#                       np.cos(pitch)*np.sin(roll)])
                
#                 # dzhi & Omega
#                 dzhi = np.dot(np.array([-1*by,bx/np.cos(roll),np.zeros(3)]),jer)/tau \
#                         +np.array([np.sin(pitch),-np.cos(pitch)*np.tan(roll),1])*dyaw
#                 S_inv = np.array([[1,0,-np.sin(pitch)],\
#                                   [0,np.cos(roll),np.cos(pitch)*np.sin(roll)],\
#                                   [0,-np.sin(roll),np.cos(pitch)*np.cos(roll)]])
#                 Omega = np.dot(S_inv,dzhi)

#                 C_inv = np.array([-1*by/tau,bx/np.cos(roll)/tau,bz])
#                 d = np.array([np.cos(yaw)*np.sin(roll)-np.cos(roll)*np.sin(yaw)*np.sin(pitch),\
#                              np.sin(yaw)*np.sin(roll)+np.cos(roll)*np.cos(yaw)*np.sin(pitch),0])*tau
#                 dtau = np.dot(bz,jer-dyaw*d)

#                 # ddzhi & dOmega
#                 dS = np.array([[0,np.cos(roll)*np.tan(pitch),-np.sin(roll)*np.tan(pitch)],\
#                             [0,-np.sin(roll),-np.cos(roll)],\
#                             [0,np.cos(roll)/np.cos(pitch),-np.sin(roll)/np.cos(pitch)]])*dzhi[0]\
#                     +np.array([[0,np.sin(roll)/np.cos(pitch)/np.cos(pitch),np.cos(roll)/np.cos(pitch)/np.cos(pitch)],\
#                             [0,0,0],\
#                             [0,np.sin(roll)*np.tan(pitch)/np.cos(pitch),np.cos(roll)*np.tan(pitch)/np.cos(pitch)]])*dzhi[1]

#                 e = 2*dtau*np.dot(np.array([-1*by,bx,0]).T,Omega)\
#                     +tau*np.dot(np.array([bx,by,bz]).T,np.array([Omega[0]*Omega[2],Omega[1]*Omega[2],-Omega[0]*Omega[0]-Omega[1]*Omega[1]]))\
#                     -tau*np.dot(np.array([-1*by,bx,0]).T,np.dot(S_inv,np.dot(dS,Omega)))

#                 ddzhi = np.dot(C_inv,sna-ddyaw*d-e)
#                 ddzhi[2] = ddyaw

#                 dOmega = -np.dot(S_inv,np.dot(dS,Omega))+np.dot(S_inv,ddzhi)
#                 Mu = np.dot(self.J,dOmega) + np.cross(Omega,np.dot(self.J,Omega))
                
#                 roll_ref[i,0] = roll
#                 pitch_ref[i,0] = pitch
#                 bx_ref[i,:] = bx
#                 by_ref[i,:] = by
#                 Omega_ref[i,:] = Omega
#                 dtau_ref[i,:] = dtau
#                 dS_ref[i,:,:] = dS
#                 e_ref[i,:] = e
#                 ddzhi_ref[i,:] = ddzhi
#                 dOmega_ref[i,:] = dOmega
#                 I_ref[i,:] = I
#                 Mu_ref[i,:] = Mu
                
#             return roll_ref, pitch_ref, bx_ref, by_ref, Omega_ref, dtau_ref, dS_ref, e_ref, ddzhi_ref, dOmega_ref, I_ref, Mu_ref
    
#         roll_ref, pitch_ref, bx_ref, by_ref, Omega_ref, dtau_ref, dS_ref, e_ref, ddzhi_ref, dOmega_ref, I_ref, Mu_ref = debug(status)
#         print(np.all(np.isclose(roll_ref, roll)))
#         print(np.all(np.isclose(pitch_ref, pitch)))
#         print(np.all(np.isclose(bx_ref, bx)))
#         print(np.all(np.isclose(by_ref, by)))
#         print(np.all(np.isclose(Omega_ref, Omega)))
#         print(np.all(np.isclose(dtau_ref, dtau)))
#         print(np.all(np.isclose(dS_ref, dS)))
#         print(np.all(np.isclose(e_ref, e)))
#         print(np.all(np.isclose(ddzhi_ref, ddzhi)))
#         print(np.all(np.isclose(dOmega_ref, dOmega)))
#         print(np.all(np.isclose(I_ref, I)))
#         print(np.all(np.isclose(Mu_ref, Mu)))