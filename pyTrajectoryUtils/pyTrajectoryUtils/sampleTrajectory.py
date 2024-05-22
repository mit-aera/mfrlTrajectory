#!/usr/bin/env python
# coding: utf-8

import os, sys
import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class SampleTrajectory:
    def __init__(self, scale = 3):
        self.points_sample = [[scale,scale],
                             [-scale,scale],
                             [scale,-scale],
                             [-scale,-scale],
                             [0,0],
                             [0,scale],
                             [scale,0],
                             [0,-scale],
                             [-scale,0]]
#         print(self.points_sample)
        self.radius_min = 0.2*scale
        self.radius_max = 0.4*scale
        self.points_margin = 0.5
        self.MAX_CORNER = 6
        self.MIN_CORNER = 2
        self.scale = scale

        self.points_sample_dist = np.zeros((len(self.points_sample),len(self.points_sample)))
        for i in range(len(self.points_sample)):
            for j in range(len(self.points_sample)):
                self.points_sample_dist[i,j] = \
                    np.linalg.norm(np.array(self.points_sample[i])-np.array(self.points_sample[j]))

        return

    def get_tangent_points(self,s1,s2,r1,r2,x1,y1,x2,y2):
        # Angle of center of the circle
        alpha = np.arctan2(y2-y1,x2-x1)
        # Angle of tangent line
        beta = np.arcsin((-s1*r1+s2*r2)/np.linalg.norm((x2-x1,y2-y1)))

        x1_t = x1 - s1*r1*np.sin(alpha+beta)
        y1_t = y1 + s1*r1*np.cos(alpha+beta)
        x2_t = x2 - s2*r2*np.sin(alpha+beta)
        y2_t = y2 + s2*r2*np.cos(alpha+beta)

        return (x1_t,y1_t), (x2_t,y2_t)

    def get_trajectory(self, NUM_CORNER=3, z_max_base=0.3, FLAG_DEBUG=False):
        while True:
            idxs = np.random.choice(range(len(self.points_sample)), NUM_CORNER, replace=False)
            np.random.shuffle(idxs)
            radius = np.random.uniform(self.radius_min,self.radius_max,NUM_CORNER)

            flag_check_waypoints = True
            for i in range(NUM_CORNER):
                for j in range(i+1,NUM_CORNER):
                    if radius[i] + radius[j] + self.points_margin > self.points_sample_dist[idxs[i],idxs[j]]:
                        flag_check_waypoints = False
            if flag_check_waypoints:
                break

        signs = 2*np.random.randint(2, size=NUM_CORNER)-1

        circles = []
        lines = []

        waypoints = np.zeros((NUM_CORNER*3,3))
        for i in range(NUM_CORNER):
            i_next = (i+1)%NUM_CORNER
            x1, y1 = self.points_sample[idxs[i]]
            x2, y2 = self.points_sample[idxs[i_next]]
            s1 = signs[i]
            s2 = signs[i_next]
            r1, r2 = radius[i], radius[i_next]
            waypoints[3*i,:2], waypoints[3*i+1,:2] = self.get_tangent_points(s1,s2,r1,r2,x1,y1,x2,y2)

            circles.append(plt.Circle((x1, y1), r1, color='b', fill=False))
            lines.append(plt.Line2D(waypoints[3*i:3*i+2,0], waypoints[3*i:3*i+2,1], color='r'))
        
        for i in range(NUM_CORNER):
            i_next = (i+1)%NUM_CORNER
            x1, y1 = waypoints[3*i+1,:2]
            x2, y2 = waypoints[3*i_next,:2]
            s2 = signs[i_next]
            r2 = radius[i_next]
            c1, c2 = self.points_sample[idxs[i_next]]

            x_tmp = -s2*(y2-y1)
            y_tmp = s2*(x2-x1)
            scale = r2/np.linalg.norm(np.array([x_tmp,y_tmp]))

            waypoints[3*i+2,:2] = [x_tmp*scale+c1,y_tmp*scale+c2]

        z = 0
        z_max = z_max_base*self.scale
        for i in range(NUM_CORNER*3):
            z += (2*np.random.rand()-1)*z_max
            waypoints[i,2] = z
            
        if FLAG_DEBUG:
            fig = plt.figure(figsize=(5,5))
            ax = fig.add_subplot(111)
            for i in range(NUM_CORNER):
                ax.add_artist(circles[i])
                ax.add_artist(lines[i])
            endp = waypoints[:,:2]
            ax.scatter(*zip(*endp), marker='o', color='r')
#             ax.set_xlim(-5, 5)
#             ax.set_ylim(-5, 5)
#             ax.legend()
            
            fig = plt.figure(figsize=(20,20))
            plot_lim = np.max(np.abs(waypoints))*1.2
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(waypoints[:,0], waypoints[:,1], waypoints[:,2], label='trajectory_t')
            ax.set_xlim(-plot_lim, plot_lim)
            ax.set_ylim(-plot_lim, plot_lim)
            ax.set_zlim(-plot_lim, plot_lim)
#             ax.legend()
            
            plt.show()
        
        return waypoints

if __name__ == "__main__":
    _trainingTraj = SampleTrajectory(scale = 1)
    waypoints = _trainingTraj.get_trajectory(NUM_CORNER=4, z_max_base=0.1, FLAG_DEBUG=True)
    
    for i in range(waypoints.shape[0]):
        print("[{}, {}, {}],".format(waypoints[i,0],waypoints[i,1],waypoints[i,2]))