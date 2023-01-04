import numpy as np
import cv2
from utils.helpers import Cam2World, World2Cam, Radar_velocity, Pos2Vel, Pos_transform


class optical_field():
    def __init__(self, optical_map, depth_map, dt, focal,cx,cy) -> None:
        # dt is the time interval between each image
        self.optical_map = optical_map
        self.depth_map = depth_map
        self.dt = dt
        self.f = focal
        self.cx = cx
        self.cy = cy
    
    def get_next_opt(self, u, v):
        return (u+round(self.optical_map[u][v][0]), v+round(self.optical_map[u][v][1]))

    def get_dep(self, u, v):
        return self.depth_map[round(u)][round(v)]

    def get_velocity(self,u,v):
        # return: [4x1]
        u1,v1 = self.get_next_opt(u,v)
        d0 = self.get_dep(u,v)
        d1 = self.get_dep(u1,v1)
        
        vx = -(d1-d0)/self.dt
        vy = (d1*(v1-self.cy)-d0*(v-self.cy))/(self.dt*self.f)
        vz = -(d1*(u1-self.cx)-d0*(u-self.cx))/(self.dt*self.f)

        return np.array([[vx],[vy],[vz],[0]])

def Ji(cVc, cVr):
    """return: compute Ji, which is the cost function for one single point"""
    alpha = np.dot(np.transpose(cVc),cVr) # should be scalar
    beta = np.dot(np.transpose(cVr),cVr) # should be scalar
    return (alpha-beta)*cVr

def J():
    """return: the final cost function J = sum(Ji^T*Ji)"""
    return

def dJ():
    """differentiate of J"""
    return

def dLieAlg(T,P):
    """The input P could be a position or a velocity"""
    return

def dalpha(cVr,cVc):
    return

def dbeta():
    
    return

def fine_optimize():
    """fine opt is implemented in a gradient descent manner"""
    return