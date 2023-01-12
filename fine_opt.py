import numpy as np
import cv2
from utils.helpers import Cam2World, World2Cam, Doppler_velocity, Pos2Vel, Pos_transform


class optical_field():
    def __init__(self, optical_map, depth_map0, depth_map1, dt, focal,cu,cv) -> None:
        # dt is the time interval between each image
        self.optical_map = optical_map
        self.depth_map0 = depth_map0
        self.depth_map1 = depth_map1
        self.dt = dt
        self.f = focal
        self.cu = cu
        self.cv = cv
    
    def get_next_opt(self, u, v):
        return (u+np.round(self.optical_map[u][v][0]), v+np.round(self.optical_map[u][v][1]))

    def get_dep0(self, u, v):
        return self.depth_map0[np.round(u)][np.round(v)]

    def get_dep1(self, u, v):
        return self.depth_map1[np.round(u)][np.round(v)]

    def get_velocity(self,u,v):
        # return: [4x1]
        u1,v1 = self.get_next_opt(u,v)
        d0 = self.get_dep0(u,v)
        d1 = self.get_dep1(u1,v1)
        
        vx = (d1-d0)/self.dt
        vy = (d1*(self.cv-v1)-d0*(self.cv-v))/(self.dt*self.f)
        vz = (d1*(self.cu-u1)-d0*(self.cu-u))/(self.dt*self.f)

        return np.array([[vx],[vy],[vz],[0]])

def Ji(cVc, cVr):
    """return: compute Ji, which is the cost function for one single point"""
    alpha = np.dot(np.transpose(cVc),cVr) # should be scalar
    beta = np.dot(np.transpose(cVr),cVr) # should be scalar
    return (alpha-beta)*cVr

def J(cVcs, cVrs):
    """return: the final cost function J = sum(Ji^T*Ji)"""
    loss = 0
    for i in range(cVcs):
        loss += Ji(cVcs[i],cVrs[i])
    return loss

def dJ():
    """differentiate of J"""
    
    return

def dLieAlg(T,P):
    """The input P could be a position or a velocity"""

    return

def dalpha(cVr,cVc):
    """Derivative of alpha"""
    
    return

def dbeta():
    """Derivative of beta"""
    
    return

def dgamma():
    """Derivative of alpha+beta"""

    return

def dopt():
    """Derivative of cVc. Using central difference"""

    return

def fine_optimize():
    """fine opt is implemented in a gradient descent manner"""

    return