import numpy as np
from numpy import dot, transpose
import cv2
from utils.helpers import Cam2World, World2Cam, Doppler_velocity, Pos2Vel, Pos_transform


class optical_field():
    def __init__(self, optical_map, depth_map0, depth_map1, dt, K) -> None:
        # dt is the time interval between each image
        self.optical_map = optical_map
        self.depth_map0 = depth_map0
        self.depth_map1 = depth_map1
        self.dt = dt
        self.k = K
    
    def get_next_opt(self, u, v):
        return (u+np.round(self.optical_map[v][u][0]), v+np.round(self.optical_map[v][u][1])) #TODO: Check, is the first output for u-axis?

    def get_dep0(self, u, v):
        return self.depth_map0[np.round(v)][np.round(u)]

    def get_dep1(self, u, v):
        return self.depth_map1[np.round(v)][np.round(u)]

    def get_velocity_uv(self,u,v):
        # return: [4x1]
        u1,v1 = self.get_next_opt(u,v)
        d0 = self.get_dep0(u,v)
        d1 = self.get_dep1(u1,v1)
        
        p0 = Cam2World(self.k, np.array([[u],[v],[d0],[1]]))
        p1 = Cam2World(self.k, np.array([[u1],[v1],[d1],[1]]))
        vel = Pos2Vel(p0,p1,self.dt)
        return vel
    
    def get_velocity(self, P):
        u,v = World2Cam(self.k, P)
        return self.get_velocity_uv(u[0],v[0])

    #We might need bilinear interpolation and central difference for optical field
    def central_difference(self, u, v):
        """
        Result should be [3*2]
        """

        return

def Ji(cVc, T, Vr):
    """return: compute Ji, which is the cost function for one single point"""
    cVr = dot(T,Vr)
    alpha = dot(transpose(cVc),cVr) # should be scalar
    beta = dot(transpose(cVr),cVr) # should be scalar
    return (alpha-beta)*cVr

def J(cVcs, T, Vrs):
    """return: the final cost function J = sum(Ji^T*Ji)"""
    loss = 0
    for i in range(cVcs):
        ji = Ji(cVcs[i], T, Vrs[i])
        loss += dot(transpose(ji),ji)
    return loss

def dJi(cVc, T, Vr):
    """differentiate of Ji"""
    cVr = dot(T,Vr)
    alpha = dot(transpose(cVc),cVr) # should be scalar
    beta = dot(transpose(cVr),cVr) # should be scalar
    gamma = alpha-beta
    result = gamma * dLieAlg(T,Vr) + dot(cVr,dgamma(T, Vr))#TODO: pass in arguments for dgamma
    return result

def dLieAlg(T,P):
    """The input P could be a position or a velocity"""
    p = P[:3] # inhomogeneous format
    R = T[:3,:3]
    t = T[:3,3]
    t = transpose(np.array([t]))
    Liealg = dot(R,p) + t
    p1,p2,p3 = transpose(Liealg)[0]
    anti_sym_mat = -np.array([[0,-p3,p2],[p3,0,-p1],[-p2,p1,0]])
    result = np.hstack((np.diag([1,1,1]),anti_sym_mat))
    result = np.vstack((result,np.zeros(6)))
    return result

def dalpha(T, Vr, cVc):#TODO: add more arguments for dopt
    """Derivative of alpha"""
    cVr = dot(T,Vr)
    result = dot(transpose(cVr),dcVc()) + dot(transpose(cVc),dLieAlg(T,Vr)) #TODO:finish dcVc

    return result

def dbeta(T,Vr):
    """Derivative of beta"""
    cVrT = transpose(dot(T,Vr))
    dliealg = dLieAlg(T, Vr)
    result = 2*dot(cVrT,dliealg)
    return result

def dgamma(T, Vr, cVc):
    """Derivative of alpha-beta"""
    result = dalpha(T,Vr, cVc) - dbeta(T,Vr) #TODO:add more arguments
    return result

def dopt(field):
    """
    Derivative of optical flow velocity. Using central difference.
    Instead of directly compute the DFM of dv/dp, it is better to transfer 3D position into a 2D camera coordinate, and then apply 2D DFM
    """
    #TODO: compute a 2-dimension central difference + chain rule
    result = dot(field.central_difference(),dP_cam())
    return result

def dP_cam():
    #TODO: compute the derivative of P_cam against P_world
    return

def dcVc(T, Pr):
    """Derivative of cVc. dcVc = dopt*dLieAlg(T,Pr) """
    Dpr = dLieAlg(T, Pr)
    result = dopt()*Dpr
    return result

def fine_optimize(M_init, P_r, K, optical_map, depth_map0, depth_map1, dt):
    """fine opt is implemented in a gradient descent/steepest ascent manner"""
    field = optical_field(optical_map, depth_map0, depth_map1, dt, K)
    for i in P_r:
        p_c = Pos_transform(M_init,i)
        u,v = World2Cam(K, p_c)


    return