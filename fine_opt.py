import numpy as np
import scipy
import cv2
from utils.helpers import Cam2World, World2Cam, Doppler_velocity, Pos2Vel, Pos_transform, build_matrix, bilinear_interpolate
import sys
import scipy.optimize as opt


class optical_field():
    def __init__(self, optical_map, depth_map0, depth_map1, K, dt, dp, dq, dr) -> None:
        # dt is the time interval between each image
        self.optical_u, self.optical_v = optical_map.numpy()
        self.depth_map0 = depth_map0
        self.depth_map1 = depth_map1
        self.dt = dt
        self.k = K
        self.dp = [dp,dq,dr] # For x,y,z
    
    def get_next_opt(self, u, v):
        du = bilinear_interpolate(self.optical_u, u, v)
        dv = bilinear_interpolate(self.optical_v, u, v)
        return (u+du, v+dv) #TODO: Check, for raft, is the first output for u-axis?

    def get_dep0(self, u, v):
        return bilinear_interpolate(self.depth_map0, u, v)

    def get_dep1(self, u, v):
        return bilinear_interpolate(self.depth_map1, u, v)

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
    def central_difference(self, P):
        """
        Result should be [4*4] ([3x3]+stack)
        """
        result = np.zeros((4,4))
        for j in range(3):
            for i in range(3):
                fdm = self.fdm(P,self.dp[j],j)
                result[i][j] = fdm[i][0]
        return np.array(result)
    
    def fdm(self, p, dp, axis):
        """
        finite difference method.
        dp should be one-dimension, while axis indicates the axis
        return (func(p+dp)-func(p-dp))/2dp
        """
        p[axis] += dp
        f0 = self.get_velocity(p)
        p[axis] -= 2*dp
        f1 = self.get_velocity(p)
        return (f0-f1)/(2*dp)

def anti_sym_mat(phi):
    """
    compute anti-symmetric matrix of the input vector. Vector should be [3x1]
    """
    p1,p2,p3 = phi
    anti_sym_mat = np.array([[0,-p3[0],p2[0]],[p3[0],0,-p1[0]],[-p2[0],p1[0],0]])
    return anti_sym_mat

def Group2Alg(T):
    """
    Compute Lie Group SE(3) to Lie Algebra se(3)

    T: [4x4] transformation matrix
    return: [6x1] Lie algebra of T
    """
    R = T[:3,:3]
    t = T[:3,3]
    t = np.array([t]).T
    theta = np.arccos((np.trace(R)-1)/2)
    eigval,eigvec = scipy.linalg.eig(R,right = True)
    for i in range(len(eigval)):
        if abs(eigval[i] - 1) < 1e-9:
            break
    a = np.real(eigvec[i]) #should be 1-dim
    a = np.array([a]).T # transform to nx1
    phi = theta*a

    J = np.sin(theta)/theta*np.eye(3) + (1-np.sin(theta)/theta)*a@a.T + (1-np.cos(theta))/theta*anti_sym_mat(a)
    rho = np.linalg.solve(J,t.T[0])
    rho = np.array([rho]).T
    return np.vstack((rho,phi))

def Alg2Group(xi):
    """
    Compute Lie Algebra se(3) to Lie Group SE(3)

    xi: [6x1] Lie alg in se(3)
    return: [4x4] Lie group of xi in SE(3)
    """
    phi = xi[3:]
    rho = xi[:3]
    theta = np.linalg.norm(phi)
    a = phi/theta
    J = np.sin(theta)/theta*np.eye(3) + (1-np.sin(theta)/theta)*a@a.T + (1-np.cos(theta))/theta*anti_sym_mat(a)
    R = np.cos(theta)*np.eye(3) + (1-np.cos(theta))*a@a.T + np.sin(theta)*anti_sym_mat(a)
    result = np.hstack((R,J@rho))
    result = np.vstack((result,np.array([0,0,0,1])))
    return result

def Ji(cVc, T, Vr):
    """
    return: compute Ji, which is the cost function for one single point
    """
    cVr = T@Vr
    alpha = cVc.T@cVr # should be scalar
    beta = cVr.T@cVr # should be scalar
    return (alpha/beta-1)*cVr

def J(cVcs, T, Vrs):
    """return: the final cost function J = sum(Ji^T*Ji)"""
    loss = 0
    for i in range(cVcs):
        ji = Ji(cVcs[i], T, Vrs[i])
        loss += ji.T@ji
    return loss

def dJi(Pr:np.ndarray, T:np.ndarray, Vr:np.ndarray, field:optical_field)->np.ndarray:
    """differentiate of Ji"""
    cVr = T@Vr
    cVc = field.get_velocity(T@Pr)
    alpha = cVc.T@cVr # should be scalar
    beta = cVr.T@cVr # should be scalar
    gamma = alpha-beta
    result = gamma * dLieAlg(T,Vr) + cVr@dgamma(T, Vr, Pr, field)
    return result

def dLieAlg(T,P):
    """The input P could be a position or a velocity"""
    p = P[:3] # inhomogeneous format
    R = T[:3,:3]
    t = T[:3,3]
    t = np.array([t]).T
    Liealg = R@p + t
    asm = -anti_sym_mat(Liealg)
    result = np.hstack((np.diag([1,1,1]),asm))
    result = np.vstack((result,np.zeros(6)))
    return result

def dalpha(T, Vr, Pr, field:optical_field):
    """Derivative of alpha"""
    cPc = T@Pr
    cVr = T@Vr
    cVc = field.get_velocity(cPc)
    result = cVr.T@(field.central_difference(cPc)@dLieAlg(T,Pr)) + cVc.T@dLieAlg(T,Vr)
    return result

def dbeta(T,Vr):
    """Derivative of beta"""#TODO: Modify this
    cVr = np.dot(T,Vr)
    dliealg = dLieAlg(T, Vr)
    result = -2*cVr.T/((cVr.T@cVr)**2)@dliealg
    return result

def dgamma(T, Vr, Pr, field:optical_field):
    """Derivative of alpha-beta"""
    cPc = T@Pr
    cVr = T@Vr
    cVc = field.get_velocity(cPc)
    alpha = cVc.T@cVr # should be scalar
    beta = cVr.T@cVr # should be scalar
    result = dalpha(T,Vr, Pr, field)*beta - alpha*dbeta(T,Vr)
    return result

# def dcVc(T, Pr, field:optical_field):
#     """Derivative of cVc. dcVc = dopt*dLieAlg(T,Pr) """
#     Dpr = dLieAlg(T, Pr)
#     result = dopt()*Dpr
#     return result


# def dopt(cPc,field:optical_field):
#     """
#     Derivative of optical flow velocity. Using central difference.
#     Instead of directly compute the FDM of dv/dp, it is better to transfer 3D position into a 2D camera coordinate, and then apply 2D FDM
#     """
#     #TODO: compute a 2-dimension central difference + chain rule
#     result = np.dot(field.central_difference(),dP_cam())
#     return result

# def dP_cam():
#     #TODO: compute the derivative of P_cam against P_world
#     return

def objective_func(x, position, fields, Vrs):
    """
    compute fun for optimization algorithms
    here, if there are n points in one image, fun returns [3nx1].
    x: [6x1] Lie Alg
    """
    x = np.array([x]).transpose()
    T = Alg2Group(x)
    cVcs = []
    for p in range(len(position)):
        v = fields[p].get_velocity(T@position[p])
        cVcs.append(v)
    result = Ji(cVcs[0], T, Vrs[0])
    for i in range(1,len(cVcs)):
        ji = Ji(cVcs[i], T, Vrs[i])
        result = np.vstack((result,ji))

    return result.transpose()[0]

def derivative(x, position, fields, Vrs):
    """
    compute the derivative for optimization algorithms
    x: optimization variaty, which is [6x1] Lie alg
    field: optical field for that x
    return: [6]
    """
    x = np.array([x]).transpose()
    T = Alg2Group(x)
    result = dJi(position[0],T,Vrs[0],fields[0])
    for i in range(1,len(fields)):
        dj = dJi(position[i],T,Vrs[i],fields[i])
        result = np.vstack((result,dj))
    result = np.array(result)
    return result

def gn(f, jac, x0, fields, Vrs, position, max_iter=1000, tol=1e-6):
    """
    Gauss Newton for optimization
    """
    x = x0
    for i in range(max_iter):
        fx = f(x, position, fields, Vrs)
        Jx = jac(x, position, fields, Vrs)
        dx = np.linalg.solve(Jx.T @ Jx, -Jx.T @ fx)
        x += dx
        #print(dx)
        if np.linalg.norm(dx) < tol:
            break
    return x

def least_squares(f, jac, x0, fields, Vrs, position, method = 'lm'):
    """
    Levenberg-Marquardt, dogleg and trf for optimization. Use scipy implementation.
    method could be {'trf','lm','dogbox'}
    """
    kwargs = {"position":position, "fields": fields, "Vrs":Vrs}
    print(callable(f))
    print(callable(jac))    
    x0 = x0.transpose()[0]
    print(x0)
    
    x = opt.least_squares(fun = f, jac=jac, x0=x0, method = method, kwargs = kwargs)
    return x

def steepest_descent():
    #TODO: Implement it if still have time
    return

def fine_optimize(M_init, Prs, Vrs, fields):
    """fine opt is implemented in a gradient descent/steepest ascent manner"""
    M_init = build_matrix(M_init)
    x0 = Group2Alg(M_init)

    res_log = least_squares(objective_func, derivative, x0, fields, Vrs, Prs)
    x=res_log.x
    print(res_log)
    #x = np.array([x]).transpose()
    #return Alg2Group(x)