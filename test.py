import numpy as np
import math
import cv2
import glob
import os
import scipy
import numpy as np
import scipy
import cv2
from utils.helpers import Cam2World, World2Cam, Doppler_velocity, Pos2Vel, Pos_transform, build_matrix, bilinear_interpolate
import sys
import scipy.optimize as opt
from utils.SA import SimulatedAnnealingBase
from utils.plot import disturbance_trans, disturbance_rot

# pt_file = 'result/radar/1672812488_633140000.npy'
# pts = np.load(pt_file)
# contain = []
# for pt in pts:
#     if pt[4] < 0.001: # larger than 0.1 cm/s
#         c = [np.array([i]) for i in pt[:3]]
#         contain.append(c)
# contain = np.array(contain)
# print(contain)



class test():
    def __init__(self,x):
        self.x = x
        self.y = addition(self.x)

    def get_y(self):
        return self.y

    def add_x(self,a):
        self.x+=a

def addition(x):
    return x+1        

def load_camera_calib(cfg):
    config = np.load(cfg)
    k = config.reshape((3,3))
    return k

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
    print(theta)
    phi = theta*a

    J = np.sin(theta)/theta*np.eye(3) + (1-np.sin(theta)/theta)*a@a.T + (1-np.cos(theta))/theta*anti_sym_mat(a)
    rho = np.linalg.solve(J,t.T[0])
    rho = np.array([rho]).T
    return np.vstack((rho,phi))

if __name__=="__main__":

    # maximum = 0
    # minimum = 1e9
    # h,w = d_image.shape
    # for i in range(h):
    #     for j in range(w):
    #         if not np.isnan(d_image[i][j]):
    #             if d_image[i][j] > maximum:
    #                 maximum = d_image[i][j]
    #             if d_image[i][j] < minimum:
    #                 minimum = d_image[i][j]
    # show_img = 255-(d_image-minimum)/(maximum-minimum)*255
    # print(maximum,minimum)
    # for i in range(h):
    #     for j in range(w):
    #         if np.isnan(show_img[i][j]):
    #             show_img[i][j] = 0
    # show_ing = show_img.astype(np.int)
    # cv2.imshow("1",show_img)
    # cv2.waitKey(0)

    m = np.array([0,0,0,1,0.1,0.1,-0.1])
    m = build_matrix(m)
    x = Group2Alg(m)






    # images = glob.glob(os.path.join(path, '*.npy'))
    # images = sorted(images)
    # for img in images:
    #     print("begin processing", img)
    #     d_image = np.load(img)
    #     d_image = BFS_nan(d_image)
    #     np.save("result/complete_depth/"+img.split('/')[-1],d_image)

