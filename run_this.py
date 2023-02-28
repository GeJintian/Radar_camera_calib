import sys
import os
from PIL import Image
import glob
import numpy as np
import argparse
import json

import torch
import cv2

from utils.raft_core.raft import RAFT
from utils.raft_core.utils import InputPadder
from utils.helpers import *
from utils.visualize import seg_mask, viz_optical, viz_mask, print_minmax, viz_pts
from coarse_opt import coarse_optimize, single_projection_problem, batch_projection_problem
from fine_opt import optical_field, fine_optimize,fine_sa
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error


DEVICE = 'cuda'


def load_depth(dfile):
    gt = cv2.imread(dfile,flags = cv2.IMREAD_COLOR)
    gt = cv2.cvtColor(gt,cv2.COLOR_BGRA2RGB)
    gt = np.array(gt)
    gray_depth = ((gt[:,:,0] + gt[:,:,1] * 256.0 + gt[:,:,2] * 256.0 * 256.0)/((256.0 * 256.0 * 256.0) - 1))
    gt = gray_depth * 1000
    # d_image = BFS_nan(d_image)
    # h,w = d_image.shape
    # for i in range(h):
    #     for j in range(w):
    #         if math.isnan(d_image[i][j]):
    #             print("encounter nan")
    return gt

def load_RGB(imfile):
    img = cv2.imread(imfile,flags = cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img,cv2.COLOR_BGRA2RGB)
    img = np.array(img)[:,:,:3]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def load_points(ptfile):
    all_points = np.load(ptfile)
    moving_points = []
    velocity = []
    for pt in all_points:
        if abs(pt[3]) > 0.001: # larger than 0.1 cm/s
            phi = -pt[0]
            theta = math.pi/2-pt[1]
            r = pt[2]
            x = r*math.sin(theta)*math.cos(phi)
            y = r*math.sin(phi)*math.sin(theta)
            z = r*math.cos(theta)
            c = [np.array([x]),np.array([y]),np.array([z])]
            c.append(np.array([1]))
            c=np.array(c)
            moving_points.append(c)
            velocity.append(pt[3])

    return moving_points, velocity

def load_camera_calib(cfg):
    config = np.load(cfg)
    k = config.reshape((3,3))
    return k

def mask(rgb,edge):
    bool_edge = edge/255
    width, height = bool_edge.shape
    for i in range(width):
        for j in range(height):
            if bool_edge[i][j]>0.5: #edge = 255
                rgb[i][j] = [255,255,255]

def load_seg(seg_file):
    img = np.array(Image.open(seg_file)).astype(np.uint8)
    h,w,_ = img.shape
    result = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            if img[i][j][2] == 142 and img[i][j][0] == 0:
                result[i][j] = 12
    return result

def load_opt(opt_file):
    result = np.load(opt_file)
    return result
                
def quaternion_to_euler_angle(w, x, y, z):
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = math.degrees(math.atan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.degrees(math.asin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = math.degrees(math.atan2(t3, t4))

    return X, Y, Z

def batch_opt(image_path, point_path,depth_path,opt_path, camera_calib_file, segment_cfg, segment_ckpts, M_t_init, seg_path,raft_ckpts,args):
    
    K = load_camera_calib(camera_calib_file)

    with torch.no_grad():
        images = glob.glob(os.path.join(image_path, '*.png')) + \
                 glob.glob(os.path.join(image_path, '*.jpg'))
        images = sorted(images)[:100]
        #print(images)

        count = 0
        problem_sets = []
        img_names = [] # store (im0,im1) for depth and raft

        # coarse optimize
        for i in range(len(images)-1):
            imfile = images[i]
            
            ptfile = os.path.join(point_path,imfile.split('/')[-1].split('.')[0]+'.npy')
            depth_map = load_depth(os.path.join(depth_path,imfile.split('/')[-1].split('.')[0]+'.png'))
            P_r, V_r = load_points(ptfile)
            if len(P_r)==0:
                #print("In this frame, there is no moving items")
                count = count + 1
                continue
            img_names.append((images[i], images[i+1]))
            seg_result = load_seg(os.path.join(seg_path,imfile.split('/')[-1].split('.')[0]+'.png'))
            remasking = remask(seg_result,12) # 12 is the idx of person
            new_mask = BFS_mask(remasking)
            #viz_pts(new_mask,imfile,P_r,M_t_init,K)
            problem = single_projection_problem(K, new_mask, depth_map, P_r, V_r, imfile.split('/')[-1])
            problem_sets.append(problem)
        
        problems = batch_projection_problem(problem_sets)
        X=[]
        Y=[]
        Z=[]
        T_X=[]
        T_Y=[]
        T_Z=[]
        gx = [0]*40
        gy = [0]*40
        gz = [-5]*40
        gtx = [0.1]*40
        gty = [0.1]*40
        gtz = [-0.1]*40
        for i in range(40):
            M_t_init = [0,0,0.00001,0.99999,8/100,8/100,-8/100]

            #print("In the beginning, the score is", problems.objective_function(M_t_init))
            M_t = coarse_optimize(M_t_init, problems)
            M_t_init = M_t
            x,y,z,w = M_t_init[:4]
            mag = np.sqrt(x*x+y*y+z*z+w*w)
            M_t_init[:4] = [x,y,z,w]/mag
            #print("Finish coarse optimize")
            #print('Coarse opt result is',M_t_init)
            qx,qy,qz,qw = M_t_init[:4]
            x,y,z = quaternion_to_euler_angle(qw,qx,qy,qz)
            t_x,t_y,t_z = M_t_init[4:]
            X.append(x)
            Y.append(y)
            Z.append(z)
            T_X.append(t_x)
            T_Y.append(t_y)
            T_Z.append(t_z)
        np.sqrt(mean_squared_error(X,gx))
        print("rmse: X = {}, Y = {}, Z={}, t_x = {}, t_y = {}, t_z = {}".format(
        np.sqrt(mean_squared_error(X,gx)),
        np.sqrt(mean_squared_error(Y,gy)),
        np.sqrt(mean_squared_error(Z,gz)),
        np.sqrt(mean_squared_error(T_X,gtx)),
        np.sqrt(mean_squared_error(T_Y,gty)),
        np.sqrt(mean_squared_error(T_Z,gtz))
        ))

        #print('Euler angles are phi = {}, theta = {}, psi = {}'.format(X,Y,Z))


        # # fine optimize
        # idx = problems.update(M_t_init)
        # model = torch.nn.DataParallel(RAFT(args))
        # model.load_state_dict(torch.load(raft_ckpts))

        # model = model.module
        # model.to(DEVICE)
        # model.eval()
        # for id in idx:
        #     img_names[id] = None

        # fields = []
        # P_rs = []
        # V_rs = []
        # print("Prepare for fine opt")
        # for idx in range(len(img_names)-1):
        #     imgs = img_names[idx]
        #     if imgs is not None:
        #         im1 = os.path.join(image_path,imgs[0].split('/')[-1])
        #         im2 = os.path.join(image_path,imgs[1].split('/')[-1])
        #         image1 = load_RGB(im1)
        #         image2 = load_RGB(im2)
        #         # padder = InputPadder(image1.shape)
        #         # image1, image2 = padder.pad(image1, image2)
        #         # _, flow_up = model(image1, image2, iters=20, test_mode=True)
        #         # flow_up = flow_up[0].cpu()
        #         flow_up = load_opt(os.path.join(opt_path,imgs[1].split('/')[-1].split('.')[0]+'.npy'))
        #         image1 = load_depth(os.path.join(depth_path,imgs[0].split('/')[-1].split('.')[0]+'.png'))
        #         image2 = load_depth(os.path.join(depth_path,imgs[1].split('/')[-1].split('.')[0]+'.png'))
        #         field = optical_field(flow_up,image1, image2, problem_sets[idx+1].mask, problem_sets[idx+1].centroid, K, 0.05, 0.01, 0.01, 0.01)
        #         P_r = problem_sets[idx].P_r
        #         V_r = problem_sets[idx].V_r
        #         for p in range(len(P_r)):
        #             pr = P_r[p]
        #             P_rs.append(pr)
        #             V_rs.append(-V_r[p]*pr/np.linalg.norm(pr))
        #             fields.append(field)
        # print("Begin fine optimization")
        # M_t_init = fine_optimize(M_t_init, P_rs, V_rs, fields)
        # print(M_t_init)
        # rot = M_t_init[:3][:3]
        # qw = np.sqrt(1+rot[0][0]+rot[1][1]+rot[2][2])/2
        # qx = (rot[2][1]-rot[1][2])/(4*qw)
        # qy = (rot[0][2]-rot[2][0])/(4*qw)
        # qz = (rot[1][0]-rot[0][1])/(4*qw)
        # print('quaternions are', [qx,qy,qz,qw])
        # X,Y,Z = quaternion_to_euler_angle(qw,qx,qy,qz)
        # print('Euler angles are phi = {}, theta = {}, psi = {}'.format(X,Y,Z))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    point_path = 'result/Carla/test/radar'
    camera_calib_file = 'result/Carla/calibration.npy'
    depth_path = 'result/Carla/test/depth'

    image_path = 'result/Carla/test/rgb'
    seg_path = 'result/Carla/test/seg'
    opt_path = 'result/Carla/test/opt'
    segment_cfg = '/mnt/e/NTU/mmlab/mmsegmentation//configs/segformer/segformer_mit-b2_512x512_160k_ade20k.py'
    segment_ckpts = 'models/b2.pth'
    raft_ckpts = 'models/raft-things.pth'
    #M_t_init = [0,0,0,0,-3/100,-5.9/100,8.75/100]
    M_t_init = [0,0,0.00001,0.99999,10/100,10/100,-10/100]

    #single_opt(image_path, point_path,depth_path, camera_calib_file, segment_cfg, segment_ckpts, M_t_init, alignment)

    batch_opt(image_path, point_path,depth_path,opt_path, camera_calib_file, segment_cfg, segment_ckpts, M_t_init, seg_path, raft_ckpts,args)
