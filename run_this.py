import sys
import os
from PIL import Image
import glob
import numpy as np
import argparse
import json

import torch
from mmseg.apis import inference_segmentor, init_segmentor
import cv2

from utils.raft_core.raft import RAFT
from utils.raft_core.utils import InputPadder
from utils.helpers import *
from utils.visualize import seg_mask, viz_optical, viz_mask, print_minmax, viz_pts
from coarse_opt import coarse_optimize, single_projection_problem, batch_projection_problem
from fine_opt import optical_field, fine_optimize, fine_sa, Alg2Group


DEVICE = 'cuda'


def load_depth(dfile):
    d_image = np.load(dfile)
    # d_image = BFS_nan(d_image)
    # h,w = d_image.shape
    # for i in range(h):
    #     for j in range(w):
    #         if math.isnan(d_image[i][j]):
    #             print("encounter nan")
    return d_image

def load_RGB(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8) # read left image on RGB format
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def load_points(ptfile):
    all_points = np.load(ptfile)
    moving_points = []
    velocity = []
    for pt in all_points:
        if abs(pt[4]) > 0.001: # larger than 0.1 cm/s
            c = [np.array([i]) for i in pt[:3]]
            c.append(np.array([1]))
            c=np.array(c)
            moving_points.append(c)
            velocity.append(pt[4])

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
                

def single_opt(image_path, point_path,depth_path, camera_calib_file, segment_cfg, segment_ckpts, M_t_init, alignment_file):
    # model = torch.nn.DataParallel(RAFT(args))
    # model.load_state_dict(torch.load(args.model))

    # model = model.module
    # model.to(DEVICE)
    # model.eval()
    
    segment_model = init_segmentor(segment_cfg, segment_ckpts, 'cuda:0')
    segment_model.eval()
    
    K = load_camera_calib(camera_calib_file)
    f = open(alignment_file,'r')
    alignment = f.read()
    alignment = json.loads(alignment)
    f.close()

    with torch.no_grad():
        images = glob.glob(os.path.join(image_path, '*.png')) + \
                 glob.glob(os.path.join(image_path, '*.jpg'))
        images = sorted(images)
        #print(images)
        #for imfile1, imfile2 in zip(images[:-1], images[1:]):
        count = 0
        for i in range(len(images)):
            #image1 = load_image(imfile1)
            #image2 = load_image(imfile2)
            #print(image1.shape)
            #padder = InputPadder(image1.shape)
            #image1, image2 = padder.pad(image1, image2)

            #_, flow_up = model(image1, image2, iters=20, test_mode=True)

            # coarse optimize
            imfile = images[i]
            ptfile = os.path.join(point_path,alignment[imfile.split('/')[-1]])
            dpfile = os.path.join(depth_path,alignment[imfile.split('/')[-1]])
            P_r = load_points(ptfile)
            #print(P_r)
            if len(P_r)==0:
                print("In this frame, there is no moving items")
                count = count + 1
                continue
            seg_result = inference_segmentor(segment_model, imfile)[0]
            remasking = remask(seg_result,12) # 12 is the idx of person
            new_mask = BFS_mask(remasking)

            problem = single_projection_problem(K, new_mask, P_r)
            
            M_t = coarse_optimize(M_t_init, problem, imfile.split('/')[-1])
            M_t_init = M_t
            x,y,z,w = M_t_init[:4]
            mag = np.sqrt(x*x+y*y+z*z+w*w) + 0.00001 #avoid 0
            M_t_init[:4] = [x,y,z,w]/mag

            viz_pts(new_mask, imfile.split('/')[-1], P_r, M_t_init,K)

        print(M_t_init)
        print("There are "+str(count)+" frames with no moving")

def batch_opt(image_path, point_path,depth_path, camera_calib_file, segment_cfg, segment_ckpts, M_t_init, alignment_file,raft_ckpts,args):
    segment_model = init_segmentor(segment_cfg, segment_ckpts, 'cuda:0')
    segment_model.eval()
    
    K = load_camera_calib(camera_calib_file)
    f = open(alignment_file,'r')
    alignment = f.read()
    alignment = json.loads(alignment)
    f.close()

    with torch.no_grad():
        images = glob.glob(os.path.join(image_path, '*.png')) + \
                 glob.glob(os.path.join(image_path, '*.jpg'))
        images = sorted(images)

        count = 0
        problem_sets = []
        img_names = [] # store (im0,im1) for depth and raft

        # coarse optimize
        for i in range(len(images)-1):
            imfile = images[i]
            
            ptfile = os.path.join(point_path,alignment[imfile.split('/')[-1]])
            P_r, V_r = load_points(ptfile)
            if len(P_r)==0:
                #print("In this frame, there is no moving items")
                count = count + 1
                continue
            img_names.append((images[i], images[i+1]))
            seg_result = inference_segmentor(segment_model, imfile)[0]
            remasking = remask(seg_result,12) # 12 is the idx of person
            new_mask = BFS_mask(remasking)
            problem = single_projection_problem(K, new_mask, P_r, V_r, imfile.split('/')[-1])
            problem_sets.append(problem)
        
        problems = batch_projection_problem(problem_sets)
        print("In the beginning, the score is", problems.objective_function(M_t_init))
        M_t = coarse_optimize(M_t_init, problems)
        M_t_init = M_t
        x,y,z,w = M_t_init[:4]
        mag = np.sqrt(x*x+y*y+z*z+w*w)
        M_t_init[:4] = [x,y,z,w]/mag
        print("Finish coarse optimize")
        print("After coarse opt, quaternions are",M_t_init)

        # fine optimize
        idx = problems.update(M_t_init)
        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(raft_ckpts))

        model = model.module
        model.to(DEVICE)
        model.eval()
        for id in idx:
            img_names[id] = None
            problem_sets[id] = None
        fields = []
        P_rs = []
        V_rs = []
        print("Prepare for fine opt")
        for idx in range(len(img_names)):
            imgs = img_names[idx]
            if imgs is not None:
                im1 = os.path.join(image_path,imgs[0].split('/')[-1])
                im2 = os.path.join(image_path,imgs[1].split('/')[-1])
                image1 = load_RGB(im1)
                image2 = load_RGB(im2)
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)
                _, flow_up = model(image1, image2, iters=20, test_mode=True)
                flow_up = flow_up[0].cpu()
                image1 = load_depth(os.path.join(depth_path,imgs[0].split('/')[-1].split('.')[0]+'.npy'))
                image2 = load_depth(os.path.join(depth_path,imgs[1].split('/')[-1].split('.')[0]+'.npy'))
                field = optical_field(flow_up,image1, image2, K, 0.1, 0.1, 0.1, 0.1)
                P_r = problem_sets[idx].P_r
                V_r = problem_sets[idx].V_r
                for p in range(len(P_r)):
                    pr = P_r[p]
                    P_rs.append(pr)
                    V_rs.append(V_r[p]*pr/np.linalg.norm(pr))
                    fields.append(field)
        print("Begin fine optimization")
        M_t_init = fine_optimize(M_t_init, P_rs, V_rs, fields)
        #M_t_init = fine_sa(M_t_init, P_rs, V_rs, fields)
        print(M_t_init)
        rot = M_t_init[:3][:3]
        qw = np.sqrt(1+rot[0][0]+rot[1][1]+rot[2][2])/2
        qx = (rot[2][1]-rot[1][2])/(4*qw)
        qy = (rot[0][2]-rot[2][0])/(4*qw)
        qz = (rot[1][0]-rot[0][1])/(4*qw)
        print('quaternions are ', [qx,qy,qz,qw])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    point_path = 'result/radar'
    camera_calib_file = 'result/calibration.npy'
    depth_path = 'result/complete_depth'

    image_path = 'result/img'
    #segment_cfg = '/home/gejintian/workspace/mmlab/mmsegmentation/configs/segformer/segformer_mit-b2_512x512_160k_ade20k.py'
    segment_cfg = '/mnt/e/NTU/mmlab/mmsegmentation//configs/segformer/segformer_mit-b2_512x512_160k_ade20k.py'
    segment_ckpts = 'models/b2.pth'
    raft_ckpts = 'models/raft-things.pth'
    #M_t_init = [0,0,0,0,-3/100,-5.9/100,8.75/100]
    M_t_init = [0,0.2,0.1,0.5,-0/100,-0/100,0/100]
    alignment = 'result/alignment.json'

    #single_opt(image_path, point_path,depth_path, camera_calib_file, segment_cfg, segment_ckpts, M_t_init, alignment)
    batch_opt(image_path, point_path,depth_path, camera_calib_file, segment_cfg, segment_ckpts, M_t_init, alignment, raft_ckpts,args)