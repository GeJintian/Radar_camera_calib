import sys
import os
from PIL import Image
import glob
import numpy as np
import argparse

import torch
from mmseg.apis import inference_segmentor, init_segmentor
import cv2

#from raft_core.raft import RAFT
#from raft_core.utils import InputPadder
from utils.helpers import *
from utils.visualize import seg_mask, viz_optical, viz_mask, print_minmax, viz_pts
from coarse_opt import coarse_optimize


DEVICE = 'cuda'

def get_image(imfile):
    # get image from camera
    # TODO: get image from camera
    return

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def load_points(ptfile):
    #TODO: load point file
    return

def load_camera_calib(cfg):
    # TODO: load camera calibration file
    return

def mask(rgb,edge):
    bool_edge = edge/255
    width, height = bool_edge.shape
    for i in range(width):
        for j in range(height):
            if bool_edge[i][j]>0.5: #edge = 255
                rgb[i][j] = [255,255,255]
                

def demo(image_path, point_path, camera_calib_file, segment_cfg, segment_ckpts, M_t_init):
    # model = torch.nn.DataParallel(RAFT(args))
    # model.load_state_dict(torch.load(args.model))

    # model = model.module
    # model.to(DEVICE)
    # model.eval()
    
    segment_model = init_segmentor(segment_cfg, segment_ckpts, 'cuda:0')
    segment_model.eval()
    
    K = load_camera_calib(camera_calib_file)


    with torch.no_grad():
        images = glob.glob(os.path.join(image_path, '*.png')) + \
                 glob.glob(os.path.join(image_path, '*.jpg'))
        images = sorted(images)
        points = glob.glob(os.path.join(image_path, '*.png')) + \
                 glob.glob(os.path.join(image_path, '*.jpg'))
        points = sorted(points)

        #for imfile1, imfile2 in zip(images[:-1], images[1:]):
        for i in len(images):
            #image1 = load_image(imfile1)
            #image2 = load_image(imfile2)
            #print(image1.shape)
            #padder = InputPadder(image1.shape)
            #image1, image2 = padder.pad(image1, image2)

            #_, flow_up = model(image1, image2, iters=20, test_mode=True)

            # coarse optimize
            imfile = images[i]
            ptfile = points[i]
            seg_result = inference_segmentor(segment_model, imfile)[0]
            remasking = remask(seg_result,12)
            new_mask = BFS(remasking)
            
            #viz_mask(new_mask,i)
            #viz_optical(image1, flow_up, seg_result, i)
            P_r = load_points(ptfile)
            M_t = coarse_optimize(M_t_init, P_r, K, new_mask)
            M_t_init = M_t

            viz_pts(new_mask, i, P_r, M_t)





if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model', help="restore checkpoint")
    # parser.add_argument('--path', help="dataset for evaluation")
    # parser.add_argument('--small', action='store_true', help='use small model')
    # parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    # parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    # args = parser.parse_args()

    point_path = ''
    camera_calib_file = ''
    image_path = '../people/images'
    segment_cfg = '/home/gejintian/workspace/mmlab/mmsegmentation/configs/segformer/segformer_mit-b2_512x512_160k_ade20k.py'
    segment_ckpts = 'models/b2.pth'
    M_t_init = []

    demo(image_path, point_path, camera_calib_file, segment_cfg, segment_ckpts, M_t_init)