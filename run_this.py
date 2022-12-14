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
from coarse_opt import SA



cfg = '/home/gejintian/workspace/mmlab/mmsegmentation/configs/segformer/segformer_mit-b2_512x512_160k_ade20k.py'
ckpts = 'models/b2.pth'
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
                
def seg_mask(rgb,seg):
    width, height = seg.shape
    #print(np.unique(seg))
    for i in range(width):
        for j in range(height):
            if seg[i][j]!=12: #edge = 255
                rgb[i][j] = [0,0,0]

def print_minmax(rgb,flo,seg_mask):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255,255,255)
    thickness = 2
    flow = np.sqrt(np.square(flo[...,0])+np.square(flo[...,1]))
    width,height=seg_mask.shape
    for i in range(width):
        for j in range(height):
            if seg_mask[i][j] !=12:
                flow[i][j] = 0
    max_idx = np.unravel_index(np.argmax(flow, axis=None), flow.shape)
    for i in range(width):
        for j in range(height):
            if seg_mask[i][j] !=12:
                flow[i][j] = 10000
    min_idx = np.unravel_index(np.argmin(flow, axis=None), flow.shape)
    #print(max_idx)
    rgb = rgb.astype(np.uint8).copy()
    #print(rgb.dtype)
    rgb = cv2.putText(rgb,"("+str(int(flo[max_idx[0]][max_idx[1]][0]))+","+str(int(flo[max_idx[0]][max_idx[1]][1]))+")",(max_idx[0],max_idx[1]+width+1),font,fontScale,color,thickness,cv2.LINE_AA)
    
    rgb = cv2.putText(rgb,"("+str(int(flo[min_idx[0]][min_idx[1]][0]))+","+str(int(flo[min_idx[0]][min_idx[1]][1]))+")",(min_idx[0],min_idx[1]+width+1),font,fontScale,color,thickness,cv2.LINE_AA)
    
    return rgb

def viz_optical(img, flo, seg_result, i):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    #flo = flow_viz.flow_to_image(flo)
    hsv = np.zeros_like(img)
    hsv[...,1]=255
    mag, ang = cv2.cartToPolar(flo[..., 0], flo[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    img = np.uint8(img)
    gray =cv2.cvtColor(img[:,:,[2,1,0]],cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,100,200)
    #mask(bgr,edges)
    seg_mask(bgr,seg_result)
    
    bgr = bgr[:,:,[2,1,0]]
    img_flo = np.concatenate([img, bgr], axis=0)
    img_flo = img_flo[:, :, [2,1,0]]
    #img_flo = print_minmax(img_flo,flo,seg_result)

    cv2.imwrite('result/'+str(i).zfill(5)+'.png', img_flo)

def viz_mask(mask,i):
    height, width = mask.shape
    img = np.ones((height,width,3))*255
    for h in range(height):
        for w in range(width):
            if mask[h][w] == 1:
                img[h][w] = [0,0,0]

    cv2.imwrite('result/'+str(i).zfill(5)+'.png',img)
    return

def demo(image_path, point_path, camera_calib_file, segment_cfg, segment_ckpts):
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
        i = 0
        #for imfile1, imfile2 in zip(images[:-1], images[1:]):
        for imfile in images:
            #image1 = load_image(imfile1)
            #image2 = load_image(imfile2)
            #print(image1.shape)
            #padder = InputPadder(image1.shape)
            #image1, image2 = padder.pad(image1, image2)

            #_, flow_up = model(image1, image2, iters=20, test_mode=True)
            seg_result = inference_segmentor(segment_model, imfile)[0]
            remasking = remask(seg_result,12)
            new_mask = BFS(remasking)
            
            viz_mask(new_mask,i)
            #viz_opt(image1, flow_up, seg_result, i)
            i = i+1

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

    demo(image_path, point_path, camera_calib_file, segment_cfg, segment_ckpts)