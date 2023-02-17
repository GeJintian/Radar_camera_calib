import numpy as np
from utils.helpers import *
import math
import cv2
import glob
import os

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

if __name__=="__main__":
    dfile = "1672812488_540169694.npy"
    d_image = np.load('result/depth/'+dfile)
    show_img = np.zeros_like(d_image)
    path = 'result/depth/'
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
    images = glob.glob(os.path.join(path, '*.npy'))
    images = sorted(images)
    for img in images:
        print("begin processing", img)
        d_image = np.load(img)
        d_image = BFS_nan(d_image)
        np.save("result/complete_depth/"+img.split('/')[-1],d_image)

