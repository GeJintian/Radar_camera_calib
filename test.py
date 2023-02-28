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



class Masking_problem():
    """
    Problem defined for masking
    """
    def __init__(self, mask, constraint):
        self.mask = mask
        self.height, self.width = self.mask.shape
        self.constraint = constraint
    def get_surroundings(self, p):
        u,v = p
        p_set = []
        if v + 1 < self.height:
            if self.mask[v+1][u]==1:
                p_set.append((u,v+1))
        if v - 1 > 0:
            if self.mask[v-1][u]==1:
                p_set.append((u,v-1))
        if u + 1 < self.width:
            if self.mask[v][u+1]==1:
                p_set.append((u+1,v))
        if u - 1 > 0:
            if self.mask[v][u-1]==1:
                p_set.append((u-1,v))
        return p_set

def BFS(mask, constraint, problem, need_contour = False):
    """
    This function will search in the mask to find the largest area with mask == 1
    return: new_mask in the same shape of mask    
    """
    v_idx, u_idx = np.where(constraint(mask))
    #print(x_idx)
    mask_set = set([])
    for i in range(len(v_idx)):
        mask_set.add((u_idx[i],v_idx[i]))
    groups = []

    while(len(mask_set) > 0):
        state_stack = Queue()
        StartState = mask_set.pop()
        state_stack.push(StartState)
        Visit = set([])
        contour = set([])
        while True:
            if state_stack.isEmpty():
                break
            state = state_stack.pop()
            if state not in Visit:
                Visit.add(state)
                successors = problem.get_surroundings(state)
                if successors is not None:
                    for successor in successors:
                        state_stack.push(successor)
                        mask_set.discard(successor)
        groups.append(Visit)

    if not need_contour:
        return groups
    else:
        return groups




