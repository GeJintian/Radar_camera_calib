import numpy as np
from utils.helpers import *


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
    M_t_init = [0,0,0,0,-3/100,-5.9/100,8.75/100]

    T = build_matrix(M_t_init)

    a=np.array([[1],[1],[1],[1]])
    k=load_camera_calib('result/calibration.npy')
    b = T@a
    c = World2Cam(k,b)
    print(c)
