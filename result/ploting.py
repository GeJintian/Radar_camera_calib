import numpy as np
import matplotlib.pyplot as plt
import math


def load_points(ptfile):
    all_points = np.load(ptfile)

    return all_points

if __name__=='__main__':
    ptfile = 'radar/1672812488_633140000.npy'
    points = load_points(ptfile)
    plt_x=[]
    plt_y=[]
    for p in points:
        x ,y, z, inten, vel = p
        plt_x.append(math.atan(y/x))
        plt_y.append(vel)
    plt_x = np.array(plt_x)
    plt_y = np.array(plt_y)
    plt.scatter(plt_x,plt_y)
    plt.show()