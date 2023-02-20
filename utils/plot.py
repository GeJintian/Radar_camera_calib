import numpy as np
import matplotlib.pyplot as plt
import math



def disturbance_trans(M_t_init, obj, Prs, Vrs, fields,idx = 0):
    plt.clf()
    x = []
    y=[]
    for i in range(2000):
        temp = np.copy(M_t_init)
        temp[idx] += (i-1000)/10000
        x.append((i-1000)/10000)
        val = obj(temp, Prs, fields, Vrs)
        y.append(0.5*np.sqrt(val.T@val))
    
    x=np.array(x)
    y = np.array(y)
    plt.plot(x,y)
    plt.savefig('axis-'+str(idx)+'.png')

def disturbance_rot(M_t_init, obj, Prs, Vrs, fields):
    p = []
    x = []
    y=[]
    idx = 5
    for i in range(1000):
        temp = np.copy(M_t_init)
        temp[idx] += (i-1000)/1000
        x.append((i-1000)/1000)
        val = obj(temp, Prs, fields, Vrs)
        y.append(0.5*np.sqrt(val.T@val))
    
    x=np.array(x)
    y = np.array(y)
    plt.plot(x,y)
    plt.savefig('axis-'+str(idx)+'.png')