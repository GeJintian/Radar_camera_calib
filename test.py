import numpy as np


pt_file = 'result/radar/1672812488_633140000.npy'
pts = np.load(pt_file)
contain = []
for pt in pts:
    if pt[4] < 0.001: # larger than 0.1 cm/s
        c = [np.array([i]) for i in pt[:3]]
        contain.append(c)
contain = np.array(contain)
print(contain)
