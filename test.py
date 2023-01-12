import numpy as np


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

if __name__=="__main__":
    a = test(1)
    print(a.get_y())
    a.add_x(2)
    print(a.get_y())