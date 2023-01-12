import numpy as np


class Queue:
    "A container with a first-in-first-out (FIFO) queuing policy."
    def __init__(self):
        self.list = []

    def push(self,item):
        "Enqueue the 'item' into the queue"
        self.list.insert(0,item)

    def pop(self):
        """
          Dequeue the earliest enqueued item still in the queue. This
          operation removes the item from the queue.
        """
        return self.list.pop()

    def isEmpty(self):
        "Returns true if the queue is empty"
        return len(self.list) == 0

def Cam2World(K,P):
    """
    K: [3x3] camera intrinsics
    P: [3x1] camera point [[u],[v],[d]]
    return: np.array([4x1]) 3D camera point [[x],[y],[z],[1]]
    """

    u,v,d = P
    f_v = K[0][0]
    f_u = K[1][1]
    cv = K[0][2]
    cu = K[1][2]
    z = d * (cu-u)/f_u
    y = d * (cv-v)/f_v
    x = d

    return np.array([x,y,z,[1]])

def World2Cam(K,P):
    """
    K: [3x3] camera intrinsics
    P: [4x1] camera point [[x],[y],[z],[1]]
    return: np.array([2x1]) 2D camera point [[u],[v]]
    """
    x,y,z = P[:-1]
    f_v = K[0][0]
    f_u = K[1][1]
    cv = K[0][2]
    cu = K[1][2]
    d = x
    u = np.int(np.round(cu - z*f_u/d))
    v = np.int(np.round(cv - f_v*y/d))

    return np.array([[u],[v]])

def Doppler_velocity(v,p):
    """
    v:[4x1] velocity, [[vx],[vy],[vz],[0]]
    p:[4x1] 3D position
    return: v_D: doppler velocity
    """

    x,y,z = p[:-1]
    vx,vy,vz = v[:-1]
    mag = np.sqrt(np.square(x)+np.square(y)+np.square(z))
    vd = 1/mag*(vx*x+vy*y+vz*z)

    return vd

def Pos2Vel(p1,p2,t):#TODO: check if this is used
    """
    p1, p2: [3x1] two positions in world coordinate
    t: time interval
    return: np.array([3x1]) velocity in this time interval    
    """

    dx = p1[0]-p2[0]
    dy = p1[1]-p2[1]
    dz = p1[2]-p2[2]

    return np.array([dx/t,dy/t,dz/t,[0]])

def read_calibration(path):

    return

def Pos_transform(T, P):
    """
    T: [4x4] extrinsics
    P: [4X1] position to be transformed
    return: P1 [4x1] transformed result    
    """

    P1 = np.dot(T,P)

    return P1

def build_matrix(t):
    """
    t:[:4], quaternion
    t:[4:], transition
    return: T[4x4]    
    """

    x,y,z,w = t[:4]
    trans = t[4:]
    mag = np.sqrt(x*x+y*y+z*z+w*w) + 0.00001
    x,y,z,w = [x,y,z,w]/mag
    Ts = np.array([[trans[0]], [trans[1]], [trans[2]]])
    M_r = np.zeros((3,3))
    M_r[0][0] = 1 - 2*(y**2) - 2*(z**2)
    M_r[0][1] = 2*x*y - 2*w*z
    M_r[0][2] = 2*x*z + 2*w*y
    M_r[1][0] = 2*x*y + 2*w*z
    M_r[1][1] = 1 - 2*(x**2) - 2*(z**2)
    M_r[1][2] = 2*y*z - 2*w*x
    M_r[2][0] = 2*x*z - 2*w*y
    M_r[2][1] = 2*y*z + 2*w*x
    M_r[2][2] = 1 - 2*(x**2) -2*(y**2)
    M_p = np.hstack((M_r,Ts))
    M_p = np.vstack((M_p,np.array([0,0,0,1])))

    return M_p

def remask(mask, idx):
    # Mask again. if mask == idx, mask = 1. 0 otherwise

    height, width = mask.shape
    for i in range(height):
        for j in range(width):
            if mask[i][j] == idx:
                mask[i][j] = 1
            else:
                mask[i][j] = 0
    return mask

class Masking_problem():
    """
    Problem defined for masking
    """
    def __init__(self, mask):
        self.mask = mask
        self.height, self.width = self.mask.shape
    def get_surroundings(self, p):
        x,y = p
        p_set = []
        if x + 1 < self.height:
            if self.mask[x+1][y] == 1:
                p_set.append((x+1,y))
        if x - 1 > 0:
            if self.mask[x-1][y] == 1:
                p_set.append((x-1,y))
        if y + 1 < self.width:
            if self.mask[x][y+1] == 1:
                p_set.append((x,y+1))
        if y - 1 > 0:
            if self.mask[x][y-1] == 1:
                p_set.append((x,y-1))
        return p_set

def BFS(mask):
    """
    This function will search in the mask to find the largest area with mask == 1
    return: new_mask in the same shape of mask    
    """

    new_mask = np.zeros_like(mask)
    x_idx, y_idx = np.where(mask==1)
    #print(x_idx)
    mask_set = set([])
    for i in range(len(x_idx)):
        mask_set.add((x_idx[i],y_idx[i]))
    groups = []
    problem = Masking_problem(mask)

    while(len(mask_set) > 0):
        state_stack = Queue()
        StartState = mask_set.pop()
        state_stack.push(StartState)
        Visit = set([])
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
    
    count = 0
    max_group =None
    for visit in groups:
        if len(visit)>count:
            count = len(visit)
            max_group = visit
    for i in max_group:
        new_mask[i[0]][i[1]] = 1

    return new_mask