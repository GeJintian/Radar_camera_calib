import numpy as np
import math
import sys

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

def bilinear_interpolate(im, x, y, p0=None, p1=None):
    """
    compute the bilinear interpolation.
    y is v, x is u
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if p0 is None:
        x0 = np.floor(x).astype(int)
        x1 = x0 + 1
        y0 = np.floor(y).astype(int)
        y1 = y0 + 1
    else:
        x0,y0 = p0
        x1,y1 = p1

    x0 = np.clip(x0, 0, im.shape[1]-1)
    x1 = np.clip(x1, 0, im.shape[1]-1)
    y0 = np.clip(y0, 0, im.shape[0]-1)
    y1 = np.clip(y1, 0, im.shape[0]-1)

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return (wa*Ia + wb*Ib + wc*Ic + wd*Id)#/((x1-x0)*(y1-y0))

def complete_depth_map(map):
    """
    Depth map contains lots of nan area. This function complete the depth map by bilinear interpolation.
    """

def Cam2World(K,P):
    """
    K: [3x3] camera intrinsics
    P: [3x1] camera point [[u],[v],[d]]
    return: np.array([4x1]) 3D camera point [[x],[y],[z],[1]]
    """
    u,v,d = P[:3]
    f_u = K[0][0]
    f_v = K[1][1]
    cu = K[0][2]
    cv = K[1][2]
    z = d * (cv-v)/f_v
    y = d * (cu-u)/f_u
    x = d

    return np.array([x,y,z,[1]])

def is_close(a, b, rel_tol=1e-09, abs_tol=1e-30):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def World2Cam(K,P):
    """
    K: [3x3] camera intrinsics
    P: [4x1] camera point [[x],[y],[z],[1]]
    return: np.array([2x1]) 2D camera point [[u],[v]]. Origin is on the image center
    """
    x,y,z = P[:-1]
    f_u = K[0][0]
    f_v = K[1][1]
    cu = K[0][2]
    cv = K[1][2]
    u = cu - f_u*y/x#TODO: consider using interpolation
    v = cv - f_v*z/x

    return np.array([u,v])

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

def Pos2Vel(p1,p2,t):
    """
    p1, p2: [3x1] two positions in world coordinate
    t: time interval
    return: np.array([3x1]) velocity in this time interval    
    """

    dx = p2[0]-p1[0]
    dy = p2[1]-p1[1]
    dz = p2[2]-p1[2]

    return np.array([dx/t,dy/t,dz/t,[0]])

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
    mag = np.sqrt(x*x+y*y+z*z+w*w)
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
    def __init__(self, mask, constraint):
        self.mask = mask
        self.height, self.width = self.mask.shape
        self.constraint = constraint
    def get_surroundings(self, p):
        u,v = p
        p_set = []
        if v + 1 < self.height:
            if self.constraint(self.mask[v+1][u]):
                p_set.append((u,v+1))
        if v - 1 > 0:
            if self.constraint(self.mask[v-1][u]):
                p_set.append((u,v-1))
        if u + 1 < self.width:
            if self.constraint(self.mask[v][u+1]):
                p_set.append((u+1,v))
        if u - 1 > 0:
            if self.constraint(self.mask[v][u-1]):
                p_set.append((u-1,v))
        return p_set
    def get_contours(self, p):
        u,v = p
        p_set = set({})
        if v + 1 < self.height:
            if not self.constraint(self.mask[v+1][u]):
                p_set.add((u,v+1))
        if v - 1 > 0:
            if not self.constraint(self.mask[v-1][u]):
                p_set.add((u,v-1))
        if u + 1 < self.width:
            if not self.constraint(self.mask[v][u+1]):
                p_set.add((u+1,v))
        if u - 1 > 0:
            if not self.constraint(self.mask[v][u-1]):
                p_set.add((u-1,v))
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
    contours = []

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
                c = problem.get_contours(state)
                #print(c)
                contour = set.union(contour, c)
                if successors is not None:
                    for successor in successors:
                        state_stack.push(successor)
                        mask_set.discard(successor)
        groups.append(Visit)
        contours.append(contour)
    if not need_contour:
        return groups
    else:
        return groups, contours

def equal1(a):
    return a==1

def BFS_mask(mask):
    """
    finding contours using bfs
    """
    new_mask = np.zeros_like(mask)
    problem = Masking_problem(mask,equal1)
    count = 0
    max_group =None
    groups = BFS(mask, equal1, problem)
    for visit in groups:
        if len(visit)>count:
            count = len(visit)
            max_group = visit
    for i in max_group:
        new_mask[i[1]][i[0]] = 1

    return new_mask

def find_closest_pos(p, contour):
    dist = 1e9
    for c in contour:
        if (p[0]-c[0])**2+(p[1]-c[1])**2 < dist:
            m = c
            dist = (p[0]-c[0])**2+(p[1]-c[1])**2
    return m

def BFS_nan(depth_map):
    """
    depth complement problem with bfs
    """
    # h,w = depth_map.shape
    # for i in range(h):
    #     for j in range(w):
    #         if math.isnan(depth_map[i][j]):
    #             depth_map[i][j] = None
    problem = Masking_problem(depth_map, np.isnan)
    
    groups,contours = BFS(depth_map,np.isnan, problem, True)
    for i in range(len(groups)):
        # p[0] is u, p[1] is v
        visit = groups[i]
        contour = contours[i]
        for p in visit:
            pc = find_closest_pos(p,contour)
            depth_map[p[1]][p[0]] = depth_map[pc[1]][pc[0]]
    return depth_map

def quaternions2euler(x,y,z,w):

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = math.sqrt(1 + 2 * (w * y - x * z))
    cosp = math.sqrt(1 - 2 * (w * y - x * z))
    pitch = 2 * math.atan2(sinp, cosp) - math.pi / 2

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw