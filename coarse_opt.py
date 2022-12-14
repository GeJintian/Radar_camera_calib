import numpy as np

from utils.helpers import Pos_transform, World2Cam
from utils.SA import SimulatedAnnealingBase

class projection_problem():
    # This is the class of projection problem. We want to make sure all projected points are within the mask
    def __init__(self, K, mask, P_r) -> None:
        self.mask = mask
        self.K = K
        self.P_r = P_r
        self.n = len(P_r)

    def get_number_of_points(self, T):
        # Given tranform matrix T and radar world points P_r, return the number of P_c= TxP_r which falls into the masking area.
        # P_r: Nx4x1: [[[x],[y],[z],[1]],[[x],[y],[z],[1]],...]
        # T: 4x4
        count = 0
        for i in self.P_r:
            p_c = Pos_transform(T,i).astype(np.int)
            u,v = World2Cam(self.K, p_c)
            if self.mask[u[0]][v[0]] == 1:
                count = count+1
        return count

    def objective_function(self,t):
        # get the objective function
        T = self.build_matrix(t)
        return self.n - self.get_number_of_points(T)

    def build_matrix(self, t):
        # t:[:4], quaternion
        # t:[4:], transition
        # return: T[4x4]

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

        return M_p

def step(T):
    # Make a small change to T.
    # Return: T after changed.

    return

def coarse_optimize(M_t_init, P_r, K, mask):
    # Run SA to maximize objective function
    # M_t_init: [7x1], quaternion + transition

    # Pre-defined parameters
    T_max = 50 # max temperature
    T_min = 1e-7 # min temperature
    k = 100 # number of success
    stop_value = 0

    problem = projection_problem(K, mask, P_r)

    sa = SimulatedAnnealingBase(problem, M_t_init, T_max, T_min, k, stop_value = stop_value)
    best_M_t, best_score = sa.run()

    print("In this round, there are "+str(best_score)+" out of the masking area.")

    return best_M_t