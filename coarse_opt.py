import numpy as np

from utils.helpers import Pos_transform, World2Cam, build_matrix, is_close
from utils.SA import SimulatedAnnealingBase

class single_projection_problem():
    """This is the class of projection problem. We want to make sure all projected points are within the mask"""
    def __init__(self, K, mask, P_r, V_r, imfile) -> None:
        self.mask = mask
        self.K = K
        self.P_r = P_r
        self.V_r = V_r # V_r is for fine opt
        self.n = len(P_r)
        self.h, self.w = self.mask.shape
        self.file = imfile
        # calculate centroid
        count = 0
        u_sum = 0
        v_sum = 0
        for i in range(self.h):
            for j in range(self.w):
                if self.mask[i][j] == 1:
                    v_sum += i
                    u_sum += j
                    count += 1
        self.centroid = (u_sum/count, v_sum/count)

    def get_number_of_points(self, T):
        """
        Given tranform matrix T and radar world points P_r, return the number of P_c= TxP_r which falls into the masking area.
        P_r: Nx4x1: [[[x],[y],[z],[1]],[[x],[y],[z],[1]],...]
        T: 4x4
        """
 
        count = 0
        sum_val = 0
        for i in self.P_r:
            #print(i)
            p_c = Pos_transform(T,i)
            u,v = np.round(World2Cam(self.K, p_c)).astype(np.int)
            if v[0] < 0 or v[0] > self.h-1 or u[0] < 0 or u[0] > self.w-1:
                sum_val += np.sqrt(((u[0]-self.centroid[0])/self.w)**2 + ((v[0]-self.centroid[1])/self.h)**2)
                continue
            if self.mask[v[0]][u[0]] == 1:
                count = count+1
            else:
                sum_val += np.sqrt(((u[0]-self.centroid[0])/self.w)**2 + ((v[0]-self.centroid[1])/self.h)**2)

        return -count*100+sum_val

    def objective_function(self,t):
        # get the objective function
        T = build_matrix(t)
        score = self.get_number_of_points(T)
        return score
    
    def get_all_pts_number(self):
        return len(self.P_r)
    
class batch_projection_problem():
    def __init__(self,problems) -> None:
        self.problems = problems

    def objective_function(self,t):
        result = 0
        for p in self.problems:
            result += p.objective_function(t)
        return result
    def get_all_pts_number(self):
        result = 0
        for p in self.problems:
            result += p.get_all_pts_number()
        return result
    def update(self,t):
        idx = []
        for i in range(len(self.problems)):
            p = self.problems[i]
            if not is_close(p.objective_function(t),-100*p.get_all_pts_number()):
                idx.append(i)
        return idx

def coarse_optimize(M_t_init, problem,imfile=None):
    """
    Run SA to maximize objective function
    M_t_init: [7x1], quaternion + transition
    """

    # Pre-defined parameters
    T_max = 50 # max temperature
    T_min = 1e-7 # min temperature
    k = 100 # number of success

    sa = SimulatedAnnealingBase(problem, M_t_init, T_max, T_min, k, stop_value=-100*problem.get_all_pts_number())
    best_M_t, best_score = sa.run()

    if imfile is not None:
        print("In "+imfile+", best score is"+str(best_score))
    else:
        print("Best score is"+str(best_score))

    return best_M_t