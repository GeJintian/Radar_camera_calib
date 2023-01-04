import numpy as np

from utils.helpers import Pos_transform, World2Cam, build_matrix
from utils.SA import SimulatedAnnealingBase

class projection_problem():
    """This is the class of projection problem. We want to make sure all projected points are within the mask"""
    def __init__(self, K, mask, P_r) -> None:
        self.mask = mask
        self.K = K
        self.P_r = P_r
        self.n = len(P_r)

    def get_number_of_points(self, T):
        """
        Given tranform matrix T and radar world points P_r, return the number of P_c= TxP_r which falls into the masking area.
        P_r: Nx4x1: [[[x],[y],[z],[1]],[[x],[y],[z],[1]],...]
        T: 4x4
        """

        count = 0
        for i in self.P_r:
            p_c = Pos_transform(T,i).astype(np.int)
            u,v = World2Cam(self.K, p_c)
            if self.mask[u[0]][v[0]] == 1:
                count = count+1
        return count

    def objective_function(self,t):
        # get the objective function
        T = build_matrix(t)
        score = self.n - self.get_number_of_points(T)
        return score


def coarse_optimize(M_t_init, P_r, K, mask):
    """
    Run SA to maximize objective function
    M_t_init: [7x1], quaternion + transition
    """

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