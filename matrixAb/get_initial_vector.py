import numpy as np
import os
import sys
from scipy.sparse import lil_matrix,coo_matrix

def get_initial_vector(initial_function_name,M_basis):
    number_of_nodes = M_basis.shape[1]
    # r = coo_matrix((number_of_nodes, 1))
    r = np.zeros((number_of_nodes,1))

    for i in range(number_of_nodes):
        x = M_basis[0, i]
        y = M_basis[1, i]
        r[i] = np.array(initial_function_name(x, y))
        # R = r.toarray()
    return r