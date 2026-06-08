import numpy as np
import os
import sys
from scipy.sparse import lil_matrix,coo_matrix
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
sys.path.insert(0, '../matrixAb')
from matrixAb.triangular_local_basis import triangular_local_basis


def FE_solution_triangle( x, y, uh, T_basis_trial, vertices, basis_type, derivative_degree_x, derivative_degree_y):
    r = 0
    number_of_local_basis = int(T_basis_trial.shape[0])
    for i in range(int(number_of_local_basis)):
        coef = triangular_local_basis(x, y, vertices, basis_type, i, derivative_degree_x,
                                              derivative_degree_y)
        r += np.array(uh[T_basis_trial[i,:], 0]).T * np.array(coef)
    return r


