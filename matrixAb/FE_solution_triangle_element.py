import numpy as np
import os
import sys
from scipy.sparse import lil_matrix,coo_matrix
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
sys.path.insert(0, '../matrixAb')
from matrixAb.triangular_local_basis_element import triangular_local_basis_element


def FE_solution_triangle_element( x, y, uh_local, vertices, basis_type, derivative_degree_x, derivative_degree_y):
    r = 0
    number_of_local_basis = len(uh_local)
    for i in range(int(number_of_local_basis)):
        coef = triangular_local_basis_element(x, y, vertices, basis_type, i, derivative_degree_x,
                                              derivative_degree_y)
        r += uh_local[i] * coef
    return r


