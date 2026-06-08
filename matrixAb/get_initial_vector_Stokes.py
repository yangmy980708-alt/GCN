import numpy as np
import os
import sys
from scipy.sparse import lil_matrix,coo_matrix

def get_initial_vector_Stokes(initial_function_name_u1,initial_function_name_u2,initial_function_name_p,M_basis_u,M_basis_p):

    number_of_nodes_u = M_basis_u.shape[1]
    number_of_nodes_p = M_basis_p.shape[1]

    # r = coo_matrix((2 * number_of_nodes_u + number_of_nodes_p, 1))
    r = np.zeros((2 * number_of_nodes_u + number_of_nodes_p, 1))
    for i in range(number_of_nodes_u):
        r[i] = initial_function_name_u1(M_basis_u[0, i], M_basis_u[1, i])

    for i in range(number_of_nodes_u):
        r[number_of_nodes_u+i] = initial_function_name_u2(M_basis_u[0, i], M_basis_u[1, i])

    for i in range(number_of_nodes_p):
        r[2*number_of_nodes_u+i] = initial_function_name_p(M_basis_u[0, i], M_basis_u[1, i])

    return r
