import numpy as np
from scipy.sparse import lil_matrix,csr_matrix,coo_matrix
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
sys.path.insert(0, '../matrixAb')
from matrixAb.generate_Gauss_local_triangle import generate_Gauss_local_triangle
from matrixAb.triangular_local_basis import triangular_local_basis

def assemble_matrix_from_volume_integral_triangle_up(coefficient_function, M_partition, T_partition, M_basis_trial,
                                                     T_basis_trial_u, T_basis_test_p,
                                                   trial_basis_type, trial_derivative_degree_x, trial_derivative_degree_y,
                                                   test_basis_type, test_derivative_degree_x, test_derivative_degree_y):

    Nb_trial = int(M_basis_trial.shape[1])
    Nb_test = int(M_partition.shape[1])  # Number of basis

    A = coo_matrix((Nb_trial, Nb_test))
    vertices = np.vstack(
        [M_partition[:, T_partition[0, :]], M_partition[:, T_partition[1, :]], M_partition[:, T_partition[2, :]]])
    Gauss_point_number = 9
    [Gauss_point_local_triangle_x, Gauss_point_local_triangle_y,
     Gauss_coefficient_local_triangle] = generate_Gauss_local_triangle(Gauss_point_number,
                                                                       vertices)
    for alpha in range(int(T_basis_test_p.shape[0])):
        for beta in range(int(T_basis_trial_u.shape[0])):
            temp = 0
            for i in range(Gauss_point_number):
                trial_basis_value = triangular_local_basis(Gauss_point_local_triangle_x[i, :],
                                                           Gauss_point_local_triangle_y[i, :], vertices,
                                                           trial_basis_type, alpha, trial_derivative_degree_x,
                                                           trial_derivative_degree_y)
                test_basis_value = triangular_local_basis(Gauss_point_local_triangle_x[i, :],
                                                          Gauss_point_local_triangle_y[i, :], vertices,
                                                          test_basis_type, beta, test_derivative_degree_x,
                                                          test_derivative_degree_y)
                coef = coefficient_function(Gauss_point_local_triangle_x[i, :], Gauss_point_local_triangle_y[i, :])
                temp += Gauss_coefficient_local_triangle[i, :] * coef * trial_basis_value * test_basis_value

                # temp = np.array(temp.flatten())
                # T_basis_trial_u[beta, :] = np.array(T_basis_trial_u[beta, :].flatten())
                # T_basis_test_p[alpha, :] = np.array(T_basis_test_p[alpha, :].flatten())
            A += coo_matrix((temp, (T_basis_trial_u[beta, :], T_basis_test_p[alpha, :])), shape=(Nb_trial, Nb_test))
            r = A.toarray()
    return r
