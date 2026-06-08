import numpy as np
import os
import sys
from scipy.sparse import lil_matrix,coo_matrix
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
sys.path.insert(0, '../matrixAb')
from matrixAb.generate_Gauss_local_triangle import generate_Gauss_local_triangle
from matrixAb.triangular_local_basis import triangular_local_basis
from matrixAb.FE_solution_triangle import FE_solution_triangle

def assemble_matrix_from_volume_integral_triangle(coefficient_function,
                                                  M_partition,T_partition,M_basis_trial,
                                                  T_basis_trial,T_basis_test,
                                                  trial_basis_type,trial_derivative_degree_x,trial_derivative_degree_y,
                                                  test_basis_type,test_derivative_degree_x,test_derivative_degree_y):
    # MS = assemble_matrix_from_volume_integral_triangle('function_one', M_partition_S, T_partition_S, M_basis_u,
    #                                                    T_basis_u, T_basis_u, 2, 0, 0, 2, 0, 0);
    Nb_trial = int(M_basis_trial.shape[1])
    Nb_test = int(M_basis_trial.shape[1])  # Number of basis
    number_of_trial_local_basis = int(T_basis_trial.shape[0])
    number_of_test_local_basis = int(T_basis_test.shape[0])
    M = coo_matrix((Nb_test, Nb_trial))
    vertices = np.vstack(
        [M_partition[:, T_partition[0, :]], M_partition[:, T_partition[1, :]], M_partition[:, T_partition[2, :]]])
    Gauss_point_number = 9
    [Gauss_point_local_triangle_x, Gauss_point_local_triangle_y,
     Gauss_coefficient_local_triangle] = generate_Gauss_local_triangle(Gauss_point_number,
                                                                       vertices)

    for alpha in range(number_of_trial_local_basis):
        for beta in range(number_of_test_local_basis):
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

                temp = np.array(temp.flatten())
                T_basis_test[beta, :] = np.array(T_basis_test[beta, :].flatten())
                T_basis_trial[alpha, :] = np.array(T_basis_trial[alpha, :].flatten())
            M += coo_matrix((temp, (T_basis_test[beta, :], T_basis_trial[alpha, :])), shape=(Nb_test, Nb_trial))
            r = M.toarray()
        return r