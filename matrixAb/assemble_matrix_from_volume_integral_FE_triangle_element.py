import numpy as np
import os
import sys
from scipy.sparse import lil_matrix,coo_matrix
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
sys.path.insert(0, '../matrixAb')
from matrixAb.generate_Gauss_local_triangle_element import generate_Gauss_local_triangle_element
from matrixAb.triangular_local_basis_element import triangular_local_basis_element
from matrixAb.FE_solution_triangle_element import FE_solution_triangle_element


def assemble_matrix_from_volume_integral_FE_triangle_element(uh, M_partition, T_partition,
                                                     M_basis_trial, T_basis_trial, T_basis_test, T_basis_FE,trial_basis_type,
                                                     trial_derivative_degree_x, trial_derivative_degree_y,
                                                     test_basis_type, test_derivative_degree_x,
                                                     test_derivative_degree_y, FE_basis_type, FE_derivative_degree_x,
                                                     FE_derivative_degree_y):

    Nb_trial = int(M_basis_trial.shape[1])
    Nb_test = int(M_basis_trial.shape[1])  # Number of basis
    number_of_trial_local_basis = int(T_basis_trial.shape[0])
    number_of_test_local_basis = int(T_basis_test.shape[0])

    # A = coo_matrix((Nb_test, Nb_trial))
    A = np.zeros((Nb_test, Nb_trial))
    for n in range(int(T_basis_trial.shape[1])):
        vertices = M_partition[:, T_partition[:, n]]
        uh_local = uh[T_basis_FE[:, n]]
        Gauss_point_number = 9
        [Gauss_point_local_triangle_x, Gauss_point_local_triangle_y,
         Gauss_coefficient_local_triangle] = generate_Gauss_local_triangle_element(Gauss_point_number,
                                                                           vertices)

        for alpha in range(number_of_trial_local_basis):
            for beta in range(number_of_test_local_basis):
                temp = 0
                for i in range(Gauss_point_number):
                    trial_basis_value = triangular_local_basis_element(
                        Gauss_point_local_triangle_x[i, :], Gauss_point_local_triangle_y[i, :],
                        vertices, trial_basis_type, alpha,
                        trial_derivative_degree_x, trial_derivative_degree_y)
                    test_basis_value = triangular_local_basis_element(
                        Gauss_point_local_triangle_x[i, :], Gauss_point_local_triangle_y[i, :],
                        vertices, test_basis_type, beta,
                        test_derivative_degree_x, test_derivative_degree_y)
                    coef_value = FE_solution_triangle_element(
                        Gauss_point_local_triangle_x[i, :], Gauss_point_local_triangle_y[i, :],
                        uh_local, vertices, FE_basis_type, FE_derivative_degree_x, FE_derivative_degree_y)

                    temp += Gauss_coefficient_local_triangle[i] * coef_value * trial_basis_value * test_basis_value
                    temp = np.array(temp.flatten())
                    T_basis_test[beta, n] = np.array(T_basis_test[beta, n].flatten())
                    T_basis_trial[alpha, n] = np.array(T_basis_trial[alpha, n].flatten())
                A[T_basis_test[beta, n], T_basis_trial[alpha, n]] = A[T_basis_test[beta, n], T_basis_trial[alpha, n]] + temp

    return A