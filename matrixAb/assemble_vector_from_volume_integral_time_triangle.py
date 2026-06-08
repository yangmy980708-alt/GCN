import numpy as np
from scipy.sparse import lil_matrix,coo_matrix
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
sys.path.insert(0, '../matrixAb')
from matrixAb.generate_Gauss_local_triangle import generate_Gauss_local_triangle
from matrixAb.Gauss_quadrature_for_volume_integral_test_triangle import Gauss_quadrature_for_volume_integral_test_triangle
from matrixAb.triangular_local_basis import triangular_local_basis

def assemble_vector_from_volume_integral_time_triangle(coefficient_function,current_time,
                                                       M_partition,T_partition,M_basis_trial,T_basis_test,
                                                       test_basis_type,
                                                       test_derivative_degree_x,test_derivative_degree_y):
    Nb = int(M_basis_trial.shape[1])
    number_of_test_local_basis = int(T_basis_test.shape[0])
    R = coo_matrix((Nb, 1))

    vertices = np.vstack(
        [M_partition[:, T_partition[0, :]], M_partition[:, T_partition[1, :]], M_partition[:, T_partition[2, :]]])
    Gauss_point_number = 9
    [Gauss_point_local_triangle_x, Gauss_point_local_triangle_y,
     Gauss_coefficient_local_triangle] = generate_Gauss_local_triangle(Gauss_point_number,
                                                                       vertices)

    for beta in range(number_of_test_local_basis):
        temp = 0
        for i in range(Gauss_point_number):
            # Perform Gauss quadrature for the volume integral for the current test basis function
            basis_value = triangular_local_basis(Gauss_point_local_triangle_x[i, :], Gauss_point_local_triangle_y[i, :], vertices,
                                       test_basis_type, beta, test_derivative_degree_x,
                                       test_derivative_degree_y)
            coef = np.array(coefficient_function(Gauss_point_local_triangle_x[i, :], Gauss_point_local_triangle_y[i, :], current_time))

            temp += np.unique(Gauss_coefficient_local_triangle[i]) * coef * basis_value
            R += coo_matrix((temp, (T_basis_test[beta, :], np.zeros_like(T_basis_test[beta, :]))), shape=(Nb, 1))
            r = R.toarray()
        return r