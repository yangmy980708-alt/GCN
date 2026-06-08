import numpy as np
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
sys.path.insert(0, '../matrixAb')
from matrixAb.generate_Gauss_local_triangle_element import generate_Gauss_local_triangle_element
from matrixAb.triangular_local_basis_element import triangular_local_basis_element
from matrixAb.FE_solution_triangle import FE_solution_triangle

def assemble_vector_from_volume_integral_2FE_triangle(uh_FE1, uh_FE2, M_partition,
                                                      T_partition, T_basis_test, T_basis_FE1, T_basis_FE2,
                                                      vector_size, test_basis_type,
                                                      test_derivative_degree_x, test_derivative_degree_y,
                                                      FE1_basis_type, FE1_derivative_degree_x, FE1_derivative_degree_y,
                                                      FE2_basis_type,
                                                      FE2_derivative_degree_x, FE2_derivative_degree_y):


    # Initialize result vector
    r = np.zeros((vector_size, 1))

    # Iterate over all elements
    for n in range(int(T_basis_test.shape[1])):
        # Get vertices and local FE solutions
        vertices = M_partition[:, T_partition[:, n]]
        uh_local_1 = uh_FE1[T_basis_FE1[:, n]]
        uh_local_2 = uh_FE2[T_basis_FE2[:, n]]

        # Generate local Gauss coefficients and points
        Gauss_point_number = 9
        [Gauss_point_local_triangle_x, Gauss_point_local_triangle_y,
         Gauss_coefficient_local_triangle] = generate_Gauss_local_triangle_element(Gauss_point_number,
                                                                                   vertices)

        # Loop over all test FE basis functions
        for beta in range(int(T_basis_test.shape[0])):
            temp = 0
            for i in range (Gauss_point_number):

                test_basis_value = triangular_local_basis_element(
                    Gauss_point_local_triangle_x[i,:], Gauss_point_local_triangle_y[i,:],
                    vertices, test_basis_type, beta, test_derivative_degree_x,
                    test_derivative_degree_y)

                coef_value_1 = FE_solution_triangle(
                    Gauss_point_local_triangle_x[i, :], Gauss_point_local_triangle_y[i, :],
                    uh_local_1, vertices, FE1_basis_type,
                    FE1_derivative_degree_x, FE1_derivative_degree_y)

                coeff_value_2 = FE_solution_triangle(
                    Gauss_point_local_triangle_x[i, :], Gauss_point_local_triangle_y[i, :],
                    uh_local_2, vertices, FE2_basis_type,
                    FE2_derivative_degree_x, FE2_derivative_degree_y)

                temp += np.unique(Gauss_coefficient_local_triangle[i]) * coef_value_1 * coeff_value_2 * test_basis_value

            # Update result vector
            r[T_basis_test[beta, n], 0] += temp

    return r
