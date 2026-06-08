import numpy as np
from scipy.sparse import lil_matrix,coo_matrix
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
sys.path.insert(0, '../matrixAb')
from matrixAb.Gauss_quadrature_for_line_integral_test_triangle import Gauss_quadrature_for_line_integral_test_triangle

def treat_Neumann_boundary_triangle(Neumann_boundary_function_name, b, boundary_edges, M_partition, T_partition, T_basis_test,
                                    number_of_test_local_basis, Gauss_coefficient_reference_1D, Gauss_point_reference_1D,
                                    test_basis_type, test_derivative_degree_x, test_derivative_degree_y):

    nbe = int(boundary_edges.shape[1])

    # Check all boundary edges
    for k in range(nbe):

        # If the kth boundary edge is a Neumann boundary edge
        if boundary_edges[0, k] == -2:
            element_index = boundary_edges[1, k]
            vertices = M_partition[:, T_partition[element_index, :]]
            end_point_1 = M_partition[:, boundary_edges[2, k]]
            end_point_2 = M_partition[:, boundary_edges[3, k]]

            # Perform the integration for each local test basis function
            for beta in range(number_of_test_local_basis):
                temp = Gauss_quadrature_for_line_integral_test_triangle(Neumann_boundary_function_name,
                                                                      Gauss_coefficient_reference_1D,
                                                                      Gauss_point_reference_1D,
                                                                      end_point_1, end_point_2, vertices,
                                                                      test_basis_type, beta,
                                                                      test_derivative_degree_x, test_derivative_degree_y)

                R = coo_matrix((temp, (T_basis_test[beta, element_index],
                                       np.zeros_like(T_basis_test[beta, element_index]))), shape=(Nb, 1))
                r = R.toarray()
                # b[T_basis_test[beta, element_index], 0] += temp

    return r
