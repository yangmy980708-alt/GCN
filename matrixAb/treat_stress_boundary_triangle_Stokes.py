import numpy as np

def treat_stress_boundary_triangle_Stokes(stress_boundary_function_name_normal, stress_boundary_function_name_tangential, b, boundary_edges, M_partition, T_partition, T_basis_test, number_of_test_local_basis, number_of_FE_nodes_u, Gauss_coefficient_reference_1D, Gauss_point_reference_1D, test_basis_type, test_derivative_degree_x, test_derivative_degree_y):
    """
    Deal with stress boundary edges and update the vector b.
    """

    # Get the number of boundary edges
    nbe = boundary_edges.shape[1]

    # Check all boundary edges
    for k in range(nbe):

        # If the kth boundary edge is a stress boundary edge in normal direction, add the corresponding line integrals to b
        if boundary_edges[0, k] == -2:
            element_index = boundary_edges[2, k]
            vertices = M_partition[:, T_partition[:, element_index]]
            end_point_1 = M_partition[:, boundary_edges[3, k]]
            end_point_2 = M_partition[:, boundary_edges[4, k]]

            for beta in range(number_of_test_local_basis):
                temp = Gauss_quadrature_for_line_integral_test_triangle(
                    stress_boundary_function_name_normal, Gauss_coefficient_reference_1D, Gauss_point_reference_1D,
                    end_point_1, end_point_2, vertices, test_basis_type, beta, test_derivative_degree_x, test_derivative_degree_y
                )

                # Update b based on the stress boundary conditions for normal direction
                b[T_basis_test[beta, element_index], 0] += boundary_edges[5, k] * temp
                b[number_of_FE_nodes_u + T_basis_test[beta, element_index], 0] += boundary_edges[6, k] * temp

        # If the kth boundary edge is a stress boundary edge in tangential direction, add the corresponding line integrals to b
        if boundary_edges[1, k] == -2:
            element_index = boundary_edges[2, k]
            vertices = M_partition[:, T_partition[:, element_index]]
            end_point_1 = M_partition[:, boundary_edges[3, k]]
            end_point_2 = M_partition[:, boundary_edges[4, k]]

            for beta in range(number_of_test_local_basis):
                temp = Gauss_quadrature_for_line_integral_test_triangle(
                    stress_boundary_function_name_tangential, Gauss_coefficient_reference_1D, Gauss_point_reference_1D,
                    end_point_1, end_point_2, vertices, test_basis_type, beta, test_derivative_degree_x, test_derivative_degree_y
                )

                # Update b based on the stress boundary conditions for tangential direction
                b[T_basis_test[beta, element_index], 0] += boundary_edges[7, k] * temp
                b[number_of_FE_nodes_u + T_basis_test[beta, element_index], 0] += boundary_edges[8, k] * temp

    return b
