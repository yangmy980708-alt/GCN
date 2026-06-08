import numpy as np

def treat_Robin_boundary_triangle_Stokes(
    stress_boundary_function_name_normal, stress_boundary_function_name_tangential,
    Robin_boundary_function_name_normal, Robin_boundary_function_name_tangential,
    A, b, boundary_edges, M_partition, T_partition, T_basis_trial, T_basis_test,
    number_of_trial_local_basis, number_of_test_local_basis, number_of_FE_nodes_u,
    Gauss_coefficient_reference_1D, Gauss_point_reference_1D, trial_basis_type,
    trial_derivative_degree_x, trial_derivative_degree_y, test_basis_type,
    test_derivative_degree_x, test_derivative_degree_y
):
    """
    Deal with Robin boundary edges for Stokes problem. This function updates the matrix A and vector b
    based on the Robin boundary conditions in both normal and tangential directions.
    """

    # Get the number of boundary edges
    nbe = boundary_edges.shape[1]

    # Check all boundary edges
    for k in range(nbe):

        # If the kth boundary edge is a Robin boundary edge in normal direction, update A and b
        if boundary_edges[0, k] == -3:
            element_index = boundary_edges[2, k]
            vertices = M_partition[:, T_partition[:, element_index]]
            end_point_1 = M_partition[:, boundary_edges[3, k]]
            end_point_2 = M_partition[:, boundary_edges[4, k]]

            for alpha in range(number_of_trial_local_basis):
                for beta in range(number_of_test_local_basis):
                    temp = Gauss_quadrature_for_line_integral_trial_test_triangle(
                        Robin_boundary_function_name_normal, Gauss_coefficient_reference_1D,
                        Gauss_point_reference_1D, end_point_1, end_point_2, vertices, trial_basis_type,
                        alpha, trial_derivative_degree_x, trial_derivative_degree_y, vertices, test_basis_type,
                        beta, test_derivative_degree_x, test_derivative_degree_y
                    )
                    # Update A matrix (from Notes for tool box of standard triangular FE)
                    A[T_basis_test[beta, element_index], T_basis_trial[alpha, element_index]] += boundary_edges[5, k] * boundary_edges[5, k] * temp
                    A[T_basis_test[beta, element_index], number_of_FE_nodes_u + T_basis_trial[alpha, element_index]] += boundary_edges[6, k] * boundary_edges[5, k] * temp
                    A[number_of_FE_nodes_u + T_basis_test[beta, element_index], T_basis_trial[alpha, element_index]] += boundary_edges[5, k] * boundary_edges[6, k] * temp
                    A[number_of_FE_nodes_u + T_basis_test[beta, element_index], number_of_FE_nodes_u + T_basis_trial[alpha, element_index]] += boundary_edges[6, k] * boundary_edges[6, k] * temp

            # Update b vector (for normal boundary condition)
            for beta in range(number_of_test_local_basis):
                temp = Gauss_quadrature_for_line_integral_test_triangle(
                    stress_boundary_function_name_normal, Gauss_coefficient_reference_1D,
                    Gauss_point_reference_1D, end_point_1, end_point_2, vertices, test_basis_type,
                    beta, test_derivative_degree_x, test_derivative_degree_y
                )
                b[T_basis_test[beta, element_index], 0] += boundary_edges[5, k] * temp
                b[number_of_FE_nodes_u + T_basis_test[beta, element_index], 0] += boundary_edges[6, k] * temp

        # If the kth boundary edge is a Robin boundary edge in tangential direction, update A and b
        if boundary_edges[1, k] == -3:
            element_index = boundary_edges[2, k]
            vertices = M_partition[:, T_partition[:, element_index]]
            end_point_1 = M_partition[:, boundary_edges[3, k]]
            end_point_2 = M_partition[:, boundary_edges[4, k]]

            for alpha in range(number_of_trial_local_basis):
                for beta in range(number_of_test_local_basis):
                    temp = Gauss_quadrature_for_line_integral_trial_test_triangle(
                        Robin_boundary_function_name_tangential, Gauss_coefficient_reference_1D,
                        Gauss_point_reference_1D, end_point_1, end_point_2, vertices, trial_basis_type,
                        alpha, trial_derivative_degree_x, trial_derivative_degree_y, vertices, test_basis_type,
                        beta, test_derivative_degree_x, test_derivative_degree_y
                    )
                    # Update A matrix (from Notes for tool box of standard triangular FE)
                    A[T_basis_test[beta, element_index], T_basis_trial[alpha, element_index]] += boundary_edges[7, k] * boundary_edges[7, k] * temp
                    A[T_basis_test[beta, element_index], number_of_FE_nodes_u + T_basis_trial[alpha, element_index]] += boundary_edges[8, k] * boundary_edges[7, k] * temp
                    A[number_of_FE_nodes_u + T_basis_test[beta, element_index], T_basis_trial[alpha, element_index]] += boundary_edges[7, k] * boundary_edges[8, k] * temp
                    A[number_of_FE_nodes_u + T_basis_test[beta, element_index], number_of_FE_nodes_u + T_basis_trial[alpha, element_index]] += boundary_edges[8, k] * boundary_edges[8, k] * temp

            # Update b vector (for tangential boundary condition)
            for beta in range(number_of_test_local_basis):
                temp = Gauss_quadrature_for_line_integral_test_triangle(
                    stress_boundary_function_name_tangential, Gauss_coefficient_reference_1D,
                    Gauss_point_reference_1D, end_point_1, end_point_2, vertices, test_basis_type,
                    beta, test_derivative_degree_x, test_derivative_degree_y
                )
                b[T_basis_test[beta, element_index], 0] += boundary_edges[7, k] * temp
                b[number_of_FE_nodes_u + T_basis_test[beta, element_index], 0] += boundary_edges[8, k] * temp

    return A, b
