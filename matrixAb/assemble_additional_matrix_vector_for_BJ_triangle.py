import numpy as np

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
sys.path.insert(0, '../matrixAb')
from matrixAb.functions_data import Functions
from matrixAb.Gauss_quadrature_for_line_integral_test_triangle import Gauss_quadrature_for_line_integral_test_triangle
from matrixAb.Gauss_quadrature_for_line_integral_trial_test_triangle import Gauss_quadrature_for_line_integral_trial_test_triangle


def assemble_additional_matrix_vector_for_BJ_triangle(A, b, interface_edges, M_partition_D, T_partition_D,
                                                      M_partition_S, T_partition_S,
                                                      T_basis_phi, T_basis_u, number_of_local_basis_phi,
                                                      number_of_local_basis_u,
                                                      number_of_unknowns_Darcy, number_of_FE_nodes_u,
                                                      Gauss_coefficient_reference_1D,
                                                      Gauss_point_reference_1D, basis_type_phi, basis_type_u):

    # Total number of interface edges
    nie = interface_edges.shape[1]

    # Loop over all interface edges
    for k in range(nie):
        Darcy_element_index = interface_edges[0, k].astype(int)
        Stokes_element_index = interface_edges[1, k].astype(int)

        Darcy_vertices = M_partition_D[:, T_partition_D[:, Darcy_element_index]]
        Stokes_vertices = M_partition_S[:, T_partition_S[:, Stokes_element_index]]

        end_point_1 = interface_edges[2:4, k]
        end_point_2 = interface_edges[4:6, k]

        # Derivative degrees for different basis functions
        derivative_degree_x_u = 0
        derivative_degree_y_u = 0

        # Process for phi basis functions (degree 1 in x, 0 in y)
        derivative_degree_x_phi = 1
        derivative_degree_y_phi = 0
        coef_data = Functions()

        # Loop over all basis functions for Darcy and Stokes
        for alpha in range(int(T_basis_phi.shape[0])):
            for beta in range(int(T_basis_u.shape[0])):
                temp = Gauss_quadrature_for_line_integral_trial_test_triangle(getattr(coef_data, 'function_k11_BJ_coe'),
                    Gauss_coefficient_reference_1D, Gauss_point_reference_1D,
                    end_point_1, end_point_2, Darcy_vertices, basis_type_phi, alpha, derivative_degree_x_phi,
                    derivative_degree_y_phi, Stokes_vertices, basis_type_u, beta, derivative_degree_x_u,
                    derivative_degree_y_u
                )
                # A15 update
                A[number_of_unknowns_Darcy + T_basis_u[beta, Stokes_element_index], T_basis_phi[
                    alpha, Darcy_element_index]] += interface_edges[8, k] ** 2 * temp
                # A19 update
                A[number_of_unknowns_Darcy + number_of_FE_nodes_u + T_basis_u[beta, Stokes_element_index], T_basis_phi[
                    alpha, Darcy_element_index]] += interface_edges[8, k] * interface_edges[9, k] * temp

        # Process for phi basis functions (degree 0 in x, 1 in y)
        derivative_degree_x_phi = 0
        derivative_degree_y_phi = 1

        for alpha in range(int(T_basis_phi.shape[0])):
            for beta in range(int(T_basis_u.shape[0])):
                temp = Gauss_quadrature_for_line_integral_trial_test_triangle(getattr(coef_data, 'function_k12_BJ_coe'),
                    Gauss_coefficient_reference_1D, Gauss_point_reference_1D,
                    end_point_1, end_point_2, Darcy_vertices, basis_type_phi, alpha, derivative_degree_x_phi,
                    derivative_degree_y_phi, Stokes_vertices, basis_type_u, beta, derivative_degree_x_u,
                    derivative_degree_y_u
                )
                # A16 update
                A[number_of_unknowns_Darcy + T_basis_u[beta, Stokes_element_index], T_basis_phi[
                    alpha, Darcy_element_index]] += interface_edges[8, k] ** 2 * temp
                # A20 update
                A[number_of_unknowns_Darcy + number_of_FE_nodes_u + T_basis_u[beta, Stokes_element_index], T_basis_phi[
                    alpha, Darcy_element_index]] += interface_edges[8, k] * interface_edges[9, k] * temp

        # Process for phi basis functions (degree 1 in x, 0 in y)
        derivative_degree_x_phi = 1
        derivative_degree_y_phi = 0

        for alpha in range(int(T_basis_phi.shape[0])):
            for beta in range(T_basis_u.shape[0]):
                temp = Gauss_quadrature_for_line_integral_trial_test_triangle(getattr(coef_data, 'function_k21_BJ_coe'),
                    Gauss_coefficient_reference_1D, Gauss_point_reference_1D,
                    end_point_1, end_point_2, Darcy_vertices, basis_type_phi, alpha, derivative_degree_x_phi,
                    derivative_degree_y_phi, Stokes_vertices, basis_type_u, beta, derivative_degree_x_u,
                    derivative_degree_y_u
                )
                # A17 update
                A[number_of_unknowns_Darcy + T_basis_u[beta, Stokes_element_index], T_basis_phi[
                    alpha, Darcy_element_index]] += interface_edges[9, k] * interface_edges[8, k] * temp
                # A21 update
                A[number_of_unknowns_Darcy + number_of_FE_nodes_u + T_basis_u[beta, Stokes_element_index], T_basis_phi[
                    alpha, Darcy_element_index]] += interface_edges[9, k] * interface_edges[9, k] * temp

        # Process for phi basis functions (degree 0 in x, 1 in y)
        derivative_degree_x_phi = 0
        derivative_degree_y_phi = 1

        for alpha in range(int(T_basis_phi.shape[0])):
            for beta in range(int(T_basis_u.shape[0])):
                temp = Gauss_quadrature_for_line_integral_trial_test_triangle(getattr(coef_data,'function_k22_BJ_coe'),
                    Gauss_coefficient_reference_1D, Gauss_point_reference_1D,
                    end_point_1, end_point_2, Darcy_vertices, basis_type_phi, alpha, derivative_degree_x_phi,
                    derivative_degree_y_phi, Stokes_vertices, basis_type_u, beta, derivative_degree_x_u,
                    derivative_degree_y_u
                )
                # A18 update
                A[number_of_unknowns_Darcy + T_basis_u[beta, Stokes_element_index], T_basis_phi[
                    alpha, Darcy_element_index]] += interface_edges[9, k] * interface_edges[8, k] * temp
                # A22 update
                A[number_of_unknowns_Darcy + number_of_FE_nodes_u + T_basis_u[beta, Stokes_element_index], T_basis_phi[
                    alpha, Darcy_element_index]] += interface_edges[9, k] * interface_edges[9, k] * temp

        # Update b vector for gravity terms
        for beta in range(int(T_basis_u.shape[0])):
            temp = Gauss_quadrature_for_line_integral_test_triangle(getattr(coef_data, 'function_gravity_depth'),
                Gauss_coefficient_reference_1D, Gauss_point_reference_1D,
                end_point_1, end_point_2, Stokes_vertices, basis_type_u, beta, derivative_degree_x_u,
                derivative_degree_y_u
            )
            # b3 update
            b[number_of_unknowns_Darcy + T_basis_u[beta, Stokes_element_index], 0] += interface_edges[6, k] * temp
            # b4 update
            b[number_of_unknowns_Darcy + number_of_FE_nodes_u + T_basis_u[beta, Stokes_element_index], 0] += \
            interface_edges[7, k] * temp

    return A, b
