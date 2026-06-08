import numpy as np
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
sys.path.insert(0, '../matrixAb')
from matrixAb.functions_data import Functions
from matrixAb.Gauss_quadrature_for_line_integral_trial_test_triangle import Gauss_quadrature_for_line_integral_trial_test_triangle

def assemble_matrix_from_interface_conditions_BJSJ_triangle(A, interface_edges, Darcy_scaling_constant, M_partition_D, T_partition_D, M_partition_S, T_partition_S, T_basis_phi, T_basis_u, number_of_local_basis_phi, number_of_local_basis_u, number_of_unknowns_Darcy, number_of_FE_nodes_u, Gauss_coefficient_reference_1D, Gauss_point_reference_1D, basis_type_phi, derivative_degree_x_phi, derivative_degree_y_phi, basis_type_u, derivative_degree_x_u, derivative_degree_y_u):
    # nie: the total number of all the interface edges.
    nie = interface_edges.shape[1]
    for k in range(int(nie)):
        Darcy_element_index = interface_edges[0, k].astype(int)
        Stokes_element_index = interface_edges[1, k].astype(int)
        # Go through all interface edges.
        Darcy_vertices = M_partition_D[:, T_partition_D[:, Darcy_element_index]]
        Stokes_vertices = M_partition_S[:, T_partition_S[:, Stokes_element_index]]
        end_point_1 = interface_edges[2:4, k]
        end_point_2 = interface_edges[4:6, k]

        coef_data = Functions()

        for alpha in range(int(T_basis_u.shape[0])):  # number_of_local_basis_u
            for beta in range(int(T_basis_phi.shape[0])):  # number_of_local_basis_phi
                temp = Gauss_quadrature_for_line_integral_trial_test_triangle(getattr(coef_data, 'function_one'),
                                                                              Gauss_coefficient_reference_1D,
                                                                              Gauss_point_reference_1D, end_point_1,
                                                                              end_point_2,
                                                                              Stokes_vertices, basis_type_u, alpha,
                                                                              derivative_degree_x_u,
                                                                              derivative_degree_y_u,
                                                                              Darcy_vertices, basis_type_phi, beta,
                                                                              derivative_degree_x_phi,
                                                                              derivative_degree_y_phi)

                # Deal with A7
                A[T_basis_phi[beta, Darcy_element_index], number_of_unknowns_Darcy + T_basis_u[
                    alpha, Stokes_element_index]] -= Darcy_scaling_constant * interface_edges[6, k] * temp
                # Deal with A8
                A[T_basis_phi[beta, Darcy_element_index], number_of_unknowns_Darcy + number_of_FE_nodes_u + T_basis_u[
                    alpha, Stokes_element_index]] -= Darcy_scaling_constant * interface_edges[7, k] * temp

        for alpha in range(int(T_basis_phi.shape[0])):
            for beta in range(int(T_basis_u.shape[0])):
                temp = Gauss_quadrature_for_line_integral_trial_test_triangle(getattr(coef_data, 'function_gravity'),
                                                                              Gauss_coefficient_reference_1D,
                                                                              Gauss_point_reference_1D,
                                                                              end_point_1, end_point_2,
                                                                              Darcy_vertices, basis_type_phi,
                                                                              alpha, derivative_degree_x_phi,
                                                                              derivative_degree_y_phi,
                                                                              Stokes_vertices, basis_type_u, beta,
                                                                              derivative_degree_x_u, derivative_degree_y_u)
                # Deal with A9
                A[number_of_unknowns_Darcy + T_basis_u[beta, Stokes_element_index], T_basis_phi[
                    alpha, Darcy_element_index]] += interface_edges[6, k] * temp
                # Deal with A12
                A[number_of_unknowns_Darcy + number_of_FE_nodes_u + T_basis_u[beta, Stokes_element_index], T_basis_phi[
                    alpha, Darcy_element_index]] += interface_edges[7, k] * temp

        for alpha in range(T_basis_u.shape[0]):
            for beta in range(T_basis_u.shape[0]):
                temp = Gauss_quadrature_for_line_integral_trial_test_triangle(getattr(coef_data, 'function_BJSJ_coefficient'), Gauss_coefficient_reference_1D, Gauss_point_reference_1D, end_point_1,
                    end_point_2,
                    Stokes_vertices, basis_type_u, alpha, derivative_degree_x_u, derivative_degree_y_u,
                    Stokes_vertices, basis_type_u, beta, derivative_degree_x_u, derivative_degree_y_u)

                # Deal with A10
                A[number_of_unknowns_Darcy + T_basis_u[beta, Stokes_element_index], number_of_unknowns_Darcy +
                  T_basis_u[alpha, Stokes_element_index]] += interface_edges[8, k] ** 2 * temp
                # Deal with A11
                A[number_of_unknowns_Darcy + T_basis_u[
                    beta, Stokes_element_index], number_of_unknowns_Darcy + number_of_FE_nodes_u + T_basis_u[
                      alpha, Stokes_element_index]] += interface_edges[9, k] * interface_edges[8, k] * temp
                # Deal with A13
                A[number_of_unknowns_Darcy + number_of_FE_nodes_u + T_basis_u[
                    beta, Stokes_element_index], number_of_unknowns_Darcy + T_basis_u[alpha, Stokes_element_index]] += \
                interface_edges[8, k] * interface_edges[9, k] * temp
                # Deal with A14
                A[number_of_unknowns_Darcy + number_of_FE_nodes_u + T_basis_u[
                    beta, Stokes_element_index], number_of_unknowns_Darcy + number_of_FE_nodes_u + T_basis_u[
                      alpha, Stokes_element_index]] += interface_edges[9, k] ** 2 * temp

    return A
