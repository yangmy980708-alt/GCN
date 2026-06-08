import numpy as np

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
sys.path.insert(0, '../Stokes_Ab')
from Stokes_Ab.Stokes_functions_data import Functions
sys.path.insert(0, '../matrixAb')
from matrixAb.Gauss_quadrature_for_line_integral_trial_test_triangle import Gauss_quadrature_for_line_integral_trial_test_triangle
from matrixAb.Gauss_quadrature_for_line_integral_test_triangle import Gauss_quadrature_for_line_integral_test_triangle
from matrixAb.assemble_matrix_from_volume_integral_triangle_global import assemble_matrix_from_volume_integral_triangle_global
from matrixAb.assemble_vector_from_volume_integral_triangle_global import assemble_vector_from_volume_integral_triangle_global
from matrixAb.assemble_matrix_from_volume_integral_triangle_up import  assemble_matrix_from_volume_integral_triangle_up


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
        coef_data = Functions.NumpyFunctions()

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

def assemble_fixed_matrix_vector_for_Poisson(M_partition, T_partition, M_basis_trial, T_basis_trial, T_basis_test, basis_type):

    coef_data = Functions.NumpyFunctions()

    A1 = assemble_matrix_from_volume_integral_triangle_global(getattr(coef_data, 'function_k11'),
                                                       M_partition, T_partition, M_basis_trial, T_basis_trial, T_basis_test,
                                                       basis_type, 1, 0, basis_type, 1, 0)
    A2 = assemble_matrix_from_volume_integral_triangle_global(getattr(coef_data, 'function_k22'),
                                                       M_partition, T_partition, M_basis_trial, T_basis_trial, T_basis_test,
                                                       basis_type, 0, 1, basis_type, 0, 1)
    A3 = assemble_matrix_from_volume_integral_triangle_global(getattr(coef_data, 'function_k12'),
                                                       M_partition, T_partition, M_basis_trial, T_basis_trial, T_basis_test,
                                                       basis_type, 0, 1, basis_type, 1, 0)
    A4 = assemble_matrix_from_volume_integral_triangle_global(getattr(coef_data, 'function_k21'),
                                                       M_partition, T_partition, M_basis_trial, T_basis_trial, T_basis_test,
                                                       basis_type, 1, 0, basis_type, 0, 1)

    A = A1 + A2 + A3 + A4
    # b = assemble_vector_from_volume_integral_triangle_global(getattr(coef_data, 'function_f_Poisson'),
    #                                                   M_partition, T_partition, M_basis_trial, T_basis_test, basis_type, 0, 0)

    return A

def assemble_fixed_matrix_vector_for_t_Stokes_Taylor_Hood(M_partition, T_partition, M_basis_trial_u, T_basis_trial_u, T_basis_test_u, T_basis_p):

    coef_data = Functions.NumpyFunctions()

    A1 = assemble_matrix_from_volume_integral_triangle_global(getattr(coef_data, 'function_nu'), M_partition, T_partition, M_basis_trial_u, T_basis_trial_u, T_basis_test_u, 2, 1, 0, 2, 1, 0)
    A2 = assemble_matrix_from_volume_integral_triangle_global(getattr(coef_data, 'function_nu'), M_partition, T_partition, M_basis_trial_u, T_basis_trial_u, T_basis_test_u, 2, 0, 1, 2, 0, 1)
    A3 = assemble_matrix_from_volume_integral_triangle_global(getattr(coef_data, 'function_nu'), M_partition, T_partition, M_basis_trial_u, T_basis_trial_u, T_basis_test_u, 2, 1, 0, 2, 0, 1)
    A4 = assemble_matrix_from_volume_integral_triangle_global(getattr(coef_data, 'function_nu'), M_partition, T_partition, M_basis_trial_u, T_basis_trial_u, T_basis_test_u, 2, 0, 1, 2, 1, 0)
    A5 = assemble_matrix_from_volume_integral_triangle_up(getattr(coef_data, 'function_negativeone'), M_partition, T_partition, M_basis_trial_u, T_basis_trial_u, T_basis_p, 1, 0, 0, 2, 1, 0)
    A6 = assemble_matrix_from_volume_integral_triangle_up(getattr(coef_data, 'function_negativeone'), M_partition, T_partition, M_basis_trial_u, T_basis_trial_u, T_basis_p, 1, 0, 0, 2, 0, 1)

    temp = np.zeros((int(M_partition.shape[1]), int(M_partition.shape[1])))
    A = np.block([[2 * A1 + A2, A3, A5], [A4, 2 * A2 + A1, A6], [A5.T, A6.T, temp]])

    #质量矩阵
    # Me = assemble_matrix_from_volume_integral_triangle_global(getattr(coef_data, 'function_one'), M_partition, T_partition, M_basis_trial_u, T_basis_trial_u, T_basis_p, 2, 0, 0, 2, 0, 0)
    # temp3 = np.zeros((int(M_basis_trial_u.shape[1]), int(M_basis_trial_u.shape[1])))
    # temp2 = np.zeros((int(M_basis_trial_u.shape[1]), int(M_partition.shape[1])))
    # M = np.block([[Me, temp3, temp2], [temp3, Me, temp2], [temp2.T, temp2.T, temp]])

    return A, A5, A6

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

        coef_data = Functions.NumpyFunctions()

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