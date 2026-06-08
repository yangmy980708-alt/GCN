import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from scipy.sparse import lil_matrix,coo_matrix
import os
import sys
# 获取当前文件的上一级目录，也就是项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
sys.path.insert(0, '../matrixAb')
from matrixAb.assemble_fixed_matrix_vector_for_Poisson import assemble_fixed_matrix_vector_for_Poisson
from matrixAb.assemble_fixed_matrix_vector_for_Stokes_Taylor_Hood import assemble_fixed_matrix_vector_for_Stokes_Taylor_Hood
from matrixAb.generate_interface_edges_StokesDarcy import generate_interface_edges_StokesDarcy
from matrixAb.generate_Gauss_reference_1D import generate_Gauss_reference_1D
from matrixAb.generate_M_T_triangle import generate_M_T_triangle
from matrixAb.assemble_matrix_from_interface_conditions_BJSJ_triangle import assemble_matrix_from_interface_conditions_BJSJ_triangle
from matrixAb.generate_Gauss_reference_triangle import generate_Gauss_reference_triangle
from matrixAb.assemble_matrix_from_volume_integral_FE_triangle import assemble_matrix_from_volume_integral_FE_triangle
from matrixAb.assemble_vector_from_volume_integral_2FE_triangle import assemble_vector_from_volume_integral_2FE_triangle
from matrixAb.Rotate_from_nt_to_Cartesian import Rotate_from_nt_to_Cartesian
from matrixAb.assemble_additional_matrix_vector_for_BJ_triangle import assemble_additional_matrix_vector_for_BJ_triangle
from matrixAb.generate_boundary_nodes_edges import generate_boundary_nodes_edges
from matrixAb.treat_Dirichlet_boundary_triangle import treat_Dirichlet_boundary_triangle
from matrixAb.generate_boundary_nodes_edges_Stokes import generate_boundary_nodes_edges_Stokes
from matrixAb.Rotate_from_Cartesian_to_nt import Rotate_from_Cartesian_to_nt
from matrixAb.treat_Dirichlet_boundary_Stokes_with_Darcy import treat_Dirichlet_boundary_Stokes_with_Darcy
from matrixAb.fix_pressure_Stokes import fix_pressure_Stokes
from matrixAb.functions_data import Functions

def steady_NS_Darcy_fem(
    left_S, right_S, bottom_S, top_S,
    left_D, right_D, bottom_D, top_D,
    h_partition_S, h_partition_D,
    fix_pressure, Dirichlet_switch,
    Darcy_scaling_constant, nonlinear_tolerance, nonlinear_max_steps
):
    coef_data = Functions()
    [M_partition_D, T_partition_D] = generate_M_T_triangle(left_D, right_D, bottom_D, top_D, h_partition_D, 1)
    [M_basis_phi, T_basis_phi] = generate_M_T_triangle(left_D, right_D, bottom_D, top_D, h_partition_D, 2)
    [M_basis_u, T_basis_u] = generate_M_T_triangle(left_S, right_S, bottom_S, top_S, h_partition_S, 2)
    [M_basis_p, T_basis_p] = generate_M_T_triangle(left_S, right_S, bottom_S, top_S, h_partition_S, 1)
    M_partition_S = M_basis_p
    T_partition_S = T_basis_p
    number_of_local_basis_phi = 6
    number_of_local_basis_u = 6

    # Step 1: Get some basic quantities which will be used in the code.
    N1_partition_D = (right_D - left_D) / h_partition_D[0]
    N2_partition_D = (top_D - bottom_D) / h_partition_D[1]
    N1_partition_S = (right_S - left_S) / h_partition_S[0]
    N2_partition_S = (top_S - bottom_S) / h_partition_S[1]

    N1_basis_phi = N1_partition_D * 2
    N2_basis_phi = N2_partition_D * 2
    N1_basis_u = N1_partition_S * 2
    N2_basis_u = N2_partition_S * 2
    N1_basis_p = N1_partition_S
    N2_basis_p = N2_partition_S

    number_of_unknowns_Darcy = int(M_basis_phi.shape[1])  # M.shape[1]
    number_of_FE_nodes_u = int(M_basis_u.shape[1])
    number_of_FE_nodes_p = int(M_basis_p.shape[1])
    number_of_unknowns_Stokes = int(2 * number_of_FE_nodes_u + number_of_FE_nodes_p)
    number_of_all_unknowns = int(number_of_unknowns_Stokes + number_of_unknowns_Darcy)
    # assemble_fixed_matrix_vector_for_Poisson(M_partition, T_partition, M_basis_trial, T_basis_trial, T_basis_test, basis_type):
    [A_Darcy, b_Darcy] = assemble_fixed_matrix_vector_for_Poisson(M_partition_D, T_partition_D, M_basis_phi,
                                                                  T_basis_phi, T_basis_phi, 2)
    # assemble_fixed_matrix_vector_for_Stokes_Taylor_Hood(M_partition, T_partition, M_basis_trial_u, T_basis_trial_u, T_basis_test_u, T_basis_p):
    [A_Stokes, b_Stokes] = assemble_fixed_matrix_vector_for_Stokes_Taylor_Hood(M_partition_S, T_partition_S, M_basis_u,
                                                                               T_basis_u, T_basis_u, T_basis_p)

    A = coo_matrix((int(number_of_all_unknowns), int(number_of_all_unknowns))).toarray()
    b = coo_matrix((int(number_of_all_unknowns), 1)).toarray()

    # 将达西区域的矩阵和向量赋值到 A 和 b 的前部分
    A[:int(number_of_unknowns_Darcy), :int(number_of_unknowns_Darcy)] = Darcy_scaling_constant * A_Darcy
    b[:number_of_unknowns_Darcy] = Darcy_scaling_constant * b_Darcy

    # 将流体区域的矩阵和向量赋值到 A 和 b 的后部分
    A[number_of_unknowns_Darcy:number_of_all_unknowns, number_of_unknowns_Darcy:number_of_all_unknowns] = A_Stokes
    b[number_of_unknowns_Darcy:number_of_all_unknowns] = b_Stokes

    del A_Darcy, b_Darcy, A_Stokes, b_Stokes

    interface_end_point_1 = [left_S, top_S]
    interface_end_point_2 = [right_S, top_S]
    interface_edges = generate_interface_edges_StokesDarcy(interface_end_point_1, interface_end_point_2, left_S,
                                                           right_S, bottom_S, top_S, left_D, right_D, bottom_D, top_D,
                                                           h_partition_S, h_partition_D)

    # Gauss points and coefficients of 1D Gauss quadratures.
    [Gauss_coefficient_reference_1D, Gauss_point_reference_1D] = generate_Gauss_reference_1D(4)

    # Assemble the matrices from the line integrals from the three interface conditions(one is BJSJ).
    A = assemble_matrix_from_interface_conditions_BJSJ_triangle(A, interface_edges, Darcy_scaling_constant,
                                                                M_partition_D, T_partition_D, M_partition_S,
                                                                T_partition_S, T_basis_phi, T_basis_u,
                                                                number_of_local_basis_phi, number_of_local_basis_u,
                                                                number_of_unknowns_Darcy, number_of_FE_nodes_u,
                                                                Gauss_coefficient_reference_1D,
                                                                Gauss_point_reference_1D, 2, 0, 0, 2, 0, 0)

    # Assemble the additional matrices and vectors from the line integrals for the difference between BJ and BJSJ interface condition.
    # [A, b] = assemble_additional_matrix_vector_for_BJ_triangle(A, b, interface_edges, M_partition_D, T_partition_D,
    #                                                            M_partition_S, T_partition_S, T_basis_phi, T_basis_u,
    #                                                            number_of_local_basis_phi, number_of_local_basis_u,
    #                                                            number_of_unknowns_Darcy, number_of_FE_nodes_u,
    #                                                            Gauss_coefficient_reference_1D, Gauss_point_reference_1D,
    #                                                            2, 2)
    # AA=A_BJSJ-A_BJ#全零
    # bb=b_BJSJ-bBJ#全零

    [boundary_nodes_D, boundary_edges_D] = generate_boundary_nodes_edges(N1_basis_phi, N2_basis_phi, N1_partition_D,
                                                                         N2_partition_D)

    # Deal with Dirichlet boundary condition of the boundary of the Darcy domain.
    [A, b] = treat_Dirichlet_boundary_triangle(getattr(coef_data, 'function_g_Poisson'), A, b, boundary_nodes_D,
                                               M_basis_phi)

    # Get the information matrices for boundary nodes and boundary edges for the Stokes Domain.
    [boundary_nodes_S, boundary_edges_S] = generate_boundary_nodes_edges_Stokes(N1_basis_u, N2_basis_u, N1_partition_S,
                                                                                N2_partition_S)

    T_basis_u_adjust = T_basis_u + number_of_unknowns_Darcy

    boundary_nodes_S[2, :] = boundary_nodes_S[2, :] + number_of_unknowns_Darcy

    # Initialize the Newton iteration for the nonlinear term
    phih = np.zeros((number_of_unknowns_Darcy, 1))
    uh1 = np.zeros((number_of_FE_nodes_u, 1))
    uh2 = np.zeros((number_of_FE_nodes_u, 1))
    ph = np.zeros((number_of_FE_nodes_p, 1))
    test_error = 1000
    nonlinear_iteration_step = 1

    temp1 = np.zeros((number_of_FE_nodes_p, number_of_FE_nodes_p))
    temp2 = np.zeros((number_of_FE_nodes_u, number_of_FE_nodes_p))
    temp3 = np.zeros((number_of_FE_nodes_p, number_of_FE_nodes_u))
    temp4 = np.zeros((number_of_FE_nodes_p, 1))
    number_of_elements_S = int(2 * N1_partition_S * N2_partition_S)
    matrix_size_uu = [number_of_FE_nodes_u, number_of_FE_nodes_u]
    vector_size_u = number_of_FE_nodes_u

    [Gauss_coefficient_reference_triangle, Gauss_point_reference_triangle] = generate_Gauss_reference_triangle(9)

    while test_error > nonlinear_tolerance:
        Ac1 = assemble_matrix_from_volume_integral_FE_triangle(uh1, M_partition_S, T_partition_S,
                                                               M_basis_u, T_basis_u, T_basis_u, T_basis_u,
                                                               2, 0, 0, 2, 0, 0, 2, 1, 0)

        Ac2 = assemble_matrix_from_volume_integral_FE_triangle(uh1, M_partition_S, T_partition_S,
                                                               M_basis_u, T_basis_u, T_basis_u, T_basis_u,
                                                               2, 1, 0, 2, 0, 0, 2, 0, 0)

        Ac3 = assemble_matrix_from_volume_integral_FE_triangle(uh2, M_partition_S, T_partition_S,
                                                               M_basis_u, T_basis_u, T_basis_u, T_basis_u,
                                                               2, 0, 1, 2, 0, 0, 2, 0, 0)

        Ac4 = assemble_matrix_from_volume_integral_FE_triangle(uh1, M_partition_S, T_partition_S,
                                                               M_basis_u, T_basis_u, T_basis_u, T_basis_u,
                                                               2, 0, 0, 2, 0, 0, 2, 0, 1)

        Ac5 = assemble_matrix_from_volume_integral_FE_triangle(uh2, M_partition_S, T_partition_S,
                                                               M_basis_u, T_basis_u, T_basis_u, T_basis_u,
                                                               2, 0, 0, 2, 0, 0, 2, 1, 0)

        Ac6 = assemble_matrix_from_volume_integral_FE_triangle(uh2, M_partition_S, T_partition_S,
                                                               M_basis_u, T_basis_u, T_basis_u, T_basis_u,
                                                               2, 0, 0, 2, 0, 0, 2, 0, 1)

        A_sum = np.zeros((number_of_all_unknowns, number_of_all_unknowns))

        A_sum[number_of_unknowns_Darcy:number_of_all_unknowns,
        number_of_unknowns_Darcy:number_of_all_unknowns] = np.block([
            [Ac1 + Ac2 + Ac3, Ac4, temp2],
            [Ac5, Ac2 + Ac3 + Ac6, temp2],
            [temp3, temp3, temp1]])
        A_sum = A_sum + A

        # Form the vector for the nonlinear term and add it to the total vector.
        # vector_size_u = int(M_basis_u.shape[1])
        bc1 = assemble_vector_from_volume_integral_2FE_triangle(uh1, uh1, M_partition_S, T_partition_S,
                                                                T_basis_u,
                                                                vector_size_u, 2, 0, 0, 2, 0, 0, 2, 1, 0)

        bc2 = assemble_vector_from_volume_integral_2FE_triangle(uh2, uh1, M_partition_S, T_partition_S,
                                                                T_basis_u,
                                                                vector_size_u, 2, 0, 0, 2, 0, 0, 2, 0, 1)

        bc3 = assemble_vector_from_volume_integral_2FE_triangle(uh1, uh2, M_partition_S, T_partition_S,
                                                                T_basis_u,
                                                                vector_size_u, 2, 0, 0, 2, 0, 0, 2, 1, 0)

        bc4 = assemble_vector_from_volume_integral_2FE_triangle(uh2, uh2, M_partition_S, T_partition_S,
                                                                T_basis_u,
                                                                vector_size_u, 2, 0, 0, 2, 0, 0, 2, 0, 1)

        b_sum = np.zeros((number_of_all_unknowns, 1))

        b_sum[number_of_unknowns_Darcy:number_of_all_unknowns, :] = np.vstack([bc1 + bc2, bc3 + bc4, temp4])
        b_sum = b_sum + b

        if Dirichlet_switch == 2:
            A_sum = Rotate_from_Cartesian_to_nt(A_sum, boundary_nodes_S, number_of_FE_nodes_u)

        # Deal with Dirichlet boundary condition for Stokes domain.
        if Dirichlet_switch == 2:
            A_sum, b_sum = treat_Dirichlet_boundary_Stokes_with_Darcy(getattr(coef_data, 'function_sn'),
                                                                      getattr(coef_data, 'function_st'),
                                                                      A_sum, b_sum, boundary_nodes_S,
                                                                      M_basis_u, number_of_FE_nodes_u,
                                                                      number_of_unknowns_Darcy)
        elif Dirichlet_switch == 1:
            A_sum, b_sum = treat_Dirichlet_boundary_Stokes_with_Darcy(getattr(coef_data, 'function_g1_Stokes'),
                                                                      getattr(coef_data, 'function_g2_Stokes'),
                                                                      A_sum, b_sum, boundary_nodes_S,
                                                                      M_basis_u, number_of_FE_nodes_u,
                                                                      number_of_unknowns_Darcy)

            # Fix pressure at one node if necessary (fix_pressure=1).
            number_of_unknows_before_p = 2 * number_of_FE_nodes_u + number_of_unknowns_Darcy
            fixed_p_index = 1
            [A_sum, b_sum] = fix_pressure_Stokes(getattr(coef_data, 'function_fix_p'), fix_pressure, A_sum, b_sum,
                                                 number_of_unknows_before_p, fixed_p_index, M_basis_p)

            # Compute the numerical solution.
            r = np.linalg.solve(A_sum, b_sum)
            r_D_boundary = r[boundary_nodes_D[1, :].astype(int), :]
            r_Stokes_boundary = r[boundary_nodes_S[2, :].astype(int), :]

            if Dirichlet_switch == 2:
                r = Rotate_from_nt_to_Cartesian(r, boundary_nodes_S, number_of_FE_nodes_u)

            # Get the finite element solutions for u1, u2 and p.
            phih_old = phih
            uh1_old = uh1
            uh2_old = uh2
            ph_old = ph

            phih = r[:number_of_unknowns_Darcy]
            uh1 = r[number_of_unknowns_Darcy:number_of_unknowns_Darcy + number_of_FE_nodes_u]
            uh2 = r[number_of_unknowns_Darcy + number_of_FE_nodes_u:number_of_unknowns_Darcy + 2 * number_of_FE_nodes_u]
            ph = r[number_of_unknowns_Darcy + 2 * number_of_FE_nodes_u:number_of_all_unknowns]

            test_error = np.linalg.norm(phih - phih_old) + np.linalg.norm(uh1 - uh1_old) + np.linalg.norm(
                uh2 - uh2_old) + np.linalg.norm(ph - ph_old)

            nonlinear_iteration_step += 1  # 相当于 nonlinear_iteration_step = nonlinear_iteration_step + 1

            if nonlinear_iteration_step > nonlinear_max_steps:
                break
    else:
        print("循环结束")
    return phih, uh1, uh2, ph, r_D_boundary, r_Stokes_boundary, boundary_nodes_D, boundary_nodes_S

    print('phih = ', phih)
    print('uh1 = ', uh1)
    print('uh2', uh2)
    print('ph = ', ph)