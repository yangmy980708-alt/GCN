import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from scipy.sparse import lil_matrix,coo_matrix
import os
import sys
# 获取当前文件的上一级目录，也就是项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
sys.path.insert(0, '../Stokes_Ab')
from Stokes_Ab.t_Ab import assemble_fixed_matrix_vector_for_Poisson, assemble_fixed_matrix_vector_for_t_Stokes_Taylor_Hood
from Stokes_Ab.Stokes_t_functions_data import Functions
from Stokes_Ab.generate_boundary_nodes_edges import generate_boundary_nodes_edges
from Stokes_Ab.generate_boundary_nodes_edges_Stokes import generate_boundary_nodes_edges_Stokes
from Stokes_Ab.generate_interface_edges_StokesDarcy import generate_interface_edges_StokesDarcy

sys.path.insert(0, '../matrixAb')
from matrixAb.generate_Gauss_reference_1D import generate_Gauss_reference_1D
from matrixAb.generate_M_T_triangle import generate_M_T_triangle
from matrixAb.get_initial_vector import get_initial_vector
from matrixAb.get_initial_vector_Stokes import get_initial_vector_Stokes
from matrixAb.fix_pressure_time_Stokes import fix_pressure_time_Stokes
from matrixAb.treat_Dirichlet_boundary_time_Stokes_with_Darcy import treat_Dirichlet_boundary_time_Stokes_with_Darcy
from matrixAb.treat_Dirichlet_boundary_time_triangle import treat_Dirichlet_boundary_time_triangle
from matrixAb.assemble_vector_from_volume_integral_time_triangle import assemble_vector_from_volume_integral_time_triangle
from matrixAb.assemble_matrix_from_interface_conditions_BJSJ_triangle import assemble_matrix_from_interface_conditions_BJSJ_triangle
from matrixAb.assemble_additional_matrix_vector_for_BJ_triangle import assemble_additional_matrix_vector_for_BJ_triangle
from matrixAb.assemble_matrix_from_volume_integral_triangle import assemble_matrix_from_volume_integral_triangle


def unsteady_Stokes_Darcy_fem(theta, initial_t, end_t, dt,
    left_S, right_S, bottom_S, top_S,
    left_D, right_D, bottom_D, top_D,
    h_partition_S, h_partition_D,
    fix_pressure, Dirichlet_switch,
    Darcy_scaling_constant):


    [M_partition_D, T_partition_D] = generate_M_T_triangle(left_D, right_D, bottom_D, top_D, h_partition_D, 1)
    [M_basis_phi, T_basis_phi] = generate_M_T_triangle(left_D, right_D, bottom_D, top_D, h_partition_D, 2)
    [M_basis_u, T_basis_u] = generate_M_T_triangle(left_S, right_S, bottom_S, top_S, h_partition_S, 2)
    [M_basis_p, T_basis_p] = generate_M_T_triangle(left_S, right_S, bottom_S, top_S, h_partition_S, 1)
    M_partition_S = M_basis_p
    T_partition_S = T_basis_p
    number_of_local_basis_phi = 6
    number_of_local_basis_u = 6
    # Step 1: Get some basic quantities which will be used in the code.
    N_t = (end_t - initial_t) / dt;
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
    A_Darcy = assemble_fixed_matrix_vector_for_Poisson(M_partition_D, T_partition_D, M_basis_phi,
                                                       T_basis_phi, T_basis_phi, 2)
    # assemble_fixed_matrix_vector_for_Stokes_Taylor_Hood(M_partition, T_partition, M_basis_trial_u, T_basis_trial_u, T_basis_test_u, T_basis_p):
    [A_Stokes, A5, A6] = assemble_fixed_matrix_vector_for_t_Stokes_Taylor_Hood(M_partition_S, T_partition_S, M_basis_u,
                                                                               T_basis_u, T_basis_u, T_basis_p)

    A = coo_matrix((int(number_of_all_unknowns), int(number_of_all_unknowns))).toarray()
    A[:int(number_of_unknowns_Darcy), :int(number_of_unknowns_Darcy)] = Darcy_scaling_constant * A_Darcy
    A[number_of_unknowns_Darcy:number_of_all_unknowns, number_of_unknowns_Darcy:number_of_all_unknowns] = A_Stokes
    del A_Darcy, A_Stokes

    interface_end_point_1 = [left_D, top_D]
    interface_end_point_2 = [right_D, top_D]
    interface_edges = generate_interface_edges_StokesDarcy(interface_end_point_1, interface_end_point_2, left_S,
                                                           right_S, bottom_S, top_S, left_D, right_D, bottom_D, top_D,
                                                           h_partition_S, h_partition_D)

    [Gauss_coefficient_reference_1D, Gauss_point_reference_1D] = generate_Gauss_reference_1D(4)

    A = assemble_matrix_from_interface_conditions_BJSJ_triangle(A, interface_edges, Darcy_scaling_constant,
                                                                M_partition_D, T_partition_D, M_partition_S,
                                                                T_partition_S, T_basis_phi, T_basis_u,
                                                                number_of_local_basis_phi, number_of_local_basis_u,
                                                                number_of_unknowns_Darcy, number_of_FE_nodes_u,
                                                                Gauss_coefficient_reference_1D,
                                                                Gauss_point_reference_1D, 2, 0, 0, 2, 0, 0)

    # b_BJ=coo_matrix((int(number_of_all_unknowns), 1)).toarray()#np.zeros(number_of_all_unknowns,1)
    # [A,b_BJ]=assemble_additional_matrix_vector_for_BJ_triangle(A,b_BJ,interface_edges,M_partition_D,T_partition_D,M_partition_S,T_partition_S,T_basis_phi,T_basis_u,number_of_local_basis_phi,number_of_local_basis_u,number_of_unknowns_Darcy,number_of_FE_nodes_u,Gauss_coefficient_reference_1D,Gauss_point_reference_1D,2,2);

    coef_data = Functions.NumpyFunctions()

    MD = assemble_matrix_from_volume_integral_triangle(getattr(coef_data, 'function_one'), M_partition_D, T_partition_D,
                                                       M_basis_phi, T_basis_phi, T_basis_phi, 2, 0, 0, 2, 0, 0);
    # assemble_matrix_from_volume_integral_triangle(coefficient_function,M_partition,T_partition,M_basis_trial,T_basis_trial,T_basis_test,trial_basis_type,trial_derivative_degree_x,trial_derivative_degree_y,test_basis_type,test_derivative_degree_x,test_derivative_degree_y):
    MS = assemble_matrix_from_volume_integral_triangle(getattr(coef_data, 'function_one'), M_partition_S, T_partition_S,
                                                       M_basis_u, T_basis_u, T_basis_u, 2, 0, 0, 2, 0, 0);
    M = coo_matrix((int(number_of_all_unknowns), int(number_of_all_unknowns))).toarray()
    M[:int(number_of_unknowns_Darcy), :int(number_of_unknowns_Darcy)] = Darcy_scaling_constant * MD;
    M[number_of_unknowns_Darcy:number_of_unknowns_Darcy + number_of_FE_nodes_u,
    number_of_unknowns_Darcy:number_of_unknowns_Darcy + number_of_FE_nodes_u] = MS;
    M[number_of_unknowns_Darcy + number_of_FE_nodes_u:number_of_unknowns_Darcy + 2 * number_of_FE_nodes_u,
    number_of_unknowns_Darcy + number_of_FE_nodes_u:number_of_unknowns_Darcy + 2 * number_of_FE_nodes_u] = MS;
    del MD, MS

    [boundary_nodes_D, boundary_edges_D] = generate_boundary_nodes_edges(N1_basis_phi, N2_basis_phi, N1_partition_D,
                                                                         N2_partition_D)

    [boundary_nodes_S, boundary_edges_S] = generate_boundary_nodes_edges_Stokes(N1_basis_u, N2_basis_u, N1_partition_S,
                                                                                N2_partition_S)
    T_basis_u_adjust = T_basis_u + number_of_unknowns_Darcy
    A_fixed = M / dt + theta * A

    X_old_Darcy = get_initial_vector(getattr(coef_data, 'function_initial_phi'), M_basis_phi)
    X_old_Stokes = get_initial_vector_Stokes(getattr(coef_data, 'function_initial_u1'),
                                             getattr(coef_data, 'function_initial_u2'),
                                             getattr(coef_data, 'function_initial_p'), M_basis_u, M_basis_p)
    X_old = np.vstack((X_old_Darcy, X_old_Stokes))

    for n in range(int(N_t)):
        current_time = initial_t + dt * (n + 1)
        # (coefficient_function,current_time,M_partition,T_partition,M_basis_trial,T_basis_test,
        # test_basis_type,
        # test_derivative_degree_x,test_derivative_degree_y)
        '''Darcy'''
        b1 = assemble_vector_from_volume_integral_time_triangle(getattr(coef_data, 'function_f_Poisson'), current_time,
                                                                M_partition_D, T_partition_D, M_basis_phi,
                                                                T_basis_phi, 2, 0, 0)
        b2 = assemble_vector_from_volume_integral_time_triangle(getattr(coef_data, 'function_f_Poisson'),
                                                                current_time - dt, M_partition_D, T_partition_D,
                                                                M_basis_phi,
                                                                T_basis_phi, 2, 0, 0)
        b_Darcy = theta * b1 + (1 - theta) * b2

        '''Stokes'''
        b1 = assemble_vector_from_volume_integral_time_triangle(getattr(coef_data, 'function_f1_Stokes'), current_time,
                                                                M_partition_S,
                                                                T_partition_S, M_basis_u, T_basis_u, 2, 0, 0)
        b2 = assemble_vector_from_volume_integral_time_triangle(getattr(coef_data, 'function_f1_Stokes'),
                                                                current_time - dt, M_partition_S,
                                                                T_partition_S, M_basis_u, T_basis_u, 2, 0, 0)
        b3 = assemble_vector_from_volume_integral_time_triangle(getattr(coef_data, 'function_f2_Stokes'), current_time,
                                                                M_partition_S,
                                                                T_partition_S, M_basis_u, T_basis_u, 2, 0, 0)
        b4 = assemble_vector_from_volume_integral_time_triangle(getattr(coef_data, 'function_f2_Stokes'),
                                                                current_time - dt, M_partition_S,
                                                                T_partition_S, M_basis_u, T_basis_u, 2, 0, 0)
        temp = np.zeros((number_of_FE_nodes_p, 1))

        b_Stokes = np.vstack((
            theta * b1 + (1 - theta) * b2,
            theta * b3 + (1 - theta) * b4,
            temp
        ))

        b = np.vstack((Darcy_scaling_constant * b_Darcy, b_Stokes))
        # b = b+b_BJ

        b = b + M * X_old / dt - (1 - theta) * A * X_old

        [A_fixed, b] = treat_Dirichlet_boundary_time_triangle(getattr(coef_data, 'function_g_Poisson'), current_time,
                                                              A_fixed, b,
                                                              boundary_nodes_D, M_basis_phi)

        if Dirichlet_switch == 1:
            [A_fixed, b] = treat_Dirichlet_boundary_time_Stokes_with_Darcy(getattr(coef_data, 'function_g1_Stokes'),
                                                                           getattr(coef_data, 'function_g2_Stokes'),
                                                                           current_time, A_fixed, b, boundary_nodes_S,
                                                                           M_basis_u, number_of_FE_nodes_u,
                                                                           number_of_unknowns_Darcy)

        if n == 1:
            b[number_of_unknowns_Darcy + 2 * number_of_FE_nodes_u:] = 0
            del A5, A6

        number_of_unknows_before_p = 2 * number_of_FE_nodes_u + number_of_unknowns_Darcy
        fixed_p_index = 1
        [A_fixed, b] = fix_pressure_time_Stokes(getattr(coef_data, 'function_fix_p'), current_time, fix_pressure,
                                                A_fixed, b,
                                                number_of_unknows_before_p, fixed_p_index, M_basis_p)

        X = np.linalg.solve(A_fixed, b)

        X_old = X

        if n == N_t:
            h_basis_phi = h_partition_D / 2
            h_basis_u = h_partition_S / 2
            h_basis_p = h_partition_S

            phih = X[:number_of_unknowns_Darcy]
            uh1 = X[number_of_unknowns_Darcy:number_of_unknowns_Darcy + number_of_FE_nodes_u]
            uh2 = X[number_of_unknowns_Darcy + number_of_FE_nodes_u:number_of_unknowns_Darcy + 2 * number_of_FE_nodes_u]
            ph = X[number_of_unknowns_Darcy + 2 * number_of_FE_nodes_u:number_of_all_unknowns]

    r_D_boundary = X[boundary_nodes_D[1, :].astype(int), :]
    r_Stokes_boundary = X[boundary_nodes_S[2, :].astype(int), :]
    phih = X[:number_of_unknowns_Darcy]
    uh1 = X[number_of_unknowns_Darcy:number_of_unknowns_Darcy + number_of_FE_nodes_u]
    uh2 = X[number_of_unknowns_Darcy + number_of_FE_nodes_u:number_of_unknowns_Darcy + 2 * number_of_FE_nodes_u]
    ph = X[number_of_unknowns_Darcy + 2 * number_of_FE_nodes_u:number_of_all_unknowns]

    return phih, uh1,uh2,ph, r_D_boundary, r_Stokes_boundary, boundary_nodes_D, boundary_nodes_S
