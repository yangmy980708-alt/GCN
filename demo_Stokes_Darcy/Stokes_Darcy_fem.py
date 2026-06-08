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
from Stokes_Ab.Ab import assemble_fixed_matrix_vector_for_Poisson, assemble_fixed_matrix_vector_for_Stokes_Taylor_Hood
from Stokes_Ab.Stokes_functions_data import Functions
from Stokes_Ab.generate_boundary_nodes_edges import generate_boundary_nodes_edges
from Stokes_Ab.generate_boundary_nodes_edges_Stokes import generate_boundary_nodes_edges_Stokes
from Stokes_Ab.generate_interface_edges_StokesDarcy import generate_interface_edges_StokesDarcy

sys.path.insert(0, '../matrixAb')
from matrixAb.generate_Gauss_reference_1D import generate_Gauss_reference_1D
from matrixAb.generate_M_T_triangle import generate_M_T_triangle
from matrixAb.assemble_matrix_from_interface_conditions_BJSJ_triangle import assemble_matrix_from_interface_conditions_BJSJ_triangle
# from matrixAb.generate_boundary_nodes_edges import generate_boundary_nodes_edges
from matrixAb.treat_Dirichlet_boundary_triangle import treat_Dirichlet_boundary_triangle
from matrixAb.treat_Dirichlet_boundary_Stokes_with_Darcy import treat_Dirichlet_boundary_Stokes_with_Darcy
from matrixAb.fix_pressure_Stokes import fix_pressure_Stokes

def steady_Stokes_Darcy_fem(
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

    A[:int(number_of_unknowns_Darcy), :int(number_of_unknowns_Darcy)] = Darcy_scaling_constant * A_Darcy
    b[:number_of_unknowns_Darcy] = Darcy_scaling_constant * b_Darcy

    A[number_of_unknowns_Darcy:number_of_all_unknowns, number_of_unknowns_Darcy:number_of_all_unknowns] = A_Stokes
    b[number_of_unknowns_Darcy:number_of_all_unknowns] = b_Stokes

    del A_Darcy, b_Darcy, A_Stokes, b_Stokes

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

    [boundary_nodes_D, boundary_edges_D] = generate_boundary_nodes_edges(N1_basis_phi, N2_basis_phi, N1_partition_D,
                                                                         N2_partition_D)

    coef_data = Functions.NumpyFunctions()
    [A, b] = treat_Dirichlet_boundary_triangle(getattr(coef_data, 'function_g_Poisson'), A, b, boundary_nodes_D,
                                               M_basis_phi)

    [boundary_nodes_S, boundary_edges_S] = generate_boundary_nodes_edges_Stokes(N1_basis_u, N2_basis_u, N1_partition_S,
                                                                                N2_partition_S)

    T_basis_u_adjust = T_basis_u + number_of_unknowns_Darcy

    boundary_nodes_S[2, :] = boundary_nodes_S[2, :] + number_of_unknowns_Darcy

    if Dirichlet_switch == 1:
        [A, b] = treat_Dirichlet_boundary_Stokes_with_Darcy(getattr(coef_data, 'function_g1_Stokes'),
                                                            getattr(coef_data, 'function_g2_Stokes'), A, b,
                                                            boundary_nodes_S, M_basis_u, number_of_FE_nodes_u,
                                                            number_of_unknowns_Darcy)

    number_of_unknows_before_p = 2 * number_of_FE_nodes_u + number_of_unknowns_Darcy
    fixed_p_index = 1
    [A, b] = fix_pressure_Stokes(getattr(coef_data, 'function_fix_p'), fix_pressure, A, b, number_of_unknows_before_p,
                                 fixed_p_index,
                                 M_basis_p)

    r = np.linalg.solve(A, b)
    r_D_boundary = r[boundary_nodes_D[1, :].astype(int), :]
    r_Stokes_boundary = r[boundary_nodes_S[2, :].astype(int), :]

    phih = r[:number_of_unknowns_Darcy]
    uh1 = r[number_of_unknowns_Darcy:number_of_unknowns_Darcy + number_of_FE_nodes_u]
    uh2 = r[number_of_unknowns_Darcy + number_of_FE_nodes_u:number_of_unknowns_Darcy + 2 * number_of_FE_nodes_u]
    ph = r[number_of_unknowns_Darcy + 2 * number_of_FE_nodes_u:number_of_all_unknowns]

    return phih, uh1,uh2,ph, r_D_boundary, r_Stokes_boundary, boundary_nodes_D, boundary_nodes_S
