import numpy as np
import torch
import matplotlib
matplotlib.use('TkAgg')
from scipy.sparse import lil_matrix,coo_matrix
import os
import sys
# 获取当前文件的上一级目录，也就是项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
sys.path.insert(0, '../matrixAb')
from matrixAb.generate_M_T_triangle import generate_M_T_triangle
from matrixAb.generate_boundary_nodes_edges import generate_boundary_nodes_edges
from matrixAb.generate_Gauss_reference_1D import generate_Gauss_reference_1D
from matrixAb.generate_boundary_nodes_edges_Stokes import generate_boundary_nodes_edges_Stokes
sys.path.insert(0, '../GCNN')
from GCNN.functions_data_torch import Functions
from GCNN.generate_interface_edges_StokesDarcy import generate_interface_edges_StokesDarcy
from GCNN.basis_Ab_uj import treat_Dirichlet_boundary_Stokes_with_Darcy, assemble_matrix_from_volume_integral_FE_triangle,\
    assemble_vector_from_volume_integral_2FE_triangle, generate_Gauss_reference_triangle,\
    treat_Dirichlet_boundary_triangle, assemble_additional_matrix_vector_for_BJ_triangle,\
    assemble_matrix_from_interface_conditions_BJSJ_triangle, fix_pressure_Stokes,ReshapeFix,\
    assemble_fixed_matrix_vector_for_Poisson, assemble_fixed_matrix_vector_for_Stokes_Taylor_Hood_uj,Double,\
assemble_matrix_from_volume_integral_triangle_global_uj,assemble_matrix_from_volume_integral_triangle_up_uj,\
assemble_vector_from_volume_integral_triangle_global_uj

def generate_residual_uj(Uf,l2gT_DS,
    left_S, right_S, bottom_S, top_S,
    left_D, right_D, bottom_D, top_D,
    h_partition_S, h_partition_D,
    fix_pressure, Dirichlet_switch,
    Darcy_scaling_constant, nonlinear_tolerance, nonlinear_max_steps
):
    coef_data = Functions()
    # 数据类型为Numpy
    [M_partition_D, T_partition_D] = generate_M_T_triangle(left_D, right_D, bottom_D, top_D, h_partition_D, 1)
    [M_basis_phi, T_basis_phi] = generate_M_T_triangle(left_D, right_D, bottom_D, top_D, h_partition_D, 2)
    [M_basis_u, T_basis_u] = generate_M_T_triangle(left_S, right_S, bottom_S, top_S, h_partition_S, 2)
    [M_basis_p, T_basis_p] = generate_M_T_triangle(left_S, right_S, bottom_S, top_S, h_partition_S, 1)
    M_partition_S = M_basis_p
    T_partition_S = T_basis_p
    number_of_local_basis_phi = 6
    number_of_local_basis_u = 6

    number_of_unknowns_Darcy = int(M_basis_phi.shape[1])  # M.shape[1]
    number_of_FE_nodes_u = int(M_basis_u.shape[1])
    number_of_FE_nodes_p = int(M_basis_p.shape[1])
    number_of_unknowns_Stokes = int(2 * number_of_FE_nodes_u + number_of_FE_nodes_p)
    number_of_all_unknowns = int(number_of_unknowns_Stokes + number_of_unknowns_Darcy)

    # 单独使用此段命令放入CUDA，但是运行时显示拼接错误
    # Darcy域
    M_partition_D = Double(M_partition_D)
    T_partition_D = Double(T_partition_D)
    M_basis_phi = Double(M_basis_phi)
    T_basis_phi = Double(T_basis_phi)
    # Stokes域
    M_partition_S = Double(M_partition_S)
    T_partition_S = Double(T_partition_S)
    M_basis_u = Double(M_basis_u)
    T_basis_u = Double(T_basis_u)
    M_basis_p = Double(M_basis_p)
    T_basis_p = Double(T_basis_p)
    # 改变形状放入CUDA
    # Darcy
    T_partition_D = T_partition_D.to(dtype=torch.long, device='cuda')
    T_basis_phi = T_basis_phi.to(dtype=torch.long, device='cuda')
    M_partition_D = M_partition_D.to(device='cuda')
    M_basis_phi = M_basis_phi.to(device='cuda')
    # Stokes
    M_partition_S = M_partition_S.to(device='cuda')
    T_partition_S = T_partition_S.to(dtype=torch.long, device='cuda')
    M_basis_u = M_basis_u.to(device='cuda')
    T_basis_u = T_basis_u.to(dtype=torch.long, device='cuda')
    M_basis_p = M_basis_p.to(device='cuda')
    T_basis_p = T_basis_p.to(dtype=torch.long, device='cuda')

    # U = ReshapeFix(Uf, [number_of_all_unknowns, 1], 'C')
    # U_D = U[:number_of_unknowns_Darcy]
    # U_u1 = U[number_of_unknowns_Darcy:number_of_unknowns_Darcy + number_of_FE_nodes_u,:]
    # U_u2 = U[number_of_unknowns_Darcy + number_of_FE_nodes_u:number_of_unknowns_Darcy + 2 * number_of_FE_nodes_u,:]
    # U_u12 = U[number_of_unknowns_Darcy:number_of_unknowns_Darcy + 2 * number_of_FE_nodes_u]
    # U_p = U[number_of_unknowns_Darcy+2*number_of_FE_nodes_u:number_of_all_unknowns,:]
    # U_NS = U[number_of_unknowns_Darcy:number_of_all_unknowns,:]

    # Uf = ReshapeFix(Uf, [number_of_all_unknowns, 1], 'C')
    # U_temp = torch.zeros([number_of_all_unknowns, 1]).double().to('cuda')
    # U = U_temp + Uf
    phih = torch.zeros((number_of_unknowns_Darcy, 1), dtype=torch.float32).to('cuda')
    uh1 = torch.zeros((number_of_FE_nodes_u, 1), dtype=torch.float32).to('cuda')
    uh2 = torch.zeros((number_of_FE_nodes_u, 1), dtype=torch.float32).to('cuda')
    ph = torch.zeros((number_of_FE_nodes_p, 1), dtype=torch.float32).to('cuda')
    temp1 = torch.zeros((number_of_FE_nodes_p, number_of_FE_nodes_p), dtype=torch.float32).to('cuda')
    temp2 = torch.zeros((number_of_FE_nodes_u, number_of_FE_nodes_p), dtype=torch.float32).to('cuda')
    temp3 = torch.zeros((number_of_FE_nodes_p, number_of_FE_nodes_u), dtype=torch.float32).to('cuda')
    temp4 = torch.zeros((number_of_FE_nodes_p, 1), dtype=torch.float32).to('cuda')
    matrix_size_uu = torch.tensor([number_of_FE_nodes_u, number_of_FE_nodes_u], dtype=torch.int32).to('cuda')
    vector_size_u = torch.tensor(number_of_FE_nodes_u, dtype=torch.int32).to('cuda')

    Re_Darcy_uj = assemble_fixed_matrix_vector_for_Poisson(phih,l2gT_DS,M_partition_D, T_partition_D, M_basis_phi,
                                                                  T_basis_phi, T_basis_phi, 2)
    # Re_stokes_uj = assemble_fixed_matrix_vector_for_Stokes_Taylor_Hood_uj(uh1, l2gT_DS, M_partition_S,
    #                                                                       T_partition_S, M_basis_u,
    #                                                                       T_basis_u, T_basis_u, T_basis_p)
    Re_A1_uj = assemble_matrix_from_volume_integral_triangle_global_uj(getattr(coef_data, 'function_nu'), uh1, l2gT_DS,
                                                                 M_partition_S, T_partition_S, M_basis_u,
                                                                          T_basis_u, T_basis_u, T_basis_p, 2, 1, 0, 2, 1, 0)
    Re_A2_uj = assemble_matrix_from_volume_integral_triangle_global_uj(getattr(coef_data, 'function_nu'), uh2, l2gT_DS,
                                                                       M_partition_S, T_partition_S, M_basis_u,
                                                                       T_basis_u, T_basis_u, T_basis_p, 2, 0, 1, 2, 0, 1)
    Re_A3_uj = assemble_matrix_from_volume_integral_triangle_global_uj(getattr(coef_data, 'function_nu'), uh2, l2gT_DS,
                                                                 M_partition_S, T_partition_S, M_basis_u,
                                                                 T_basis_u, T_basis_u, T_basis_p, 2, 1, 0, 2, 0, 1)
    Re_A4_uj = assemble_matrix_from_volume_integral_triangle_global_uj(getattr(coef_data, 'function_nu'), uh1, l2gT_DS,
                                                                 M_partition_S, T_partition_S, M_basis_u,
                                                                 T_basis_u, T_basis_u, T_basis_p, 2, 0, 1, 2, 1, 0)
    Re_A5_uj = assemble_matrix_from_volume_integral_triangle_up_uj(getattr(coef_data, 'function_negativeone'), ph, l2gT_DS,
                                                             M_partition_S, T_partition_S, M_basis_u, T_basis_u,
                                                             T_basis_p, 1, 0, 0, 2, 1, 0)
    Re_A6_uj = assemble_matrix_from_volume_integral_triangle_up_uj(getattr(coef_data, 'function_negativeone'), ph, l2gT_DS,
                                                             M_partition_S, T_partition_S, M_basis_u, T_basis_u,
                                                             T_basis_p, 1, 0, 0, 2, 0, 1)
    temp_uj = 1e-8 * assemble_matrix_from_volume_integral_triangle_global_uj(getattr(coef_data, 'function_nu'), ph, l2gT_DS,
                                                                 M_partition_S, T_partition_S, M_basis_p,
                                                                          T_basis_p, T_basis_p, T_basis_p, 1, 0, 0, 1, 0, 0)
    b1 = assemble_vector_from_volume_integral_triangle_global_uj(getattr(coef_data, 'function_f1_Stokes'), l2gT_DS,
                                                                 M_partition_S, T_partition_S,
                                                                 M_basis_u, T_basis_u, 2, 0, 0)
    b2 = assemble_vector_from_volume_integral_triangle_global_uj(getattr(coef_data, 'function_f2_Stokes'), l2gT_DS,
                                                                 M_partition_S, T_partition_S,
                                                                 M_basis_u, T_basis_u, 2, 0, 0)
    temp_b = 1e-8 * assemble_vector_from_volume_integral_triangle_global_uj(getattr(coef_data, 'function_f2_Stokes'), l2gT_DS,
                                                                 M_partition_S, T_partition_S,
                                                                 M_basis_p, T_basis_p, 2, 0, 0)#全零维度为（Nbp,1）

    Re_stokes_1 = 2 * Re_A1_uj + Re_A3_uj + Re_A5_uj - b1
    Re_stokes_2 = Re_A4_uj + 2 * Re_A2_uj + Re_A1_uj + Re_A6_uj - b2
    Re_stokes_3 = Re_A5_uj.T + Re_A6_uj.T + temp_uj + temp_b

    interface_end_point_1 = torch.tensor([left_S, top_S], dtype=torch.float32, device='cuda')
    interface_end_point_2 = torch.tensor([left_S, top_S], dtype=torch.float32, device='cuda')
    interface_edges = generate_interface_edges_StokesDarcy(interface_end_point_1, interface_end_point_2, left_S,
                                                           right_S, bottom_S, top_S, left_D, right_D, bottom_D, top_D,
                                                           h_partition_S, h_partition_D)

    [Gauss_coefficient_reference_1D, Gauss_point_reference_1D] = generate_Gauss_reference_1D(4)
    Gauss_coefficient_reference_1D = Double(Gauss_coefficient_reference_1D)
    Gauss_point_reference_1D = Double(Gauss_point_reference_1D)

    # Assemble the matrices from the line integrals from the three interface conditions(one is BJSJ).
    #边界如何处理
    A = assemble_matrix_from_interface_conditions_BJSJ_triangle(U,l2gT_DS,A, interface_edges, Darcy_scaling_constant,
                                                                M_partition_D, T_partition_D, M_partition_S,
                                                                T_partition_S, T_basis_phi, T_basis_u,
                                                                number_of_local_basis_phi, number_of_local_basis_u,
                                                                number_of_unknowns_Darcy, number_of_FE_nodes_u,
                                                                Gauss_coefficient_reference_1D,
                                                                Gauss_point_reference_1D, 2, 0, 0, 2, 0, 0)

    # [A, b] = assemble_additional_matrix_vector_for_BJ_triangle(A, b, interface_edges, M_partition_D, T_partition_D,
    #                                                                M_partition_S, T_partition_S, T_basis_phi, T_basis_u,
    #                                                                number_of_local_basis_phi, number_of_local_basis_u,
    #                                                                number_of_unknowns_Darcy, number_of_FE_nodes_u,
    #                                                                Gauss_coefficient_reference_1D, Gauss_point_reference_1D,
    #                                                                2, 2)
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
    [boundary_nodes_D, boundary_edges_D] = generate_boundary_nodes_edges(N1_basis_phi, N2_basis_phi, N1_partition_D,
                                                                         N2_partition_D)
    boundary_nodes_D = Double(boundary_nodes_D)
    boundary_edges_D = Double(boundary_edges_D)
    boundary_edges_D = boundary_edges_D.to(dtype=torch.long, device='cuda')
    boundary_nodes_D = boundary_nodes_D.to(dtype=torch.long, device='cuda')

    [A, b] = treat_Dirichlet_boundary_triangle(getattr(coef_data, 'function_g_Poisson'), A, b, boundary_nodes_D,
                                               M_basis_phi)

    # Get the information matrices for boundary nodes and boundary edges for the Stokes Domain.
    [boundary_nodes_S, boundary_edges_S] = generate_boundary_nodes_edges_Stokes(N1_basis_u, N2_basis_u, N1_partition_S,
                                                                                N2_partition_S)
    boundary_edges_S = Double(boundary_edges_S)
    boundary_nodes_S = Double(boundary_nodes_S)
    boundary_nodes_S = boundary_nodes_S.to(dtype=torch.long, device='cuda')
    boundary_edges_S = boundary_edges_S.to(dtype=torch.long, device='cuda')

    T_basis_u_adjust = T_basis_u + number_of_unknowns_Darcy
    boundary_nodes_S[2, :] = boundary_nodes_S[2, :] + number_of_unknowns_Darcy

    test_error = 1000
    nonlinear_iteration_step = 1
    number_of_elements_S = int(2 * N1_partition_S * N2_partition_S)

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

        A_sum = torch.zeros((number_of_all_unknowns, number_of_all_unknowns))

        A_sum[number_of_unknowns_Darcy:number_of_all_unknowns,
        number_of_unknowns_Darcy:number_of_all_unknowns] = torch.cat([
            torch.cat([Ac1 + Ac2 + Ac3, Ac4, temp2], dim=1),
            torch.cat([Ac5, Ac2 + Ac3 + Ac6, temp2], dim=1),
            torch.cat([temp3, temp3, temp1], dim=1)], dim=0)
        A_sum = A_sum.to('cuda')
        A = A.to('cuda')
        A_sum = A_sum + A

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

        b_sum = torch.zeros((number_of_all_unknowns, 1))

        b_sum[number_of_unknowns_Darcy:number_of_all_unknowns, :] = torch.cat([bc1 + bc2, bc3 + bc4, temp4], dim=0)
        b_sum = b_sum.to('cuda')
        b = b.to('cuda')
        b_sum = b_sum + b
        if Dirichlet_switch == 1:
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

        r = torch.linalg.solve(A_sum, b_sum)
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

    Re = A_sum - b
    return Re


