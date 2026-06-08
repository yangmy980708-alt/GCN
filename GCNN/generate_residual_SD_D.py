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
from GCNN.Stokes_functions_data_torch import Functions_torch
from GCNN.generate_interface_edges_StokesDarcy import generate_interface_edges_StokesDarcy
from GCNN.Stokes_basis_Ab import treat_Dirichlet_boundary_Stokes_with_Darcy, assemble_matrix_from_volume_integral_FE_triangle,\
    assemble_vector_from_volume_integral_2FE_triangle, generate_Gauss_reference_triangle,\
    treat_Dirichlet_boundary_triangle, assemble_additional_matrix_vector_for_BJ_triangle,\
    assemble_matrix_from_interface_conditions_BJSJ_triangle, fix_pressure_Stokes,\
    assemble_fixed_matrix_vector_for_Poisson, assemble_fixed_matrix_vector_for_Stokes_Taylor_Hood,Double

def generate_residual_D(U,
    left_S, right_S, bottom_S, top_S,
    left_D, right_D, bottom_D, top_D,
    h_partition_S, h_partition_D,
    fix_pressure, Dirichlet_switch,
    Darcy_scaling_constant,r_D_boundary, r_Stokes_boundary):

    coef_data = Functions_torch()
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

    M_partition_S = M_partition_S.to(device='cuda')
    T_partition_S = T_partition_S.to(dtype=torch.long, device='cuda')
    M_basis_u = M_basis_u.to(device='cuda')
    T_basis_u = T_basis_u.to(dtype=torch.long, device='cuda')
    M_basis_p = M_basis_p.to(device='cuda')
    T_basis_p = T_basis_p.to(dtype=torch.long, device='cuda')
    [A_Darcy, b_Darcy] = assemble_fixed_matrix_vector_for_Poisson(M_partition_D, T_partition_D, M_basis_phi,
                                                                  T_basis_phi, T_basis_phi, 2)
    # 在 CUDA 上创建零矩阵
    A = torch.zeros((int(number_of_all_unknowns), int(number_of_all_unknowns)), device="cuda")
    b = torch.zeros((int(number_of_all_unknowns), 1), device="cuda")
    # 计算 Darcy 相关部分
    A[:int(number_of_unknowns_Darcy), :int(number_of_unknowns_Darcy)] = Darcy_scaling_constant * A_Darcy
    b[:number_of_unknowns_Darcy] = Darcy_scaling_constant * b_Darcy

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

    A = A.cpu().numpy()
    # 释放不再使用的变量
    del A_Darcy, b_Darcy

    U = U.to(device='cuda')
    A = Double(A)
    b = Double(b)
    # Re = A * U - b #这里算出来是（51，51）维度错误，应该是（51，1）
    Re_D = torch.matmul(A, U) - b
    r_D_boundary = Double(r_D_boundary)
    r_Stokes_boundary = Double(r_Stokes_boundary)
    return Re_D, boundary_nodes_D,  r_D_boundary, r_Stokes_boundary