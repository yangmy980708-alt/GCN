import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.linalg import solve
import matplotlib
matplotlib.use('TkAgg')
from scipy.sparse import lil_matrix,coo_matrix
import matplotlib.pyplot as plt
from scipy.io import loadmat
import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset, DataLoader
from torch import tensor
import os
import sys
# 获取当前文件的上一级目录，也就是项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
sys.path.insert(0, '../matrixAb')
from matrixAb.triangular_reference_basis import triangular_reference_basis
from matrixAb.generate_Gauss_local_triangle import generate_Gauss_local_triangle
sys.path.insert(0, '../GCNN')
from GCNN.functions_data_torch import Functions

def assemble_fixed_matrix_vector_for_Poisson(U,M_partition, T_partition, M_basis_trial, T_basis_trial, T_basis_test, basis_type):
    coef_data = Functions()

    A1_uj = assemble_matrix_from_volume_integral_triangle_global_uj(getattr(coef_data, 'function_k11'),U,
                                                       M_partition, T_partition, M_basis_trial, T_basis_trial, T_basis_test,
                                                       basis_type, 1, 0, basis_type, 1, 0)
    A2_uj = assemble_matrix_from_volume_integral_triangle_global_uj(getattr(coef_data, 'function_k22'),U,
                                                       M_partition, T_partition, M_basis_trial, T_basis_trial, T_basis_test,
                                                       basis_type, 0, 1, basis_type, 0, 1)
    A3_uj = assemble_matrix_from_volume_integral_triangle_global_uj(getattr(coef_data, 'function_k12'),U,
                                                       M_partition, T_partition, M_basis_trial, T_basis_trial, T_basis_test,
                                                       basis_type, 0, 1, basis_type, 1, 0)
    A4_uj = assemble_matrix_from_volume_integral_triangle_global_uj(getattr(coef_data, 'function_k21'),U,
                                                       M_partition, T_partition, M_basis_trial, T_basis_trial, T_basis_test,
                                                       basis_type, 1, 0, basis_type, 0, 1)

    A1_uj = Double(A1_uj)
    A2_uj = Double(A2_uj)
    A3_uj = Double(A3_uj)
    A4_uj = Double(A4_uj)

    b_uj = assemble_vector_from_volume_integral_triangle_global_uj(getattr(coef_data, 'function_f_Poisson'),
                                                      M_partition, T_partition, M_basis_trial, T_basis_trial, basis_type, 0, 0)
    b_uj = Double(b_uj)
    Re_Darcy_uj = Double(A1_uj + A2_uj + A3_uj + A4_uj - b_uj)
    return Re_Darcy_uj

def assemble_matrix_from_volume_integral_triangle_global_uj(coefficient_function,U,M_partition, T_partition,
                                                            M_basis_trial, T_basis_trial, T_basis_test,
                                                   trial_basis_type, trial_derivative_degree_x, trial_derivative_degree_y,
                                                   test_basis_type, test_derivative_degree_x, test_derivative_degree_y):

    Nb_trial = int(M_basis_trial.shape[1])
    Nb_test = int(M_basis_trial.shape[1])#Number of basis
    number_of_trial_local_basis = int(T_basis_trial.shape[0])
    number_of_test_local_basis = int(T_basis_test.shape[0])

    A = coo_matrix((Nb_test, Nb_trial))
    vertices = torch.cat([M_partition[:, T_partition[0, :]], M_partition[:, T_partition[1, :]],
                          M_partition[:, T_partition[2, :]]])
    # vertices = np.vstack([M_partition[:, T_partition[0, :]], M_partition[:, T_partition[1, :]], M_partition[:, T_partition[2, :]]])
    Gauss_point_number = 9
    [Gauss_point_local_triangle_x,Gauss_point_local_triangle_y, Gauss_coefficient_local_triangle] = generate_Gauss_local_triangle(Gauss_point_number,
                                                                   vertices)
    for alpha in range(number_of_trial_local_basis):
           for beta in range(number_of_test_local_basis):
               temp = 0
               for i in range(Gauss_point_number):
                   trial_basis_value_uj = FE_solution_triangle(Gauss_point_local_triangle_x[i, :],
                                                               Gauss_point_local_triangle_y[i, :],
                                                               U,T_basis_trial,vertices,trial_basis_type,
                                                               trial_derivative_degree_x,
                                                               trial_derivative_degree_y)
                   test_basis_value = triangular_local_basis(Gauss_point_local_triangle_x[i, :],
                                                             Gauss_point_local_triangle_y[i, :], vertices,
                                                             test_basis_type, beta, test_derivative_degree_x,
                                                             test_derivative_degree_y)
                   coef = coefficient_function(Gauss_point_local_triangle_x[i, :],Gauss_point_local_triangle_y[i, :])
                   temp += Gauss_coefficient_local_triangle[i,:] * coef * trial_basis_value_uj * test_basis_value# * u_j
                   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                   T_basis_test[beta, :] = T_basis_test[beta, :].flatten().to(device)
                   T_basis_trial[alpha, :] = T_basis_trial[alpha, :].flatten().to(device)
               #
               # # A += coo_matrix((temp, (T_basis_test[beta, :], T_basis_trial[alpha, :])), shape=(Nb_test, Nb_trial))
               A += sp.coo_matrix((temp.cpu().numpy(),
                                   (T_basis_test[beta, :].cpu().numpy(), T_basis_trial[alpha, :].cpu().numpy())),
                             shape=(Nb_test, Nb_trial))
               # # A += torch.sparse_coo_tensor(([T_basis_test[beta, :], T_basis_trial[alpha, :]]), temp, size=(Nb_test, Nb_trial), device='cuda')
               r = A.toarray()
               r = Double(r)
    return r

def assemble_vector_from_volume_integral_triangle_global_uj(coefficient_function, M_partition, T_partition, M_basis_trial, T_basis_test,
                                                 test_basis_type, test_derivative_degree_x, test_derivative_degree_y):

    Nb = int(M_basis_trial.shape[1])
    number_of_test_local_basis = int(T_basis_test.shape[0])
    R = coo_matrix((Nb, 1))
    vertices = torch.cat([M_partition[:, T_partition[0, :]], M_partition[:, T_partition[1, :]],
                          M_partition[:, T_partition[2, :]]])
    # vertices = np.vstack(
    # [M_partition[:, T_partition[0, :]], M_partition[:, T_partition[1, :]], M_partition[:, T_partition[2, :]]])
    Gauss_point_number = 9
    [Gauss_point_local_triangle_x,Gauss_point_local_triangle_y,Gauss_coefficient_local_triangle] = generate_Gauss_local_triangle(Gauss_point_number,
                                                                                                   vertices)
    for beta in range(number_of_test_local_basis):
        temp = 0
        for i in range(Gauss_point_number):
            basis_value = triangular_local_basis(Gauss_point_local_triangle_x[i, :], Gauss_point_local_triangle_y[i, :],
                                                 vertices,
                                       test_basis_type, beta, test_derivative_degree_x,
                                       test_derivative_degree_y)
            coef = coefficient_function(Gauss_point_local_triangle_x[i, :], Gauss_point_local_triangle_y[i, :])

            temp += Gauss_coefficient_local_triangle[i] * coef * basis_value
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        row_indices = T_basis_test[beta, :].to(device)
        col_indices = torch.zeros_like(row_indices, device=device)  # 保持 GPU 计算
        R += sp.coo_matrix((temp.cpu().numpy(), (row_indices.cpu().numpy(), col_indices.cpu().numpy())), shape=(Nb, 1))
        # R += coo_matrix((temp.cpu().numpy(), (T_basis_test[beta, :].cpu().numpy(), np.zeros_like(T_basis_test[beta, :]))), shape=(Nb, 1))
        r = R.toarray()
        r = Double(r)
    return r

def assemble_fixed_matrix_vector_for_Stokes_Taylor_Hood_uj(U,M_partition, T_partition, M_basis_trial_u, T_basis_trial_u,
                                                        T_basis_test_u, T_basis_p):
    coef_data = Functions()

    A1 = assemble_matrix_from_volume_integral_triangle_global_uj(getattr(coef_data, 'function_nu'), U,M_partition, T_partition, M_basis_trial_u, T_basis_trial_u,
                                                        T_basis_test_u, 2, 1, 0, 2, 1, 0)
    A2 = assemble_matrix_from_volume_integral_triangle_global_uj(getattr(coef_data, 'function_nu'), U,M_partition, T_partition, M_basis_trial_u, T_basis_trial_u,
                                                        T_basis_test_u, 2, 0, 1, 2, 0, 1)
    A3 = assemble_matrix_from_volume_integral_triangle_global_uj(getattr(coef_data, 'function_nu'), U,M_partition, T_partition, M_basis_trial_u, T_basis_trial_u,
                                                        T_basis_test_u, 2, 1, 0, 2, 0, 1)
    A4 = assemble_matrix_from_volume_integral_triangle_global_uj(getattr(coef_data, 'function_nu'), U,M_partition, T_partition, M_basis_trial_u, T_basis_trial_u,
                                                        T_basis_test_u, 2, 0, 1, 2, 1, 0)
    A5 = assemble_matrix_from_volume_integral_triangle_up_uj(getattr(coef_data, 'function_negativeone'), U,M_partition, T_partition, M_basis_trial_u, T_basis_trial_u, T_basis_p, 1, 0, 0, 2, 1, 0)
    A6 = assemble_matrix_from_volume_integral_triangle_up_uj(getattr(coef_data, 'function_negativeone'), U,M_partition, T_partition, M_basis_trial_u, T_basis_trial_u, T_basis_p, 1, 0, 0, 2, 0, 1)
    temp = np.zeros((int(M_partition.shape[1]), int(M_partition.shape[1])))
    A1 = Double(A1)
    A2 = Double(A2)
    A3 = Double(A3)
    A4 = Double(A4)
    A5 = Double(A5)
    A6 = Double(A6)
    temp = Double(temp)

    # 计算矩阵
    A1_uj = 2 * A1 + A2 + A3 + A5
    A2_uj = A4 + 2 * A2 + A1 + A6
    A3_uj = A5.T + A6.T#temp=0
    # A_uj = torch.cat([
    #     torch.cat([2 * A1 + A2, A3, A5], dim=1),
    #     torch.cat([A4, 2 * A2 + A1, A6], dim=1),
    #     torch.cat([A5.T, A6.T, temp], dim=1)
    # ], dim=0)
    A_uj= torch.cat([A1_uj,A2_uj,A3_uj], dim=0)
    A_uj = Double(A_uj)

    b1 = assemble_vector_from_volume_integral_triangle_global_uj(getattr(coef_data, 'function_f1_Stokes'),M_partition, T_partition,
                                                       M_basis_trial_u, T_basis_trial_u, 2, 0, 0)
    b2 = assemble_vector_from_volume_integral_triangle_global_uj(getattr(coef_data, 'function_f2_Stokes'),M_partition, T_partition,
                                                       M_basis_trial_u, T_basis_trial_u, 2, 0, 0)
    temp_b = np.zeros((int(M_partition.shape[1]),1))
    b1 = Double(b1)
    b2 = Double(b2)
    temp_b = Double(temp_b)
    b_uj = torch.cat([b1, b2, temp_b], dim=0)
    b_uj = Double(b_uj)
    Re_uj = A_uj - b_uj
    return Re_uj

def assemble_matrix_from_volume_integral_triangle_up_uj(coefficient_function,U,M_partition, T_partition, M_basis_trial,
                                                     T_basis_trial_u, T_basis_test_p,
                                                   trial_basis_type, trial_derivative_degree_x, trial_derivative_degree_y,
                                                   test_basis_type, test_derivative_degree_x, test_derivative_degree_y):

    Nb_trial = int(M_basis_trial.shape[1])
    Nb_test = int(M_partition.shape[1])  # Number of basis

    A = coo_matrix((Nb_trial, Nb_test))
    vertices = torch.cat([M_partition[:, T_partition[0, :]], M_partition[:, T_partition[1, :]],
                          M_partition[:, T_partition[2, :]]])
    # vertices = np.vstack(
    #     [M_partition[:, T_partition[0, :]], M_partition[:, T_partition[1, :]], M_partition[:, T_partition[2, :]]])
    Gauss_point_number = 9
    [Gauss_point_local_triangle_x, Gauss_point_local_triangle_y,
     Gauss_coefficient_local_triangle] = generate_Gauss_local_triangle(Gauss_point_number,
                                                                       vertices)
    for alpha in range(int(T_basis_test_p.shape[0])):
        for beta in range(int(T_basis_trial_u.shape[0])):
            temp = 0
            for i in range(Gauss_point_number):
                # trial_basis_value = triangular_local_basis(Gauss_point_local_triangle_x[i, :],
                #                                            Gauss_point_local_triangle_y[i, :], vertices,
                #                                            trial_basis_type, alpha, trial_derivative_degree_x,
                #                                            trial_derivative_degree_y)
                trial_basis_value_uj = FE_solution_triangle(Gauss_point_local_triangle_x[i, :],
                                                               Gauss_point_local_triangle_y[i, :],
                                                               U,T_basis_test_p,vertices,trial_basis_type,
                                                               trial_derivative_degree_x,
                                                               trial_derivative_degree_y)#sum(p_j * phi_j)
                test_basis_value = triangular_local_basis(Gauss_point_local_triangle_x[i, :],
                                                          Gauss_point_local_triangle_y[i, :], vertices,
                                                          test_basis_type, beta, test_derivative_degree_x,
                                                          test_derivative_degree_y)
                coef = coefficient_function(Gauss_point_local_triangle_x[i, :], Gauss_point_local_triangle_y[i, :])
                temp += Gauss_coefficient_local_triangle[i, :] * coef * trial_basis_value_uj * test_basis_value
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                T_basis_test_p[alpha, :] = T_basis_test_p[alpha, :].flatten().to(device)
                T_basis_trial_u[beta, :] = T_basis_trial_u[beta, :].flatten().to(device)
            A += sp.coo_matrix(
                (temp.cpu().numpy(), (T_basis_trial_u[beta, :].cpu().numpy(), T_basis_test_p[alpha, :].cpu().numpy())),
                shape=(Nb_trial, Nb_test))
            # A += coo_matrix((temp, (T_basis_trial_u[beta, :], T_basis_test_p[alpha, :])), shape=(Nb_trial, Nb_test))
            r = A.toarray()
            r = Double(r)
    return r

def assemble_matrix_from_interface_conditions_BJSJ_triangle(U,A, interface_edges, Darcy_scaling_constant, M_partition_D, T_partition_D, M_partition_S, T_partition_S, T_basis_phi, T_basis_u, number_of_local_basis_phi, number_of_local_basis_u, number_of_unknowns_Darcy, number_of_FE_nodes_u, Gauss_coefficient_reference_1D, Gauss_point_reference_1D, basis_type_phi, derivative_degree_x_phi, derivative_degree_y_phi, basis_type_u, derivative_degree_x_u, derivative_degree_y_u):
    nie = interface_edges.shape[1]
    for k in range(int(nie)):
        # Darcy_element_index = interface_edges[0, k].astype(int)
        Darcy_element_index = interface_edges[0, k].to(dtype=torch.int)
        Stokes_element_index = interface_edges[1, k].to(dtype=torch.int)
        # Go through all interface edges.
        Darcy_vertices = M_partition_D[:, T_partition_D[:, Darcy_element_index]]
        Stokes_vertices = M_partition_S[:, T_partition_S[:, Stokes_element_index]]
        end_point_1 = interface_edges[2:4, k]
        end_point_2 = interface_edges[4:6, k]
        coef_data = Functions()

        for alpha in range(int(T_basis_u.shape[0])):  # number_of_local_basis_u
            for beta in range(int(T_basis_phi.shape[0])):  # number_of_local_basis_phi
                temp = Gauss_quadrature_for_line_integral_trial_test_triangle(getattr(coef_data, 'function_one'),U,
                                                                              M_partition_D, T_partition_D,
                                                                              M_partition_S, T_partition_S, T_basis_phi,
                                                                              T_basis_u,
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
                temp = Gauss_quadrature_for_line_integral_trial_test_triangle(getattr(coef_data, 'function_gravity'),U,
                                                                              M_partition_D, T_partition_D,
                                                                              M_partition_S, T_partition_S, T_basis_phi,
                                                                              T_basis_u,
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
                temp = Gauss_quadrature_for_line_integral_trial_test_triangle(getattr(coef_data, 'function_BJSJ_coefficient'), U,
                                                                              M_partition_D, T_partition_D,
                                                                              M_partition_S, T_partition_S, T_basis_phi,
                                                                              T_basis_u,
                                                                              Gauss_coefficient_reference_1D, Gauss_point_reference_1D,
                                                                              end_point_1,end_point_2,
                                                                              Stokes_vertices, basis_type_u, alpha,
                                                                              derivative_degree_x_u, derivative_degree_y_u,
                                                                              Stokes_vertices, basis_type_u, beta,
                                                                              derivative_degree_x_u, derivative_degree_y_u)

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
    A = Double(A)
    return A

def Gauss_quadrature_for_line_integral_trial_test_triangle(
    coefficient_function_name,U,
M_partition_D, T_partition_D, M_partition_S, T_partition_S, T_basis_phi, T_basis_u,
    Gauss_coefficient_reference_1D,
    Gauss_point_reference_1D,
    end_point_1,
    end_point_2,
    trial_vertices,
    trial_basis_type,
    trial_basis_index,
    trial_derivative_degree_x,
    trial_derivative_degree_y,
    test_vertices,
    test_basis_type,
    test_basis_index,
    test_derivative_degree_x,
    test_derivative_degree_y
):

    Gpn = len(Gauss_coefficient_reference_1D)
    r = 0

    if end_point_1[1] == end_point_2[1]:  # The line is horizontal
        lower_bound = min(end_point_1[0], end_point_2[0])
        upper_bound = max(end_point_1[0], end_point_2[0])
        Gauss_coefficient_local_1D, Gauss_point_local_1D = generate_Gauss_local_1D(
            Gpn, lower_bound, upper_bound)
        for i in range(Gpn):
            coef = coefficient_function_name(Gauss_point_local_1D[i], end_point_1[1])
            # r_1 = triangular_local_basis_element(
            #         Gauss_point_local_1D[i], end_point_1[1],
            #         trial_vertices, trial_basis_type, trial_basis_index,
            #         trial_derivative_degree_x, trial_derivative_degree_y)
            r1_uj = FE_solution_triangle(Gauss_point_local_1D[i], end_point_1[1],
                                 U, T_basis_u, trial_vertices, trial_basis_type,
                                 trial_derivative_degree_x,
                                 trial_derivative_degree_y)
            r_2 = triangular_local_basis_element(
                    Gauss_point_local_1D[i], end_point_1[1],
                    test_vertices, test_basis_type, test_basis_index,
                    test_derivative_degree_x, test_derivative_degree_y)
            r += Gauss_coefficient_local_1D[i] * coef * r1_uj * r_2

    elif end_point_1[0] == end_point_2[0]:  # The line is vertical
        lower_bound = min(end_point_1[1], end_point_2[1])
        upper_bound = max(end_point_1[1], end_point_2[1])
        Gauss_coefficient_local_1D, Gauss_point_local_1D = generate_Gauss_local_1D(
            Gpn, lower_bound, upper_bound)
        for i in range(Gpn):
            coef = coefficient_function_name(end_point_1[0], Gauss_point_local_1D[i])
            # r_1 = triangular_local_basis_element(
            #         end_point_1[0], Gauss_point_local_1D[i],
            #         trial_vertices, trial_basis_type, trial_basis_index,
            #         trial_derivative_degree_x, trial_derivative_degree_y)
            r1_uj = FE_solution_triangle(end_point_1[0], Gauss_point_local_1D[i],
                                 U, T_basis_u, trial_vertices, trial_basis_type,
                                 trial_derivative_degree_x,
                                 trial_derivative_degree_y)
            r_2 = triangular_local_basis_element(
                    end_point_1[0], Gauss_point_local_1D[i],
                    test_vertices, test_basis_type, test_basis_index,
                    test_derivative_degree_x, test_derivative_degree_y)
            r += Gauss_coefficient_local_1D[i] * coef * r1_uj * r_2

    else:  # The slope of the edge is in (0, infinity)
        lower_bound = min(end_point_1[0], end_point_2[0])
        upper_bound = max(end_point_1[0], end_point_2[0])
        Gauss_coefficient_local_1D, Gauss_point_local_1D = generate_Gauss_local_1D(
            Gpn, lower_bound, upper_bound
        )
        slope = (end_point_2[1] - end_point_1[1]) / (end_point_2[0] - end_point_1[0])
        Jacobi = np.sqrt(1 + slope**2)
        for i in range(Gpn):
            x = Gauss_point_local_1D[i]
            y = slope * (x - end_point_1[0]) + end_point_1[1]
            coef = coefficient_function_name(x, y)
            # r_1 = triangular_local_basis_element(
            #         x, y, trial_vertices, trial_basis_type, trial_basis_index,
            #         trial_derivative_degree_x, trial_derivative_degree_y)
            r1_uj = FE_solution_triangle(x, y,
                                         U, T_basis_u, trial_vertices, trial_basis_type,
                                         trial_derivative_degree_x,
                                         trial_derivative_degree_y)
            r_2 = triangular_local_basis_element(
                    x, y, test_vertices, test_basis_type, test_basis_index,
                    test_derivative_degree_x, test_derivative_degree_y)
            r += Gauss_coefficient_local_1D[i] * Jacobi * coef * r1_uj * r_2
    r = Double(r)
    return r

def assemble_additional_matrix_vector_for_BJ_triangle(A, b, interface_edges, M_partition_D, T_partition_D,
                                                      M_partition_S, T_partition_S,
                                                      T_basis_phi, T_basis_u, number_of_local_basis_phi,
                                                      number_of_local_basis_u,
                                                      number_of_unknowns_Darcy, number_of_FE_nodes_u,
                                                      Gauss_coefficient_reference_1D,
                                                      Gauss_point_reference_1D, basis_type_phi, basis_type_u):
    # # 数据全部转换为numpy
    # A = A.cpu().numpy()
    # T_basis_u = T_basis_u.cpu().numpy()
    # T_basis_phi = T_basis_phi.cpu().numpy()
    # interface_edges = interface_edges.cpu().numpy()
    # Darcy_element_index = Darcy_element_index.cpu().numpy()
    # Stokes_element_index = Stokes_element_index.cpu().numpy()

    nie = interface_edges.shape[1]
    coef_data = Functions()
    for k in range(nie):
        Darcy_element_index = interface_edges[0, k].to(dtype=torch.int)
        Stokes_element_index = interface_edges[1, k].to(dtype=torch.int)

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
                a1 = number_of_unknowns_Darcy + T_basis_u[beta, Stokes_element_index]
                a2 = T_basis_phi[alpha, Darcy_element_index]
                A[a1,a2] += interface_edges[8, k] ** 2 * temp
                A[number_of_unknowns_Darcy + T_basis_u[beta, Stokes_element_index],
                  T_basis_phi[alpha, Darcy_element_index]] += interface_edges[8, k] ** 2 * temp
                # A19 update
                A[number_of_unknowns_Darcy + number_of_FE_nodes_u + T_basis_u[beta, Stokes_element_index],
                T_basis_phi[alpha, Darcy_element_index]] += interface_edges[8, k] * interface_edges[9, k] * temp

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
    A = Double(A)
    b = Double(b)
    return A, b

def Gauss_quadrature_for_line_integral_test_triangle(
    coefficient_function_name,
    Gauss_coefficient_reference_1D,
    Gauss_point_reference_1D,
    end_point_1,
    end_point_2,
    test_vertices,
    test_basis_type,
    test_basis_index,
    test_derivative_degree_x,
    test_derivative_degree_y
):
    Gpn = len(Gauss_coefficient_reference_1D)
    r = 0
    if end_point_1[1] == end_point_2[1]:
        lower_bound = min(end_point_1[0], end_point_2[0])
        upper_bound = max(end_point_1[0], end_point_2[0])
        Gauss_coefficient_local_1D, Gauss_point_local_1D = generate_Gauss_local_1D(Gpn, lower_bound, upper_bound)
        for i in range(Gpn):
            coef = coefficient_function_name(Gauss_point_local_1D[i], end_point_1[1])
            r_1 = triangular_local_basis_element(
                    Gauss_point_local_1D[i],
                    end_point_1[1],
                    test_vertices,
                    test_basis_type,
                    test_basis_index,
                    test_derivative_degree_x,
                    test_derivative_degree_y)
            r += Gauss_coefficient_local_1D[i] * coef * r_1

    elif end_point_1[0] == end_point_2[0]:
        lower_bound = min(end_point_1[1], end_point_2[1])
        upper_bound = max(end_point_1[1], end_point_2[1])
        Gauss_coefficient_local_1D, Gauss_point_local_1D = generate_Gauss_local_1D(Gpn, lower_bound, upper_bound)
        for i in range(Gpn):
            coef = coefficient_function_name(end_point_1[0], Gauss_point_local_1D[i])
            r_1 = triangular_local_basis_element(end_point_1[0], Gauss_point_local_1D[i],
                                               test_vertices,
                                               test_basis_type,
                                               test_basis_index,
                                               test_derivative_degree_x,
                                               test_derivative_degree_y)
            r += Gauss_coefficient_local_1D[i] * coef * r_1

    else:
        lower_bound = min(end_point_1[0], end_point_2[0])
        upper_bound = max(end_point_1[0], end_point_2[0])
        Gauss_coefficient_local_1D, Gauss_point_local_1D = generate_Gauss_local_1D(Gpn, lower_bound, upper_bound)
        slope = (end_point_2[1] - end_point_1[1]) / (end_point_2[0] - end_point_1[0])
        Jacobi = (1 + slope**2) ** 0.5
        for i in range(Gpn):
            x = Gauss_point_local_1D[i]
            y = slope * (x - end_point_1[0]) + end_point_1[1]
            coef = coefficient_function_name(x,y)
            r_1 = triangular_local_basis_element(x,
                    y,
                    test_vertices,
                    test_basis_type,
                    test_basis_index,
                    test_derivative_degree_x,
                    test_derivative_degree_y)
            r += Gauss_coefficient_local_1D[i] * Jacobi * coef * r_1
    r = Double(r)
    return r

def treat_Dirichlet_boundary_triangle(Dirichlet_boundary_function_name, A, b, boundary_nodes, M_basis):
    nbn = int(boundary_nodes.shape[1])
    for k in range(nbn):
        if boundary_nodes[0, k] == -1:
            i = int(boundary_nodes[1, k])
            A[i, :] = 0
            A[i, i] = 1
            b[i, 0] = Dirichlet_boundary_function_name(M_basis[0, i], M_basis[1, i])
    A = Double(A)
    b = Double(b)
    return A, b

def Gauss_quadrature_for_volume_integral_trial_test_FE_triangle(
    coefficient_function_name,
    uh_local,
    Gauss_coefficient_local,
    Gauss_point_local_x,
    Gauss_point_local_y,
    vertices,
    trial_basis_type,
    trial_basis_index,
    trial_derivative_degree_x,
    trial_derivative_degree_y,
    test_basis_type,
    test_basis_index,
    test_derivative_degree_x,
    test_derivative_degree_y,
    FE_basis_type,
    FE_derivative_degree_x,
    FE_derivative_degree_y
):
    Gpn = len(Gauss_coefficient_local)
    r = 0
    x = Gauss_point_local_x
    y = Gauss_point_local_y
    for i in range(Gpn):
        coeff_value = coefficient_function_name(x[i,:], y[i,:], uh_local, vertices, FE_basis_type, FE_derivative_degree_x, FE_derivative_degree_y)
        trial_basis_value = triangular_local_basis(x[i,:], y[i,:], vertices, trial_basis_type, trial_basis_index,
            trial_derivative_degree_x, trial_derivative_degree_y)
        test_basis_value = triangular_local_basis(x[i,:], y[i,:], vertices, test_basis_type, test_basis_index,
            test_derivative_degree_x, test_derivative_degree_y)
        r += Gauss_coefficient_local[i] * coeff_value * trial_basis_value * test_basis_value
    r = Double(r)
    return r


def Gauss_quadrature_for_volume_integral_test_2FE_triangle(
        coefficient_function_name,
        uh_local_1,
        uh_local_2,
        Gauss_coefficient_local,
        Gauss_point_local,
        vertices,
        test_basis_type,
        test_basis_index,
        test_derivative_degree_x,
        test_derivative_degree_y,
        FE1_basis_type,
        FE1_derivative_degree_x,
        FE1_derivative_degree_y,
        FE2_basis_type,
        FE2_derivative_degree_x,
        FE2_derivative_degree_y,
):
    Gpn = len(Gauss_coefficient_local)
    r = 0.0
    for i in range(Gpn):
        x, y = Gauss_point_local[i, 0], Gauss_point_local[i, 1]
        coeff_value_1 = coefficient_function_name(x, y, uh_local_1, vertices, FE1_basis_type, FE1_derivative_degree_x, FE1_derivative_degree_y)
        coeff_value_2 = coefficient_function_name(x, y, uh_local_2, vertices, FE2_basis_type, FE2_derivative_degree_x, FE2_derivative_degree_y)
        test_basis_value = triangular_local_basis(x, y, vertices, test_basis_type, test_basis_index, test_derivative_degree_x, test_derivative_degree_y)
        r += Gauss_coefficient_local[i] * coeff_value_1 * coeff_value_2 * test_basis_value
    r = Double(r)
    return r

def treat_Dirichlet_boundary_Stokes_with_Darcy(Dirichlet_boundary_function_name_u1, Dirichlet_boundary_function_name_u2,
                                               A, b, boundary_nodes, M_basis_u,
                                               number_of_FE_nodes_u, number_of_unknowns_Darcy):
    nbn = int(boundary_nodes.shape[1])
    for k in range(nbn):
        if boundary_nodes[0, k] == -1:
            global_index = int(boundary_nodes[2, k])
            i = global_index - number_of_unknowns_Darcy
            A[global_index, :] = 0
            A[global_index, global_index] = 1
            b[global_index, 0] = Dirichlet_boundary_function_name_u1(M_basis_u[0, i], M_basis_u[1, i])
        if boundary_nodes[1, k] == -1:
            global_index = int(boundary_nodes[2, k])
            u2_index = number_of_FE_nodes_u + global_index
            i = global_index - number_of_unknowns_Darcy
            A[u2_index, :] = 0
            A[u2_index, u2_index] = 1
            b[u2_index, 0] = Dirichlet_boundary_function_name_u2(M_basis_u[0, i], M_basis_u[1, i])
    A = Double(A)
    b = Double(b)
    return A, b

def fix_pressure_Stokes(Dirichlet_boundary_function_name_p, fix_pressure, A, b, number_of_unknows_before_p,
                        fixed_p_index, M_basis_p):
    if fix_pressure == 1:
        A[number_of_unknows_before_p + fixed_p_index - 1, :] = 0
        A[number_of_unknows_before_p + fixed_p_index - 1, number_of_unknows_before_p + fixed_p_index - 1] = 1
        b[number_of_unknows_before_p + fixed_p_index - 1, 0] = Dirichlet_boundary_function_name_p(
            M_basis_p[0, fixed_p_index-1], M_basis_p[1, fixed_p_index-1])
    A = Double(A)
    b = Double(b)
    return A, b

def triangular_local_basis(x, y, vertices, basis_type, basis_index, derivative_degree_x, derivative_degree_y):
    vertices = Double(vertices)
    x1 = Double(vertices[0, :])
    y1 = Double(vertices[1, :])
    x2 = Double(vertices[2, :])
    y2 = Double(vertices[3, :])
    x3 = Double(vertices[4, :])
    y3 = Double(vertices[5, :])
    J_11 = Double(x2 - x1)
    J_12 = Double(x3 - x1)
    J_21 = Double(y2 - y1)
    J_22 = Double(y3 - y1)
    J_det = Double(J_11 * J_22 - J_12 * J_21)
    x_hat = Double((J_22 * (x - vertices[0, :]) - J_12 * (y - vertices[1, :])) / J_det)
    y_hat = Double((-J_21 * (x - vertices[0, :]) + J_11 * (y - vertices[1, :])) / J_det)

    if derivative_degree_x == 0 and derivative_degree_y == 0:
        r = triangular_reference_basis(x_hat, y_hat, basis_type, basis_index, 0, 0)
    elif derivative_degree_x == 1 and derivative_degree_y == 0:
        r = (triangular_reference_basis(x_hat, y_hat, basis_type, basis_index, 1, 0) * J_22 +
             triangular_reference_basis(x_hat, y_hat, basis_type, basis_index, 0, 1) * (-J_21)) / J_det
    elif derivative_degree_x == 0 and derivative_degree_y == 1:
        r = (triangular_reference_basis(x_hat, y_hat, basis_type, basis_index, 1, 0) * (-J_12) +
             triangular_reference_basis(x_hat, y_hat, basis_type, basis_index, 0, 1) * J_11) / J_det
    elif derivative_degree_x == 2 and derivative_degree_y == 0:
        r = (triangular_reference_basis(x_hat, y_hat, basis_type, basis_index, 2, 0) * J_22 ** 2 +
             triangular_reference_basis(x_hat, y_hat, basis_type, basis_index, 0, 2) * J_21 ** 2 +
             triangular_reference_basis(x_hat, y_hat, basis_type, basis_index, 1, 1) * (-2 * J_21 * J_22)) / J_det ** 2
    elif derivative_degree_x == 0 and derivative_degree_y == 2:
        r = (triangular_reference_basis(x_hat, y_hat, basis_type, basis_index, 2, 0) * J_12 ** 2 +
             triangular_reference_basis(x_hat, y_hat, basis_type, basis_index, 0, 2) * J_11 ** 2 +
             triangular_reference_basis(x_hat, y_hat, basis_type, basis_index, 1, 1) * (-2 * J_11 * J_12)) / J_det ** 2
    elif derivative_degree_x == 1 and derivative_degree_y == 1:
        r = (triangular_reference_basis(x_hat, y_hat, basis_type, basis_index, 2, 0) * (-J_22 * J_12) +
             triangular_reference_basis(x_hat, y_hat, basis_type, basis_index, 0, 2) * (-J_21 * J_11) +
             triangular_reference_basis(x_hat, y_hat, basis_type, basis_index, 1, 1) * (
                     J_21 * J_12 + J_11 * J_22)) / J_det ** 2
    # r = Double(r)
    return r


def triangular_local_basis_element(x, y, vertices, basis_type, basis_index, derivative_degree_x, derivative_degree_y):
    vertices = Double(vertices)
    J_11 = Double(vertices[0,1] - vertices[0,0])
    J_12 = Double(vertices[0,2] - vertices[0,0])
    J_21 = Double(vertices[1,1] - vertices[1,0])
    J_22 = Double(vertices[1,2] - vertices[1,0])
    J_det = Double(J_11 * J_22 - J_12 * J_21)
    x_hat = Double((J_22 * (x - vertices[0, 0]) - J_12 * (y - vertices[1, 0])) / J_det)
    y_hat = Double((-J_21 * (x - vertices[0, 0]) + J_11 * (y - vertices[1, 0])) / J_det)
    if derivative_degree_x == 0 and derivative_degree_y == 0:
        r = triangular_reference_basis(x_hat, y_hat, basis_type, basis_index, 0, 0)
    elif derivative_degree_x == 1 and derivative_degree_y == 0:
        r = (triangular_reference_basis(x_hat, y_hat, basis_type, basis_index, 1, 0) * J_22 +
             triangular_reference_basis(x_hat, y_hat, basis_type, basis_index, 0, 1) * (-J_21)) / J_det
    elif derivative_degree_x == 0 and derivative_degree_y == 1:
        r = (triangular_reference_basis(x_hat, y_hat, basis_type, basis_index, 1, 0) * (-J_12) +
             triangular_reference_basis(x_hat, y_hat, basis_type, basis_index, 0, 1) * J_11) / J_det
    elif derivative_degree_x == 2 and derivative_degree_y == 0:
        r = (triangular_reference_basis(x_hat, y_hat, basis_type, basis_index, 2, 0) * J_22 ** 2 +
             triangular_reference_basis(x_hat, y_hat, basis_type, basis_index, 0, 2) * J_21 ** 2 +
             triangular_reference_basis(x_hat, y_hat, basis_type, basis_index, 1, 1) * (-2 * J_21 * J_22)) / J_det ** 2
    elif derivative_degree_x == 0 and derivative_degree_y == 2:
        r = (triangular_reference_basis(x_hat, y_hat, basis_type, basis_index, 2, 0) * J_12 ** 2 +
             triangular_reference_basis(x_hat, y_hat, basis_type, basis_index, 0, 2) * J_11 ** 2 +
             triangular_reference_basis(x_hat, y_hat, basis_type, basis_index, 1, 1) * (-2 * J_11 * J_12)) / J_det ** 2
    elif derivative_degree_x == 1 and derivative_degree_y == 1:
        r = (triangular_reference_basis(x_hat, y_hat, basis_type, basis_index, 2, 0) * (-J_22 * J_12) +
             triangular_reference_basis(x_hat, y_hat, basis_type, basis_index, 0, 2) * (-J_21 * J_11) +
             triangular_reference_basis(x_hat, y_hat, basis_type, basis_index, 1, 1) * (
                         J_21 * J_12 + J_11 * J_22)) / J_det ** 2

    return r

def Double(A):
    if isinstance(A, torch.Tensor):  # 如果 A 是 PyTorch 张量
        return A.double().to('cuda')  # 转为 double 并放到 GPU
    elif isinstance(A, np.ndarray):  # 如果 A 是 NumPy 数组
        return torch.from_numpy(A).double().to('cuda')  # 转为 PyTorch 张量并放到 GPU
    else:
        raise TypeError("Input must be a NumPy array or PyTorch tensor.")


def generate_Gauss_local_triangle(Gauss_point_number,vertices):
    device = torch.device("cuda")  # 设定计算设备
    if Gauss_point_number == 4:
        # Gauss points and coefficients for 4-point Gauss quadrature
        Gauss_coefficient_reference_triangle = torch.tensor([
            (1 - 1 / torch.sqrt(torch.tensor(3.0))) / 8,
            (1 - 1 / torch.sqrt(torch.tensor(3.0))) / 8,
            (1 + 1 / torch.sqrt(torch.tensor(3.0))) / 8,
            (1 + 1 / torch.sqrt(torch.tensor(3.0))) / 8
        ], device=device)

        Gauss_point_reference_triangle = torch.tensor([
            [(1 / torch.sqrt(torch.tensor(3.0)) + 1) / 2, (1 - 1 / torch.sqrt(torch.tensor(3.0))) * (1 + 1 / torch.sqrt(torch.tensor(3.0))) / 4],
            [(1 / torch.sqrt(torch.tensor(3.0)) + 1) / 2, (1 - 1 / torch.sqrt(torch.tensor(3.0))) * (1 - 1 / torch.sqrt(torch.tensor(3.0))) / 4],
            [(-1 / torch.sqrt(torch.tensor(3.0)) + 1) / 2, (1 + 1 / torch.sqrt(torch.tensor(3.0))) * (1 + 1 / torch.sqrt(torch.tensor(3.0))) / 4],
            [(-1 / torch.sqrt(torch.tensor(3.0)) + 1) / 2, (1 + 1 / torch.sqrt(torch.tensor(3.0))) * (1 - 1 / torch.sqrt(torch.tensor(3.0))) / 4]
        ], device=device)

    elif Gauss_point_number == 9:
        # Gauss points and coefficients for 9-point Gauss quadrature
        Gauss_coefficient_reference_triangle = torch.tensor([
            64 / 81 * (1 - 0) / 8, 100 / 324 * (1 - torch.sqrt(torch.tensor(3.0 / 5))) / 8,
            100 / 324 * (1 - torch.sqrt(torch.tensor(3.0 / 5))) / 8,
            100 / 324 * (1 + torch.sqrt(torch.tensor(3.0 / 5))) / 8,
            100 / 324 * (1 + torch.sqrt(torch.tensor(3.0 / 5))) / 8,
            40 / 81 * (1 - 0) / 8, 40 / 81 * (1 - 0) / 8,
            40 / 81 * (1 - torch.sqrt(torch.tensor(3.0 / 5))) / 8,
            40 / 81 * (1 + torch.sqrt(torch.tensor(3.0 / 5))) / 8
        ], device=device)

        Gauss_point_reference_triangle = torch.tensor([
            [(1 + 0) / 2, (1 - 0) * (1 + 0) / 4],
            [(1 + torch.sqrt(torch.tensor(3.0 / 5))) / 2,
             (1 - torch.sqrt(torch.tensor(3.0 / 5))) * (1 + torch.sqrt(torch.tensor(3.0 / 5))) / 4],
            [(1 + torch.sqrt(torch.tensor(3.0 / 5))) / 2,
             (1 - torch.sqrt(torch.tensor(3.0 / 5))) * (1 - torch.sqrt(torch.tensor(3.0 / 5))) / 4],
            [(1 - torch.sqrt(torch.tensor(3.0 / 5))) / 2,
             (1 + torch.sqrt(torch.tensor(3.0 / 5))) * (1 + torch.sqrt(torch.tensor(3.0 / 5))) / 4],
            [(1 - torch.sqrt(torch.tensor(3.0 / 5))) / 2,
             (1 + torch.sqrt(torch.tensor(3.0 / 5))) * (1 - torch.sqrt(torch.tensor(3.0 / 5))) / 4],
            [(1 + 0) / 2, (1 - 0) * (1 + torch.sqrt(torch.tensor(3.0 / 5))) / 4],
            [(1 + 0) / 2, (1 - 0) * (1 - torch.sqrt(torch.tensor(3.0 / 5))) / 4],
            [(1 + torch.sqrt(torch.tensor(3.0 / 5))) / 2, (1 - torch.sqrt(torch.tensor(3.0 / 5))) * (1 + 0) / 4],
            [(1 - torch.sqrt(torch.tensor(3.0 / 5))) / 2, (1 + torch.sqrt(torch.tensor(3.0 / 5))) * (1 + 0) / 4]
        ], device=device)

    elif Gauss_point_number == 3:
        # Gauss points and coefficients for 3-point Gauss quadrature
        Gauss_coefficient_reference_triangle = torch.tensor([1 / 6, 1 / 6, 1 / 6], device=device)

        Gauss_point_reference_triangle = torch.tensor([
            [1 / 2, 0],
            [1 / 2, 1 / 2],
            [0, 1 / 2]
        ], device=device)

    # Extract the vertices of the local triangle
    vertices = vertices.to(device)
    x1 = vertices[0, :]
    y1 = vertices[1, :]
    x2 = vertices[2, :]
    y2 = vertices[3, :]
    x3 = vertices[4, :]
    y3 = vertices[5, :]

    # Calculate the Jacobian (area of the reference triangle)
    Jacobi = torch.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

    # Gauss_coefficient_local_triangle = np.array(Gauss_coefficient_reference_triangle).reshape(-1, 1) * np.array(
    #     Jacobi).reshape(1, -1)
    Gauss_coefficient_local_triangle = Gauss_coefficient_reference_triangle.reshape(-1, 1) * Jacobi.reshape(1, -1)

    # 计算局部高斯点坐标
    # Gauss_point_local_triangle = np.zeros((Gauss_point_reference_triangle.shape[0], vertices.shape[1]))
    Gauss_point_local_triangle_x = x1.reshape(1, -1) + \
                                   (x2 - x1).reshape(1, -1) * Gauss_point_reference_triangle[:, 0].reshape(-1, 1) + \
                                   (x3 - x1).reshape(1, -1) * Gauss_point_reference_triangle[:, 1].reshape(-1, 1)
    Gauss_point_local_triangle_y = y1.reshape(1, -1) + \
                                   (y2 - y1).reshape(1, -1) * Gauss_point_reference_triangle[:, 0].reshape(-1, 1) + \
                                   (y3 - y1).reshape(1, -1) * Gauss_point_reference_triangle[:, 1].reshape(-1, 1)


    return Gauss_point_local_triangle_x,Gauss_point_local_triangle_y, Gauss_coefficient_local_triangle

def generate_Gauss_reference_triangle(Gauss_point_number):
    device = torch.device("cuda")  # 设定计算设备
    if Gauss_point_number == 4:
        # Gauss points and coefficients for 4-point Gauss quadrature
        Gauss_coefficient_reference_triangle = torch.tensor([
            (1 - 1 / torch.sqrt(torch.tensor(3.0))) / 8,
            (1 - 1 / torch.sqrt(torch.tensor(3.0))) / 8,
            (1 + 1 / torch.sqrt(torch.tensor(3.0))) / 8,
            (1 + 1 / torch.sqrt(torch.tensor(3.0))) / 8
        ], device=device)

        Gauss_point_reference_triangle = torch.tensor([
            [(1 / torch.sqrt(torch.tensor(3.0)) + 1) / 2,
             (1 - 1 / torch.sqrt(torch.tensor(3.0))) * (1 + 1 / torch.sqrt(torch.tensor(3.0))) / 4],
            [(1 / torch.sqrt(torch.tensor(3.0)) + 1) / 2,
             (1 - 1 / torch.sqrt(torch.tensor(3.0))) * (1 - 1 / torch.sqrt(torch.tensor(3.0))) / 4],
            [(-1 / torch.sqrt(torch.tensor(3.0)) + 1) / 2,
             (1 + 1 / torch.sqrt(torch.tensor(3.0))) * (1 + 1 / torch.sqrt(torch.tensor(3.0))) / 4],
            [(-1 / torch.sqrt(torch.tensor(3.0)) + 1) / 2,
             (1 + 1 / torch.sqrt(torch.tensor(3.0))) * (1 - 1 / torch.sqrt(torch.tensor(3.0))) / 4]
        ], device=device)

    elif Gauss_point_number == 9:
        # Gauss points and coefficients for 9-point Gauss quadrature
        Gauss_coefficient_reference_triangle = torch.tensor([
            64 / 81 * (1 - 0) / 8, 100 / 324 * (1 - torch.sqrt(torch.tensor(3.0 / 5))) / 8,
            100 / 324 * (1 - torch.sqrt(torch.tensor(3.0 / 5))) / 8,
            100 / 324 * (1 + torch.sqrt(torch.tensor(3.0 / 5))) / 8,
            100 / 324 * (1 + torch.sqrt(torch.tensor(3.0 / 5))) / 8,
            40 / 81 * (1 - 0) / 8, 40 / 81 * (1 - 0) / 8,
            40 / 81 * (1 - torch.sqrt(torch.tensor(3.0 / 5))) / 8,
            40 / 81 * (1 + torch.sqrt(torch.tensor(3.0 / 5))) / 8
        ], device=device)

        Gauss_point_reference_triangle = torch.tensor([
            [(1 + 0) / 2, (1 - 0) * (1 + 0) / 4],
            [(1 + torch.sqrt(torch.tensor(3.0 / 5))) / 2,
             (1 - torch.sqrt(torch.tensor(3.0 / 5))) * (1 + torch.sqrt(torch.tensor(3.0 / 5))) / 4],
            [(1 + torch.sqrt(torch.tensor(3.0 / 5))) / 2,
             (1 - torch.sqrt(torch.tensor(3.0 / 5))) * (1 - torch.sqrt(torch.tensor(3.0 / 5))) / 4],
            [(1 - torch.sqrt(torch.tensor(3.0 / 5))) / 2,
             (1 + torch.sqrt(torch.tensor(3.0 / 5))) * (1 + torch.sqrt(torch.tensor(3.0 / 5))) / 4],
            [(1 - torch.sqrt(torch.tensor(3.0 / 5))) / 2,
             (1 + torch.sqrt(torch.tensor(3.0 / 5))) * (1 - torch.sqrt(torch.tensor(3.0 / 5))) / 4],
            [(1 + 0) / 2, (1 - 0) * (1 + torch.sqrt(torch.tensor(3.0 / 5))) / 4],
            [(1 + 0) / 2, (1 - 0) * (1 - torch.sqrt(torch.tensor(3.0 / 5))) / 4],
            [(1 + torch.sqrt(torch.tensor(3.0 / 5))) / 2, (1 - torch.sqrt(torch.tensor(3.0 / 5))) * (1 + 0) / 4],
            [(1 - torch.sqrt(torch.tensor(3.0 / 5))) / 2, (1 + torch.sqrt(torch.tensor(3.0 / 5))) * (1 + 0) / 4]
        ], device=device)

    elif Gauss_point_number == 3:
        # Gauss points and coefficients for 3-point Gauss quadrature
        Gauss_coefficient_reference_triangle = torch.tensor([1 / 6, 1 / 6, 1 / 6], device=device)

        Gauss_point_reference_triangle = torch.tensor([
            [1 / 2, 0],
            [1 / 2, 1 / 2],
            [0, 1 / 2]
        ], device=device)
    return Gauss_coefficient_reference_triangle, Gauss_point_reference_triangle

def generate_Gauss_local_1D(Gauss_point_number, lower_bound, upper_bound):
    device = torch.device("cuda")
    if Gauss_point_number == 4:
        Gauss_coefficient_reference_1D = torch.tensor([0.3478548451, 0.3478548451, 0.6521451549, 0.6521451549], device=device)
        Gauss_point_reference_1D = torch.tensor([0.8611363116, -0.8611363116, 0.3399810436, -0.3399810436], device=device)
    elif Gauss_point_number == 8:
        Gauss_coefficient_reference_1D = torch.tensor([0.1012285363, 0.1012285363, 0.2223810345, 0.2223810345,
                                                   0.3137066459, 0.3137066459, 0.3626837834, 0.3626837834], device=device)
        Gauss_point_reference_1D = torch.tensor([0.9602898565, -0.9602898565, 0.7966664774, -0.7966664774,
                                             0.5255324099, -0.5255324099, 0.1834346425, -0.1834346425], device=device)
    elif Gauss_point_number == 2:
        Gauss_coefficient_reference_1D = torch.tensor([1, 1], device=device)
        Gauss_point_reference_1D = torch.tensor([-1 / torch.sqrt(torch.tensor(3.0, device=device)),
                                                 1 / torch.sqrt(torch.tensor(3.0, device=device))], device=device)

    Gauss_coefficient_local_1D = (upper_bound - lower_bound) * torch.tensor(
        Gauss_coefficient_reference_1D).clone().detach().to(device) / 2
    Gauss_point_local_1D = (upper_bound - lower_bound) * torch.tensor(Gauss_point_reference_1D).clone().detach().to(
        device) / 2 + (upper_bound + lower_bound) / 2
    return Gauss_coefficient_local_1D, Gauss_point_local_1D

'''(u·D)u'''
def assemble_matrix_from_volume_integral_FE_triangle(uh, M_partition, T_partition,
                                                     M_basis_trial, T_basis_trial, T_basis_test, T_basis_FE,trial_basis_type,
                                                     trial_derivative_degree_x, trial_derivative_degree_y,
                                                     test_basis_type, test_derivative_degree_x,
                                                     test_derivative_degree_y, FE_basis_type, FE_derivative_degree_x,
                                                     FE_derivative_degree_y):

    Nb_trial = int(M_basis_trial.shape[1])
    Nb_test = int(M_basis_trial.shape[1])  # Number of basis
    number_of_trial_local_basis = int(T_basis_trial.shape[0])
    number_of_test_local_basis = int(T_basis_test.shape[0])

    A = coo_matrix((Nb_test, Nb_trial))
    vertices = torch.cat([M_partition[:, T_partition[0, :]], M_partition[:, T_partition[1, :]], M_partition[:, T_partition[2, :]]])
    Gauss_point_number = 9
    [Gauss_point_local_triangle_x, Gauss_point_local_triangle_y,
     Gauss_coefficient_local_triangle] = generate_Gauss_local_triangle(Gauss_point_number,
                                                                               vertices)
    for alpha in range(number_of_trial_local_basis):
        for beta in range(number_of_test_local_basis):
            temp = 0
            for i in range(Gauss_point_number):
                trial_basis_value = triangular_local_basis(
                    Gauss_point_local_triangle_x[i, :], Gauss_point_local_triangle_y[i, :],
                    vertices, trial_basis_type, alpha,
                    trial_derivative_degree_x, trial_derivative_degree_y)
                test_basis_value = triangular_local_basis(
                    Gauss_point_local_triangle_x[i, :], Gauss_point_local_triangle_y[i, :],
                    vertices, test_basis_type, beta,
                    test_derivative_degree_x, test_derivative_degree_y)
                coef_value = FE_solution_triangle(
                    Gauss_point_local_triangle_x[i, :], Gauss_point_local_triangle_y[i, :],
                    uh, T_basis_trial, vertices, FE_basis_type, FE_derivative_degree_x, FE_derivative_degree_y)

                temp += Gauss_coefficient_local_triangle[i,:] * coef_value * trial_basis_value * test_basis_value
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                T_basis_test[beta, :] = T_basis_test[beta, :].flatten().to(device)
                T_basis_trial[alpha, :] = T_basis_trial[alpha, :].flatten().to(device)

            A += sp.coo_matrix((temp.cpu().numpy(),
                                (T_basis_test[beta, :].cpu().numpy(), T_basis_trial[alpha, :].cpu().numpy())),
                                shape=(Nb_test, Nb_trial))
            r = A.toarray()
            r = Double(r)
    return r

def assemble_vector_from_volume_integral_2FE_triangle(uh_FE1, uh_FE2, M_partition,
                                                      T_partition, T_basis_test,
                                                      vector_size, test_basis_type,
                                                      test_derivative_degree_x, test_derivative_degree_y,
                                                      FE1_basis_type, FE1_derivative_degree_x, FE1_derivative_degree_y,
                                                      FE2_basis_type,
                                                      FE2_derivative_degree_x, FE2_derivative_degree_y):


    vector_size = vector_size.cpu().numpy()
    R = coo_matrix((vector_size, 1))
    vertices = torch.cat(
        [M_partition[:, T_partition[0, :]], M_partition[:, T_partition[1, :]], M_partition[:, T_partition[2, :]]])
    Gauss_point_number = 9
    [Gauss_point_local_triangle_x, Gauss_point_local_triangle_y,
     Gauss_coefficient_local_triangle] = generate_Gauss_local_triangle(Gauss_point_number,
                                                                       vertices)
    # Loop over all test FE basis functions
    for beta in range(int(T_basis_test.shape[0])):
        temp = 0
        for i in range(Gauss_point_number):
            test_basis_value = triangular_local_basis(
                Gauss_point_local_triangle_x[i, :], Gauss_point_local_triangle_y[i, :],
                vertices, test_basis_type, beta, test_derivative_degree_x,
                test_derivative_degree_y)

            coef_value_1 = FE_solution_triangle(
                Gauss_point_local_triangle_x[i, :], Gauss_point_local_triangle_y[i, :],
                uh_FE1, T_basis_test, vertices, FE1_basis_type,
                FE1_derivative_degree_x, FE1_derivative_degree_y)

            coeff_value_2 = FE_solution_triangle(
                Gauss_point_local_triangle_x[i, :], Gauss_point_local_triangle_y[i, :],
                uh_FE2, T_basis_test, vertices, FE2_basis_type,
                FE2_derivative_degree_x, FE2_derivative_degree_y)

            temp += Gauss_coefficient_local_triangle[i,:] * coef_value_1 * coeff_value_2 * test_basis_value

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            row_indices = T_basis_test[beta, :].to(device)
            col_indices = torch.zeros_like(row_indices, device=device)  # 保持 GPU 计算
        R += sp.coo_matrix((temp.cpu().numpy(), (row_indices.cpu().numpy(), col_indices.cpu().numpy())),
                               shape=(vector_size, 1))
        # R += coo_matrix((temp, (T_basis_test[beta, :].cpu().numpy, np.zeros_like(T_basis_test[beta, :]))), shape=(vector_size, 1))
        r = R.toarray()
        r = Double(r)
    return r

def FE_solution_triangle( x, y, uh, T_basis_trial, vertices, basis_type, derivative_degree_x, derivative_degree_y):
    r = 0
    number_of_local_basis = int(T_basis_trial.shape[0])
    for i in range(int(number_of_local_basis)):
        coef = triangular_local_basis(x, y, vertices, basis_type, i, derivative_degree_x,
                                              derivative_degree_y)
        r += uh[T_basis_trial[i,:], 0].T * coef
        r = Double(r)
    return r

def treat_Dirichlet_boundary_Stokes_with_Darcy(Dirichlet_boundary_function_name_u1, Dirichlet_boundary_function_name_u2,
                                               A, b, boundary_nodes, M_basis_u,
                                               number_of_FE_nodes_u, number_of_unknowns_Darcy):

    nbn = int(boundary_nodes.shape[1])

    for k in range(nbn):

        if boundary_nodes[0, k] == -1:
            global_index = int(boundary_nodes[2, k])
            i = global_index - number_of_unknowns_Darcy
            A[global_index, :] = 0
            A[global_index, global_index] = 1
            b[global_index, 0] = Dirichlet_boundary_function_name_u1(M_basis_u[0, i], M_basis_u[1, i])

        if boundary_nodes[1, k] == -1:
            global_index = int(boundary_nodes[2, k])
            u2_index = number_of_FE_nodes_u + global_index
            i = global_index - number_of_unknowns_Darcy
            A[u2_index, :] = 0
            A[u2_index, u2_index] = 1
            b[u2_index, 0] = Dirichlet_boundary_function_name_u2(M_basis_u[0, i], M_basis_u[1, i])

    return A, b

def ReshapeFix(input, Shape,order='F'):
	if order=='F':
		return torch.reshape(input.T,[Shape[len(Shape)-1-i] \
			                for i in range(len(Shape))]).permute([len(Shape)-1-i \
							for i in range(len(Shape))])
	elif order=='C':
		return torch.reshape(input, Shape)
	else:
		raise ValueError('Reshape Only Support Fortran or C')
