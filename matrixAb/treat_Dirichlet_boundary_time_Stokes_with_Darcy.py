import numpy as np


def treat_Dirichlet_boundary_time_Stokes_with_Darcy(Dirichlet_boundary_function_name_u1, Dirichlet_boundary_function_name_u2,
                                               t, A, b, boundary_nodes, M_basis_u,
                                               number_of_FE_nodes_u, number_of_unknowns_Darcy):

    # Get the total number of boundary nodes
    nbn = int(boundary_nodes.shape[1])

    # Check all boundary nodes of FE
    for k in range(nbn):

        # If the kth boundary node is a Dirichlet boundary node in the normal direction (u1)
        if boundary_nodes[0, k] == -1:
            global_index = int(boundary_nodes[2, k])
            i = global_index - number_of_unknowns_Darcy
            A[global_index, :] = 0
            A[global_index, global_index] = 1
            b[global_index, 0] = Dirichlet_boundary_function_name_u1(M_basis_u[0, i], M_basis_u[1, i], t)

        # If the kth boundary node is a Dirichlet boundary node in the tangential direction (u2)
        if boundary_nodes[1, k] == -1:
            global_index = int(boundary_nodes[2, k])
            u2_index = number_of_FE_nodes_u + global_index
            i = global_index - number_of_unknowns_Darcy
            A[u2_index, :] = 0
            A[u2_index, u2_index] = 1
            b[u2_index, 0] = Dirichlet_boundary_function_name_u2(M_basis_u[0, i], M_basis_u[1, i], t)

    return A, b
