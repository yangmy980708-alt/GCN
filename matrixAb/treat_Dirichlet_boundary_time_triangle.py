import numpy as np


def treat_Dirichlet_boundary_time_triangle(Dirichlet_boundary_function_name, current_time, A, b, boundary_nodes, M_basis):

    nbn = int(boundary_nodes.shape[1])

    # Check all boundary nodes
    for k in range(nbn):

        # If the k-th boundary node is a Dirichlet boundary node
        if boundary_nodes[0, k] == -1:
            i = int(boundary_nodes[1, k])

            # Reset the i-th equation in the linear system: 1 * u_i = u(X_i)
            A[i, :] = 0
            A[i, i] = 1
            # Evaluate the Dirichlet boundary function at the node's coordinates and assign to b
            b[i, 0] = Dirichlet_boundary_function_name(M_basis[0, i], M_basis[1, i], current_time)

    return A, b
