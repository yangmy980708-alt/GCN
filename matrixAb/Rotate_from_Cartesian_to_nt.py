import numpy as np


def Rotate_from_Cartesian_to_nt(A, boundary_nodes, number_of_nodes_u):


    # Get the number of boundary nodes
    nbn = boundary_nodes.shape[1]

    # Rotate for all boundary nodes of FE
    for k in range(nbn):
        # Get the global index of the boundary nodes for u1 and u2
        u1_index = boundary_nodes[2, k]
        u2_index = u1_index + number_of_nodes_u

        # Perform the rotation according to the notes "Notes for tool box of standard triangular FE" (*31)
        temp = A[:, u1_index] * boundary_nodes[3, k] + A[:, u2_index] * boundary_nodes[4, k]
        A[:, u2_index] = A[:, u1_index] * boundary_nodes[5, k] + A[:, u2_index] * boundary_nodes[6, k]
        A[:, u1_index] = temp

    return A
