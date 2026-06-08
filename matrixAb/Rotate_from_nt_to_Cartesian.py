import numpy as np


def Rotate_from_nt_to_Cartesian(r, boundary_nodes, number_of_nodes_u):


    # Get the total number of boundary nodes
    nbn = boundary_nodes.shape[1]

    # Rotate for all boundary nodes of FE in the backward index
    for k in range(nbn - 1, -1, -1):
        u1_index = boundary_nodes[2, k]
        u2_index = u1_index + number_of_nodes_u

        # Rotate the components (implement (*32) in section 2-3-3-5)
        temp = r[u1_index] * boundary_nodes[3, k] + r[u2_index] * boundary_nodes[5, k]
        r[u2_index] = r[u1_index] * boundary_nodes[4, k] + r[u2_index] * boundary_nodes[6, k]
        r[u1_index] = temp

    return r
