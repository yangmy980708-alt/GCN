import numpy as np


def generate_interface_edges_StokesDarcy(interface_end_point_1, interface_end_point_2,
                                         left_S, right_S, bottom_S, top_S,
                                         left_D, right_D, bottom_D, top_D,
                                         h_partition_S, h_partition_D):

    N1_partition_D = (right_D - left_D) / h_partition_D[0]
    N2_partition_D = (top_D - bottom_D) / h_partition_D[1]
    N2_partition_S = (top_S - bottom_S) / h_partition_S[1]

    # Total number of interface edges.
    nie = int(N1_partition_D)

    # Initialize the interface_edges array (10 rows, nie columns)
    interface_edges = np.zeros((10, nie))

    # Loop through each interface edge
    for k in range(1, int(N1_partition_D) + 1):
        # Darcy domain element index (in Darcy domain, each element is indexed by k)
        interface_edges[0, k - 1] = (k - 1) * 2 * N2_partition_D
        # Stokes domain element index (in Stokes domain, each element is indexed by k)
        interface_edges[1, k - 1] = k * 2 * N2_partition_S - 1

        # Coordinates of the first end of the interface edge
        interface_edges[2, k - 1] = interface_end_point_1[0] + (k - 1) * h_partition_D[0]
        interface_edges[3, k - 1] = interface_end_point_1[1]

        # Coordinates of the second end of the interface edge
        interface_edges[4, k - 1] = interface_end_point_1[0] + k * h_partition_D[0]
        interface_edges[5, k - 1] = interface_end_point_1[1]

        # Unit outer normal vector on the interface edge from Stokes side
        interface_edges[6, k - 1] = 0
        interface_edges[7, k - 1] = 1

    # The tangential vector is counterclockwise to the normal vector.
    # So tau_1 = -n_2, tau_2 = n_1
    interface_edges[8, :] = -interface_edges[7, :]
    interface_edges[9, :] = interface_edges[6, :]

    return interface_edges

