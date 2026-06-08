import numpy as np


def generate_interface_edges_StokesDarcy(interface_end_point_1, interface_end_point_2,
                                         left_S, right_S, bottom_S, top_S,
                                         left_D, right_D, bottom_D, top_D,
                                         h_partition_S, h_partition_D):
    N1_partition_D = int((right_D - left_D) / h_partition_D[0])
    N2_partition_D = int((top_D - bottom_D) / h_partition_D[1])
    N2_partition_S = int((top_S - bottom_S) / h_partition_S[1])

    # Number of interface edges
    nie = N1_partition_D
    interface_edges = np.zeros((10, nie))
    # Generate interface edges information
    for k in range(nie):
        # Element indices
        interface_edges[0, k] = (k + 1) * 2 * N2_partition_D - 1  # Darcy domain element
        interface_edges[1, k] = k * 2 * N2_partition_S  # Stokes domain element

        # Endpoint coordinates
        interface_edges[2, k] = interface_end_point_1[0] + k * h_partition_D[0]  # x1
        interface_edges[3, k] = interface_end_point_1[1]  # y1
        interface_edges[4, k] = interface_end_point_1[0] + (k + 1) * h_partition_D[0]  # x2
        interface_edges[5, k] = interface_end_point_1[1]  # y2

        # Normal vector (pointing outward from Stokes side)
        interface_edges[6:8, k] = [0, -1]
    interface_edges[8, :] = -interface_edges[7, :]
    interface_edges[9, :] = interface_edges[6, :]

    return interface_edges

