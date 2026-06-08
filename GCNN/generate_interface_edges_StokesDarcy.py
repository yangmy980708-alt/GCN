import torch


def generate_interface_edges_StokesDarcy(interface_end_point_1, interface_end_point_2,
                                         left_S, right_S, bottom_S, top_S,
                                         left_D, right_D, bottom_D, top_D,
                                         h_partition_S, h_partition_D, device='cuda'):
    """
    Generate the information matrix for interface edges in the Stokes-Darcy system using CUDA.
    """
    # Ensure computations are done on GPU
    interface_end_point_1 = torch.tensor(interface_end_point_1, device=device)
    interface_end_point_2 = torch.tensor(interface_end_point_2, device=device)
    h_partition_S = torch.tensor(h_partition_S, device=device)
    h_partition_D = torch.tensor(h_partition_D, device=device)

    N1_partition_D = (right_D - left_D) / h_partition_D[0]
    N2_partition_D = (top_D - bottom_D) / h_partition_D[1]
    N2_partition_S = (top_S - bottom_S) / h_partition_S[1]

    # Total number of interface edges.
    nie = int(N1_partition_D)

    # Initialize the interface_edges tensor (10 rows, nie columns) on GPU
    interface_edges = torch.zeros((10, nie), device=device)

    k_values = torch.arange(1, nie + 1, device=device)

    # Darcy domain element index
    interface_edges[0, :] = (k_values - 1) * 2 * N2_partition_D
    # Stokes domain element index
    interface_edges[1, :] = k_values * 2 * N2_partition_S - 1

    # Coordinates of the first end of the interface edge
    interface_edges[2, :] = interface_end_point_1[0] + (k_values - 1) * h_partition_D[0]
    interface_edges[3, :] = interface_end_point_1[1]

    # Coordinates of the second end of the interface edge
    interface_edges[4, :] = interface_end_point_1[0] + k_values * h_partition_D[0]
    interface_edges[5, :] = interface_end_point_1[1]

    # Unit outer normal vector on the interface edge from Stokes side
    interface_edges[6, :] = 0
    interface_edges[7, :] = 1

    # The tangential vector
    interface_edges[8, :] = -interface_edges[7, :]
    interface_edges[9, :] = interface_edges[6, :]

    return interface_edges
