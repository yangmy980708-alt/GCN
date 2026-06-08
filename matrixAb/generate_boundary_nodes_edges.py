import numpy as np

def generate_boundary_nodes_edges(N1_basis, N2_basis, N1_partition, N2_partition):
    N1_basis = int(N1_basis)
    N1_partition = int(N1_partition)
    N2_basis = int(N2_basis)
    N2_partition = int(N2_partition)

    nbn = 2 * (N1_basis + N2_basis)
    nbe = 2 * (N1_partition + N2_partition)

    # Initialize boundary_nodes and boundary_edges
    boundary_nodes = np.zeros((2, nbn))
    boundary_edges = np.zeros((4, nbe))

    # Set all boundary nodes to Dirichlet (-1)
    boundary_nodes[0, :] = -1

    # Change bottom boundary nodes to non-boundary nodes (-100)
    for k in range(1, N1_basis):
        boundary_nodes[0, k] = -100

        # Bottom boundary nodes (using FE index)
    for k in range(N1_basis):
        boundary_nodes[1, k] = k * (N2_basis + 1)

        # Right boundary nodes
    for k in range(N1_basis, N1_basis + N2_basis):
        boundary_nodes[1, k] = N1_basis * (N2_basis + 1) + k - N1_basis

        # Top boundary nodes
    for k in range(N1_basis + N2_basis, 2 * N1_basis + N2_basis):
        boundary_nodes[1, k] = (2 * N1_basis + N2_basis - k + 1) * (N2_basis + 1) - 1

        # Left boundary nodes
    for k in range(2 * N1_basis + N2_basis, nbn):
        boundary_nodes[1, k] = 2 * N1_basis + 2 * N2_basis - k

        # Set all boundary edges to Dirichlet (-1)
    boundary_edges[0, :] = -1

    # Change bottom boundary edges to non-boundary edges (-100)
    for k in range(N1_partition):
        boundary_edges[0, k] = -100

        # Bottom boundary edges (using partition index)
    for k in range(N1_partition):
        boundary_edges[1, k] = k * 2 * N2_partition
        boundary_edges[2, k] = k * (N2_partition + 1)
        boundary_edges[3, k] = (k + 1) * (N2_partition + 1)

        # Right boundary edges
    for k in range(N1_partition, N1_partition + N2_partition):
        boundary_edges[1, k] = (N1_partition - 1) * 2 * N2_partition + 2 * (k - N1_partition + 1) - 1
        boundary_edges[2, k] = N1_partition * (N2_partition + 1) + (k - N1_partition + 1) - 1
        boundary_edges[3, k] = N1_partition * (N2_partition + 1) + (k - N1_partition + 1)

        # Top boundary edges
    for k in range(N1_partition + N2_partition, 2 * N1_partition + N2_partition):
        boundary_edges[1, k] = (2 * N1_partition + N2_partition - k) * 2 * N2_partition - 1
        boundary_edges[2, k] = (2 * N1_partition + N2_partition + 1 - k) * (N2_partition + 1) - 1
        boundary_edges[3, k] = (2 * N1_partition + N2_partition - k) * (N2_partition + 1) - 1

        # Left boundary edges
    for k in range(2 * N1_partition + N2_partition, nbe):
        boundary_edges[1, k] = 2 * (2 * N1_partition + 2 * N2_partition - k) - 2
        boundary_edges[2, k] = 2 * N1_partition + 2 * N2_partition - k
        boundary_edges[3, k] = 2 * N1_partition + 2 * N2_partition - k - 1

    return boundary_nodes, boundary_edges
