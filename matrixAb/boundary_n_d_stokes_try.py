import numpy as np

N1_basis=8; N2_basis=8; N1_partition=8; N2_partition=8
# Number of boundary nodes (nbn) and boundary edges (nbe)
nbn = int(2 * (N1_basis + N2_basis))
nbe = int(2 * (N1_partition + N2_partition))

    # Initialize boundary nodes matrix (3 x nbn)
boundary_nodes = np.zeros((7, nbn))

    # Set all boundary nodes to Dirichlet boundary nodes (u1 and u2 directions)
boundary_nodes[0, :] = -1
boundary_nodes[1, :] = -1

    # Set top boundary nodes to non-boundary nodes
for k in range(N1_basis + N2_basis + 1, 2 * N1_basis + N2_basis):
    boundary_nodes[0, k] = -100
    boundary_nodes[1, k] = -100

    # Bottom boundary nodes
for k in range(N1_basis):
    boundary_nodes[2, k] = k * (N2_basis + 1)
    boundary_nodes[3:5, k] = [0, -1]

    # Right boundary nodes
for k in range(N1_basis, N1_basis + N2_basis):
    boundary_nodes[2, k] = N1_basis * (N2_basis + 1) + k - N1_basis
    boundary_nodes[3:5, k] = [1, 0]

    # Top boundary nodes
for k in range(N1_basis + N2_basis, 2 * N1_basis + N2_basis):
    boundary_nodes[2, k] = (2 * N1_basis + N2_basis + 1 - k) * (N2_basis + 1) - 1
    boundary_nodes[3:5, k] = [0, 1]

    # Left boundary nodes
for k in range(2 * N1_basis + N2_basis, nbn):
    boundary_nodes[2, k] = 2 * N1_basis + 2 * N2_basis - k
    boundary_nodes[3:5, k] = [-1, 0]

    # Correct the normal direction at the corners
boundary_nodes[3:5, 0] = [-1, -1] / np.sqrt(2)  # left-bottom corner
boundary_nodes[3:5, N1_basis] = [1, -1] / np.sqrt(2)  # right-bottom corner
boundary_nodes[3:5, N1_basis + N2_basis] = [1, 1] / np.sqrt(2)  # right-top corner
boundary_nodes[3:5, 2 * N1_basis + N2_basis] = [-1, 1] / np.sqrt(2)  # left-top corner

    # Tangential vectors (counterclockwise: tau = [-n2, n1])
boundary_nodes[5, :] = -boundary_nodes[4, :]
boundary_nodes[6, :] = boundary_nodes[3, :]

    # Initialize boundary edges matrix (9 x nbe)
boundary_edges = np.zeros((9, nbe))

    # Set all boundary edges to Dirichlet boundary edges (normal and tangential directions)
boundary_edges[0, :] = -1
boundary_edges[1, :] = -1

    # Set top boundary edges to non-boundary edges
for k in range(N1_partition + N2_partition, 2 * N1_partition + N2_partition):
    boundary_edges[0, k] = -100
    boundary_edges[1, k] = -100

    # Bottom boundary edges
for k in range(N1_partition):
    boundary_edges[2, k] = k * 2 * N2_partition
    boundary_edges[3, k] = k * (N2_partition + 1)
    boundary_edges[4, k] = (k + 1) * (N2_partition + 1)
    boundary_edges[5:7, k] = [0, -1]

    # Right boundary edges
for k in range(N1_partition, N1_partition + N2_partition):
    boundary_edges[2, k] = (N1_partition - 1) * 2 * N2_partition + 2 * (k - N1_partition) + 1
    boundary_edges[3, k] = N1_partition * (N2_partition + 1) + (k - N1_partition)
    boundary_edges[4, k] = N1_partition * (N2_partition + 1) + (k - N1_partition) + 1
    boundary_edges[5:7, k] = [1, 0]

    # Top boundary edges
for k in range(N1_partition + N2_partition, 2 * N1_partition + N2_partition):
    boundary_edges[2, k] = (2 * N1_partition + N2_partition - k) * 2 * N2_partition - 1
    boundary_edges[3, k] = (2 * N1_partition + N2_partition + 1 - k) * (N2_partition + 1) - 1
    boundary_edges[4, k] = (2 * N1_partition + N2_partition - k) * (N2_partition + 1) - 1
    boundary_edges[5:7, k] = [0, 1]

    # Left boundary edges
for k in range(2 * N1_partition + N2_partition, nbe):
    boundary_edges[2, k] = 2 * (2 * N1_partition + 2 * N2_partition- k) - 2
    boundary_edges[3, k] = 2 * N1_partition + 2 * N2_partition - k
    boundary_edges[4, k] = 2 * N1_partition + 2 * N2_partition - k - 1
    boundary_edges[5:7, k] = [-1, 0]

    # Tangential vectors (counterclockwise: tau = [-n2, n1])
boundary_edges[7, :] = -boundary_edges[6, :]
boundary_edges[8, :] = boundary_edges[5, :]
print(boundary_edges)