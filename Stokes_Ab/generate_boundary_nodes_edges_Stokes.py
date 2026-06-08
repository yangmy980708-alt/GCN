import numpy as np

def generate_boundary_nodes_edges_Stokes(N1_basis, N2_basis, N1_partition, N2_partition):
    # Number of boundary nodes (nbn) and boundary edges (nbe)
    N1_basis = int(N1_basis)
    N1_partition = int(N1_partition)
    N2_basis = int(N2_basis)
    N2_partition = int(N2_partition)
    nbn = 2 * (N1_basis + N2_basis)
    boundary_nodes = np.zeros((7, nbn))

    # Set all boundary nodes as Dirichlet (-1) by default for both directions
    boundary_nodes[0, :] = -1  # Normal direction (u1)
    boundary_nodes[1, :] = -1  # Tangential direction (u2)

    # Change bottom boundary nodes to non-boundary (-100)
    for k in range(1, N1_basis):
        boundary_nodes[0, k] = -100
        boundary_nodes[1, k] = -100

    # Bottom boundary nodes
    for k in range(N1_basis):
        boundary_nodes[2, k] = k * (N2_basis + 1)  # Global index
        boundary_nodes[3:5, k] = [0, -1]  # Normal vector

    # Right boundary nodes
    for k in range(N1_basis, N1_basis + N2_basis):
        boundary_nodes[2, k] = N1_basis * (N2_basis + 1) + (k - N1_basis + 1) - 1
        boundary_nodes[3:5, k] = [1, 0]

    # Top boundary nodes
    for k in range(N1_basis + N2_basis, 2 * N1_basis + N2_basis):
        boundary_nodes[2, k] = (2 * N1_basis + N2_basis + 1 - k) * (N2_basis + 1) - 1
        boundary_nodes[3:5, k] = [0, 1]

    # Left boundary nodes
    for k in range(2 * N1_basis + N2_basis, nbn):
        boundary_nodes[2, k] = 2 * N1_basis + 2 * N2_basis - k
        boundary_nodes[3:5, k] = [-1, 0]

    # Correct normal direction at corners
    # Left-bottom corner
    boundary_nodes[3:5, 0] = [-1, -1] / np.sqrt(2)
    # Right-bottom corner
    boundary_nodes[3:5, N1_basis] = [1, -1] / np.sqrt(2)
    # Right-top corner
    boundary_nodes[3:5, N1_basis + N2_basis] = [1, 1] / np.sqrt(2)
    # Left-top corner
    boundary_nodes[3:5, 2 * N1_basis + N2_basis] = [-1, 1] / np.sqrt(2)

    # Calculate tangential vectors (τ₁ = -n₂, τ₂ = n₁)
    boundary_nodes[5, :] = -boundary_nodes[4, :]
    boundary_nodes[6, :] = boundary_nodes[3, :]

    # Initialize boundary edges matrix (9 rows x nbe columns)
    nbe = 2 * (N1_partition + N2_partition)
    boundary_edges = np.zeros((9, nbe))

    # Set all boundary edges as Dirichlet (-1) by default for both directions
    boundary_edges[0, :] = -1  # Normal direction
    boundary_edges[1, :] = -1  # Tangential direction

    # Change bottom boundary edges to non-boundary (-100)
    for k in range(N1_partition):
        boundary_edges[0, k] = -100
        boundary_edges[1, k] = -100

    # Bottom boundary edges
    for k in range(N1_partition):
        boundary_edges[2, k] = k * 2 * N2_partition  # Element index
        boundary_edges[3, k] = k * (N2_partition + 1)  # Endpoint 1
        boundary_edges[4, k] = (k + 1) * (N2_partition + 1)  # Endpoint 2
        boundary_edges[5:7, k] = [0, -1]  # Normal vector

    # Right boundary edges
    for k in range(N1_partition, N1_partition + N2_partition):
        boundary_edges[2, k] = (N1_partition - 1) * 2 * N2_partition + 2 * (k - N1_partition + 1) - 1
        boundary_edges[3, k] = N1_partition * (N2_partition + 1) + (k - N1_partition + 1) - 1
        boundary_edges[4, k] = N1_partition * (N2_partition + 1) + (k - N1_partition + 2) - 1
        boundary_edges[5:7, k] = [1, 0]

    # Top boundary edges
    for k in range(N1_partition + N2_partition, 2 * N1_partition + N2_partition):
        boundary_edges[2, k] = (2 * N1_partition + N2_partition - k) * 2 * N2_partition - 1
        boundary_edges[3, k] = (2 * N1_partition + N2_partition + 1 - k) * (N2_partition + 1) - 1
        boundary_edges[4, k] = (2 * N1_partition + N2_partition - k) * (N2_partition + 1) - 1
        boundary_edges[5:7, k] = [0, 1]

    # Left boundary edges
    for k in range(2 * N1_partition + N2_partition, nbe):
        boundary_edges[2, k] = 2 * (2 * N1_partition + 2 * N2_partition - k) - 2
        boundary_edges[3, k] = 2 * N1_partition + 2 * N2_partition - k
        boundary_edges[4, k] = 2 * N1_partition + 2 * N2_partition - k - 1
        boundary_edges[5:7, k] = [-1, 0]

    # Calculate tangential vectors (τ₁ = -n₂, τ₂ = n₁)
    boundary_edges[7, :] = -boundary_edges[6, :]
    boundary_edges[8, :] = boundary_edges[5, :]

    return boundary_nodes, boundary_edges
