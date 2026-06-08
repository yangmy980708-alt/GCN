import numpy as np

left_S=0;right_S=2;bottom_S=-2;top_S=0
left_D=0;right_D=2;bottom_D=0;top_D=2
h_partition_S=(2,2);h_partition_D=(2,2)
fix_pressure=1;Dirichlet_switch=1;Darcy_scaling_constant=1
nonlinear_tolerance=0.1;nonlinear_max_steps=10

interface_end_point_1 = [left_S, top_S]
interface_end_point_2 = [right_S, top_S]

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

print('int_face_edge=',interface_edges)

