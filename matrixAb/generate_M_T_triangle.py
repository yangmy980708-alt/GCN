import numpy as np


def generate_M_T_triangle(left, right, bottom, top, h_partition, basis_type):
    h = np.array(h_partition)
    # print("h=", h)

    if basis_type == 1:
        N1 = int((right - left) / h[0])
        N2 = int((top - bottom) / h[1])
        tnp = int((N1 + 1) * (N2 + 1))
        M = np.zeros((2, tnp))
        T = np.zeros((3, 2 * N1 * N2), dtype=int)
        Q = np.zeros((N1 + 1, N2 + 1), dtype=int)

        # Set coordinates of the mesh grid
        for j in range(tnp):
            if (j + 1) % (N2 + 1) == 0:

                M[0, j] = left + ((j + 1) / (N2 + 1) - 1) * h[0]
                M[1, j] = top
            else:
                M[0, j] = left + int((j + 1) / (N2 + 1)) * h[0]
                M[1, j] = bottom + (((j + 1) % (N2 + 1)) - 1) * h[1]

            # Create global indices for the mesh grid
        for i in range(N1 + 1):
            for j in range(N2 + 1):
                Q[i, j] = i * (N2 + 1) + (j + 1) - 1

            # Assign triangle indices to the elements
        for n in range(N1 * N2):
            if ((n + 1) % N2) == 0:
                row = N2
                column = int((n + 1) / N2)
            else:
                row = (n + 1) % N2
                column = int((n + 1) / N2) + 1

            T[0, 2 * n] = Q[column - 1, row - 1]
            T[1, 2 * n] = Q[column, row - 1]
            T[2, 2 * n] = Q[column - 1, row]

            T[0, 2 * n + 1] = Q[column - 1, row]
            T[1, 2 * n + 1] = Q[column, row - 1]
            T[2, 2 * n + 1] = Q[column, row]


    elif basis_type == 2:
        N1 = int((right - left) / h[0])
        N2 = int((top - bottom) / h[1])
        dh = h / 2
        dN1 = N1 * 2
        dN2 = N2 * 2
        tnp = (dN1 + 1) * (dN2 + 1)
        M = np.zeros((2, tnp))
        T = np.zeros((6, 2 * N1 * N2), dtype=int)
        Q = np.zeros((dN1 + 1, dN2 + 1), dtype=int)

        # Set coordinates of the mesh grid
        for j in range(tnp):
            if ((j + 1) % (dN2 + 1)) == 0:
                M[0, j] = left + ((j + 1) / (dN2 + 1) - 1) * dh[0]
                M[1, j] = top
            else:
                M[0, j] = left + np.fix((j + 1) / (dN2 + 1)) * dh[0]
                M[1, j] = bottom + ((j + 1) % (dN2 + 1) - 1) * dh[1]

            # Create global indices for the mesh grid
        for i in range(dN1 + 1):
            for j in range(dN2 + 1):
                Q[i, j] = i * (dN2 + 1) + (j + 1) - 1

            # Assign triangle indices to the elements
        for n in range(N1 * N2):
            if ((n + 1) % N2) == 0:
                row = N2
                column = int((n + 1) / N2)
            else:
                row = (n + 1) % N2
                column = int(np.fix((n + 1) / N2) + 1)

            T[0, 2 * n] = Q[2 * column - 2, 2 * row - 2]
            T[1, 2 * n] = Q[2 * column, 2 * row - 2]
            T[2, 2 * n] = Q[2 * column - 2, 2 * row]
            T[3, 2 * n] = Q[2 * column - 1, 2 * row - 2]
            T[4, 2 * n] = Q[2 * column - 1, 2 * row - 1]
            T[5, 2 * n] = Q[2 * column - 2, 2 * row - 1]

            T[0, 2 * n + 1] = Q[2 * column - 2, 2 * row]
            T[1, 2 * n + 1] = Q[2 * column, 2 * row - 2]
            T[2, 2 * n + 1] = Q[2 * column, 2 * row]
            T[3, 2 * n + 1] = Q[2 * column - 1, 2 * row - 1]
            T[4, 2 * n + 1] = Q[2 * column, 2 * row - 1]
            T[5, 2 * n + 1] = Q[2 * column - 1, 2 * row]

    return M, T