import numpy as np

def get_2D_solution_and_maximum_error(solution_1D, N1_basis, N2_basis, left, bottom, h_basis, phi_exact_solution):
    """
    Transfer the 1D solution into 2D solution and compute the maximum error at all nodes.
    This function is for pure Dirichlet boundary condition.

    :param solution_1D: The 1D solution vector.
    :param N1_basis: The number of sub-intervals of the partition in the x-direction.
    :param N2_basis: The number of sub-intervals of the partition in the y-direction.
    :param left: The left boundary of the domain.
    :param bottom: The bottom boundary of the domain.
    :param h_basis: The step size for the finite element nodes (not the partition).
    :param phi_exact_solution: Function to compute the exact solution at a given (x, y) point.

    :return: solution_2D (the 2D solution grid) and maxerror (the maximum error at all nodes).
    """

    maxerror = 0
    solution_2D = np.zeros((N2_basis+1, N1_basis+1))  # Create a 2D solution grid

    for i in range(N1_basis + 1):
        for j in range(N2_basis + 1):
            # Assign the 1D solution to the 2D grid
            solution_2D[j, i] = solution_1D[i * (N2_basis + 1) + j]

            # Compute the error at the current point
            temp = solution_2D[j, i] - phi_exact_solution(left + (i) * h_basis[0], bottom + (j) * h_basis[1])

            # Update the maximum error if necessary
            if abs(maxerror) < abs(temp):
                maxerror = temp

    return solution_2D, maxerror
