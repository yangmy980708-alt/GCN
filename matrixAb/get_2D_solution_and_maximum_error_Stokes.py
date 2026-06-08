import numpy as np

def get_2D_solution_and_maximum_error_Stokes(uh1, uh2, ph, N1_basis_u, N2_basis_u, N1_basis_p, N2_basis_p, left, bottom, h_basis_u, h_basis_p, u1_exact_solution, u2_exact_solution, p_exact_solution):
    """
    Transfer the 1D solution into 2D solution and compute the maximum error at all nodes for Stokes equation.
    This function is for pure Dirichlet boundary condition.

    :param uh1: The 1D velocity solution in the x-direction (uh1).
    :param uh2: The 1D velocity solution in the y-direction (uh2).
    :param ph: The 1D pressure solution (ph).
    :param N1_basis_u: The number of sub-intervals of the partition in the x-direction for velocity.
    :param N2_basis_u: The number of sub-intervals of the partition in the y-direction for velocity.
    :param N1_basis_p: The number of sub-intervals of the partition in the x-direction for pressure.
    :param N2_basis_p: The number of sub-intervals of the partition in the y-direction for pressure.
    :param left: The left boundary of the domain.
    :param bottom: The bottom boundary of the domain.
    :param h_basis_u: The step size for the finite element nodes in the velocity field.
    :param h_basis_p: The step size for the finite element nodes in the pressure field.
    :param u1_exact_solution: Function to compute the exact solution for u1 (x-component of velocity).
    :param u2_exact_solution: Function to compute the exact solution for u2 (y-component of velocity).
    :param p_exact_solution: Function to compute the exact solution for pressure.

    :return: solution_2D_uh1, solution_2D_uh2, solution_2D_ph: The 2D solutions for u1, u2, and pressure.
             maxerror_uh: The maximum error for velocity.
             maxerror_ph: The maximum error for pressure.
    """

    # For velocity component u1
    maxerror_uh1 = 0
    solution_2D_uh1 = np.zeros((N2_basis_u + 1, N1_basis_u + 1))
    for i in range(N1_basis_u + 1):
        for j in range(N2_basis_u + 1):
            solution_2D_uh1[j, i] = uh1[i * (N2_basis_u + 1) + j]
            temp = solution_2D_uh1[j, i] - u1_exact_solution(left + i * h_basis_u[0], bottom + j * h_basis_u[1])
            maxerror_uh1 = max(maxerror_uh1, abs(temp))

    # For velocity component u2
    maxerror_uh2 = 0
    solution_2D_uh2 = np.zeros((N2_basis_u + 1, N1_basis_u + 1))
    for i in range(N1_basis_u + 1):
        for j in range(N2_basis_u + 1):
            solution_2D_uh2[j, i] = uh2[i * (N2_basis_u + 1) + j]
            temp = solution_2D_uh2[j, i] - u2_exact_solution(left + i * h_basis_u[0], bottom + j * h_basis_u[1])
            maxerror_uh2 = max(maxerror_uh2, abs(temp))

    maxerror_uh = max(maxerror_uh1, maxerror_uh2)

    # For pressure field
    maxerror_ph = 0
    solution_2D_ph = np.zeros((N2_basis_p + 1, N1_basis_p + 1))
    for i in range(N1_basis_p + 1):
        for j in range(N2_basis_p + 1):
            solution_2D_ph[j, i] = ph[i * (N2_basis_p + 1) + j]
            temp = solution_2D_ph[j, i] - p_exact_solution(left + i * h_basis_p[0], bottom + j * h_basis_p[1])
            maxerror_ph = max(maxerror_ph, abs(temp))

    return solution_2D_uh1, solution_2D_uh2, solution_2D_ph, maxerror_uh, maxerror_ph
