import numpy as np


def fix_pressure_time_Stokes(Dirichlet_boundary_function_name_p, current_time, fix_pressure, A, b, number_of_unknows_before_p,
                        fixed_p_index, M_basis_p):

    if fix_pressure == 1:
        # Fix the pressure at the fixed_p_index FE node
        A[number_of_unknows_before_p + fixed_p_index - 1, :] = 0
        A[number_of_unknows_before_p + fixed_p_index - 1, number_of_unknows_before_p + fixed_p_index - 1] = 1
        b[number_of_unknows_before_p + fixed_p_index - 1, 0] = Dirichlet_boundary_function_name_p(
            M_basis_p[0, fixed_p_index-1], M_basis_p[1, fixed_p_index-1], current_time)

    return A, b