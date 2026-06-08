import numpy as np
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
sys.path.insert(0, '../matrixAb')
from matrixAb.generate_Gauss_local_1D import generate_Gauss_local_1D
from matrixAb.triangular_local_basis_element import triangular_local_basis_element


def Gauss_quadrature_for_line_integral_trial_test_triangle(
    coefficient_function_name,
    Gauss_coefficient_reference_1D,
    Gauss_point_reference_1D,
    end_point_1,
    end_point_2,
    trial_vertices,
    trial_basis_type,
    trial_basis_index,
    trial_derivative_degree_x,
    trial_derivative_degree_y,
    test_vertices,
    test_basis_type,
    test_basis_index,
    test_derivative_degree_x,
    test_derivative_degree_y
):
    """
    Use Gauss quadrature to compute a line integral on an edge for a matrix.

    Parameters and their descriptions are the same as in the MATLAB version.
    """
    Gpn = len(Gauss_coefficient_reference_1D)
    r = 0

    if end_point_1[1] == end_point_2[1]:  # The line is horizontal
        lower_bound = min(end_point_1[0], end_point_2[0])
        upper_bound = max(end_point_1[0], end_point_2[0])
        Gauss_coefficient_local_1D, Gauss_point_local_1D = generate_Gauss_local_1D(
            Gpn, lower_bound, upper_bound)
        for i in range(Gpn):
            coef = coefficient_function_name(Gauss_point_local_1D[i], end_point_1[1])
            r_1 = triangular_local_basis_element(
                    Gauss_point_local_1D[i], end_point_1[1],
                    trial_vertices, trial_basis_type, trial_basis_index,
                    trial_derivative_degree_x, trial_derivative_degree_y)
            r_2 = triangular_local_basis_element(
                    Gauss_point_local_1D[i], end_point_1[1],
                    test_vertices, test_basis_type, test_basis_index,
                    test_derivative_degree_x, test_derivative_degree_y)
            r += np.unique(Gauss_coefficient_local_1D[i]) * coef * r_1 * r_2

    elif end_point_1[0] == end_point_2[0]:  # The line is vertical
        lower_bound = min(end_point_1[1], end_point_2[1])
        upper_bound = max(end_point_1[1], end_point_2[1])
        Gauss_coefficient_local_1D, Gauss_point_local_1D = generate_Gauss_local_1D(
            Gpn, lower_bound, upper_bound)
        for i in range(Gpn):
            coef = coefficient_function_name(end_point_1[0], Gauss_point_local_1D[i])
            r_1 = triangular_local_basis_element(
                    end_point_1[0], Gauss_point_local_1D[i],
                    trial_vertices, trial_basis_type, trial_basis_index,
                    trial_derivative_degree_x, trial_derivative_degree_y)
            r_2 = triangular_local_basis_element(
                    end_point_1[0], Gauss_point_local_1D[i],
                    test_vertices, test_basis_type, test_basis_index,
                    test_derivative_degree_x, test_derivative_degree_y)
            r += np.unique(Gauss_coefficient_local_1D[i]) * coef * r_1 * r_2

    else:  # The slope of the edge is in (0, infinity)
        lower_bound = min(end_point_1[0], end_point_2[0])
        upper_bound = max(end_point_1[0], end_point_2[0])
        Gauss_coefficient_local_1D, Gauss_point_local_1D = generate_Gauss_local_1D(
            Gpn, lower_bound, upper_bound
        )
        slope = (end_point_2[1] - end_point_1[1]) / (end_point_2[0] - end_point_1[0])
        Jacobi = np.sqrt(1 + slope**2)
        for i in range(Gpn):
            x = Gauss_point_local_1D[i]
            y = slope * (x - end_point_1[0]) + end_point_1[1]
            coef = coefficient_function_name(x, y)
            r_1 = triangular_local_basis_element(
                    x, y, trial_vertices, trial_basis_type, trial_basis_index,
                    trial_derivative_degree_x, trial_derivative_degree_y)
            r_2 = triangular_local_basis_element(
                    x, y, test_vertices, test_basis_type, test_basis_index,
                    test_derivative_degree_x, test_derivative_degree_y)
            r += np.unique(Gauss_coefficient_local_1D[i]) * Jacobi * coef * r_1 * r_2

    return r
