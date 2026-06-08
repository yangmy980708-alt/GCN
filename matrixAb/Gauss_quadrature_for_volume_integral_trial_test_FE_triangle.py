import numpy as np
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
sys.path.insert(0, '../matrixAb')
from matrixAb.triangular_local_basis import triangular_local_basis

def Gauss_quadrature_for_volume_integral_trial_test_FE_triangle(
    coefficient_function_name,
    uh_local,
    Gauss_coefficient_local,
    Gauss_point_local_x,
    Gauss_point_local_y,
    vertices,
    trial_basis_type,
    trial_basis_index,
    trial_derivative_degree_x,
    trial_derivative_degree_y,
    test_basis_type,
    test_basis_index,
    test_derivative_degree_x,
    test_derivative_degree_y,
    FE_basis_type,
    FE_derivative_degree_x,
    FE_derivative_degree_y
):

    Gpn = len(Gauss_coefficient_local)
    r = 0
    x = Gauss_point_local_x
    y = Gauss_point_local_y
    for i in range(Gpn):

        coeff_value = coefficient_function_name(
            x[i,:], y[i,:], uh_local, vertices, FE_basis_type, FE_derivative_degree_x, FE_derivative_degree_y
        )
        trial_basis_value = triangular_local_basis(
            x[i,:], y[i,:], vertices, trial_basis_type, trial_basis_index,
            trial_derivative_degree_x, trial_derivative_degree_y
        )
        test_basis_value = triangular_local_basis(
            x[i,:], y[i,:], vertices, test_basis_type, test_basis_index,
            test_derivative_degree_x, test_derivative_degree_y
        )
        r += Gauss_coefficient_local[i] * coeff_value * trial_basis_value * test_basis_value

    return r
