import numpy as np
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
sys.path.insert(0, '../matrixAb')
from matrixAb.triangular_local_basis import triangular_local_basis

def Gauss_quadrature_for_volume_integral_test_2FE_triangle(
    coefficient_function_name,
    uh_local_1,
    uh_local_2,
    Gauss_coefficient_local,
    Gauss_point_local,
    vertices,
    test_basis_type,
    test_basis_index,
    test_derivative_degree_x,
    test_derivative_degree_y,
    FE1_basis_type,
    FE1_derivative_degree_x,
    FE1_derivative_degree_y,
    FE2_basis_type,
    FE2_derivative_degree_x,
    FE2_derivative_degree_y,
):

    Gpn = len(Gauss_coefficient_local)
    r = 0.0

    for i in range(Gpn):
        x, y = Gauss_point_local[i, 0], Gauss_point_local[i, 1]

        coeff_value_1 = coefficient_function_name(
            x, y, uh_local_1, vertices, FE1_basis_type, FE1_derivative_degree_x, FE1_derivative_degree_y
        )
        coeff_value_2 = coefficient_function_name(
            x, y, uh_local_2, vertices, FE2_basis_type, FE2_derivative_degree_x, FE2_derivative_degree_y
        )
        test_basis_value = triangular_local_basis(
            x, y, vertices, test_basis_type, test_basis_index, test_derivative_degree_x, test_derivative_degree_y
        )

        r += Gauss_coefficient_local[i] * coeff_value_1 * coeff_value_2 * test_basis_value

    return r
