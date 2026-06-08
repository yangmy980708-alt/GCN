import numpy as np
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
sys.path.insert(0, '../matrixAb')
from matrixAb.triangular_local_basis import triangular_local_basis

def Gauss_quadrature_for_volume_integral_test_triangle(coefficient_function_name, Gauss_coefficient_local,
                                                       Gauss_point_local_x,Gauss_point_local_y, vertices,
                                                       test_basis_type, test_basis_index,
                                                       test_derivative_degree_x, test_derivative_degree_y):
    """
    Use Gauss quadrature to compute a volume integral on a local triangular element T for a vector.
    The integrand of the volume integral must be in the following format:
    a coefficient function * a test FE basis function (or its derivatives).

    Parameters:
    coefficient_function_name: the coefficient function of the integrand.
    Gauss_coefficient_local: Gauss coefficients on the triangular element T.
    Gauss_point_local: Gauss points on the triangular element T.
    vertices: coordinates of all vertices of the triangular element T.
    test_basis_type: the type of the test FE basis function.
    test_basis_index: the index of the test FE basis function to specify which test FE basis function to use.
    test_derivative_degree_x: the derivative degree of the test FE basis function with respect to x.
    test_derivative_degree_y: the derivative degree of the test FE basis function with respect to y.

    Returns:
    r: the result of the volume integral.
    """
    Gpn = len(Gauss_coefficient_local)  # Number of Gauss points
    r = 0

    # Loop over each Gauss point and compute the integral
    for i in range(Gpn):
        # Evaluate the coefficient function at the Gauss point
        # coefficient_value = coefficient_function_name(Gauss_point_local[i, 0], Gauss_point_local[i, 1])

        # Evaluate the test basis function and its derivatives at the Gauss point
        basis_value = np.array(triangular_local_basis(Gauss_point_local_x[i,:], Gauss_point_local_y[i,:], vertices,
                                             test_basis_type, test_basis_index, test_derivative_degree_x,
                                             test_derivative_degree_y)).reshape(1,-1)

        # Compute the contribution to the integral at this Gauss point
        r += np.unique(Gauss_coefficient_local[i]) * coefficient_function_name * basis_value

    return r
