import numpy as np
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
sys.path.insert(0, '../matrixAb')
from matrixAb.triangular_local_basis import triangular_local_basis

def Gauss_quadrature_for_volume_integral_trial_test_triangle(coefficient_function_name, Gauss_coefficient_local,
                                                             Gauss_point_local_x,Gauss_point_local_y,
                                                             vertices, trial_basis_type, trial_basis_index,
                                                             trial_derivative_degree_x,
                                                             trial_derivative_degree_y, test_basis_type,
                                                             test_basis_index,
                                                             test_derivative_degree_x, test_derivative_degree_y):
    """
    Use Gauss quadrature to compute a volume integral on a local triangular element for a matrix.

    :param coefficient_function_name: The coefficient function of the integrand.
    :param Gauss_coefficient_local: Gauss coefficients for the local triangular element.
    :param Gauss_point_local: Gauss points for the local triangular element.
    :param vertices: The coordinates of the vertices of the local triangular element.
    :param trial_basis_type: Type of the trial FE basis function (1 for linear, 2 for quadratic).
    :param trial_basis_index: Index of the trial FE basis function.
    :param trial_derivative_degree_x: Degree of the derivative of the trial FE basis function with respect to x.
    :param trial_derivative_degree_y: Degree of the derivative of the trial FE basis function with respect to y.
    :param test_basis_type: Type of the test FE basis function (1 for linear, 2 for quadratic).
    :param test_basis_index: Index of the test FE basis function.
    :param test_derivative_degree_x: Degree of the derivative of the test FE basis function with respect to x.
    :param test_derivative_degree_y: Degree of the derivative of the test FE basis function with respect to y.

    :return: The result of the Gauss quadrature volume integral.
    """
    # Gpn: The number of Gauss points
    Gpn = len(Gauss_coefficient_local)

    result = 0

    # Loop through all Gauss points
    for i in range(Gpn):
        # Evaluate the coefficient function at the Gauss point
        # coefficient_value = coefficient_function_name(Gauss_point_local[i, 0], Gauss_point_local[i, 1])#按积分式计算的系数函数


        # Get the trial and test basis function values at the Gauss point
        trial_basis_value = np.array(triangular_local_basis(Gauss_point_local_x[i, :], Gauss_point_local_y[i, :], vertices,
                                                   trial_basis_type, trial_basis_index, trial_derivative_degree_x,
                                                   trial_derivative_degree_y)).reshape(1,-1)
        test_basis_value = np.array(triangular_local_basis(Gauss_point_local_x[i, :], Gauss_point_local_y[i, :], vertices,
                                                  test_basis_type, test_basis_index, test_derivative_degree_x,
                                                  test_derivative_degree_y)).reshape(1,-1)
        G_w_i = Gauss_coefficient_local[i]

        # Add the contribution of the current Gauss point to the result
        result += np.unique(Gauss_coefficient_local[i]) * coefficient_function_name * trial_basis_value * test_basis_value

    return result
