import numpy as np
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
sys.path.insert(0, '../matrixAb')
from matrixAb.triangular_reference_basis import triangular_reference_basis

def triangular_local_basis_element(x, y, vertices, basis_type, basis_index, derivative_degree_x, derivative_degree_y):
   # Affine mapping matrix
    J_11 = vertices[0,1] - vertices[0,0]
    J_12 = vertices[0,2] - vertices[0,0]
    J_21 = vertices[1,1] - vertices[1,0]
    J_22 = vertices[1,2] - vertices[1,0]
    J_det = J_11 * J_22 - J_12 * J_21
    # print("J_22=", J_22)

    # Affine mapping to reference element (x_hat, y_hat)
    x_hat = (J_22 * (x - vertices[0, 0]) - J_12 * (y - vertices[1, 0])) / J_det
    y_hat = (-J_21 * (x - vertices[0, 0]) + J_11 * (y - vertices[1, 0])) / J_det

    # Based on the derivative degrees, select the appropriate basis function and its derivative
    if derivative_degree_x == 0 and derivative_degree_y == 0:
        r = triangular_reference_basis(x_hat, y_hat, basis_type, basis_index, 0, 0)
    elif derivative_degree_x == 1 and derivative_degree_y == 0:
        r = (triangular_reference_basis(x_hat, y_hat, basis_type, basis_index, 1, 0) * J_22 +
             triangular_reference_basis(x_hat, y_hat, basis_type, basis_index, 0, 1) * (-J_21)) / J_det
    elif derivative_degree_x == 0 and derivative_degree_y == 1:
        r = (triangular_reference_basis(x_hat, y_hat, basis_type, basis_index, 1, 0) * (-J_12) +
             triangular_reference_basis(x_hat, y_hat, basis_type, basis_index, 0, 1) * J_11) / J_det
    elif derivative_degree_x == 2 and derivative_degree_y == 0:
        r = (triangular_reference_basis(x_hat, y_hat, basis_type, basis_index, 2, 0) * J_22 ** 2 +
             triangular_reference_basis(x_hat, y_hat, basis_type, basis_index, 0, 2) * J_21 ** 2 +
             triangular_reference_basis(x_hat, y_hat, basis_type, basis_index, 1, 1) * (-2 * J_21 * J_22)) / J_det ** 2
    elif derivative_degree_x == 0 and derivative_degree_y == 2:
        r = (triangular_reference_basis(x_hat, y_hat, basis_type, basis_index, 2, 0) * J_12 ** 2 +
             triangular_reference_basis(x_hat, y_hat, basis_type, basis_index, 0, 2) * J_11 ** 2 +
             triangular_reference_basis(x_hat, y_hat, basis_type, basis_index, 1, 1) * (-2 * J_11 * J_12)) / J_det ** 2
    elif derivative_degree_x == 1 and derivative_degree_y == 1:
        r = (triangular_reference_basis(x_hat, y_hat, basis_type, basis_index, 2, 0) * (-J_22 * J_12) +
             triangular_reference_basis(x_hat, y_hat, basis_type, basis_index, 0, 2) * (-J_21 * J_11) +
             triangular_reference_basis(x_hat, y_hat, basis_type, basis_index, 1, 1) * (
                         J_21 * J_12 + J_11 * J_22)) / J_det ** 2

    return r
