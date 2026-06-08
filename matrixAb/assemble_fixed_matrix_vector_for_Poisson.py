import numpy as np
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
sys.path.insert(0, '../matrixAb')
from matrixAb.generate_M_T_triangle import generate_M_T_triangle
from matrixAb.assemble_matrix_from_volume_integral_triangle import assemble_matrix_from_volume_integral_triangle
from matrixAb.assemble_vector_from_volume_integral_triangle import assemble_vector_from_volume_integral_triangle
from matrixAb.functions_data import Functions
from matrixAb.assemble_matrix_from_volume_integral_triangle_global import assemble_matrix_from_volume_integral_triangle_global
from matrixAb.assemble_vector_from_volume_integral_triangle_global import assemble_vector_from_volume_integral_triangle_global

def assemble_fixed_matrix_vector_for_Poisson(M_partition, T_partition, M_basis_trial, T_basis_trial, T_basis_test, basis_type):

    coef_data = Functions()

    A1 = assemble_matrix_from_volume_integral_triangle_global(getattr(coef_data, 'function_k11'),
                                                       M_partition, T_partition, M_basis_trial, T_basis_trial, T_basis_test,
                                                       basis_type, 1, 0, basis_type, 1, 0)
    A2 = assemble_matrix_from_volume_integral_triangle_global(getattr(coef_data, 'function_k22'),
                                                       M_partition, T_partition, M_basis_trial, T_basis_trial, T_basis_test,
                                                       basis_type, 0, 1, basis_type, 0, 1)
    A3 = assemble_matrix_from_volume_integral_triangle_global(getattr(coef_data, 'function_k12'),
                                                       M_partition, T_partition, M_basis_trial, T_basis_trial, T_basis_test,
                                                       basis_type, 0, 1, basis_type, 1, 0)
    A4 = assemble_matrix_from_volume_integral_triangle_global(getattr(coef_data, 'function_k21'),
                                                       M_partition, T_partition, M_basis_trial, T_basis_trial, T_basis_test,
                                                       basis_type, 1, 0, basis_type, 0, 1)

    A = A1 + A2 + A3 + A4
    b = assemble_vector_from_volume_integral_triangle_global(getattr(coef_data, 'function_f_Poisson'),
                                                      M_partition, T_partition, M_basis_trial, T_basis_test, basis_type, 0, 0)

    return A, b