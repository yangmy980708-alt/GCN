import numpy as np
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
sys.path.insert(0, '../matrixAb')

from matrixAb.generate_M_T_triangle import generate_M_T_triangle
from matrixAb.generate_Gauss_local_triangle import generate_Gauss_local_triangle
from matrixAb.assemble_matrix_from_volume_integral_triangle_global import assemble_matrix_from_volume_integral_triangle_global
from matrixAb.assemble_matrix_from_volume_integral_triangle_up import assemble_matrix_from_volume_integral_triangle_up
from matrixAb.assemble_vector_from_volume_integral_triangle_global import assemble_vector_from_volume_integral_triangle_global
from matrixAb.assemble_vector_from_volume_integral_triangle import assemble_vector_from_volume_integral_triangle
from matrixAb.functions_data import Functions
def assemble_fixed_matrix_vector_for_Stokes_Taylor_Hood(M_partition, T_partition, M_basis_trial_u, T_basis_trial_u, T_basis_test_u, T_basis_p):

    coef_data = Functions()

    A1 = assemble_matrix_from_volume_integral_triangle_global(getattr(coef_data, 'function_nu'), M_partition, T_partition, M_basis_trial_u, T_basis_trial_u, T_basis_test_u, 2, 1, 0, 2, 1, 0)
    A2 = assemble_matrix_from_volume_integral_triangle_global(getattr(coef_data, 'function_nu'), M_partition, T_partition, M_basis_trial_u, T_basis_trial_u, T_basis_test_u, 2, 0, 1, 2, 0, 1)
    A3 = assemble_matrix_from_volume_integral_triangle_global(getattr(coef_data, 'function_nu'), M_partition, T_partition, M_basis_trial_u, T_basis_trial_u, T_basis_test_u, 2, 1, 0, 2, 0, 1)
    A4 = assemble_matrix_from_volume_integral_triangle_global(getattr(coef_data, 'function_nu'), M_partition, T_partition, M_basis_trial_u, T_basis_trial_u, T_basis_test_u, 2, 0, 1, 2, 1, 0)
    A5 = assemble_matrix_from_volume_integral_triangle_up(getattr(coef_data, 'function_negativeone'), M_partition, T_partition, M_basis_trial_u, T_basis_trial_u, T_basis_p, 1, 0, 0, 2, 1, 0)
    A6 = assemble_matrix_from_volume_integral_triangle_up(getattr(coef_data, 'function_negativeone'), M_partition, T_partition, M_basis_trial_u, T_basis_trial_u, T_basis_p, 1, 0, 0, 2, 0, 1)

    temp = np.zeros((int(M_partition.shape[1]), int(M_partition.shape[1])))
    A = np.block([[2 * A1 + A2, A3, A5], [A4, 2 * A2 + A1, A6], [A5.T, A6.T, temp]])

    # Assemble the load vector
    b1 = assemble_vector_from_volume_integral_triangle_global(getattr(coef_data, 'function_f1_Stokes'), M_partition, T_partition,
                                                       M_basis_trial_u, T_basis_trial_u, 2, 0, 0)
    b2 = assemble_vector_from_volume_integral_triangle_global(getattr(coef_data, 'function_f2_Stokes'),  M_partition, T_partition,
                                                       M_basis_trial_u, T_basis_trial_u, 2, 0, 0)

    temp_b = np.zeros((int(M_partition.shape[1]),1))
    b = np.concatenate([b1, b2, temp_b])

    return A, b
