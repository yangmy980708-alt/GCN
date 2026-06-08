import numpy as np


def generate_Gauss_reference_triangle(Gauss_point_number):

    if Gauss_point_number == 4:
        # Gauss points and coefficients for 4-point Gauss quadrature
        Gauss_coefficient_reference_triangle = np.array([
            (1 - 1 / np.sqrt(3)) / 8,
            (1 - 1 / np.sqrt(3)) / 8,
            (1 + 1 / np.sqrt(3)) / 8,
            (1 + 1 / np.sqrt(3)) / 8
        ])

        Gauss_point_reference_triangle = np.array([
            [(1 / np.sqrt(3) + 1) / 2, (1 - 1 / np.sqrt(3)) * (1 + 1 / np.sqrt(3)) / 4],
            [(1 / np.sqrt(3) + 1) / 2, (1 - 1 / np.sqrt(3)) * (1 - 1 / np.sqrt(3)) / 4],
            [(-1 / np.sqrt(3) + 1) / 2, (1 + 1 / np.sqrt(3)) * (1 + 1 / np.sqrt(3)) / 4],
            [(-1 / np.sqrt(3) + 1) / 2, (1 + 1 / np.sqrt(3)) * (1 - 1 / np.sqrt(3)) / 4]
        ])

    elif Gauss_point_number == 9:
        # Gauss points and coefficients for 9-point Gauss quadrature
        Gauss_coefficient_reference_triangle = np.array([
            64 / 81 * (1 - 0) / 8, 100 / 324 * (1 - np.sqrt(3 / 5)) / 8, 100 / 324 * (1 - np.sqrt(3 / 5)) / 8,
            100 / 324 * (1 + np.sqrt(3 / 5)) / 8, 100 / 324 * (1 + np.sqrt(3 / 5)) / 8,
            40 / 81 * (1 - 0) / 8, 40 / 81 * (1 - 0) / 8, 40 / 81 * (1 - np.sqrt(3 / 5)) / 8,
            40 / 81 * (1 + np.sqrt(3 / 5)) / 8
        ])

        Gauss_point_reference_triangle = np.array([
            [(1 + 0) / 2, (1 - 0) * (1 + 0) / 4],
            [(1 + np.sqrt(3 / 5)) / 2, (1 - np.sqrt(3 / 5)) * (1 + np.sqrt(3 / 5)) / 4],
            [(1 + np.sqrt(3 / 5)) / 2, (1 - np.sqrt(3 / 5)) * (1 - np.sqrt(3 / 5)) / 4],
            [(1 - np.sqrt(3 / 5)) / 2, (1 + np.sqrt(3 / 5)) * (1 + np.sqrt(3 / 5)) / 4],
            [(1 - np.sqrt(3 / 5)) / 2, (1 + np.sqrt(3 / 5)) * (1 - np.sqrt(3 / 5)) / 4],
            [(1 + 0) / 2, (1 - 0) * (1 + np.sqrt(3 / 5)) / 4],
            [(1 + 0) / 2, (1 - 0) * (1 - np.sqrt(3 / 5)) / 4],
            [(1 + np.sqrt(3 / 5)) / 2, (1 - np.sqrt(3 / 5)) * (1 + 0) / 4],
            [(1 - np.sqrt(3 / 5)) / 2, (1 + np.sqrt(3 / 5)) * (1 + 0) / 4]
        ])

    elif Gauss_point_number == 3:
        # Gauss points and coefficients for 3-point Gauss quadrature
        Gauss_coefficient_reference_triangle = np.array([1 / 6, 1 / 6, 1 / 6])

        Gauss_point_reference_triangle = np.array([
            [1 / 2, 0],
            [1 / 2, 1 / 2],
            [0, 1 / 2]
        ])

    return Gauss_coefficient_reference_triangle, Gauss_point_reference_triangle
