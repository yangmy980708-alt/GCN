import numpy as np

def generate_Gauss_local_triangle(Gauss_point_number,vertices):

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

    # Extract the vertices of the local triangle
    x1 = vertices[0, :]
    y1 = vertices[1, :]
    x2 = vertices[2, :]
    y2 = vertices[3, :]
    x3 = vertices[4, :]
    y3 = vertices[5, :]

    # Calculate the Jacobian (area of the reference triangle)
    Jacobi = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    # Gauss_coefficient_reference_triangle=np.array(Gauss_coefficient_reference_triangle).reshape(-1, 1)
    # print("Gauss_coefficient_reference_triangle",Gauss_coefficient_reference_triangle)
    # Jacobi = np.array(Jacobi).reshape(1,-1)
    # print("Jacobi=",Jacobi)

    Gauss_coefficient_local_triangle = np.array(Gauss_coefficient_reference_triangle).reshape(-1, 1) * np.array(
        Jacobi).reshape(1, -1)

    # 计算局部高斯点坐标
    # Gauss_point_local_triangle = np.zeros((Gauss_point_reference_triangle.shape[0], vertices.shape[1]))
    Gauss_point_local_triangle_x = np.array(x1).reshape(1, -1) + \
                                   np.array((x2 - x1)).reshape(1, -1) * np.array(
        Gauss_point_reference_triangle[:, 0]).reshape(-1, 1) + \
                                   np.array((x3 - x1)).reshape(1, -1) * np.array(
        Gauss_point_reference_triangle[:, 1]).reshape(-1, 1)
    Gauss_point_local_triangle_y = np.array(y1).reshape(1, -1) + \
                                   np.array((y2 - y1)).reshape(1, -1) * np.array(
        Gauss_point_reference_triangle[:, 0]).reshape(-1, 1) + \
                                   np.array((y3 - y1)).reshape(1, -1) * np.array(
        Gauss_point_reference_triangle[:, 1]).reshape(-1, 1)


    return Gauss_point_local_triangle_x,Gauss_point_local_triangle_y, Gauss_coefficient_local_triangle
