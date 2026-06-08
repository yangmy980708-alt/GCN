import numpy as np
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
sys.path.insert(0, '../matrixAb')
from matrixAb.generate_M_T_triangle import generate_M_T_triangle



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


M_partition, T_partition = generate_M_T_triangle(0,2,0,4, (2,2), 2)


# Gaussian quadrature points and weights
vertices = np.vstack(
[M_partition[:, T_partition[0, :]], M_partition[:, T_partition[1, :]], M_partition[:, T_partition[2, :]]])
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

Gauss_coefficient_local_triangle = np.array(Gauss_coefficient_reference_triangle).reshape(-1, 1) * np.array(Jacobi).reshape(1,-1)

    # 计算局部高斯点坐标
    # Gauss_point_local_triangle = np.zeros((Gauss_point_reference_triangle.shape[0], vertices.shape[1]))
Gauss_point_local_triangle_x = np.array(x1).reshape(1,-1) + \
                                       np.array((x2 - x1)).reshape(1, -1) * np.array(Gauss_point_reference_triangle[:, 0]).reshape(-1, 1) +\
                                        np.array((x3 - x1)).reshape(1, -1) * np.array(Gauss_point_reference_triangle[:, 0]).reshape(-1, 1)
Gauss_point_local_triangle_y = np.array(y1).reshape(1,-1) + \
                                       np.array((y2 - y1)).reshape(1, -1) * np.array(Gauss_point_reference_triangle[:, 0]).reshape(-1, 1) +\
                                       np.array((y3 - y1)).reshape(1, -1) * np.array(Gauss_point_reference_triangle[:, 0]).reshape(-1, 1)

print("Gauss_point_local_triangle_x=",Gauss_point_local_triangle_x)
print("Gauss_point_local_triangle_y",Gauss_point_local_triangle_y)
print("Gauss_coefficient_local_triangle",Gauss_coefficient_local_triangle)
for k in range(9):
    Gauss_coefficient_local_triangle_K = Gauss_coefficient_local_triangle[k]
    Gauss_coefficient_local_triangle_KK = np.unique(Gauss_coefficient_local_triangle_K)

print(Gauss_coefficient_local_triangle_K)
