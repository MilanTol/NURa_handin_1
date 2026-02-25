import sys
import copy 

sys.path.append("/home/milan/Desktop/NURa")

import numpy as np

from algorithms.linear_systems.matrix import Matrix

mat = np.array([
    [3, 8, 1, -12, -4],
    [1, 0, 0, -1, -0],
    [4, 4, 3, -40, -3],
    [0, 2, 1, -3, -2],
    [0, 1, 0, -12, -0]
])

# mat = np.array([
#     [3, 8, 1, -12, -4],
#     [0, 1, 0, -1, -0],
#     [0, 0, 3, -40, -3],
#     [0, 0, 0, -3, -2],
#     [0, 0, 0, -0, 1]
# ])

mat = Matrix(mat)

b= np.array([2, 0, 1, 0, 0])
b = Matrix(b)

x_inv = mat.inverse()@b
print(x_inv)
print(mat@x_inv)

print("")

x_LU = mat.solve(b)
print(x_LU)
print(mat@x_LU)

print("")

print(mat.LU_decomposition())

