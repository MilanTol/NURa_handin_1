import sys
import copy 

sys.path.append("/home/milan/Desktop/NURa")

import numpy as np

from algorithms.linear_systems.matrix import Matrix
from algorithms.linear_systems.Gauss import solve_Gauss

mat = np.array([
    [3, 8, 1, -12, -4],
    [1, 0, 0, -1, -0],
    [4, 4, 3, -40, -3],
    [0, 2, 1, -3, -2],
    [0, 1, 0, -12, -0]
])

mat = np.array([
    [0, 1],
    [5, 0],
])

mat = Matrix(mat)


b= np.array([2, 1])
b = Matrix(b)

x = solve_Gauss(mat, b)

print(x)
print(b)

print(mat@x)


