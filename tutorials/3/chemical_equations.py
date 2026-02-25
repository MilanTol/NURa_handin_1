import sys
import copy 

sys.path.append("/home/milan/Desktop/NURa")

import numpy as np

from algorithms.linear_systems.matrix import Matrix
from algorithms.linear_systems.Gauss import Gauss

mat = np.array([
    [3, 8, 1, -12, -4],
    [1, 0, 0, -1, -0],
    [4, 4, 3, -40, -3],
    [0, 2, 1, -3, -2],
    [0, 1, 0, -12, -0]
])
mat = Matrix(mat)

b= np.array([2, 0, 1, 0, 0])
b = Matrix(b)

gauss = Gauss(mat, copy.deepcopy(b))
gauss.cast()

x = gauss.solve()

print(x)
print(b)

print(mat@x)


