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

gauss = Gauss(mat.copy(), b.copy())
gauss.cast()

x = gauss.solve()

print(x)
print(b)

print(mat@x)

[Matrix([  3.,   8.,   1., -12.,  -4.]), 
 Matrix([ 0.        , -2.66666667, -0.33333333,  3.        ,  1.33333333]), 
 Matrix([  0. ,   0. ,   2.5, -31.5,  -1. ]), 
 Matrix([ 0.00000000e+00,  0.00000000e+00, -1.11022302e-16,
         8.70000000e+00, -7.00000000e-01]), 
         Matrix([ 0.00000000e+00,  0.00000000e+00, -1.58876743e-16,
        -1.77635684e-15, -5.51724138e-01])]
