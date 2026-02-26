import sys
import copy 
import time

sys.path.append("/home/milan/Desktop/NURa")

import numpy as np

from algorithms.linear_systems.matrix import Matrix


N = 20


mat = np.random.random(size=(N, N))
mat = np.array([
    [3, 8, 1, -12, -4],
    [1, 0, 0, -1, -0],
    [4, 4, 3, -40, -3],
    [0, 2, 1, -3, -2],
    [0, 1, 0, -12, -0]
])

mat = np.array([
    [3, 8, 1],
    [1, 0, 0],
    [4, 4, 3]
])

mat = Matrix(mat)


b = np.random.random(size=(N,))
b= np.array([2, 0, 1, 0, 0])
b= np.array([2, 0, 1])
b = Matrix(b)

x_inv = mat.inverse()@b
x_LU = mat.solve(b)

print(x_inv)
print(x_LU.data)

def gauss_method():
    return mat.inverse()@b

def LU_method():
    return mat.solve(b)


time1 = time.time()

for i in range(10):
    gauss_method()

time2 = time.time()

for i in range(10):
    LU_method()

time3 = time.time()

print("Gauss_method:", time2 - time1)
print("LU_method:", time3 - time2)
