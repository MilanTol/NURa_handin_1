import sys
import copy 
import time
import timeit

sys.path.append("/home/milan/Desktop/NURa")

import numpy as np
import matplotlib.pyplot as plt

from algorithms.optimization.optimizer import Optimizer

def func(x):
    return x**4 + 10*x**3 + 10*(x-2)**2

x_vals = np.linspace(-8, 4, 400)

opt_obj = Optimizer(func, -8, 4)

bracket1 = opt_obj.bracket(-8, -4)
print(bracket1)
minimum1 = opt_obj.tighten(bracket1, 1e-8)
print(minimum1)

bracket2 = opt_obj.bracket(0, 2)
print(bracket2)
minimum2 = opt_obj.tighten(bracket2, 1e-8)
print(minimum2)


plt.plot(x_vals, func(x_vals))
plt.show()