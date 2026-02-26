import sys
import copy 
import time

sys.path.append("/home/milan/Desktop/NURa")

import numpy as np
import matplotlib.pyplot as plt

from algorithms.differentiation.differentiation import central_difference, Ridders


def f(x):
    return x**2 * np.sin(x)

def f_prime(x):
    return 2*x*np.sin(x) + x**2 * np.cos(x)


x_vals = np.linspace(0, 2*np.pi, 100)


for h in [1, 0.1, 0.01, 0.001]:
    y_vals = central_difference(f, x_vals, h)
    plt.plot(x_vals, y_vals, label=f'cd: h = {h}')

y_vals = Ridders(f, x_vals, 1, 10, 4)
plt.plot(x_vals, y_vals, label=f'ridders')

plt.plot(x_vals, f_prime(x_vals), label='analytic', linestyle='--')
plt.legend()

plt.show()



