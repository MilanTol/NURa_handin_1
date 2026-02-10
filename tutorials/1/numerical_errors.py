import numpy as np

def factorial(x: np.int64):
    res = 1
    for i in range(x):
        res *= (i + 1)
    return res

def sinc(x: np.double, order: np.int64):
    res = 0
    for n in range(order):
        numerator = (-1)**n * x**(2*n)
        denominator = factorial(2*n + 1)
        res += numerator/denominator
    return res

for order in np.linspace(1, 10, 10).astype(np.int64):
    # print("np version:", np.sinc(7/np.pi))
    # print("custom version:", sinc(7, order))
    print("error: ", np.sinc(7/np.pi) - sinc(7, order))