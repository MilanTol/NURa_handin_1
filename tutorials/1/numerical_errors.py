import numpy as np

def sinc(x: np.double, order: np.int64):
    for n in range(order):
        numerator = 1