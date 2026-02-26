import numpy as np

def central_difference(f: callable, x, h):
    return 1/(2*h) * (f(x+h) - f(x-h))

def Ridders(f: callable, x, h_init, d, m:int=5):
    """
    calculate f' at x using Ridders method.
    combines central differences for h values of h_init/d**m.
    (so using m approximations)
    """
    D  = central_difference(f, x, h_init / d**np.linspace(1, m, m))

    for j in range(m - 1):
        D = (d**(2*j)*D[1:] - D[:-1]) / (d**(2*j) - 1)  

    return D[0]



