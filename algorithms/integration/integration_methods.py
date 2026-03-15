import numpy as np

def trapezoid(ys: np.ndarray, dx):
    """
    calculates area for sample points ys. Need to be equally spaced: dx
    """
    return dx * 0.5*  (np.sum(ys[:-1]) + np.sum(ys[1:])) 
    

def simpson(ys: np.ndarray, dx):
    S1 = trapezoid(ys, dx)

    dx = 2 * dx
    ys = ys[0:-1:2]
    S0 = trapezoid(ys, dx)

    return (4*S1 - S0)/3


def romberg(f: callable, a: np.float64, b: np.float64, order: np.int32):
    """
    computes the romberg integral of some function between endpoints a and b,
    using 2**order + 1 samples.
    """
    r = np.ndarray((order))

    h =b-a
    xs = np.array([a, b])
    r[0] = trapezoid(f(xs), xs[1] - xs[0])
    
    N_p = 1 #start by computing one additional point in between a, b
    for i in range(1, order): #ranges from 1 to m-1
        Delta = h
        h = h/2 #Delta must be 2*h so that you dont recompute points from previous iterations!
        xs = np.linspace(a+h, b-h, N_p)
        r[i] = 0.5* (r[i-1] + Delta*np.sum(f(xs)))
        N_p *= 2

    N_p = 1
    for i in range(1, order):
        N_p *= 4
        for j in range(order - i):
            r[j] = (N_p * r[j+1] - r[j]) / (N_p - 1)

    return r[0], np.abs(r[0] - r[1])


