import numpy as np

def bisection(x, x_vals, M=2):
    """
    Returns index of the -M/2 x_val closest to x
    
    :param x:
    :param x_vals: 
    :param M: 
    """
    N = len(x_vals)
    
    #Check for monotonicity of x_samples!
    for i in range(N - 1):
        if not (x_vals[i] < x_vals[i+1]):
            print("ERROR: x_vals are not monotonic")
            return 

    #check for x value being at the edge:
    if x < x_vals[M - 1]:
        return 0
    if x > x_vals[-M - 1]:
        return len(x_vals) - M
    
    i_low = 0
    iteration = 0
    while x_vals[i_low + 1] < x:
        iteration += 1
        stepsize = int(0.5**iteration *N) + 1
        if (x > x_vals[i_low + stepsize]):
            i_low += stepsize
        else:
            pass

    return i_low

