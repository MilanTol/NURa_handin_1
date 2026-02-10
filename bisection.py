import numpy as np

def bisection(x, x_vals, M=4):
    """
    Returns index of the -M/2 x_val closest to x
    
    :param x:
    :param x_vals: 
    :param M: 
    """
    N = len(x_vals)
    
    for i in range(N - 1):
        if not (x_vals[i] < x_vals[i+1]):
            print("ERROR: x_vals are not monotonic")
            return 

    x_vals_cut = x_vals
    i_low = 0
    
    while x_vals_cut[1] != x_vals_cut[-1]:
        if (x < x_vals_cut[int(0.5*N)]):
            x_vals_cut = x_vals_cut[:int(0.5*N)]
        else:
            x_vals_cut = x_vals_cut[int(0.5*N):]
            i_low += int(0.5*N)

        N = len(x_vals_cut)

    return i_low - int(M/2)


print(bisection(59.5, np.linspace(0, 101, 101), 8))
