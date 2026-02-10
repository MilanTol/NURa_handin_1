from .bisection import bisection

def linear_interpolator(x, x_samples, y_samples):
    i = bisection(x, x_samples) #find index of lower closest x_sample, add 1 for upper 
    return (
        (y_samples[i + 1] - y_samples[i]) 
        / (x_samples[i + 1] - x_samples[i])
        * (x - x_samples[i])
        + y_samples[i]
        )

def Neville(x, x_samples, x_l, x_u, P_l, P_u):
    if x_u > x_l:
        return ((x_u - x)*P_l(x) + (x - x_l)*P_u(x)) / (x_u - x_l)
    else:
        print("Error: x_u < x_l")
        return 
    
def Neville_interpolator(x, x_samples, y_samples, M=2):
    i = bisection(x, x_samples) 
    
    for m in range(M):
        
        P_i = Neville(x, x_samples, y_samples, P_l, P_u)


