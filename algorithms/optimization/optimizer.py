import numpy as np
from ..interpolation.VDM import vandermonde_solve_coefficients, evaluate_polynomial
from ..linear_systems.matrix import Matrix


def parabola_minimum(a, b, c, f_a, f_b, f_c):
    """
    returns x at which quadratic fit to data has a minimum.
    """
    num = 0.5*(b-a)**2*(f_b - f_c) - (b-c)**2 * (f_b - f_a)
    den = (b-a)*(f_b - f_c) - (b-c)*(f_b - f_a)
    return b - num/den


class Optimizer:
    def __init__(self, func: callable, x_min:float, x_max:float, args:tuple=()):
        self.func=func
        self.x_min = x_min
        self.x_max = x_max
        self.args = args

    def bracket(self, a:float, b:float, maxit:int=100) -> tuple[float, float, float]:
        """
        returns a bracket (a,b,c) containing a minimum.
        """
        f_a = self.func(a, *self.args)
        f_b = self.func(b, *self.args)

        if f_a < f_b:
            a, b = b, a
            f_a, f_b = f_b, f_a

        phi = 1.618
        c = b + (b - a)*phi
        f_c = self.func(c, *self.args)

        for i in range(maxit):
            if f_c > f_b:
                return a, b, c
            
            d = parabola_minimum(a, b, c, f_a, f_b, f_c)
            f_d = self.func(d, *self.args)

            if b < d and d < c:
                if f_d < f_c:
                    return b, d, c
                if f_d > f_b: 
                    return a, b, d
                else:
                    d = c + (c-b)*phi
            else:
                if np.abs(d-b)>100*np.abs(c-b):
                    d = c + (c-b)*phi
            
            a, b, c = b, c, d
            f_a, f_b, f_c = f_b, f_c, f_d

        raise Exception(f'could not find bracket in {maxit} iterations')
        

    def tighten(self, bracket:tuple[float,float,float], target_accuracy, maxit:int=100) -> float:
        """
        returns the minimum inside bracket by tightening algorithm.
        """
        a, b, c = bracket
        f_b = self.func(b, *self.args)

        if np.abs(c-b) > np.abs(b-a): #identify larger half of the bracket
            x = c
        else:
            x = a

        for i in range(maxit):
            w = 0.38197
            d = b + (x-b)*w
            f_d = self.func(d, *self.args)

            if np.abs(c-a) < target_accuracy:
                if f_d < f_b:
                    return d
                else:
                    return b
            
            if f_d < f_b:
                if x==a:
                    c = b
                    b = d # Note that x remains a!
                else:
                    a = b
                    b = d # #Note that x remains c!
            else:
                if x==a:
                    a = d #
                    x = c #change x to c since that is now the largest gap
                else:
                    c = d
                    x = a #change x to a since that is now the largest gap




