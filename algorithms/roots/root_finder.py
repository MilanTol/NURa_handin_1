

class RootFinder:
    def __init__(self, func: callable, args:tuple = ()):
        """
        initialize rootfinder object.

        Parameters
        ----------

        func : callable
            Function to find root of
        """
        self.func = func
        self.args = args

       
    def bisection(self,
            bracket: tuple,
            atol: float = 1e-6,
            rtol: float = 1e-6,
            max_iters: int = 100,
        ) -> tuple[float, float, float]:
        """
        Find a root of a function using bisection

        Parameters
        ----------

        bracket : tuple
            Bracket for which to find first secant
        atol : float, optional
            Absolute tolerance.
            The default is 1e-6
        rtol : float, optional
            Relative tolerance.
            The default is 1e-6
        max_iters: int, optional
            Maximum number of iterations.
            The default is 100

        Returns
        -------
        root : float
            Approximate root
        aerr : float
            Absolute error
        rerr : float
            Relative error
        """
        a, b = bracket
        f_a = self.func(a, *self.args)
        f_b = self.func(b, *self.args)
        if f_a*f_b > 0: # check whether inputted bracket is in fact a bracket
            raise Exception("bracket does not contain root")
        
        Delta0 = b - a
        Delta = Delta0 
        
        c = 0.5*(a+b) #  set c as midpoint between a and b
        for i in range(max_iters):
            f_c = self.func(c, *self.args)
            if f_a*f_c < 0: #check whether root is within [a,c]
                b = c # overwrite b with c
                f_b = f_c # reassign function value
            else: # otherwise the root must lie within [c, b]
                a = c # overwrite a with c
                f_a = f_c
            c = 0.5*(a+b) # find new c
            Delta *= 0.5 #bisection always decreases bracket width by factor 0.5
            if Delta < atol:
                break
            if Delta < rtol * c:
                break

        if i == max_iters:
            raise Warning("requested tolerance not reached")
        
        return c, Delta, Delta/c


    def false_position(self,
            bracket: tuple,
            atol: float = 1e-6,
            rtol: float = 1e-6,
            max_iters: int = 100,
        ) -> tuple[float, float, float]:
        """
        Find a root of a function using false position algorithm

        Parameters
        ----------

        bracket : tuple
            Bracket for which to find first secant
        atol : float, optional
            Absolute tolerance.
            The default is 1e-6
        rtol : float, optional
            Relative tolerance.
            The default is 1e-6
        max_iters: int, optional
            Maximum number of iterations.
            The default is 100

        Returns
        -------
        root : float
            Approximate root
        aerr : float
            Absolute error
        rerr : float
            Relative error
        """
    
        a, b = bracket
        f_a = self.func(a, *self.args)
        f_b = self.func(b, *self.args)
        if f_a*f_b > 0: # check whether inputted bracket is in fact a bracket
            raise Exception("bracket does not contain root")
        
        Delta0 = b - a
        Delta = Delta0 
        
        c = b - (b - a)/(f_b - f_a) *f_b # finds root using slope 
        for i in range(max_iters):
            f_c = self.func(c, *self.args)
            if f_a*f_c < 0: #check whether root is within [a,c]
                b = c # overwrite b with c
                f_b = f_c # reassign function value
            else: # otherwise the root must lie within [c, b]
                a = c # overwrite a with c
                f_a = f_c
            c = b - (b - a)/(f_b - f_a) *f_b # find new c

            if c < a or c > b: # if c is outside of bracket, use bisection instead
                c = 0.5*(a + b)
            Delta = b - a 
            if Delta < atol:
                break
            if Delta < rtol * c:
                break

        if i == max_iters:
            raise Warning("requested tolerance not reached")
        
        return c, Delta, Delta/c



