import numpy as np
import timeit

G = np.double(6.67e-11)
c = np.double(3e8)

def Rs_division(M):
    return 2*G*M/c**2

c_factor = 1/c**2
def Rs_multiplication(M):
    return 2*G*M * c_factor

mass_samples = np.random.normal(loc=1e6, scale=1e5, size=10000)

def Rs_divided():
    Rs_division(mass_samples)

def Rs_multiplied():
    Rs_multiplication(mass_samples)

print("division time:", timeit.timeit(Rs_divided))
print("multiplication time:", timeit.timeit(Rs_multiplied))