import numpy as np
import timeit

G = np.double(6.67e-11)
c = np.double(3e8)

def Rs_division(M):
    resulting_list = []
    for m in M:
        temp =  2*G*m/c**2
        resulting_list.append(temp)
    return resulting_list

c_inv2 = 1/c**2
def Rs_multiplication(M):
    resulting_list = []
    for m in M:
        temp =  2*G*m * c_inv2
        resulting_list.append(temp)
    return resulting_list

mass_samples = np.random.normal(loc=1e6, scale=1e5, size=100)

def Rs_divided():
    Rs_division(mass_samples)

def Rs_multiplied():
    Rs_multiplication(mass_samples)

print("division time:", timeit.timeit(Rs_divided))
print("multiplication time:", timeit.timeit(Rs_multiplied))