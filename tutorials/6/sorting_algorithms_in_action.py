import sys
import copy 
import time
import timeit

sys.path.append("/home/milan/Desktop/NURa")

import numpy as np
import matplotlib.pyplot as plt


#question a)

from algorithms.sorting.sorter import Sorter

N = 40
rand_arr = np.random.randint(1, N, N)
print(rand_arr)

sorter_obj = Sorter(rand_arr)
sorted_arr = sorter_obj.selection_sort()

# print(sorted_arr)
# print(indx)

# print(rand_arr[indx])

# print()

#disadadvantage is that it is not a stable sorting algorithm. It is also slow: always N^2


# sortarr = Sorter(np.array([4, 2, 1, 1, 4]))

#question b)
sorted_arr, indx = sorter_obj.quicksort(make_indx=True)
print('sorted by quicksort:', sorted_arr)
print('index array', indx)

print('indexed random array', rand_arr[indx])
print(timeit.timeit(sorter_obj.selection_sort, number=10))

print(timeit.timeit(sorter_obj.quicksort, number=10))
