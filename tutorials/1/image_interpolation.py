import numpy as np

import os
import sys
sys.path.append("/home/milan/Desktop/NURa")

from matplotlib.image import imread
image=imread('files/M42_128.jpg').astype(np.double)
first_row = image[:,0]
x_samples = np.arange(len(first_row))

from algorithms.interpolation.interpolators import linear_interpolator
interpolated_first_row = [linear_interpolator(x, x_samples, first_row) for x in x_samples]

import matplotlib.pyplot as plt
plt.scatter(x_samples, first_row, label='data', color='black', s=4)
plt.plot(x_samples, interpolated_first_row, label='interpolated')
plt.legend()
plt.show()



