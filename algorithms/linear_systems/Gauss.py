from .matrix import Matrix

import numpy as np

class Gauss:
    def __init__(self, mat: Matrix, b: Matrix):
        if mat.shape[0] != mat.shape[1]:
            print("error: matrix not square")
            return

        self.mat = mat
        self.b = b
        self.N = mat.shape[0]

    def cast(self):
        """
        casts the matrix into Gauss form.
        """
        N = self.N

        for column in range(N):
            self.set_pivot(column)
            self.empty_lower_rows(column)
     

    def set_pivot(self, column):
        for row in range(self.N)[column:]: #loop over rows with row >= column
            if self.mat[row, column] != 0:
                if row != column:
                    self.mat.swap_rows(column, row) #swap row with index=column with pivot row
                    self.b.swap_rows(column, row)
                break


    def empty_lower_rows(self, column):
        """
        subtracts pivot row from lower rows, such that there are 0s in column.
        """
        
        pivot_inv = 1/self.mat[column, column]

        for row in range(self.N)[column + 1:]:

            if self.mat[row, column] != 0:
                self.mat[row, :] -= self.mat[row, column] * pivot_inv * self.mat[column, :]

                if self.b.ndim == 1:
                    self.b[row] -= self.mat[row, column] * pivot_inv * self.b[column]
                else:
                    self.b[row, :] -= self.mat[row, column] * pivot_inv * self.b[column, :]


    def solve(self):
        for i in range(self.N):
            self.b[i] = 1/self.mat[i, i] * (self.b[i] - np.sum(self.mat[i, i:]*self.b[i:]))
        return self.b
    