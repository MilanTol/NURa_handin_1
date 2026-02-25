from .matrix import Matrix

import numpy as np

def cast_Gauss(mat: Matrix, b: Matrix):
    """
    cast matrix into Gauss form, transform b vector (or matrix) along.
    Returns transformed matrix and b.
    """

    if mat.shape[0] != mat.shape[1]: #check matrix is square
        print("error: matrix not square")
        return
    else:
        N = mat.shape[0]
    

    #set pivots across diagonal:
    for column in range(N):
        pivot_inv = 1/mat[column, column]

        for row in range(N)[column + 1:]:

            if mat[column, column] == 0:
                if mat[row, column] != 0: #check entry is 0, if not --> set as pivot
                    if row != column:
                        mat.swap_rows(column, row) #swap row with index=column with pivot row
                        b.swap_rows(column, row)
                    break

                elif row == N:
                    raise Exception("Matrix is singular")
                
            else: 
                mat[row, :] -= mat[row, column]*pivot_inv * mat[column, :] #subtract entry/pivot times pivot row of current row
                
                if b.ndim == 1:
                    b[row] -= mat[row, column] *pivot_inv * b[column]
                else:
                    b[row, :] -= mat[row, column] * pivot_inv * b[column, :]

    return mat, b


def solve_Gauss(A: Matrix, b: Matrix):
    """
    solves the system Ax = b, using gauss elimination.
    returns x.
    """

    A, b = cast_Gauss(A.copy(), b.copy())

    for i in range(A.ndim):
        b[i] = 1/A[i, i] * (b[i] - np.sum(A[i, i+1:]*b[i+1:]))

    return b





    