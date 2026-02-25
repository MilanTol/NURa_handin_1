import numpy as np
import copy


class Matrix(list):

    def __init__(self, list: list):
        self.rows = len(list)
        self.columns = len(list[0])


    def shape(self):
        return (self.rows, self.columns)
        

    def copy(self):
        return copy.deepcopy(self)
    

    def swap_rows(self, row1: int, row2: int):
        self[[row1, row2]] = self[[row1, row2]]


    def inverse(self):
        """
        returns inverse of matrix, using gauss jordan elimination.
        """
        mat = self.copy()
        if mat.shape[1] != mat.shape[0]:
            raise Exception("Matrix not square, does not have inverse")
        
        N = self.shape[0] #NxN matrix
        inv = Matrix(np.eye(N))

        for i in range(N): #loop over columns,

            for row in range(N)[i: ]:
                if mat[row, i] != 0: #select first row with nonzero pivot entry
                    if mat[row, i] != 1:
                        inv[row, :] /= mat[row, i]
                        mat[row, :] /= mat[row, i]
                    inv.swap_rows(row, i)
                    mat.swap_rows(row, i) #swap rows such that pivot is in [i, i]
                    break

                elif row == N - 1:
                    raise Exception("Matrix is singular")
        
            for row in range(N): #subtract pivot row from all other rows with non-zero elements in column i
                if row == i:
                    continue
                if mat[row, i] == 0:
                    continue
                else:
                    inv[row, :] -= mat[row, i] * inv[i, :]
                    mat[row, :] -= mat[row, i] * mat[i, :] 

        return inv


    def LU_decomposition(self):
        """
        returns LU matrix from LU_decomposition of matrix (with alpha_ii = 1, so can be stored in 1 matrix!).
        """

        mat = self.copy()
        if mat.shape[1] != mat.shape[0]:
            raise Exception("Matrix not square")
        
        N = self.shape[0]

        for j in range(N): #loop over columns j
            for i in range(N):
                if i <= j:
                    mat[i, j] -= np.sum(mat[i, :i] * mat[:i, j])
                else:
                    mat[i, j] = 1/mat[j, j] * (mat[i, j] - np.sum(mat[i, :j] * mat[:j, j]))

        return mat
    
    
    def solve(self, b: np.ndarray):
        if self.shape[1] != b.shape[0]:
            raise Exception("shape of b does not match")

        N = b.shape[0]
        LU = self.LU_decomposition()
        b = b.copy()
        
        #forward substitution
        for i in range(N):
            b[i] -= np.sum(LU[i, :i] * b[:i])

        #backward substitution
        for i in range(N-1, -1, -1): #start at N-1, go up to and including 0, with steps -1
            b[i] = 1/LU[i,i] * ( b[i] - np.sum(LU[i, i+1:]*b[i+1:]) )

        return b




        

