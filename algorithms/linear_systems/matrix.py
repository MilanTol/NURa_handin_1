import numpy as np
import copy


class Matrix:

    def __init__(self, data):
        self.data = np.array(data, dtype=float)
        self.LU = None

    @property
    def shape(self):
        return self.data.shape

    def copy(self):
        return Matrix(self.data.copy())

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value
       
    def __matmul__(self, other):
        if not isinstance(other, Matrix):
            raise Exception("Error, other object not matrix")
        
        if self.shape[1] != other.shape[0]:
            raise Exception("matrix shapes are incompatible for matrix multiplication")
        
        if len(other.shape) > 2:
            raise Exception("matrix multiplication not defined for tensors (yet)")
        
        A = self
        B = other
        
        if len(other.shape) == 2: #check for matrix
            # say A has shape (m , n) and B has (n, p)
            # we want (A@B)_ij = sum_k A_ik * B_kj, with shape of (A@B) is (m, p)
            # by creating a matrix such that C_ikj = A_ik * B_kj, the multiplication can be done in vectorized way
            # by adding dimensions: A --> A[i, k, None], B --> B[None, k, j], multiplication is done correctly
            C = A[:, :, None] * B[None, :, :]
            # Now shape of C is (m, n, p) and C_ikj = A_ik * B_kj.
            # So to get matrix product we sum over the [1] axis.
            return np.sum(C, axis=1)   
        
        elif len(other.shape) == 1: #check for a vector
            C = A[:, :] * B[None, :] #same story but then for vectors
            return np.sum(C, axis=1)

        raise Exception("Matrix shape is not properly defined")
        

    def swap_rows(self, row1: int, row2: int):
        """
        swaps all elements of row1 with row2.
        """
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

        LU = self.copy() #copy matrix to not modify the matrix itself
        if LU.shape[1] != LU.shape[0]: #check whether matrix is square
            raise Exception("Matrix not square")
        
        N = LU.shape[0]
    
        for j in range(N): #loop over columns j
            for i in range(N):
                if i <= j:
                    LU[i, j] -= np.sum(LU[i, :i] * LU[:i, j])
                else:
                    LU[i, j] = 1/LU[j, j] * (LU[i, j] - np.sum(LU[i, :j] * LU[:j, j]))

        self.LU = LU #store LU matrix for future computations
        return LU
    
    
    def solve(self, b):
        """
        Solves for x given the equation Ax = b, where A is current matrix object.
        b maybe be passed as a vector or as a matrix.

        In case b is a matrix. The code returns a matrix where column i is the solution 
        to the equation Ax = b_i where b_i is the column vector given by the ith column of the matrix b.
        """

        if self.shape[1] != b.shape[0]:
            raise Exception("shape of b (number of rows) does not match matrix (number of columns)")
        
        N = b.shape[0]
        if self.LU is None: #check whether LU matrix has been computed before
            self.LU_decomposition()
        
        b = b.copy()
        #forward substitution
        for i in range(N):
            b[i] -= np.sum(self.LU[i, :i] * b[:i])

        #backward substitution
        for i in range(N-1, -1, -1): #start at N-1, go up to and including 0, with steps -1
            b[i] = 1/self.LU[i,i] * ( b[i] - np.sum(self.LU[i, i+1:]*b[i+1:]) )

        return b




        

