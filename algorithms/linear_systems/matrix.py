import numpy as np
import copy


class Matrix:

    def __init__(self, data):
        self.data = np.array(data, dtype=float)

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
        
        if len(other.shape) == 2: 
            # say A = (m , n) and B = (n, p)
            # we want (A@B)_ij = sum_k A_ik * B_kj, with (A@B) = (m, p)
            # the multiplication can be done in vectorized way
            # by creating a matrix such that C_ikj = A_ik * B_kj
            # by adding dimensions: A --> A[i, k, None], B --> B[None, k, j] 
            # we can do this using * operator.
            C = A[:, :, None] * B[None, :, :]
            # Now C = (m, n, p) and C_ikj = A_ik * B_kj.
            # So to get the sum we just sum over the [1] axis.
            return np.sum(C, axis=1)   
        
        elif len(other.shape) == 1: #then it must be a vector
            C = A[:, :] * B[None, :] #same story but then for vectors
            return np.sum(C, axis=1)

        raise
        

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

        LU = np.zeros_like(self.data)
        if LU.shape[1] != LU.shape[0]:
            raise Exception("Matrix not square")
        
        N = LU.shape[0]

        for j in range(N): 
        #loop over columns j (must be done in a for loop since it depends on previously calculated j vals)

            # note that in crouts algorithm, the sum is similar to matrix multiplication
            # except that the matrices alpha has columns i till N removed, and beta has rows i till N removed.
            # we instead create a zeros-filled LU matrix such that for i <= j we can simply use full matrix multiplication
            # while for compute i > j (alpha values), we can use j, which we loop over. 

            #Since we use one matrix only, we must make a distinction for i <= j and i > j, since alpha_ii != beta_ii
            LU[:j+1, j] = self[:j+1, j] - LU[:j+1, :] @ LU[:, j] # i <= j
            LU[j+1:, j] = 1/LU[j, j] * (self[j+1:, j] - LU[j+1:, :j] @ LU[:j, j]) # i > j

        return Matrix(LU)
    
    
    def solve(self, b):
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




        

