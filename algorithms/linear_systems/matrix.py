
import numpy as np

class Matrix:

    def __init__(self, input_arr, type=float):
        self.arr = np.array(input_arr, dtype=type)

    @property
    def shape(self):
        return self.arr.shape
    
    @property
    def ndim(self):
        return self.arr.ndim
    
    def copy(self):
        return self.arr.copy()

    def __repr__(self):
        return f"{self.arr}"
    
    def __matmul__(self, other):
        return self.arr @ other
    
    
    def swap_rows(self, row1: int, row2: int):
        """
        swap rows (row1) and (row2)
        """
        if self.ndim == 1:
            self[row1], self[row2] = self[row2], self[row1]
        else:
            self[[row1, row2], :] = self[[row2, row1], :]


    def swap_cols(self, col1: int, col2: int):
        """
        swap columns (column1) and (column2)
        """
        self[:, [col1, col2]] = self[:, [col2, col1]]


    

