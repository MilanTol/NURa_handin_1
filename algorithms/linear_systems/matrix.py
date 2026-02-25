
import numpy as np

class Matrix(np.ndarray):

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj


    def __array_finalize__(self, obj):
        if obj is None:
            return


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


    def scale_row(self, row: int, scalar: np.float64):
        self[row] *= scalar


    def scale_col(self, col: int, scalar: np.float64):
        self[col] *= scalar
           
    

