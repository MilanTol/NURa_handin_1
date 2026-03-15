import numpy as np
from ..linear_systems.matrix import Matrix

def construct_vandermonde_matrix(x: np.ndarray) -> Matrix:
    """
    Construct the Vandermonde matrix V with V[i,j] = x[i]^j.

    Parameters
    ----------
    x : np.ndarray, x-values.

    Returns
    -------
    V : Matrix, Vandermonde matrix.
    """
    shape = (len(x), len(x))
    V = Matrix(np.ndarray(shape))  # initialize vandermonde matrix
    for j in range(len(x)):  # sum over columns
        V[:, j] = x**j

    return V


def vandermonde_solve_coefficients(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Solve for polynomial coefficients c from data (x,y) using the Vandermonde matrix.

    Parameters
    ----------
    x : np.ndarray
        x-values.
    y : np.ndarray
        y-values.

    Returns
    -------
    c : np.ndarray
        Polynomial coefficients.
    """

    V = construct_vandermonde_matrix(x)
    return V.solve(y)  


def evaluate_polynomial(c: np.ndarray, x_eval: np.ndarray):
    """
    Evaluate y(x) = sum_j c[j] * x^j.

    Parameters
    ----------
    c : np.ndarray
        Polynomial coefficients.
    x_eval : np.ndarray
        Evaluation points.

    Returns
    -------
    y_eval : Matrix
        vector (as matrix object) containing Polynomial values.
    """
    # note that with a matrix M similar to the vandermonde matrix we can write
    # y(x) = sum_j c[j] * x^j  as   y[i] = sum_j M_ij c_j
    # note that this is simply matrix vector multiplication

    # for this we require M[i, j] = x_i ** j
    # so M has shape (len(x_eval), len(c))

    shape = (len(x_eval), len(c))
    M = Matrix(np.ndarray(shape))  # initialize vandermonde matrix
    for j in range(len(c)):  # sum over columns
        M[:, j] = x_eval**j

    y = M @ Matrix(c)

    return y  