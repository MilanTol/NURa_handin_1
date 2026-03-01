import os
import sys
import timeit

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from matrix import Matrix
from bisection import bisection

mpl.rcParams["font.size"] = 20
mpl.rcParams["axes.labelsize"] = 20
mpl.rcParams["xtick.labelsize"] = 20
mpl.rcParams["ytick.labelsize"] = 20


def load_data():
    """
    Function to load the data from Vandermonde.txt.

    Returns
    ------------
    x (np.ndarray): Array of x data points.

    y (np.ndarray): Array of y data points.
    """
    data = np.genfromtxt(
        os.path.join(sys.path[0], "Vandermonde.txt"), comments="#", dtype=np.float64
    )
    x = data[:, 0]
    y = data[:, 1]
    return x, y


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
    return V.solve(y)  # Replace with your solution


def evaluate_polynomial(c: np.ndarray, x_eval: np.ndarray) -> Matrix:
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
    # TODO:
    # evaluate the polynomial at x_eval using the coefficients c from vandermonde_solve_coefficients

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

    return y  # Replace with your results
    

def neville(x_data: np.ndarray, y_data: np.ndarray, x_interp: float, M: int = None) -> float:
    """
    Function that applies Nevilles algorithm to calculate the function value at x_interp.

    Parameters
    ------------
    x_data (np.ndarray): Array of x data points.
    y_data (np.ndarray): Array of y data points.
    x_interp (float): The x value at which to interpolate.
    M (int): the order of interpolation, if M is None, uses M = x_data.shape[0]

    Returns
    ------------
    float: The interpolated y value at x_interp.
    """
    if M is None:
        M = x_data.shape[0]

    #compute the lowest index of the M points from x_data closest to x_interp using bisection:
    i_lowest = bisection(x_interp, x_data, M)

    P = y_data.copy()

    #first we slice the data since we only use M data points
    x_data = x_data[i_lowest : i_lowest+M+1]
    P = P[i_lowest : i_lowest+M+1]
    
    # note that we can access the i+1 and ith element simulatenously in a vectorized way 
    # as follows x_i+1 - x_i "=" x_data[1:] - x_data[:-1]
    # we can use a similar to trick to compare P_10 with P_12 etc.
    # Since the P array becomes shorter for each iteration we dont increase the indexing like we do for x_data.
     
    for i in range(1, M):
        P = (x_data[i:] - x_interp) * P[:-1] + (x_interp - x_data[:-i]) * P[1:]
        P /= x_data[i:] - x_data[:-i] 

    return P[0]


# you can merge the function below with LU_decomposition to make it more efficient
def run_LU_iterations(
    x: np.ndarray,
    y: np.ndarray,
    iterations: int = 11,
    coeffs_output_path: str = "Coefficients_per_iteration.txt",
) -> list:
    """
    Iteratively improves computation of coefficients c.

    Parameters
    ----------
    x : np.ndarray
        x-values.
    y : np.ndarray
        y-values.
    iterations : int
        Number of iterations.
    coeffs_output_path : str
        File to write coefficient values per iteration.

    Returns
    -------
    coeffs_history :
        List of coefficient vectors.
    """

    V = construct_vandermonde_matrix(x) 
    # instantiate a list where coefficient values for each iterations are stored
    coeff_values_list = []

    #note that the matrix class stores the LU decomposition so that we only compute it once!
    coeff = V.solve(y)
    coeff_values_list.append(coeff)

    for i in range(iterations):
        # Coeff is not determined perfectly: coeff = coeff_true + error
        # Notice then that: 
        # V@coeff - y = V@coeff - V@coeff_true = V@(coeff - coeff_true) = V@error

        # So we can try to solve for the error
        V_matmul_err = V@Matrix(coeff) - y 
        err = V.solve(V_matmul_err) 

        # and subtract it from our previous best estimate
        coeff -= err
        coeff_values_list.append(coeff)

    with open(coeffs_output_path, "w") as f:
        for i, coeff in enumerate(coeff_values_list):
            f.write(f"Iteration {i}:\n") #make a section header saying "iteration {i}"
            for j, c in enumerate(coeff):
                f.write(f"  c{j} = {c:.16e}\n") #write the values contained in the coefficients into the text file
            f.write("\n") #skip line for readability

    return coeff_values_list  


def plot_part_a(
    x_data: np.ndarray,
    y_data: np.ndarray,
    coeffs_c: np.ndarray,
    plots_dir: str = "Plots",
) -> None:
    """
    Ploting routine for part (a) results.

    Parameters
    ----------
    x_data : np.ndarray
        x-values.
    y_data : np.ndarray
        y-values.
    coeffs_c : np.ndarray
        Polynomial coefficients c.
    plots_dir : str
        Directory to save plots.

    Returns
    -------
    None
    """
    xx = np.linspace(x_data[0], x_data[-1], 1001)
    yy = evaluate_polynomial(coeffs_c, xx)
    y_at_data = evaluate_polynomial(coeffs_c, x_data)

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, hspace=0, height_ratios=[2.0, 1.0])
    axs = gs.subplots(sharex=True, sharey=False)

    axs[0].plot(x_data, y_data, marker="o", linewidth=0)
    axs[0].plot(xx, yy, linewidth=3)
    axs[0].set_xlim(
        np.floor(xx[0]) - 0.01 * (xx[-1] - xx[0]),
        np.ceil(xx[-1]) + 0.01 * (xx[-1] - xx[0]),
    )
    axs[0].set_ylim(-400, 400)
    axs[0].set_ylabel("$y$")
    axs[0].legend(["data", "Via LU decomposition"], frameon=False, loc="lower left")

    axs[1].set_ylim(1e-16, 1e1)
    axs[1].set_yscale("log")
    axs[1].set_ylabel(r"$|y - y_i|$")
    axs[1].set_xlabel("$x$")
    axs[1].plot(x_data, np.abs(y_data - y_at_data), linewidth=3)

    plt.savefig(os.path.join(plots_dir, "vandermonde_sol_2a.pdf"))
    plt.close()


def plot_part_b(
    x_data: np.ndarray,
    y_data: np.ndarray,
    plots_dir: str = "Plots",
) -> None:
    """
    Ploting routine for part (b) results.

    Parameters
    ----------
    x_data : np.ndarray
        x-values.
    y_data : np.ndarray
        y-values.
    plots_dir : str
        Directory to save plots.

    Returns
    -------
    None
    """
    xx = np.linspace(x_data[0], x_data[-1], 1001)
    yy = np.array([neville(x_data, y_data, x) for x in xx], dtype=np.float64)
    y_at_data = np.array([neville(x_data, y_data, x) for x in x_data], dtype=np.float64)

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, hspace=0, height_ratios=[2.0, 1.0])
    axs = gs.subplots(sharex=True, sharey=False)

    axs[0].plot(x_data, y_data, marker="o", linewidth=0)
    axs[0].plot(xx, yy, linestyle="dashed", linewidth=3)
    axs[0].set_xlim(
        np.floor(xx[0]) - 0.01 * (xx[-1] - xx[0]),
        np.ceil(xx[-1]) + 0.01 * (xx[-1] - xx[0]),
    )
    axs[0].set_ylim(-400, 400)
    axs[0].set_ylabel("$y$")
    axs[0].legend(["data", "Via Neville's algorithm"], frameon=False, loc="lower left")

    axs[1].set_ylim(1e-16, 1e1)
    axs[1].set_yscale("log")
    axs[1].set_ylabel(r"$|y - y_i|$")
    axs[1].set_xlabel("$x$")
    axs[1].plot(x_data, np.abs(y_data - y_at_data), linestyle="dashed", linewidth=3)

    plt.savefig(os.path.join(plots_dir, "vandermonde_sol_2b.pdf"))
    plt.close()


def plot_part_c(
    x_data: np.ndarray,
    y_data: np.ndarray,
    coeffs_history: list[np.ndarray],
    iterations_num: list[int] = [0, 1, 10],
    plots_dir: str = "Plots",
) -> None:
    """
    Ploting routine for part (c) results.

    Parameters
    ----------
    x_data : np.ndarray
        x-values.
    y_data : np.ndarray
        y-values.
    coeffs_history : list[np.ndarray]
        Coefficients per iteration.
    iterations_num : list[int]
        Iteration numbers to plot.
    plots_dir : str
        Directory to save plots.

    Returns
    -------
    None
    """

    linstyl = ["solid", "dashed", "dotted"]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    xx = np.linspace(x_data[0], x_data[-1], 1001)

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, hspace=0, height_ratios=[2.0, 1.0])
    axs = gs.subplots(sharex=True, sharey=False)

    axs[0].plot(x_data, y_data, marker="o", linewidth=0, color="black", label="data")

    for i, k in enumerate(iterations_num):
        if k >= len(coeffs_history):
            continue
        c = coeffs_history[k]
        yy = evaluate_polynomial(c, xx)
        y_at_data = evaluate_polynomial(c, x_data)
        diff = np.abs(y_at_data - y_data)

        axs[0].plot(
            xx,
            yy,
            linestyle=linstyl[i],
            color=colors[i],
            linewidth=3,
            label=f"Iteration {k}",
        )
        axs[1].plot(x_data, diff, linestyle=linstyl[i], color=colors[i], linewidth=3)

    axs[0].set_xlim(
        np.floor(xx[0]) - 0.01 * (xx[-1] - xx[0]),
        np.ceil(xx[-1]) + 0.01 * (xx[-1] - xx[0]),
    )
    axs[0].set_ylim(-400, 400)
    axs[0].set_ylabel("$y$")
    axs[0].legend(frameon=False, loc="lower left")

    axs[1].set_ylim(1e-16, 1e1)
    axs[1].set_yscale("log")
    axs[1].set_ylabel(r"$|y - y_i|$")
    axs[1].set_xlabel("$x$")

    plt.savefig(os.path.join(plots_dir, "vandermonde_sol_2c.pdf"))
    plt.close()


def main():
    os.makedirs("Plots", exist_ok=True)
    x_data, y_data = load_data()

    # compute times
    number = 1000

    t_a = (
        timeit.timeit(
            # note that we reconstruct our vandermonde matrix for each iteration, so its not cheating the time.
            stmt=lambda: vandermonde_solve_coefficients(x_data, y_data),
            number=number,
        )
        / number
    )

    xx = np.linspace(x_data[0], x_data[-1], 1001)
    t_b = (
        timeit.timeit(
            stmt=lambda: np.array(
                [neville(x_data, y_data, x) for x in xx], dtype=np.float64
            ),
            number=number,
        )
        / number
    )

    t_c = (
        timeit.timeit(
            stmt=lambda: run_LU_iterations(x_data, y_data, iterations=11),
            number=number,
        )
        / number
    )

    # write all timing
    with open("Execution_times.txt", "w", encoding="utf-8") as f:
        f.write(f"\\item Execution time for part (a): {t_a:.5f} seconds\n")
        f.write(f"\\item Execution time for part (b): {t_b:.5f} seconds\n")
        f.write(f"\\item Execution time for part (c): {t_c:.5f} seconds\n")

    c_a = vandermonde_solve_coefficients(x_data, y_data)
    plot_part_a(x_data, y_data, c_a)

    formatted_c = [f"{coef:.3e}" for coef in c_a]
    with open("Coefficients_output.txt", "w", encoding="utf-8") as f:
        for i, coef in enumerate(formatted_c):
            f.write(f"c$_{i+1}$ = {coef}, ")

    plot_part_b(x_data, y_data)

    coeffs_history = run_LU_iterations(
        x_data,
        y_data,
        iterations=11,
        coeffs_output_path="Coefficients_per_iteration.txt",
    )
    plot_part_c(x_data, y_data, coeffs_history, iterations_num=[0, 1, 10])


if __name__ == "__main__":
    main()
