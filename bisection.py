import numpy as np


def bisection(x, x_vals: np.ndarray, M:int) -> int:
    """
    Returns lowest index of the M sample_points closest to input point

    :param x: input point
    :param x_vals: sample_points
    :param M: amount of closest sample_points
    """

    N = len(x_vals)

    # Check for monotonicity of x_samples!
    for i in range(N - 1):
        if not (x_vals[i] < x_vals[i + 1]):
            print("ERROR: x_vals are not monotonic")
            return

    # check for x value being at the edge:
    if x < x_vals[M - 1]:
        return 0
    if x > x_vals[-M - 1]:
        return N - M

    # iteratively increase i_low if input point is in "second half^iteration of considered sample_points"
    i_low = 0
    iteration = 0
    while x_vals[i_low + 1] < x:
        iteration += 1
        stepsize = int(0.5**iteration * N) + 1
        if x > x_vals[i_low + stepsize]:
            i_low += stepsize
        else:
            pass

    return i_low
