import numpy as np


def Poisson(k: np.float32, lmbda: np.float32) -> np.float32:
    """
    Calculate the Poisson probability for k occurrences with mean lmbda.
    Parameters:
        k (np.float32): The number of occurrences.
        lmbda (np.float32): The mean number of occurrences.
    Returns:
        np.float32: The probability of observing k occurrences given the mean lmbda.
    """

    # Note that we value memory usage over computational speed.
    # We do not compute lmbda^k and k! separately, producing 2 massive numbers.
    # Instead we overwrite lmbda^k/k! during a larger number of operations, keeping the numbers "small"

    # we compute it using a large number of divisions, the tradeoff is computational speed.
    temp1 = 1
    for i in range(k):
        i = i + 1
        temp1 *= lmbda / i

    return temp1 * np.exp(-lmbda)


def main() -> None:
    # (lambda, k) pairs:
    values = [
        (np.float32(1.0), np.int32(0)),
        (np.float32(5.0), np.int32(10)),
        (np.float32(3.0), np.int32(21)),
        (np.float32(2.6), np.int32(40)),
        (np.float32(100.0), np.int32(5)),
        (np.float32(101.0), np.int32(200)),
    ]
    with open("Poisson_output.txt", "w") as file:
        for i, (lmbda, k) in enumerate(values):
            P = Poisson(k, lmbda)
            if i < len(values) - 1:
                file.write(f"{lmbda:.1f} & {k} & {P:.6e} \\\\ \\hline \n")
            else:
                file.write(f"{lmbda:.1f} & {k} & {P:.6e} \n")


if __name__ == "__main__":
    main()
