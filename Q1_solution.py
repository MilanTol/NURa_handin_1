import numpy as np


def Poisson(k: np.int32, lmbda: np.float32) -> np.float32:
    """
    Calculate the Poisson probability for k occurrences with mean lmbda: P_lmbda(k)
    Parameters:
        k (np.int32): The number of occurrences.
        lmbda (np.float32): The mean number of occurrences.
    Returns:
        np.float32: The probability of observing k occurrences given the mean lmbda.
    """

    # Note that we value memory usage over computational speed!
    # components of the calculation can overflow, or underflow.

    # An observation we can make is that 0 <= P_lmbda(k) <= 1 for all k, k >= 0, and lmbda >= 0.
    # So all numbers that we are working with are larger than 0, which means we can safely work in logspace
    # This makes our range of numbers go from (min, max) to (exp(min), exp(max)).

    # Assuming that our log(final number) is within np.float32 range, we do the calculation as follows.

    # log(P_lmbda(k)) = log(lmbda^k) + log(exp(-lmbda)) - log(k!) = k*log(lmbda) - lmbda - sum_{i=0}^k i
    # we compute it using a sum over a linspace, the tradeoff is computational speed.
    term1 = k*np.log(lmbda) - lmbda
    term2 = 0
    for i in range(1, k+1):
        term2 += np.log(i) #overwrite term2, rather than np.sum over a linspace since that would require more memory.

    logP = term1 - term2

    # we can now check whether logP is beyond the range that np.float32 can handle using numpy.finfo() 
    # log a warning to the user if necessary.
    if logP < np.log(np.finfo('float32').tiny):
        print("Warning, P is smaller than minimum supported value by np.float32")
    #we dont have to check whether logP is too large since P_lmbda(k) <= 1.
     
    P = np.exp(logP)
    return P


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
