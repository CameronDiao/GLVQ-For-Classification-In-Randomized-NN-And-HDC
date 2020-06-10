import numpy as np

def binding(a, b):
    """
    Performs binding operation on the row vectors of matrices a and b
    Binding operation is the Hadamard product
    :param a: a numpy 2-D array
    :param b: a numpy 2-D array
    :return: a numpy 2-D array whose rows are composed of bound vectors
    """
    return np.multiply(a, b)

def bundling(matrix, kappa):
    """
    Performs bundling operation on row vectors of matrix
    :param matrix: a numpy 2-D array
    :param kappa: an int value representing the threshold parameter of the clipping function
    :return: h: a numpy 1-D array representing the bundled vector
    """
    clipping = lambda x: np.piecewise(x, [x <= -kappa, x > -kappa and x < kappa, x >= kappa],
                                      [lambda t: -kappa, lambda t: t, lambda t: kappa])
    clipping = np.vectorize(clipping)
    col_sum = matrix.sum(axis = 0)
    h = clipping(col_sum)
    return h