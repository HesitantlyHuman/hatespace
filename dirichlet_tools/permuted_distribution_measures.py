import numpy as np
from scipy.optimize import linear_sum_assignment

def permuted_normalized_frobenius(matrix_a, matrix_b):
    c = np.dot(matrix_a.T, matrix_b)
    _, permutation = linear_sum_assignment(c, maximize = True)
    dist = np.power(np.sum(np.power(matrix_a[:, permutation] - matrix_b, 2)) / matrix_a.shape[0], 0.5)

    return permutation, dist