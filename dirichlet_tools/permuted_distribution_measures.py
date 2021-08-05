import numpy as np
from scipy.optimize import lsq_linear, linear_sum_assignment

def permuted_average_l2(matrix_a, matrix_b):
    c = np.dot(matrix_a.T, matrix_b)
    _, permutation = linear_sum_assignment(c, maximize=True)
    dist = np.sum(np.linalg.norm(matrix_a[:, permutation] - matrix_b, axis = 1)) / matrix_a.shape[0]

    return permutation, dist