import numpy as np
from scipy.signal import convolve2d

# Define the input matrices
matrix1 = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])

matrix2 = np.array([[2, 0, 1],
                    [1, 2, 3],
                    [0, 1, 2]])

# Perform convolution
result = convolve2d(matrix1, matrix2, mode='full')

print(result)

number = 2
matrix = [[1, 2], [3, 4]]

result = np.linalg.matrix_power(matrix, number)
print(result)


def ca_cfar_tresholding(self, matrix, sliding_window, false_alarm_prob):
    sliding_sum = signal.convolve2d(matrix, sliding_window, mode='same')
    contribution_matrix = signal.convolve2d(np.ones(matrix.shape), sliding_window, mode='same')

    alpha_matrix = (false_alarm_prob ** (-1 / contribution_matrix)) - 1

    # FORMULA: threshold = alpha * sigma_est, sigma_est = sliding_sum
    threshold_matrix = np.multiply(sliding_sum, alpha_matrix)

    return threshold_matrix