'''
Given an n×n NumPy matrix:
Extract the main diagonal (top-left to bottom-right)
Extract the anti-diagonal (top-right to bottom-left)
Flip the matrix vertically and horizontally
Print all results with labels
'''

import numpy as np


def cross_diagonals(a: np.ndarray):
    matrix_details = {}
    r, c = a.shape
    matrix_details['main diagonal'] = a[np.arange(0, min(r, c), 1), np.arange(0, min(r, c), 1)]
    matrix_details['anti diagonal'] = a[np.arange(0, min(r, c), 1), np.arange(c-1, abs(r-c)-1, -1)]
    matrix_details['vertical flip'] = a[np.arange(r-1, -1, -1), :]
    matrix_details['horizontal flip'] = a[:, np.arange(c-1, -1, -1)]
    print_details(matrix_details)


def print_details(matrix_details):
    for key, value in matrix_details.items():
        print(f"\n\n{key}:\n{value}")


if __name__ == "__main__":
    cross_diagonals(np.array([
        [1,  2,  3,  4,  5],
        [6,  7,  8,  9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25]
    ]))
