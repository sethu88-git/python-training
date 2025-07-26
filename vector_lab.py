'''
Objective:
    Write a Python script that:
        Creates three NumPy arrays:
            v1: shape (5,), values from 1 to 5
            v2: shape (5,), values from 5 to 1 (descending)
            v3: shape (5,), alternating 1 and -1 starting from 1
        Calculates and prints:
            Dot product of v1 and v2
            Element-wise product of v1 and v3
            L2 norm of v2
            Cosine similarity between v1 and v2
    All outputs must be clearly labeled and printed in a tidy format
'''

import numpy as np


def vector_lab(v1, v2, v3):
    vector_dict = {}
    vector_dict['dot product'] = np.dot(v1, v2)
    vector_dict['element-wise product'] = v1*v3
    vector_dict['L2 norm'] = round(np.linalg.norm(v2, ord=2), 4)
    vector_dict['is_orthogonal'] = np.dot(v1, v3) == 0
    vector_dict['cosine_similarilty'] = round(
        np.dot(v1, v2)/(np.linalg.norm(v1) * np.linalg.norm(v2)), 4)
    vector_print(vector_dict)


def vector_print(vector_dict):
    for key, value in vector_dict.items():
        print(f"\n{key} = {value}")


if __name__ == "__main__":
    v1 = np.arange(1, 6, 1)
    v2 = np.arange(5, 0, -1)
    v3 = np.array([(-1)**x for x in range(2, 7, 1)])
    vector_lab(v1, v2, v3)
