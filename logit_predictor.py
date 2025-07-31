'''
Implement the forward-pass of logistic regression using:

Feature matrix X

Weight vector w

Bias b

Sigmoid activation
Create:
X: shape (10 × 5)
w: shape (5 × 1)
b: scalar bias = 0.5

Compute:
Z = X @ w + b
A = sigmoid(Z)

Print:
Shapes of X, w, Z, A
First 5 values of A, rounded to 4 decimals

'''


import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))


def activation_function(w, b, x):
    return np.matmul(x, w) + b


if __name__ == "__main__":
    x = np.random.rand(10, 5)
    w = np.random.rand(5, 1)
    b = np.array([0.5])
    z = activation_function(w, b, x)
    A = np.round(sigmoid(z), 4)
    print(f"A={A[:5]}\nShapes: x: {x.shape}, w: {w.shape}, z:{z.shape}, A:{A.shape}")
