'''
Write a generic function that:
Takes a 2D NumPy array A and a tuple block_shape = (rows, cols)
Validates that the shape of A is divisible by the block_shape
Returns all non-overlapping blocks of that shape as a flat list
Each block must be a NumPy array of shape block_shape
No loops for slicing — use nested comprehensions or pure slicing
Validate divisibility and raise ValueError if incompatible
Each block must be a separate copy, not just a view (use .copy())
'''
import numpy as np


def block_extractor(a: np.ndarray, block_shape: tuple):
    R, C = a.shape
    r, c = block_shape
    if R % r != 0 or C % c != 0:
        raise ValueError("Given matrix is indivisble to the given block size")
    split_blocks = [
        a[i:i+r, j:j+c].copy()
        for i in range(0, R, r)
        for j in range(0, C, c)
    ]
    print_blocks(split_blocks)


def print_blocks(split_blocks):
    for i in range(len(split_blocks)):
        print(f"\n\nSplit {i+1}:\n{split_blocks[i]}")


if __name__ == "__main__":
    block_extractor(np.random.randint(low=1, high=10, size=(6, 6)), (3, 3))
