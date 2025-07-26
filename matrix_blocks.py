'''
Write a script that:
Creates a 6×6 NumPy array filled with numbers from 1 to 36 in row-major order.
Splits this into 4 equal non-overlapping 3×3 blocks:
Top-left
Top-right
Bottom-left
Bottom-right
Print each block clearly with a title.
Use slicing only. No loops.
No reshape or for-loops to extract blocks.
Label each block in output.
'''

import numpy as np


def matrix_blocks(input_matrix):
    split_blocks = []
    split_blocks.append(input_matrix[:-3, :-3])
    split_blocks.append(input_matrix[:-3, -3:])
    split_blocks.append(input_matrix[-3:, :-3])
    split_blocks.append(input_matrix[-3:, -3:])
    print_split(split_blocks)


def print_split(split_blocks):
    print(f'''\nTop left:\n{split_blocks[0]}\nTop right:\n{
        split_blocks[1]}\nBottom left:\n{split_blocks[2]}\nBottom right:\n{split_blocks[3]}''')


if __name__ == "__main__":
    a = np.arange(1, 37, 1).reshape((6, 6))
    matrix_blocks(a)
