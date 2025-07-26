import numpy as np


def matrix_inspector(a):
    dict_report = {}
    dict_report['transpose'] = np.transpose(a)
    dict_report['determinant'] = np.linalg.det(a)
    dict_report['rank'] = np.linalg.matrix_rank(a)
    dict_report['is_symmetric'] = np.array_equal(dict_report['transpose'], a)
    try:
        dict_report['inverse'] = np.linalg.inv(a)
    except np.linalg.LinAlgError:
        dict_report['inverse'] = "Not invertible"
    return (dict_report)


def print_report(dict_report):
    for key, value in dict_report.items():
        print(f"\n{key}:\n{value}")


if __name__ == "__main__":
    np_array = np.random.randint(low=0, high=100, size=(4, 4))
    print_report(matrix_inspector(np_array))
