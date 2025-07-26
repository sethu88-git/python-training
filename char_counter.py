'''
Create a script that:

Takes a string as input

Counts how many times each character (case-sensitive) appears in the string

Ignores spaces entirely

Prints the result as a sorted list of tuples: [(char1, count1), (char2, count2), ...]

Sorted by character in ascending order

'''


def char_counter(str_input):
    char_dict = {}
    str_input = str.replace(str_input, " ", "")
    for character in str_input:
        char_dict[character] = char_dict.get(character, 0)+1
    print_charcounts(char_dict)


def print_charcounts(char_dict):
    char_list = []
    for key in sorted(char_dict.keys()):
        char_list.append(tuple((key, char_dict[key])))
    print(char_list)


if __name__ == "__main__":
    in_string = input("Enter string: ").strip()
    char_counter(in_string)
