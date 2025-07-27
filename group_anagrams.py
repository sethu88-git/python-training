'''
Given a list of words, group all anagrams together.
'''


def group_anagrams(str_list):
    anagram_dict = {}
    for word in str_list:
        pivot = "".join(sorted(word))
        if pivot not in anagram_dict:
            anagram_dict[pivot] = [word]
        else:
            anagram_dict[pivot].append(word)
    print_anagrams(anagram_dict)


def print_anagrams(anagram_dict):
    print(f"{[value for value in anagram_dict.values()]}")


if __name__ == "__main__":
    in_string = list(input().strip().split())
    group_anagrams(in_string)
