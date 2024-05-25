# -*- coding: utf-8 -*-
# .\src\arabic_utilities.py

# Example usage with Arabic alphabet
arabic_alphabet = [
    "ا", "ب", "ت", "ث", "ج", "ح", "خ", "د", "ذ", "ر", "ز", "س", "ش", "ص", "ض", "ط", "ظ",
    "ع", "غ", "ف", "ق", "ك", "ل", "م", "ن", "ه", "و", "ي"
]

def get_next_generation(initial_string: str, alphabet: list) -> list:
    """
    Generate the next generation of strings by appending each character from the alphabet to the initial string.

    :param initial_string: The initial string to which each character from the alphabet will be appended.
    :param alphabet: A list of characters representing the alphabet.
    :return: A list of strings where each string is the initial string followed by a character from the alphabet.
    """
    return [initial_string + char for char in alphabet]

