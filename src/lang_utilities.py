# -*- coding: utf-8 -*-
# src/lang_utilities.py

# Import necessary libraries
import sqlite3
import re

# Example usage with a comprehensive Arabic alphabet
arabic_alphabet = [
    "\u0627", # Alef "ا"
    "\u0628", # Baa  "ب"
    "\u062a", # Taa  "ت"
    "\u062b", # Thaa "ث"
    "\u062c", # Jeem "ج"
    "\u062d", # Haa  "ح"
    "\u062e", # Khaa "خ"
    "\u062f", # Dal  "د"
    "\u0630", # Thal "ذ"
    "\u0631", # Ra   "ر"
    "\u0632", # Zay  "ز"
    "\u0633", # Seen "س"
    "\u0634", # Sheen"ش"
    "\u0635", # Sad  "ص"
    "\u0636", # Dad  "ض"
    "\u0637", # Tah  "ط"
    "\u0638", # Zah  "ظ"
    "\u0639", # Ain  "ع"
    "\u063a", # Ghain"غ"
    "\u0641", # Feh  "ف"
    "\u0642", # Qaf  "ق"
    "\u0643", # Kaf  "ك"
    "\u0644", # Lam  "ل"
    "\u0645", # Meem "م"
    "\u0646", # Noon "ن"
    "\u0647", # Heh  "ه"
    "\u0648", # Waw  "و"
    "\u064a", # Yeh  "ي"
    "\u0623", # Alef with Hamza Above "أ"
    "\u0625", # Alef with Hamza Below "إ"
    "\u0622", # Alef with Madda "آ"
    "\u0621", # Hamza "ء"
    "\u0624", # Waw with Hamza "ؤ"
    "\u0626", # Yeh with Hamza "ئ"
]

arabic_digits = [
    "\u0660", # Arabic-Indic Digit Zero "٠"
    "\u0661", # Arabic-Indic Digit One "١"
    "\u0662", # Arabic-Indic Digit Two "٢"
    "\u0663", # Arabic-Indic Digit Three "٣"
    "\u0664", # Arabic-Indic Digit Four "٤"
    "\u0665", # Arabic-Indic Digit Five "٥"
    "\u0666", # Arabic-Indic Digit Six "٦"
    "\u0667", # Arabic-Indic Digit Seven "٧"
    "\u0668", # Arabic-Indic Digit Eight "٨"
    "\u0669", # Arabic-Indic Digit Nine "٩"
]

# Define the Arabic characters regex pattern to include only letters and digits
arabic_letters_pattern = re.compile(
    r'[\u0621-\u063A\u0641-\u064A\u0660-\u0669]'
)

# Unicode control characters
ZWJ = "\u200D"  # Zero Width Joiner
ZWNJ = "\u200C"  # Zero Width Non-Joiner

def generate_arabic_shapes_dynamic(alphabet):
    """
    Generate the four basic shapes (isolated, initial, medial, final) of all Arabic characters in the given alphabet
    dynamically using Unicode control characters.

    :param alphabet: List of Arabic characters.
    :return: Dictionary with characters as keys and their shapes as values.
    """
    shapes_dict = {}
    for char in alphabet:
        shapes_dict[char] = [
            char,                   # Isolated form
            char + ZWJ,             # Initial form
            ZWJ + char + ZWJ,       # Medial form
            ZWJ + char              # Final form
        ]
    return shapes_dict


def regex_count_arabic_letters(word):
    """
    Count the actual Arabic letters in a word, excluding control characters and diacritics.
    
    :param word: The word to count the letters in.
    :return: The count of Arabic letters.
    """
    return len(re.findall(arabic_letters_pattern, word))

def manual_count_arabic_letters(word, arabic_alphabet):
    """
    Count Arabic letters manually by checking each character in the word against the Arabic alphabet.
    
    :param word: The word to count the letters in.
    :param arabic_alphabet: List of Arabic letters.
    :return: The count of Arabic letters.
    """
    return sum(1 for char in word if char in arabic_alphabet)

def count_arabic_letters(word):
    return manual_count_arabic_letters(word, arabic_alphabet)
