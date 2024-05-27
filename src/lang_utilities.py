# -*- coding: utf-8 -*-
# src/lang_utilities.py

# Import necessary libraries
import sqlite3

# Example usage with a comprehensive Arabic alphabet
arabic_alphabet = [
    r"\u0627", # Alef "ا"
    r"\u0628", # Baa  "ب"
    r"\u062a", # Taa  "ت"
    r"\u062b", # Thaa "ث"
    r"\u062c", # Jeem "ج"
    r"\u062d", # Haa  "ح"
    r"\u062e", # Khaa "خ"
    r"\u062f", # Dal  "د"
    r"\u0630", # Thal "ذ"
    r"\u0631", # Ra   "ر"
    r"\u0632", # Zay  "ز"
    r"\u0633", # Seen "س"
    r"\u0634", # Sheen"ش"
    r"\u0635", # Sad  "ص"
    r"\u0636", # Dad  "ض"
    r"\u0637", # Tah  "ط"
    r"\u0638", # Zah  "ظ"
    r"\u0639", # Ain  "ع"
    r"\u063a", # Ghain"غ"
    r"\u0641", # Feh  "ف"
    r"\u0642", # Qaf  "ق"
    r"\u0643", # Kaf  "ك"
    r"\u0644", # Lam  "ل"
    r"\u0645", # Meem "م"
    r"\u0646", # Noon "ن"
    r"\u0647", # Heh  "ه"
    r"\u0648", # Waw  "و"
    r"\u064a", # Yeh  "ي"
    r"\u0623", # Alef with Hamza Above "أ"
    r"\u0625", # Alef with Hamza Below "إ"
    r"\u0622", # Alef with Madda "آ"
    r"\u0621", # Hamza "ء"
    r"\u0624", # Waw with Hamza "ؤ"
    r"\u0626", # Yeh with Hamza "ئ"
]

arabic_digits = [
    r"\u0660", # Arabic-Indic Digit Zero "٠"
    r"\u0661", # Arabic-Indic Digit One "١"
    r"\u0662", # Arabic-Indic Digit Two "٢"
    r"\u0663", # Arabic-Indic Digit Three "٣"
    r"\u0664", # Arabic-Indic Digit Four "٤"
    r"\u0665", # Arabic-Indic Digit Five "٥"
    r"\u0666", # Arabic-Indic Digit Six "٦"
    r"\u0667", # Arabic-Indic Digit Seven "٧"
    r"\u0668", # Arabic-Indic Digit Eight "٨"
    r"\u0669", # Arabic-Indic Digit Nine "٩"
]

# Unicode control characters
ZWJ = r"\u200D"  # Zero Width Joiner
ZWNJ = r"\u200C"  # Zero Width Non-Joiner

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
