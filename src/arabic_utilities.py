# -*- coding: utf-8 -*-
# .\src\arabic_utilities.py

def generate_valid_arabic_ligatures():
    # Define Arabic letters
    arabic_letters = [
        "ا", "ب", "ت", "ث", "ج", "ح", "خ", "د", "ذ", "ر", "ز", "س", "ش", "ص", "ض", "ط", "ظ",
        "ع", "غ", "ف", "ق", "ك", "ل", "م", "ن", "ه", "و", "ي"
    ]
    
    # Define letters that cannot connect to the following letter
    non_connecting_letters = ["ا", "د", "ذ", "ر", "ز", "و"]
    
    # Define a set of common ligatures
    common_ligatures = {
        "لا", "لله", "بب", "مم", "نن", "للا"  # Extend this set with more predefined ligatures
    }
    
    # Generate all possible 2-letter and 3-letter combinations
    possible_combinations = [(a, b) for a in arabic_letters for b in arabic_letters] + \
                            [(a, b, c) for a in arabic_letters for b in arabic_letters for c in arabic_letters]
    
    # Filter valid connections for 2-letter and 3-letter combinations
    valid_combinations = [
        comb for comb in possible_combinations
        if all(comb[i] not in non_connecting_letters for i in range(len(comb) - 1))
    ]
    
    # Identify valid ligatures
    valid_ligatures = [
        "".join(comb) for comb in valid_combinations
        if "".join(comb) in common_ligatures
    ]
    
    return valid_ligatures

# Call the function and print the results
ligatures = generate_valid_arabic_ligatures()
print(ligatures)
