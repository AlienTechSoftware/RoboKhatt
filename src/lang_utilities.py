# -*- coding: utf-8 -*-
# .\src\lang_utilities.py

from tqdm import tqdm
import multiprocessing as mp
import time

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

def get_next_generation(initial_string: str, alphabet: list) -> list:
    """
    Generate the next generation of strings by appending each character from the alphabet to the initial string.

    :param initial_string: The initial string to which each character from the alphabet will be appended.
    :param alphabet: A list of characters representing the alphabet.
    :return: A list of strings where each string is the initial string followed by a character from the alphabet.
    """
    return [initial_string + char for char in alphabet]

def calculate_combinations(alphabet_size, max_length):
    return sum(alphabet_size ** k for k in range(max_length + 1))

def benchmark_generation(alphabet, max_length):
    start_time = time.time()
    generate_all_combinations_iter(alphabet, 3)  # Smaller subset for benchmarking
    end_time = time.time()
    time_taken = end_time - start_time

    # Calculate total combinations for benchmark length and target length
    small_combinations = calculate_combinations(len(alphabet), 3)
    large_combinations = calculate_combinations(len(alphabet), max_length)

    # Estimate time for max_length
    estimated_time = time_taken * (large_combinations / small_combinations)
    return estimated_time

def generate_combinations_worker(args):
    initial_string, alphabet, max_length = args
    result = []
    queue = [initial_string]

    while queue:
        current = queue.pop(0)
        result.append(current)
        if len(current) < max_length:
            next_gen = get_next_generation(current, alphabet)
            queue.extend(next_gen)

    return result

def generate_all_combinations_iter(alphabet: list, max_length: int) -> list:
    """
    Generate all possible combinations of the alphabet up to a specified maximum length.

    :param alphabet: A list of characters representing the alphabet.
    :param max_length: The maximum length of the generated strings.
    :return: A list of all possible combinations of the alphabet up to the specified maximum length.
    """
    initial_combinations = get_next_generation("", alphabet)
    total_combinations = calculate_combinations(len(alphabet), max_length)

    # Create a pool of workers
    pool = mp.Pool(mp.cpu_count())

    try:
        # Divide work among workers
        tasks = [(initial, alphabet, max_length) for initial in initial_combinations]

        results = []
        with tqdm(total=total_combinations, desc="Generating combinations") as pbar:
            for i, result in enumerate(pool.imap_unordered(generate_combinations_worker, tasks)):
                results.extend(result)
                if (i + 1) % 100 == 0:
                    pbar.update(len(result))
                    pbar.refresh()  # Force refresh the progress bar
                    time.sleep(0)  # Yield to the OS to allow console update

            pbar.update(total_combinations % 100)  # Update the remaining progress if any

        # Remove duplicates to ensure correctness
        unique_results = list(set(results))

        # Add the initial empty string if not already included
        if "" not in unique_results:
            unique_results.append("")

        return unique_results

    except KeyboardInterrupt:
        print("Process interrupted. Returning generated combinations so far.")
        pool.terminate()
        pool.join()
        return results

    finally:
        pool.close()
        pool.join()

def generate_all_combinations(alphabet: list, max_length: int) -> list:
    """
    Estimate time and generate all possible combinations of the alphabet up to a specified maximum length.

    :param alphabet: A list of characters representing the alphabet.
    :param max_length: The maximum length of the generated strings.
    :return: A list of all possible combinations of the alphabet up to the specified maximum length.
    """
    estimated_time = benchmark_generation(alphabet, max_length)
    print(f"Estimated time for generating all combinations up to length {max_length}: {estimated_time:.2f} seconds")

    if estimated_time > 60:  # threshold for warning (60 seconds)
        confirm = input(f"The estimated time is {estimated_time/60:.2f} minutes. Do you want to proceed? (y/n): ")
        if confirm.lower() != 'y':
            print("Process aborted by user.")
            return []

    return generate_all_combinations_iter(alphabet, max_length)
