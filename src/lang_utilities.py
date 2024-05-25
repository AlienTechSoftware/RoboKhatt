# -*- coding: utf-8 -*-
# .\src\lang_utilities.py

from tqdm import tqdm
import multiprocessing as mp
import time

# Example usage with a comprehensive Arabic alphabet
arabic_alphabet = [
    "ا", "ب", "ت", "ث", "ج", "ح", "خ", "د", "ذ", "ر", "ز", "س", "ش", "ص", "ض", "ط", "ظ",
    "ع", "غ", "ف", "ق", "ك", "ل", "م", "ن", "ه", "و", "ي",
    "أ", "إ", "آ", "ء", "ؤ", "ئ"
]

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
