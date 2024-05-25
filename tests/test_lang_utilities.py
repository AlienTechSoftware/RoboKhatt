# -*- coding: utf-8 -*-
# .\tests\test_lang_utilities.py

import unittest
from src.lang_utilities import generate_all_combinations, arabic_alphabet, get_next_generation

class TestGenerateAllCombinations(unittest.TestCase):

    def test_generate_all_combinations_small(self):
        alphabet = ["ا", "ب", "ت"]  # Limit to a smaller subset for this test
        max_length = 3
        combinations = generate_all_combinations(alphabet, max_length)
        expected_count = sum(len(alphabet) ** i for i in range(max_length + 1))
        self.assertEqual(len(combinations), expected_count)

    def test_generate_all_combinations_arabic(self):
        alphabet = arabic_alphabet
        max_length = 2
        combinations = generate_all_combinations(alphabet, max_length)
        expected_count = sum(len(alphabet) ** i for i in range(max_length + 1))
        self.assertEqual(len(combinations), expected_count)

    def test_generate_all_combinations_arabic_max_length_3(self):
        alphabet = arabic_alphabet
        max_length = 4
        combinations = generate_all_combinations(alphabet, max_length)
        expected_count = sum(len(alphabet) ** i for i in range(max_length + 1))
        self.assertEqual(len(combinations), expected_count)

    def test_progress_and_interrupt(self):
        alphabet = ["ا", "ب", "ت"]  # Limit to a smaller subset for this test
        max_length = 5
        try:
            combinations = []
            queue = [""]
            while queue:
                current = queue.pop(0)
                if len(current) < max_length:
                    next_gen = get_next_generation(current, alphabet)
                    combinations.extend(next_gen)
                    queue.extend(next_gen)
        except KeyboardInterrupt:
            print("Interrupted")
            self.assertTrue(len(combinations) > 0)  # Ensure we still got some combinations

if __name__ == "__main__":
    unittest.main()
