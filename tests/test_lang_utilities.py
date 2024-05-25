# -*- coding: utf-8 -*-
# .\tests\test_lang_utilities.py

import unittest
from src.lang_utilities import get_next_generation, arabic_alphabet

class TestGetNextGeneration(unittest.TestCase):

    def setUp(self):
        # Setup code
        pass

    def test_empty_initial_string(self):
        result = get_next_generation("", arabic_alphabet)
        self.assertEqual(result, arabic_alphabet)

    def test_non_empty_initial_string(self):
        initial_string = "أ"
        expected_result = ["أ" + char for char in arabic_alphabet]
        result = get_next_generation(initial_string, arabic_alphabet)
        self.assertEqual(result, expected_result)

    def test_generic_alphabet(self):
        english_alphabet = ["a", "b", "c", "d"]
        initial_string = "test"
        expected_result = ["testa", "testb", "testc", "testd"]
        result = get_next_generation(initial_string, english_alphabet)
        self.assertEqual(result, expected_result)

if __name__ == "__main__":
    unittest.main()
