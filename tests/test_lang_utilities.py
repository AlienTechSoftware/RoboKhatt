# -*- coding: utf-8 -*-
# .\tests\test_lang_utilities.py

import unittest
from src.diffusion.dataset import load_arabic_datasources
from src.lang_utilities import arabic_alphabet

class TestGenerateAllCombinations(unittest.TestCase):

    def test_load_arabic_datasources(self):
        words_by_length = load_arabic_datasources(force_reload=True)
        # Check if the database has been loaded correctly
        for length, words in words_by_length.items():
            self.assertTrue(len(words) > 0)  # Ensure there are words of each length

if __name__ == "__main__":
    unittest.main()
