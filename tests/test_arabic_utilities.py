# -*- coding: utf-8 -*-
# .\tests\test_arabic_utilities.py

import unittest
from src.arabic_utilities import generate_valid_arabic_ligatures

class TestGenerateValidArabicLigatures(unittest.TestCase):

    def test_generate_valid_arabic_ligatures(self):
        # Call the function
        ligatures = generate_valid_arabic_ligatures()

        # Define expected results
        expected_ligatures = {"لا", "لله", "بب", "مم", "نن", "للا"}  # This should match your common_ligatures

        # Check that all expected ligatures are in the result
        for ligature in expected_ligatures:
            self.assertIn(ligature, ligatures)

        # Check that all ligatures in the result are valid
        for ligature in ligatures:
            self.assertTrue(ligature in expected_ligatures)

        # Additional checks for invalid ligatures
        invalid_ligatures = ["اا", "دد", "وو", "زر", "ذو"]
        for ligature in invalid_ligatures:
            self.assertNotIn(ligature, ligatures)

        # Check that the function does not include non-common ligatures
        self.assertNotIn("باب", ligatures)

if __name__ == "__main__":
    unittest.main()
