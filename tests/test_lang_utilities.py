# -*- coding: utf-8 -*-
# tests/test_lang_utilities.py

import unittest
import re
from src.lang_utilities import regex_count_arabic_letters, manual_count_arabic_letters, arabic_alphabet, arabic_letters_pattern, count_arabic_letters

class TestCountArabicLetters(unittest.TestCase):

    def test_regex_count_arabic_letters(self):
        self.assertEqual(regex_count_arabic_letters("سلام"), 4)
        self.assertEqual(regex_count_arabic_letters("السّلام"), 6)
        self.assertEqual(regex_count_arabic_letters("الأصدقاء"), 8)
        self.assertEqual(regex_count_arabic_letters("العالم"), 6)
        self.assertEqual(regex_count_arabic_letters("إختبار"), 6)
        self.assertEqual(regex_count_arabic_letters("س\u200Dلام"), 4)  # Including Zero Width Joiner (ZWJ)
        self.assertEqual(regex_count_arabic_letters("س\u200Cلام"), 4)  # Including Zero Width Non-Joiner (ZWNJ)

    def test_manual_count_arabic_letters(self):
        self.assertEqual(manual_count_arabic_letters("سلام", arabic_alphabet), 4)
        self.assertEqual(manual_count_arabic_letters("السّلام", arabic_alphabet), 6)
        self.assertEqual(manual_count_arabic_letters("الأصدقاء", arabic_alphabet), 8)
        self.assertEqual(manual_count_arabic_letters("العالم", arabic_alphabet), 6)
        self.assertEqual(manual_count_arabic_letters("إختبار", arabic_alphabet), 6)
        self.assertEqual(manual_count_arabic_letters("س\u200Dلام", arabic_alphabet), 4)  # Including Zero Width Joiner (ZWJ)
        self.assertEqual(manual_count_arabic_letters("س\u200Cلام", arabic_alphabet), 4)  # Including Zero Width Non-Joiner (ZWNJ)

    def test_count_arabic_letters(self):
        self.assertEqual(count_arabic_letters("سلام"), 4)
        self.assertEqual(count_arabic_letters("السّلام"), 6)
        self.assertEqual(count_arabic_letters("الأصدقاء"), 8)
        self.assertEqual(count_arabic_letters("العالم"), 6)
        self.assertEqual(count_arabic_letters("إختبار"), 6)
        self.assertEqual(count_arabic_letters("س\u200Dلام"), 4)  # Including Zero Width Joiner (ZWJ)
        self.assertEqual(count_arabic_letters("س\u200Cلام"), 4)  # Including Zero Width Non-Joiner (ZWNJ)

    def test_count_with_control_characters(self):
        self.assertEqual(regex_count_arabic_letters("س\u200Dلام"), 4)  # Including Zero Width Joiner (ZWJ)
        self.assertEqual(regex_count_arabic_letters("س\u200Cلام"), 4)  # Including Zero Width Non-Joiner (ZWNJ)
        self.assertEqual(manual_count_arabic_letters("س\u200Dلام", arabic_alphabet), 4)  # Including Zero Width Joiner (ZWJ)
        self.assertEqual(manual_count_arabic_letters("س\u200Cلام", arabic_alphabet), 4)  # Including Zero Width Non-Joiner (ZWNJ)

if __name__ == "__main__":
    unittest.main()
