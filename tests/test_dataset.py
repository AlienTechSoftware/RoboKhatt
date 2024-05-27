# -*- coding: utf-8 -*-
# tests/test_dataset.py

import unittest
import os
import sqlite3
from src.diffusion.dataset import ArabicDataset

class TestArabicDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_database_path = '.generated/test_arabic_words.db'
        cls.arabic_dataset = ArabicDataset(db_path=cls.test_database_path)
        cls.conn = sqlite3.connect(cls.test_database_path)
        cls.cursor = cls.conn.cursor()
        cls.test_word_file = '.datasets/test-word-list.txt'

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()
        if os.path.exists(cls.test_database_path):
            pass
            # os.remove(cls.test_database_path)

    def test_load_words_from_file(self):
        self.arabic_dataset.load_words_from_file(self.test_word_file)
        words_by_length = self.arabic_dataset.load_arabic_datasources()
        total_words = sum(len(words) for words in words_by_length.values())
        self.cursor.execute('SELECT COUNT(*) FROM words')
        db_word_count = self.cursor.fetchone()[0]
        self.assertEqual(total_words, db_word_count)

    def test_force_reload(self):
        # Load initial words from file
        self.arabic_dataset.load_words_from_file(self.test_word_file)
        initial_words_by_length = self.arabic_dataset.load_arabic_datasources()
        print(f"Initial words by length: {len(initial_words_by_length)}")

        # Force reload the datasource
        self.arabic_dataset.load_arabic_datasources(force_reload=True)
        words_by_length_after_insert = self.arabic_dataset.load_arabic_datasources()
        print(f"Words by length after reload: {len(words_by_length_after_insert)}")

        # Check that we have non-zero words in each size from 1-6
        for length in range(1, 7):
            print(f"index {length}")
            self.assertGreater(len(words_by_length_after_insert.get(length, [])), 0)

if __name__ == '__main__':
    unittest.main()
