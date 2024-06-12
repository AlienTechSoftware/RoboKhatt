# -*- coding: utf-8 -*-
# tests/test_dataset.py

import unittest
import sqlite3
import os
import torch
from torch.utils.data import DataLoader
from src.diffusion.dataset import ArabicDataset, create_database, load_words_from_file, load_generated_arabic_shapes, load_arabic_datasources
from src.img_utilities import TextImageDataset
from src.lang_utilities import arabic_alphabet

class TestArabicDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_database_path = '.generated/test_arabic_words.db'
        create_database(cls.test_database_path)
        cls.conn = sqlite3.connect(cls.test_database_path)
        cls.cursor = cls.conn.cursor()
        cls.arabic_dataset = ArabicDataset(arabic_alphabet, 2, 'arial.ttf', 30, (512, 128), is_arabic=True, database_path=cls.test_database_path)

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()
        os.remove(cls.test_database_path)

    def test_load_words_from_file(self):
        load_words_from_file('.datasets/test-word-list.txt', self.test_database_path)
        self.cursor.execute('SELECT COUNT(*) FROM words')
        count = self.cursor.fetchone()[0]
        self.assertGreater(count, 0)

    def test_load_generated_arabic_shapes(self):
        load_generated_arabic_shapes(self.test_database_path)
        self.cursor.execute('SELECT COUNT(*) FROM words')
        count = self.cursor.fetchone()[0]
        self.assertGreater(count, 0)

    def test_force_reload(self):
        # Load initial words from file
        load_words_from_file('.datasets/test-word-list.txt', self.test_database_path)
        initial_words_by_length = load_arabic_datasources(database_path=self.test_database_path)

        # Force reload the datasource
        load_arabic_datasources(force_reload=True, database_path=self.test_database_path)
        words_by_length_after_insert = load_arabic_datasources(database_path=self.test_database_path)

        # Check that we have non-zero counts for lengths 1-6
        for length in range(1, 7):
            self.assertGreater(len(words_by_length_after_insert.get(length, [])), 0)

    def test_word_count_and_length(self):
        load_words_from_file('.datasets/test-word-list.txt', self.test_database_path)
        words_by_length = load_arabic_datasources(database_path=self.test_database_path)
        total_words = sum(len(words) for words in words_by_length.values())
        self.cursor.execute('SELECT COUNT(*) FROM words')
        db_word_count = self.cursor.fetchone()[0]
        self.assertEqual(total_words, db_word_count)

    def test_dataset_generation(self):
        # Set alphabet for this test
        alphabet = arabic_alphabet
        max_length = 2
        font_name = "arial.ttf"
        font_size = 30
        image_size = (512, 128)
        is_arabic = True

        # Create dataset and dataloader
        dataset = TextImageDataset(alphabet, max_length, font_name, font_size, image_size, is_arabic)
        
        # Test if dataset generates the correct number of samples
        self.assertEqual(len(dataset), len(list(dataset.texts)))

        # Test if dataset returns images and text
        for i in range(len(dataset)):
            image, text = dataset[i]
            self.assertIsInstance(image, torch.Tensor)
            self.assertIsInstance(text, str)

if __name__ == "__main__":
    unittest.main()
