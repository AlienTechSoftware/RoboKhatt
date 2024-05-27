# src/diffusion/dataset.py

import sqlite3
import os
from src.lang_utilities import generate_arabic_shapes_dynamic, arabic_alphabet, count_arabic_letters

class ArabicDataset:
    def __init__(self, db_path='.generated/test_arabic_words.db'):
        self.db_path = db_path
        self.create_database()

    def create_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS words (
            id INTEGER PRIMARY KEY,
            word TEXT NOT NULL,
            char_length INTEGER,
            arabic_letter_count INTEGER
        )
        ''')
        conn.commit()
        conn.close()

    def load_words_from_file(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                word = line.strip()
                char_length = len(word)
                arabic_letter_count = count_arabic_letters(word)
                cursor.execute('INSERT INTO words (word, char_length, arabic_letter_count) VALUES (?, ?, ?)',
                               (word, char_length, arabic_letter_count))
        conn.commit()
        conn.close()

    def load_generated_arabic_shapes(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Generate Arabic shapes
        shapes_dict = generate_arabic_shapes_dynamic(arabic_alphabet)
        for char, shapes in shapes_dict.items():
            for shape in shapes:
                char_length = len(shape)
                arabic_letter_count = count_arabic_letters(shape)
                cursor.execute('INSERT INTO words (word, char_length, arabic_letter_count) VALUES (?, ?, ?)',
                               (shape, char_length, arabic_letter_count))
        conn.commit()
        conn.close()

    def load_arabic_datasources(self, force_reload=False):
        """
        Load and prepare Arabic data sources, generating the dynamic shapes if necessary.

        :param force_reload: Boolean flag to force reload and clean up existing data.
        :return: Dictionary with word lengths as keys and lists of words of that length as values.
        """
        self.create_database()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if force_reload:
            # Clear existing data
            cursor.execute('DELETE FROM words')
            conn.commit()

            # Reload the data from files
            self.load_generated_arabic_shapes()
            self.load_words_from_file('.datasets/test-word-list.txt')

        # Retrieve words from the database and organize them by length
        cursor.execute('SELECT word, arabic_letter_count FROM words')
        words = cursor.fetchall()
        words_by_length = {}
        for word, count in words:
            if count not in words_by_length:
                words_by_length[count] = []
            words_by_length[count].append(word)

        conn.close()
        return words_by_length
