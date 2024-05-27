# -*- coding: utf-8 -*-
# src/diffusion/dataset.py

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from src.lang_utilities import generate_arabic_shapes_dynamic, arabic_alphabet, count_arabic_letters
import sqlite3
import os

# Define the directory and ensure it exists
generated_dir = '.generated'
os.makedirs(generated_dir, exist_ok=True)

# Connect to SQLite database (or create it if it doesn't exist)
DATABASE_PATH = os.path.join(generated_dir, 'arabic_words.db')

def create_database(database_path=DATABASE_PATH):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    # Create table for storing Arabic words if it doesn't exist
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

def load_words_from_file(file_path, database_path=DATABASE_PATH):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    conn = sqlite3.connect(database_path)
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

def load_generated_arabic_shapes(database_path=DATABASE_PATH):
    conn = sqlite3.connect(database_path)
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

def load_arabic_datasources(force_reload=False, database_path=DATABASE_PATH):
    """
    Load and prepare Arabic data sources, generating the dynamic shapes if necessary.

    :param force_reload: Boolean flag to force reload and clean up existing data.
    :return: Dictionary with word lengths as keys and lists of words of that length as values.
    """
    create_database(database_path)
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    if force_reload:
        # Clear existing data
        cursor.execute('DELETE FROM words')
        conn.commit()

    # Check if the table is empty
    cursor.execute('SELECT COUNT(*) FROM words')
    count = cursor.fetchone()[0]

    if count == 0:
        load_generated_arabic_shapes(database_path)
        load_words_from_file('.datasets/word-list.txt', database_path)

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

def generate_all_combinations():
    words_by_length = load_arabic_datasources()
    for length, words in words_by_length.items():
        for word in words:
            yield word

class ArabicDataset(Dataset):
    def __init__(self, alphabet, max_length, font_name, font_size, image_size, is_arabic=False, database_path=DATABASE_PATH):
        self.alphabet = alphabet
        self.max_length = max_length
        self.font_name = font_name
        self.font_size = font_size
        self.image_size = image_size
        self.is_arabic = is_arabic
        self.database_path = database_path
        self.texts = list(generate_all_combinations())
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        image = self.render_text_image(text)
        image = self.transform(image)
        return image, text

    def render_text_image(self, text):
        # Create a blank image with white background
        image = Image.new('RGB', self.image_size, (255, 255, 255))
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype(self.font_name, self.font_size)
        except IOError:
            font = ImageFont.load_default()

        # Calculate text size and position
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_position = ((self.image_size[0] - text_width) // 2, (self.image_size[1] - text_height) // 2)

        # Draw the text on the image
        draw.text(text_position, text, font=font, fill=(0, 0, 0))

        return image
