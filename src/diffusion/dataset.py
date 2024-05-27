# src/diffusion/dataset.py

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from src.lang_utilities import generate_arabic_shapes_dynamic, arabic_alphabet, arabic_digits
import sqlite3
import os

# Define the directory and ensure it exists
generated_dir = '.generated'
os.makedirs(generated_dir, exist_ok=True)

# Connect to SQLite database (or create it if it doesn't exist)
db_path = os.path.join(generated_dir, 'arabic_words.db')

# Unicode control characters
ZWJ = "\u200D"  # Zero Width Joiner
ZWNJ = "\u200C"  # Zero Width Non-Joiner

def load_arabic_datasources(force_reload=False):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create table for storing Arabic words if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS words (
        id INTEGER PRIMARY KEY,
        word TEXT NOT NULL
    )
    ''')
    conn.commit()

    if force_reload:
        # Clear existing data
        cursor.execute('DELETE FROM words')
        conn.commit()

    # Check if the table is empty
    cursor.execute('SELECT COUNT(*) FROM words')
    count = cursor.fetchone()[0]

    if count == 0:
        # Generate Arabic shapes
        shapes_dict = generate_arabic_shapes_dynamic(arabic_alphabet + arabic_digits)

        # Insert words into the database
        for char, shapes in shapes_dict.items():
            for shape in shapes:
                cursor.execute('INSERT INTO words (word) VALUES (?)', (shape,))
        conn.commit()

    # Retrieve words from the database and organize them by length
    cursor.execute('SELECT word FROM words')
    words = cursor.fetchall()
    words_by_length = {}
    for word in words:
        length = len(word[0].replace(ZWJ, '').replace(ZWNJ, ''))  # Ignore control characters in length
        if length not in words_by_length:
            words_by_length[length] = []
        words_by_length[length].append(word[0])

    conn.close()
    return words_by_length

def generate_all_combinations():
    words_by_length = load_arabic_datasources()
    for length, words in words_by_length.items():
        for word in words:
            yield word

class TextImageDataset(Dataset):
    def __init__(self, alphabet, max_length, font_name, font_size, image_size, is_arabic=False):
        self.alphabet = alphabet
        self.max_length = max_length
        self.font_name = font_name
        self.font_size = font_size
        self.image_size = image_size
        self.is_arabic = is_arabic
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
