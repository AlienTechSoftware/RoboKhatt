# -*- coding: utf-8 -*-
# load_database.py

import os
from src.diffusion.dataset import create_database, load_words_from_file, load_generated_arabic_shapes

# Ensure the necessary directories and files exist
datasets_dir = '.datasets'
generated_dir = '.generated'
os.makedirs(datasets_dir, exist_ok=True)
os.makedirs(generated_dir, exist_ok=True)

# Path to the large dataset file
large_dataset_path = os.path.join(datasets_dir, 'arabic-words.txt')

if not os.path.exists(large_dataset_path):
    raise FileNotFoundError(f"The file {large_dataset_path} does not exist. Please ensure the large dataset file is present in the '.datasets' directory.")

# Create the database and load the large dataset
create_database()
load_words_from_file(large_dataset_path)
load_generated_arabic_shapes()

print("Database loaded with large dataset.")
