# -*- coding: utf-8 -*-
# load_database.py

import os
import sqlite3
from src.diffusion.dataset import create_database, load_words_from_file, load_generated_arabic_shapes

# Ensure the necessary directories and files exist
datasets_dir = '.datasets'
generated_dir = '.generated'
os.makedirs(datasets_dir, exist_ok=True)
os.makedirs(generated_dir, exist_ok=True)

# Path to the large dataset file
words_dataset_path = os.path.join(datasets_dir, 'large-word-list.txt')
database_path = os.path.join(generated_dir, 'arabic_words.db')

if not os.path.exists(words_dataset_path):
    raise FileNotFoundError(f"The file {words_dataset_path} does not exist. Please ensure the large dataset file is present in the '.datasets' directory.")

# Delete the database file if it exists
if os.path.exists(database_path):
    os.remove(database_path)

# Create the database and load the large dataset
create_database(database_path)
load_words_from_file(words_dataset_path, database_path)
load_generated_arabic_shapes(database_path)

# Connect to the database to count the records
conn = sqlite3.connect(database_path)
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM words')
total_record_count = cursor.fetchone()[0]

# Group by arabic_letter_count
cursor.execute('SELECT arabic_letter_count, COUNT(*) FROM words GROUP BY arabic_letter_count')
record_counts_by_letter = cursor.fetchall()
conn.close()

print(f"Database loaded with dataset from {words_dataset_path}")
print(f"Total number of records in the database: {total_record_count}")
print("Record counts by Arabic letter count:")
for count, num_records in record_counts_by_letter:
    print(f"Arabic letter count {count}: {num_records} records")
