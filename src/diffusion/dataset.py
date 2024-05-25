# -*- coding: utf-8 -*-
# src/diffusion/dataset.py

from torch.utils.data import Dataset
from torchvision import transforms
from .utilities import render_text_image
from src.lang_utilities import generate_all_combinations

class TextImageDataset(Dataset):
    def __init__(self, alphabet, max_length, font_name, font_size, image_size, is_arabic=False):
        self.alphabet = alphabet
        self.max_length = max_length
        self.font_name = font_name
        self.font_size = font_size
        self.image_size = image_size
        self.is_arabic = is_arabic
        self.texts = generate_all_combinations(alphabet, max_length)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        image = render_text_image(text, self.image_size, self.font_name, self.font_size, self.is_arabic)
        image = self.transform(image)
        return image, text
