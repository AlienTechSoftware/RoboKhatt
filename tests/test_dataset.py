# -*- coding: utf-8 -*-
# tests/test_dataset.py

import unittest
from src.diffusion.dataset import TextImageDataset
from PIL import ImageFont

class TestTextImageDataset(unittest.TestCase):
    def setUp(self):
        self.alphabet = [
            "\u0627", "\u0628", "\u062a", "\u062b", "\u062c", "\u062d", "\u062e",
            "\u062f", "\u0630", "\u0631", "\u0632", "\u0633", "\u0634", "\u0635",
            "\u0636", "\u0637", "\u0638", "\u0639", "\u063a", "\u0641", "\u0642",
            "\u0643", "\u0644", "\u0645", "\u0646", "\u0647", "\u0648", "\u064a",
            "\u0623", "\u0625", "\u0622", "\u0621", "\u0624", "\u0626"
        ]
        self.max_length = 2
        self.font_name = "arial"
        self.font_size = 32
        self.image_size = (512, 128)
        self.is_arabic = True

        # Check if font file exists
        try:
            self.font = ImageFont.truetype(self.font_name, self.font_size)
        except IOError:
            self.skipTest(f"Font '{self.font_name}' is not available.")

        self.dataset = TextImageDataset(
            self.alphabet, self.max_length, self.font_name, self.font_size, self.image_size, self.is_arabic
        )

    def test_len(self):
        expected_length = sum(len(self.alphabet) ** i for i in range(1, self.max_length + 1)) + 1
        self.assertEqual(len(self.dataset), expected_length)

    def test_getitem(self):
        for i in range(len(self.dataset)):
            image, text = self.dataset[i]
            self.assertEqual(image.shape, (3, self.image_size[1], self.image_size[0]))

    def test_image_rendering(self):
        for i in range(min(10, len(self.dataset))):
            image, text = self.dataset[i]
            image = image.numpy().transpose((1, 2, 0)) * 255
            self.assertEqual(image.shape, (self.image_size[1], self.image_size[0], 3))

if __name__ == '__main__':
    unittest.main()
