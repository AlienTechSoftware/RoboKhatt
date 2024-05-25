# -*- coding: utf-8 -*-
# .\tests\test_img_utilities.py

import unittest
import os
from PIL import ImageFont, ImageDraw, Image
from src.img_utilities import render_text_image
import easyocr

class TestRenderTextImage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.output_dir = ".generated"
        os.makedirs(cls.output_dir, exist_ok=True)
        cls.reader = easyocr.Reader(['en'], gpu=False)  # Initialize EasyOCR

    def setUp(self):
        self.text = "Test"
        self.image_size = (400, 300)
        self.font_name = "arial.ttf"
        self.font_size = 30
        self.positions = [
            'top-left', 'top-center', 'top-right',
            'center-left', 'center', 'center-right',
            'bottom-left', 'bottom-center', 'bottom-right'
        ]

    def test_render_text_image(self):
        for position in self.positions:
            with self.subTest(position=position):
                img = render_text_image(self.text, self.image_size, self.font_name, self.font_size, position)
                self.assertEqual(img.size, self.image_size)
                # Save the generated image
                img.save(os.path.join(self.output_dir, f"test_image_{position}.png"))

                # Simple validation that the image is not empty
                pixels = list(img.getdata())
                non_white_pixels = sum(1 for pixel in pixels if pixel != (255, 255, 255))
                self.assertGreater(non_white_pixels, 0)

                # Check if font is correctly loaded
                font = ImageFont.truetype(self.font_name, self.font_size)
                self.assertIsNotNone(font)

    def test_invalid_position(self):
        img = render_text_image(self.text, self.image_size, self.font_name, self.font_size, 'invalid-position')
        self.assertEqual(img.size, self.image_size)
        # Save the generated image
        img.save(os.path.join(self.output_dir, "test_image_invalid_position.png"))

        # Simple validation that the image is not empty
        pixels = list(img.getdata())
        non_white_pixels = sum(1 for pixel in pixels if pixel != (255, 255, 255))
        self.assertGreater(non_white_pixels, 0)

    def test_different_text_lengths(self):
        texts = ["Short", "A bit longer text", "This is a much longer text to test wrapping or clipping"]
        for text in texts:
            with self.subTest(text=text):
                img = render_text_image(text, self.image_size, self.font_name, self.font_size, 'center')
                self.assertEqual(img.size, self.image_size)
                # Save the generated image
                img.save(os.path.join(self.output_dir, f"test_image_{text.replace(' ', '_')}.png"))

                # Simple validation that the image is not empty
                pixels = list(img.getdata())
                non_white_pixels = sum(1 for pixel in pixels if pixel != (255, 255, 255))
                self.assertGreater(non_white_pixels, 0)

                font = ImageFont.truetype(self.font_name, self.font_size)
                self.assertIsNotNone(font)

    def test_ocr_validation_for_english(self):
        img = render_text_image(self.text, self.image_size, self.font_name, self.font_size, 'center')
        self.assertEqual(img.size, self.image_size)
        img_path = os.path.join(self.output_dir, "test_image_ocr.png")
        img.save(img_path)

        # Use EasyOCR to perform OCR on the image
        result = self.reader.readtext(img_path)
        
        # Extract the recognized text
        recognized_text = " ".join([res[1] for res in result]).strip()

        # Verify the recognized text matches the original text
        self.assertEqual(recognized_text, self.text)

if __name__ == "__main__":
    unittest.main()
