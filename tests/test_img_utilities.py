# -*- coding: utf-8 -*-
# .\tests\test_img_utilities.py

import unittest
import os
from PIL import ImageFont
from src.img_utilities import render_text_image, TextImageDataset
from src.lang_utilities import generate_arabic_shapes_dynamic, arabic_alphabet
import easyocr

class TestRenderTextImage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.output_dir = ".generated"
        os.makedirs(cls.output_dir, exist_ok=True)
        cls.reader_en = easyocr.Reader(['en'], gpu=False)  # Initialize EasyOCR for English
        cls.reader_ar = easyocr.Reader(['ar'], gpu=False)  # Initialize EasyOCR for Arabic

    def setUp(self):
        self.text_en = "Test"
        self.text_ar = "اختبار"  # Arabic for "Test"
        self.image_size = (400, 300)
        self.font_name = "arial.ttf"
        self.font_size = 30
        self.positions = [
            'top-left', 'top-center', 'top-right',
            'center-left', 'center', 'center-right',
            'bottom-left', 'bottom-center', 'bottom-right'
        ]

    def test_render_text_image_english(self):
        for position in self.positions:
            with self.subTest(position=position):
                img = render_text_image(self.text_en, self.image_size, self.font_name, self.font_size, position)
                self.assertEqual(img.size, self.image_size)
                # Save the generated image
                img.save(os.path.join(self.output_dir, f"test_image_{position}_en.png"))

                # Simple validation that the image is not empty
                pixels = list(img.getdata())
                non_white_pixels = sum(1 for pixel in pixels if pixel != (255, 255, 255))
                self.assertGreater(non_white_pixels, 0)

                # Check if font is correctly loaded
                font = ImageFont.truetype(self.font_name, self.font_size)
                self.assertIsNotNone(font)

    def test_render_text_image_arabic(self):
        for position in self.positions:
            with self.subTest(position=position):
                img = render_text_image(self.text_ar, self.image_size, self.font_name, self.font_size, position, is_arabic=True)
                self.assertEqual(img.size, self.image_size)
                # Save the generated image
                img.save(os.path.join(self.output_dir, f"test_image_{position}_ar.png"))

                # Simple validation that the image is not empty
                pixels = list(img.getdata())
                non_white_pixels = sum(1 for pixel in pixels if pixel != (255, 255, 255))
                self.assertGreater(non_white_pixels, 0)

                # Check if font is correctly loaded
                font = ImageFont.truetype(self.font_name, self.font_size)
                self.assertIsNotNone(font)

    def test_invalid_position(self):
        img = render_text_image(self.text_en, self.image_size, self.font_name, self.font_size, 'invalid-position')
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
                img = render_text_image(text, self.image_size, self.font_name, self.font_size, 'bottom-right')
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
        img = render_text_image(self.text_en, self.image_size, self.font_name, self.font_size, 'bottom-right')
        self.assertEqual(img.size, self.image_size)
        img_path = os.path.join(self.output_dir, "test_image_ocr_en.png")
        img.save(img_path)

        # Use EasyOCR to perform OCR on the image
        result = self.reader_en.readtext(img_path)
        
        # Extract the recognized text
        recognized_text = " ".join([res[1] for res in result]).strip()

        # Verify the recognized text matches the original text
        self.assertEqual(recognized_text, self.text_en)

    def test_ocr_validation_for_arabic(self):
        img = render_text_image(self.text_ar, self.image_size, self.font_name, self.font_size, 'center', is_arabic=True)
        self.assertEqual(img.size, self.image_size)
        img_path = os.path.join(self.output_dir, "test_image_ocr_ar.png")
        img.save(img_path)

        # Use EasyOCR to perform OCR on the image
        result = self.reader_ar.readtext(img_path)
        
        # Extract the recognized text
        recognized_text = " ".join([res[1] for res in result]).strip()

        # Verify the recognized text matches the original text
        self.assertEqual(recognized_text, self.text_ar)

    def test_generate_arabic_shapes_dynamic(self):
        shapes_dict = generate_arabic_shapes_dynamic(arabic_alphabet)
        os.makedirs(self.output_dir, exist_ok=True)
        for char, shapes in shapes_dict.items():
            for shape in shapes:
                img = render_text_image(shape, self.image_size, self.font_name, self.font_size, 'bottom-right', is_arabic=True)
                self.assertEqual(img.size, self.image_size)
                # Save the generated image
                img.save(os.path.join(self.output_dir, f"test_shape_{char}_{shapes.index(shape)}.png"))

                # Simple validation that the image is not empty
                pixels = list(img.getdata())
                non_white_pixels = sum(1 for pixel in pixels if pixel != (255, 255, 255))
                self.assertGreater(non_white_pixels, 0)

if __name__ == "__main__":
    unittest.main()
