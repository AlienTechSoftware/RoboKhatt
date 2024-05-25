# -*- coding: utf-8 -*-
# .\tests\test_diffusion_utilities.py

import unittest
import torch
import os
from torch.utils.data import DataLoader
from PIL import Image
from src.diffusion_utilities import TextImageDataset, render_text_image, train_diffusion_model, evaluate_model
from src.lang_utilities import arabic_alphabet, generate_all_combinations
from src.diffusion_utilities import ContextUnet

class TestDiffusionUtilities(unittest.TestCase):

    def setUp(self):
        # Parameters for the tests
        self.alphabet = arabic_alphabet
        self.max_length = 4  # Reduced to 4 letters for testing
        self.font_name = "arial.ttf"
        self.font_size = 30
        self.image_size = (512, 128)
        self.is_arabic = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_path = ".generated/test_trained_model.pth"

        # Create dataset and dataloader
        self.dataset = TextImageDataset(self.alphabet, self.max_length, self.font_name, self.font_size, self.image_size, self.is_arabic)
        self.dataloader = DataLoader(self.dataset, batch_size=2, shuffle=True)

        # Initialize the model
        self.model = ContextUnet(in_channels=3, n_feat=64, n_cfeat=5, height=self.image_size[1]).to(self.device)

    def test_dataset_generation(self):
        # Test if dataset generates the correct number of samples
        self.assertEqual(len(self.dataset), len(generate_all_combinations(self.alphabet, self.max_length)))

        # Test if dataset returns images and text
        for i in range(len(self.dataset)):
            image, text = self.dataset[i]
            self.assertIsInstance(image, torch.Tensor)
            self.assertIsInstance(text, str)

    def test_render_text_image(self):
        # Test rendering of a text image
        text = "اختبار"
        image = render_text_image(text, self.image_size, self.font_name, self.font_size, self.is_arabic)
        self.assertIsInstance(image, Image.Image)

    def test_train_diffusion_model(self):
        # Test the training process
        trained_model = train_diffusion_model(self.model, self.dataloader, epochs=1, device=self.device, save_path=self.save_path)
        self.assertIsInstance(trained_model, ContextUnet)
        self.assertTrue(os.path.exists(self.save_path))

    def test_evaluate_model(self):
        # Test the evaluation process
        text_to_generate = "اختبار"
        generated_image = evaluate_model(self.model, text_to_generate, self.font_name, self.font_size, self.image_size, self.is_arabic, self.device)
        self.assertIsInstance(generated_image, torch.Tensor)

if __name__ == "__main__":
    unittest.main()
