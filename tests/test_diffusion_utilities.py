# -*- coding: utf-8 -*-
# .\tests\test_diffusion_utilities.py

import unittest
import torch
import os
from torch.utils.data import DataLoader
from src.diffusion import TextImageDataset, render_text_image, ContextUnet
from src.lang_utilities import arabic_alphabet, generate_all_combinations
from PIL import Image

class TestDiffusionUtilities(unittest.TestCase):

    def setUp(self):
        # Parameters for the tests
        self.alphabet = arabic_alphabet
        self.max_length = 2  # Reduced to 2 letters for testing
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
        text = "بم"
        image = render_text_image(text, self.image_size, self.font_name, self.font_size, self.is_arabic)
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.size, self.image_size)

    @unittest.skip("Skipping training test temporarily")
    def test_train_diffusion_model(self):
        # Test the training process
        trained_model = train_diffusion_model(self.model, self.dataloader, epochs=1, device=self.device, save_path=self.save_path)
        self.assertIsNotNone(trained_model)
        self.assertTrue(os.path.exists(self.save_path))

    @unittest.skip("Skipping evaluation test temporarily")
    def test_evaluate_model(self):
        # Test the evaluation process
        text_to_generate = "بم"
        generated_image = evaluate_model(self.model, text_to_generate, self.font_name, self.font_size, self.image_size, self.is_arabic, self.device)
        self.assertIsInstance(generated_image, torch.Tensor)
        self.assertEqual(generated_image.shape[2:], self.image_size)

    def test_model_forward(self):
        # Test the forward pass of the model
        batch_size = 2
        images, _ = next(iter(self.dataloader))
        images = images.to(self.device)
        t = torch.rand(batch_size, 1, 1, 1).to(self.device)
        output = self.model(images, t)
        self.assertEqual(output.shape, images.shape)

    def test_intermediate_shapes(self):
        # Test the shapes of intermediate outputs
        batch_size = 2
        images, _ = next(iter(self.dataloader))
        images = images.to(self.device)
        t = torch.rand(batch_size, 1, 1, 1).to(self.device)
        x = self.model.init_conv(images)
        self.assertEqual(x.shape, (batch_size, 64, 128, 512))
        down1 = self.model.down1(x)
        self.assertEqual(down1.shape, (batch_size, 128, 64, 256))
        down2 = self.model.down2(down1)
        self.assertEqual(down2.shape, (batch_size, 256, 32, 128))
        down3 = self.model.down3(down2)
        self.assertEqual(down3.shape, (batch_size, 512, 16, 64))
        hiddenvec = self.model.to_vec(down3)
        self.assertEqual(hiddenvec.shape, (batch_size, 512, 4, 16))

    @unittest.skip("Skipping multi-epoch training test temporarily")
    def test_training_with_multiple_epochs(self):
        # Test the training process with multiple epochs
        trained_model = train_diffusion_model(self.model, self.dataloader, epochs=3, device=self.device, save_path=self.save_path)
        self.assertIsNotNone(trained_model)
        self.assertTrue(os.path.exists(self.save_path))

    @unittest.skip("Skipping Arabic test with 4 characters")
    def test_evaluate_arabic_text(self):
        # Test the evaluation process with Arabic text
        text_to_generate = "اختبار"
        generated_image = evaluate_model(self.model, text_to_generate, self.font_name, self.font_size, self.image_size, self.is_arabic, self.device)
        self.assertIsInstance(generated_image, torch.Tensor)
        self.assertEqual(generated_image.shape[2:], self.image_size)

    def tearDown(self):
        # Clean up any created files
        if os.path.exists(self.save_path):
            os.remove(self.save_path)

if __name__ == "__main__":
    unittest.main()
