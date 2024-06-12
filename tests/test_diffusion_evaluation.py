# -*- coding: utf-8 -*-
# tests/test_diffusion_evaluation.py

import unittest
import torch
import os
from src.img_utilities import TextImageDataset
from src.diffusion.diffusion_model import load_model, evaluate_model, TextToImageModel
from src.lang_utilities import arabic_alphabet
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestDiffusionEvaluation(unittest.TestCase):

    def setUp(self):
        self.font_name = "arial.ttf"
        self.font_size = 30
        self.image_size = (512, 128)  # Correct image size as a tuple
        self.is_arabic = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"TestDiffusionEvaluation: running on {self.device}")
        self.save_path = ".generated/test_trained_model.pth"

        # Initialize the model
        self.model = TextToImageModel(in_channels=3, n_feat=64, n_cfeat=10, height=self.image_size[1], vocab_size=10000, embed_dim=256).to(self.device)

    def check_if_exists_or_train(self, model_path, device):
        if os.path.exists(model_path):
            try:
                logger.info(f"Model found at {model_path}. Attempting to load.")
                return load_model(model_path, device)
            except RuntimeError as e:
                logger.error(f"Error loading model: {e}")
                logger.info(f"Deleting the corrupted model file at {model_path}.")
                os.remove(model_path)

        logger.info(f"Model not found at {model_path} or corrupted. Training new model.")
        alphabet = arabic_alphabet
        max_length = 2
        dataset = TextImageDataset(
            alphabet=alphabet,
            max_length=max_length,
            font_name=self.font_name,
            font_size=self.font_size,
            image_size=self.image_size,
            is_arabic=self.is_arabic
        )
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        self.model.to(device)

        train_diffusion_model(
            self.model,
            dataloader,
            epochs=3,
            device=device,
            save_path=model_path
        )
        return self.model
            
    def _common_evaluate_model(self, model_path, device, text_to_generate):
        model = self.check_if_exists_or_train(model_path, device)
        padded_text = self.pad_text_to_length(text_to_generate, 16)  # Adjust the length to 16 characters
        generated_image = evaluate_model(model, padded_text, self.font_name, self.font_size, self.image_size, self.is_arabic, device)
        self.assertIsInstance(generated_image, torch.Tensor)
        self.assertTrue(
            generated_image.shape[1:] == self.image_size or generated_image.shape[1:] == self.image_size[::-1],
            f"Expected shape {self.image_size} or {self.image_size[::-1]}, but got {generated_image.shape[1:]}"
        )

    @unittest.skip("Skipping training test temporarily")
    def test_evaluate_model(self):
        text_to_generate = "تم"
        generated_image = evaluate_model(self.model, text_to_generate, self.font_name, self.font_size, self.image_size, self.is_arabic, self.device)
        self.assertIsInstance(generated_image, torch.Tensor)
        self.assertEqual(generated_image.shape[2:], self.image_size)

    @unittest.skip("Skipping evaluation test temporarily")
    def test_evaluate_model_cpu(self):
        model_path = ".generated/trained_model_cpu.pth"
        self._common_evaluate_model(model_path, torch.device("cpu"), "اختبار")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
    @unittest.skip("Skipping evaluation test temporarily")
    def test_evaluate_model_cuda(self):
        model_path = ".generated/trained_model_cuda.pth"
        self._common_evaluate_model(model_path, torch.device("cuda"), "اختبار")

    def pad_text_to_length(self, text, length):
        return text.ljust(length)

if __name__ == "__main__":
    unittest.main()
