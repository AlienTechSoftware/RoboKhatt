# -*- coding: utf-8 -*-
# tests/test_diffusion_utilities.py

import unittest
import torch
import os
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
from src.img_utilities import TextImageDataset
from src.diffusion.diffusion_model import load_model, train_diffusion_model, evaluate_model, TextToImageModel
from src.lang_utilities import arabic_alphabet
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def check_if_exists_or_train(model, model_path, device, font_name, font_size, image_size, is_arabic, epochs=3):
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
        font_name=font_name,
        font_size=font_size,
        image_size=image_size,
        is_arabic=is_arabic
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Ensure the model is on the correct device
    model.to(device)

    # Train the model
    train_diffusion_model(
        model,
        dataloader,
        epochs=epochs,
        device=device,
        save_path=model_path
    )
    return model

    def pad_text_to_length(self, text, length):
        """
        Pads the input text to the specified length.

        Args:
            text (str): The input text.
            length (int): The length to pad the text to.

        Returns:
            str: The padded text.
        """
        return text.ljust(length)
    
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

    def tearDown(self):
        if os.path.exists(self.save_path):
            os.remove(self.save_path)
if __name__ == "__main__":
    unittest.main()
