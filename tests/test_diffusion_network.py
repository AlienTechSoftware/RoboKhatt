# -*- coding: utf-8 -*-
# tests/test_diffusion_network.py

import unittest
import torch
from torch.utils.data import DataLoader
from src.img_utilities import TextImageDataset
from src.diffusion.diffusion_model import TextToImageModel
from src.lang_utilities import arabic_alphabet
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestDiffusionNetwork(unittest.TestCase):

    def setUp(self):
        self.font_name = "arial.ttf"
        self.font_size = 30
        self.image_size = (512, 128)  # Correct image size as a tuple
        self.is_arabic = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"TestDiffusionNetwork: running on {self.device}")

        # Initialize the model
        self.model = TextToImageModel(in_channels=3, n_feat=64, n_cfeat=10, height=self.image_size[1], vocab_size=10000, embed_dim=256).to(self.device)

    def test_model_forward(self):
        alphabet = arabic_alphabet
        max_length = 2
        dataset = TextImageDataset(alphabet, max_length, self.font_name, self.font_size, self.image_size, self.is_arabic)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        batch_size = 2
        images, _ = next(iter(dataloader))
        images = images.to(self.device)
        t = torch.rand(batch_size, 1, 1, 1).to(self.device)
        logger.debug(f"test_model_forward: images shape: {images.shape}, t shape: {t.shape}")
        output = self.model(images, t, torch.tensor([[0] * 10]).to(self.device))  # Pass a zero tensor for context if not available
        logger.debug(f"test_model_forward: output shape: {output.shape}")
        self.assertEqual(output.shape, images.shape)

    def test_intermediate_shapes(self):
        alphabet = arabic_alphabet
        max_length = 2
        dataset = TextImageDataset(alphabet, max_length, self.font_name, self.font_size, self.image_size, self.is_arabic)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        batch_size = 2
        images, _ = next(iter(dataloader))
        images = images.to(self.device)
        t = torch.rand(batch_size, 1, 1, 1).to(self.device)
        x = self.model.context_unet.init_conv(images)  # Access init_conv through context_unet
        logger.debug(f"test_intermediate_shapes: x shape after init_conv: {x.shape}")
        self.assertEqual(x.shape, (batch_size, 64, 128, 512))
        down1 = self.model.context_unet.down1(x)
        logger.debug(f"test_intermediate_shapes: down1 shape: {down1.shape}")
        self.assertEqual(down1.shape, (batch_size, 128, 64, 256))
        down2 = self.model.context_unet.down2(down1)
        logger.debug(f"test_intermediate_shapes: down2 shape: {down2.shape}")
        self.assertEqual(down2.shape, (batch_size, 256, 32, 128))
        down3 = self.model.context_unet.down3(down2)
        logger.debug(f"test_intermediate_shapes: down3 shape: {down3.shape}")
        self.assertEqual(down3.shape, (batch_size, 512, 16, 64))
        hiddenvec = self.model.context_unet.to_vec(down3)
        logger.debug(f"test_intermediate_shapes: hiddenvec shape: {hiddenvec.shape}")
        self.assertEqual(hiddenvec.shape, (batch_size, 512, 4, 16))

if __name__ == "__main__":
    unittest.main()
