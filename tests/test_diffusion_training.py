# -*- coding: utf-8 -*-
# tests/test_diffusion_training.py

import unittest
import torch
import os
from torch.utils.data import DataLoader
from torch.profiler import profile, ProfilerActivity
from src.img_utilities import TextImageDataset
from src.diffusion.diffusion_model import train_diffusion_model, TextToImageModel
from src.lang_utilities import arabic_alphabet
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestDiffusionTraining(unittest.TestCase):

    def setUp(self):
        self.font_name = "arial.ttf"
        self.font_size = 30
        self.image_size = (512, 128)  # Correct image size as a tuple
        self.is_arabic = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"TestDiffusionTraining: running on {self.device}")
        self.save_path = ".generated/test_trained_model.pth"

        self.model = TextToImageModel(in_channels=3, n_feat=64, n_cfeat=10, height=self.image_size[1], vocab_size=10000, embed_dim=256).to(self.device)

    def _train_diffusion_model_for_device(self, device):
        alphabet = arabic_alphabet
        max_length = 2
        dataset = TextImageDataset(alphabet, max_length, self.font_name, self.font_size, self.image_size, self.is_arabic)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        save_path = f".generated/test_trained_model_{device}.pth".replace(":", "_")
        logger.info(f"Model will be saved to: {save_path}")

        self.model.to(device)

        trained_model = train_diffusion_model(self.model, dataloader, epochs=3, device=device, save_path=save_path)
        trained_model.to(device)

        self.assertIsNotNone(trained_model)
        self.assertTrue(os.path.exists(save_path), f"Model not found at {save_path}")

    @unittest.skip("Skipping training test temporarily")
    def test_train_diffusion_model_cpu(self):
        with profile(activities=[ProfilerActivity.CPU], schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1), on_trace_ready=torch.profiler.tensorboard_trace_handler('./log_cpu')) as prof:
            self._train_diffusion_model_for_device("cpu")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
    @unittest.skip("Skipping training test temporarily")
    def test_train_diffusion_model_cuda(self):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1), on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')) as prof:
            self._train_diffusion_model_for_device("cuda:0")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    @unittest.skip("Skipping multiple_epochs test temporarily")
    def test_training_with_multiple_epochs(self):
        alphabet = arabic_alphabet
        max_length = 2
        dataset = TextImageDataset(alphabet, max_length, self.font_name, self.font_size, self.image_size, self.is_arabic)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        trained_model = train_diffusion_model(self.model, dataloader, epochs=3, device=self.device, save_path=self.save_path)
        self.assertIsNotNone(trained_model)
        self.assertTrue(os.path.exists(self.save_path))

    def tearDown(self):
        if os.path.exists(self.save_path):
            os.remove(self.save_path)

if __name__ == "__main__":
    unittest.main()
