# -*- coding: utf-8 -*-
# tests/test_diffusion_utilities.py

import unittest
import torch
import os
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
from src.img_utilities import TextImageDataset, render_text_image
from src.diffusion.diffusion_model import load_model, train_diffusion_model, evaluate_model, TextToImageModel
from src.lang_utilities import arabic_alphabet
from PIL import Image
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestDiffusionUtilities(unittest.TestCase):

    def setUp(self):
        # Parameters for the tests
        self.font_name = "arial.ttf"
        self.font_size = 30
        self.image_size = (512, 128)  # Correct image size as a tuple
        self.is_arabic = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"TestDiffusionUtilities: running on {self.device}")
        self.save_path = ".generated/test_trained_model.pth"

        # Initialize the model
        self.model = TextToImageModel(in_channels=3, n_feat=64, n_cfeat=10, height=self.image_size[1], vocab_size=10000, embed_dim=256).to(self.device)

    def test_dataset_generation(self):
        # Set alphabet for this test
        alphabet = arabic_alphabet
        max_length = 2

        # Create dataset and dataloader
        dataset = TextImageDataset(alphabet, max_length, self.font_name, self.font_size, self.image_size, self.is_arabic)
        
        # Test if dataset generates the correct number of samples
        self.assertEqual(len(dataset), len(list(dataset.texts)))

        # Test if dataset returns images and text
        for i in range(len(dataset)):
            image, text = dataset[i]
            self.assertIsInstance(image, torch.Tensor)
            self.assertIsInstance(text, str)

    def test_render_text_image(self):
        text = "تم"
        image = render_text_image(text, self.image_size, self.font_name, self.font_size, 'bottom-right', self.is_arabic)
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.size, self.image_size)

    def _train_diffusion_model_for_device(self, device):
        alphabet = arabic_alphabet
        max_length = 2
        dataset = TextImageDataset(alphabet, max_length, self.font_name, self.font_size, self.image_size, self.is_arabic)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        # Log the save path
        save_path = f".generated/test_trained_model_{device}.pth".replace(":", "_")
        logger.info(f"Model will be saved to: {save_path}")

        # Ensure the model is on the correct device
        self.model.to(device)

        # Training the model
        trained_model = train_diffusion_model(self.model, dataloader, epochs=1, device=device, save_path=save_path)
        
        # Ensure that the trained model is on the correct device
        trained_model.to(device)

        self.assertIsNotNone(trained_model)
        self.assertTrue(os.path.exists(save_path), f"Model not found at {save_path}")

    @unittest.skip("Skipping training test temporarily")
    def test_train_diffusion_model_cpu(self):
        with profile(activities=[ProfilerActivity.CPU],
                    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log_cpu')) as prof:
            self._train_diffusion_model_for_device("cpu")
        # Print profiler results for CPU
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
    @unittest.skip("Skipping training test temporarily")
    def test_train_diffusion_model_cuda(self):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')) as prof:
            self._train_diffusion_model_for_device("cuda:0")
        # Print profiler results for GPU
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


    @unittest.skip("Skipping training test temporarily")
    def test_evaluate_model(self):
        text_to_generate = "تم"
        generated_image = evaluate_model(self.model, text_to_generate, self.font_name, self.font_size, self.image_size, self.is_arabic, self.device)
        self.assertIsInstance(generated_image, torch.Tensor)
        self.assertEqual(generated_image.shape[2:], self.image_size)

    @unittest.skip("Skipping training test temporarily")
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

    @unittest.skip("Skipping evaluation test temporarily")
    def test_training_with_multiple_epochs(self):
        alphabet = arabic_alphabet
        max_length = 2
        dataset = TextImageDataset(alphabet, max_length, self.font_name, self.font_size, self.image_size, self.is_arabic)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        trained_model = train_diffusion_model(self.model, dataloader, epochs=3, device=self.device, save_path=self.save_path)
        self.assertIsNotNone(trained_model)
        self.assertTrue(os.path.exists(self.save_path))

    def _common_evaluate_model(self, model_path, device, text_to_generate):
        model = load_model(model_path, device)
        padded_text = self.pad_text_to_length(text_to_generate, 16)  # Adjust the length to 16 characters
        generated_image = evaluate_model(model, padded_text, self.font_name, self.font_size, self.image_size, self.is_arabic, device)
        self.assertIsInstance(generated_image, torch.Tensor)
        self.assertEqual(generated_image.shape[1:], self.image_size)

    @unittest.skip("Skipping training test temporarily")
    def test_evaluate_model_cpu(self):
        model_path = ".generated/trained_model_cpu.pth"
        self._common_evaluate_model(model_path, torch.device("cpu"), "اختبار")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
    @unittest.skip("Skipping training test temporarily")
    def test_evaluate_model_cuda(self):
        model_path = ".generated/trained_model_cuda.pth"
        self._common_evaluate_model(model_path, torch.device("cuda"), "اختبار")

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

    def tearDown(self):
        if os.path.exists(self.save_path):
            os.remove(self.save_path)

if __name__ == "__main__":
    unittest.main()
