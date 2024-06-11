# -*- coding: utf-8 -*-
# train_production_model.py

import os
import json
import torch
from torch.utils.data import DataLoader
from src.diffusion.diffusion_model import TextToImageModel, train_diffusion_model
from src.img_utilities import TextImageDataset
from src.lang_utilities import arabic_alphabet
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load configuration from the JSON file in the same directory
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Load parameters from config
    text_image_dataset_config = config['TextImageDataset']
    context_unet_config = config['ContextUnet']
    training_config = config['DiffusionModelTraining']

    # Create the dataset and dataloader
    dataset = TextImageDataset(
        alphabet=arabic_alphabet,
        max_length=text_image_dataset_config['max_length'],
        font_name=text_image_dataset_config['font_name'],
        font_size=text_image_dataset_config['font_size'],
        image_size=tuple(text_image_dataset_config['image_size']),
        is_arabic=text_image_dataset_config['is_arabic']
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize the model
    model = TextToImageModel(
        in_channels=context_unet_config['in_channels'],
        n_feat=context_unet_config['n_feat'],
        n_cfeat=context_unet_config['n_cfeat'],
        height=context_unet_config['height'],
        vocab_size=10000,  # Assuming a fixed vocab size
        embed_dim=256      # Assuming a fixed embed dimension
    )

    # Determine device and save path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = "prod_trained_model_cuda.pth" if torch.cuda.is_available() else "prod_trained_model_cpu.pth"
    logger.info(f"Training on device: {device}, model will be saved to: {save_path}")

    # Train the model
    train_diffusion_model(
        model=model,
        dataloader=dataloader,
        epochs=training_config['epochs'],
        device=device,
        save_path=save_path
    )

if __name__ == "__main__":
    main()
