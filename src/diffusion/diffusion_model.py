# -*- coding: utf-8 -*-
# src/diffusion/diffusion_model.py

import torch
import torch.nn as nn
import os
import logging
from src.img_utilities import render_text_image
from torchvision import transforms

logger = logging.getLogger(__name__)

def train_diffusion_model(model, dataloader, epochs, device, save_path):
    model.to(device)
    logger.info(f"Training on device: {device}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    mse_loss = nn.MSELoss()
    model.to(device)  # Ensure the model is on the device
    model.train()
    
    for epoch in range(epochs):
        for images, _ in dataloader:
            images = images.to(device)
            t = torch.rand(images.size(0), 1, 1, 1).to(device)
            optimizer.zero_grad()
            noise = torch.randn_like(images)
            # noised_images = images * t + noise * (1 - t)          
            noised_images = images + t * torch.randn_like(images).to(device)
            predicted_noise = model(noised_images, t)
            # loss = mse_loss(predicted_noise, noise)
            loss = ((predicted_noise - torch.randn_like(predicted_noise).to(device)) ** 2).mean()
            loss.backward()
            optimizer.step()
        
            logger.debug(f"Train - Epoch {epoch+1}, Loss: {loss.item()}")

        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item()}")

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")
    return model

@torch.no_grad()
def evaluate_model(model, text, font_name, font_size, image_size, is_arabic, device):
    image = render_text_image(text, image_size, font_name, font_size, 'bottom-right', is_arabic)
    image = transforms.ToTensor()(image).unsqueeze(0).to(device)
    t = torch.ones(1, 1, 1, 1).to(device)
    logger.debug(f"evaluate_model: input image shape: {image.shape}, t shape: {t.shape}")
    with torch.no_grad():
        model.to(device)  # Ensure the model is on the device
        model.eval()
        output = model(image, t)
        logger.debug(f"evaluate_model: output shape: {output.shape}")
    return output
