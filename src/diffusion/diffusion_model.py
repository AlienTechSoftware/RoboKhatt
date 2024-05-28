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
    
    model.train()
    
    for epoch in range(epochs):
        for images, _ in dataloader:
            images = images.to(device)
            t = torch.rand(images.size(0), 1, 1, 1).to(device)
            
            optimizer.zero_grad()
            noise = torch.randn_like(images).to(device)
            noised_images = images * t + noise * (1 - t)
            
            # Log the devices of the input tensors and model parameters
            logger.debug(f"train_diffusion_model: noised_images device: {noised_images.device}, model parameters device: {next(model.parameters()).device}")

            predicted_noise = model(noised_images, t)
            loss = mse_loss(predicted_noise, noise)
            loss.backward()
            optimizer.step()
        
            logger.debug(f"Train - Epoch {epoch+1}, Loss: {loss.item()}")

        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item()}")

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")

    return model

def evaluate_model(model, text, font_name, font_size, image_size, is_arabic, device):
    model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    with torch.no_grad():
        text_image = render_text_image(text, image_size, font_name, font_size, 'center', is_arabic)
        text_tensor = transform(text_image).unsqueeze(0).to(device)
        t = torch.rand(1, 1, 1, 1).to(device)
        
        noise = torch.randn_like(text_tensor).to(device)
        noised_images = text_tensor * t + noise * (1 - t)

        predicted_noise = model(noised_images, t)
        
        generated_image = text_tensor - predicted_noise * t
        generated_image = generated_image.squeeze().cpu()

        return generated_image
