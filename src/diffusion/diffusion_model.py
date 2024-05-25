# -*- coding: utf-8 -*-
# src/diffusion/diffusion_model.py

import torch
import torch.nn as nn
import os
import logging

logger = logging.getLogger(__name__)

def train_diffusion_model(model, dataloader, epochs, device, save_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()
    model.train()

    for epoch in range(epochs):
        for images, _ in dataloader:
            images = images.to(device)
            optimizer.zero_grad()
            
            t = torch.rand(images.size(0), 1, 1, 1, device=device)
            noise = torch.randn_like(images)
            noised_images = images * t + noise * (1 - t)
            
            predicted_noise = model(noised_images, t)
            loss = mse_loss(predicted_noise, noise)
            
            loss.backward()
            optimizer.step()

            logger.debug(f"Train - Epoch {epoch+1}, Loss: {loss.item()}")
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item()}")

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(model.state_dict(), save_path)

    return model

@torch.no_grad()
def evaluate_model(model, text, font_name, font_size, image_size, is_arabic, device):
    model.eval()
    image = render_text_image(text, image_size, font_name, font_size, is_arabic)
    image = transforms.ToTensor()(image).unsqueeze(0).to(device)
    t = torch.tensor([1.0])[:, None, None, None].to(device)
    generated_image = model(image, t)
    return generated_image
