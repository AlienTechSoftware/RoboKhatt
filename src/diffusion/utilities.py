# -*- coding: utf-8 -*-
# .\src\diffusion\utilities.py

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display
import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def render_text_image(text, image_size, font_name, font_size, is_arabic=False):
    img = Image.new('RGB', image_size, 'white')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_name, font_size)

    if is_arabic:
        reshaped_text = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped_text)
    else:
        bidi_text = text

    text_bbox = draw.textbbox((0, 0), bidi_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    if is_arabic:
        x, y = image_size[0] - text_width, image_size[1] - text_height - font.getmetrics()[1]
    else:
        x, y = 0, image_size[1] - text_height - font.getmetrics()[1]

    draw.text((x, y), bidi_text, font=font, fill='black')
    return img

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
    logger.debug(f"evaluate_model: input image shape: {image.shape}, t shape: {t.shape}")
    generated_image = model(image, t)
    logger.debug(f"evaluate_model: generated image shape: {generated_image.shape}")
    return generated_image
