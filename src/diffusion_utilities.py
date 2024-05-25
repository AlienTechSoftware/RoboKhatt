# -*- coding: utf-8 -*-
# .\src\diffusion_utilities.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display
import numpy as np
import os
import logging

from src.lang_utilities import generate_all_combinations, arabic_alphabet

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_res=False):
        super().__init__()
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            if self.same_channels:
                out = x + x2
            else:
                shortcut = nn.Conv2d(x.shape[1], x2.shape[1], kernel_size=1, stride=1, padding=0).to(x.device)
                out = shortcut(x) + x2
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2

class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            ResidualConvBlock(in_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        return self.model(x)

class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            ResidualConvBlock(out_channels * 2, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        )

    def forward(self, x, skip):
        x = self.up(x)
        logger.debug(f"UnetUp: after up - x shape: {x.shape}, skip shape: {skip.shape}")
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        x = F.pad(x, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        x = torch.cat((skip, x), dim=1)
        logger.debug(f"UnetUp: after cat - x shape: {x.shape}")
        return self.conv(x)

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super().__init__()
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=256, n_cfeat=10, height=128):
        super().__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.h = height

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.to_vec = nn.Sequential(nn.AvgPool2d(kernel_size=4), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, 1 * n_feat)
        self.contextembed1 = EmbedFC(n_cfeat, 2 * n_feat)
        self.contextembed2 = EmbedFC(n_cfeat, 1 * n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, kernel_size=(height // 32), stride=(height // 32)),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, t, c=None):
        logger.debug(f"ContextUnet: input x shape: {x.shape}, t shape: {t.shape}, c shape: {c.shape if c is not None else 'None'}")
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)
        logger.debug(f"ContextUnet: hiddenvec shape: {hiddenvec.shape}")

        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat).to(x.device)
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hiddenvec)
        logger.debug(f"ContextUnet: up1 shape: {up1.shape}")
        up2 = self.up1(cemb1 * up1 + temb1, down2)
        logger.debug(f"ContextUnet: up2 shape: {up2.shape}")
        up3 = self.up2(cemb2 * up2 + temb2, down1)
        logger.debug(f"ContextUnet: up3 shape: {up3.shape}")
        out = self.out(torch.cat((up3, x), 1))
        logger.debug(f"ContextUnet: out shape: {out.shape}")
        return out

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

class TextImageDataset(Dataset):
    def __init__(self, alphabet, max_length, font_name, font_size, image_size, is_arabic=False):
        self.alphabet = alphabet
        self.max_length = max_length
        self.font_name = font_name
        self.font_size = font_size
        self.image_size = image_size
        self.is_arabic = is_arabic
        self.texts = generate_all_combinations(alphabet, max_length)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        image = render_text_image(text, self.image_size, self.font_name, self.font_size, self.is_arabic)
        image = self.transform(image)
        return image, text

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

# Parameters
alphabet = arabic_alphabet
max_length = 4
font_name = "arial.ttf"
font_size = 30
image_size = (512, 128)
is_arabic = True
save_path = ".generated/trained_model.pth"

dataset = TextImageDataset(alphabet, max_length, font_name, font_size, image_size, is_arabic)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ContextUnet(in_channels=3, n_feat=64, n_cfeat=5, height=image_size[1]).to(device)

trained_model = train_diffusion_model(model, dataloader, epochs=10, device=device, save_path=save_path)

# Example evaluation
def example_evaluation():
    text_to_generate = "اختبار"
    generated_image = evaluate_model(trained_model, text_to_generate, font_name, font_size, image_size, is_arabic, device)
    plt.imshow(generated_image.squeeze().cpu().permute(1, 2, 0).numpy())
    plt.show()
