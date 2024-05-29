# -*- coding: utf-8 -*-
# src/diffusion/diffusion_model.py

import os
import torch
import torch.nn as nn
import logging
from .context_unet import ContextUnet
from torchvision import transforms
from src.img_utilities import render_text_image

logger = logging.getLogger(__name__)

class TextToImageModel(nn.Module):
    def __init__(self, in_channels, n_feat=64, n_cfeat=10, height=128, vocab_size=10000, embed_dim=256):
        super().__init__()
        # Text embedding layer
        self.text_embed = nn.Embedding(vocab_size, embed_dim)
        # Contextual UNet model for image generation
        self.context_unet = ContextUnet(in_channels, n_feat, n_cfeat, height)
        # Fully connected layer to map text embeddings to context features
        self.fc = nn.Linear(embed_dim, n_cfeat)

    def forward(self, x, t, text):
        # Embed the input text
        text_embedded = self.text_embed(text)
        # Average the embeddings over the sequence length dimension to get a fixed-size vector
        text_embedded = text_embedded.mean(dim=1)
        # Map text embeddings to context features
        text_condition = self.fc(text_embedded)
        # Pass the image, time step, and text condition to the UNet
        return self.context_unet(x, t, text_condition)

def train_diffusion_model(model, dataloader, epochs, device, save_path):
    """
    Trains the diffusion model.

    Args:
        model (nn.Module): The diffusion model to be trained.
        dataloader (DataLoader): DataLoader for the training data.
        epochs (int): Number of training epochs.
        device (torch.device): Device to train the model on (CPU or CUDA).
        save_path (str): Path to save the trained model.

    Returns:
        nn.Module: The trained model.
    """
    model.to(device)
    logger.info(f"Training on device: {device}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    mse_loss = nn.MSELoss()
    
    model.train()
    
    for epoch in range(epochs):
        for images, texts in dataloader:
            # Move images to the specified device
            images = images.to(device)
            
            # Convert texts to indices and then to a tensor
            text_indices = [text_to_indices(text) for text in texts]
            text_indices = torch.tensor(text_indices).to(device)
            
            # Generate random time steps
            t = torch.rand(images.size(0), 1, 1, 1).to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Add noise to the images
            noise = torch.randn_like(images).to(device)
            noised_images = images * t + noise * (1 - t)
            
            # Log the devices of the input tensors and model parameters
            logger.debug(f"train_diffusion_model: noised_images device: {noised_images.device}, model parameters device: {next(model.parameters()).device}")

            # Predict the noise
            predicted_noise = model(noised_images, t, text_indices)  # Pass texts as conditioning input
            
            # Compute the loss
            loss = mse_loss(predicted_noise, noise)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
        
            logger.debug(f"Train - Epoch {epoch+1}, Loss: {loss.item()}")

        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item()}")

    # Save the model
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")

    return model

def evaluate_model(model, text, font_name, font_size, image_size, is_arabic, device):
    """
    Evaluates the model by generating an image from the given text.

    Args:
        model (nn.Module): The trained diffusion model.
        text (str): The input text to generate an image for.
        font_name (str): The name of the font to use.
        font_size (int): The font size.
        image_size (tuple): The size of the generated image.
        is_arabic (bool): Whether the text is in Arabic.
        device (torch.device): Device to perform the evaluation on (CPU or CUDA).

    Returns:
        PIL.Image: The generated image.
    """
    model.to(device)
    model.eval()
    
    # Transform to normalize the image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    with torch.no_grad():
        # Render text image and convert to tensor using bottom-right alignment for Arabic
        text_image = render_text_image(text, image_size, font_name, font_size, 'bottom-right' if is_arabic else 'center', is_arabic)
        text_tensor = transform(text_image).unsqueeze(0).to(device)
        
        # Convert text to indices, padding or truncating to max length of 16
        text_indices = torch.tensor(text_to_indices(text, max_length=16)).unsqueeze(0).to(device)
        
        # Generate random time step tensor
        t = torch.rand(1, 1, 1, 1).to(device)
        
        # Generate random noise
        noise = torch.randn_like(text_tensor).to(device)
        noised_images = text_tensor * t + noise * (1 - t)

        # Predict the noise
        predicted_noise = model(noised_images, t, text_indices)
        
        # Generate the image by removing the predicted noise
        generated_image = text_tensor - predicted_noise * t
        generated_image = generated_image.squeeze().cpu()

        return generated_image

def load_model(model_path, device):
    """
    Loads a trained model from the specified path.

    Args:
        model_path (str): Path to the saved model.
        device (torch.device): Device to load the model on (CPU or CUDA).

    Returns:
        nn.Module: The loaded model.
    """
    # Load the TextToImageModel with the correct parameters
    model = TextToImageModel(in_channels=3, n_feat=64, n_cfeat=10, height=128, vocab_size=10000, embed_dim=256)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set model to evaluation mode
    return model

# Helper function to convert text to indices
def text_to_indices(text, max_length=16, vocab={'<PAD>': 0, '<UNK>': 1}):
    """
    Converts text to a list of indices, padding or truncating to the specified maximum length.

    Args:
        text (str): The input text to convert.
        max_length (int): The maximum length of the output list.
        vocab (dict): A dictionary mapping characters to indices.

    Returns:
        list: A list of indices representing the text.
    """
    indices = [vocab.get(char, vocab['<UNK>']) for char in text]
    indices += [vocab['<PAD>']] * (max_length - len(indices))
    return indices[:max_length]
