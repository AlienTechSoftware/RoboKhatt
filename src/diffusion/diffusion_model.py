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
    # Move the model to the specified device (CPU or CUDA)
    model.to(device)
    logger.info(f"Training on device: {device}")
    
    # Set up the optimizer with Adam algorithm and a learning rate of 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Define Mean Squared Error (MSE) loss function
    mse_loss = nn.MSELoss()
    
    # Set the model to training mode
    model.train()
    
    # Training loop over the specified number of epochs
    for epoch in range(epochs):
        # Iterate over the DataLoader
        for images, _ in dataloader:
            # Move images to the specified device
            images = images.to(device)
            
            # Generate a random time step tensor
            t = torch.rand(images.size(0), 1, 1, 1).to(device)  # Random time step tensor

            # Zero the gradients before backpropagation
            optimizer.zero_grad()
            
            # Generate noise tensor of the same shape as images
            noise = torch.randn_like(images).to(device)
            
            # Create noised images by blending images and noise based on the time step t
            # The images are multiplied by t, and noise is multiplied by (1 - t)
            # This creates a mixture where t controls the influence of the original image and noise
            noised_images = images * t + noise * (1 - t)
            
            # Forward pass through the model
            # The model aims to predict the noise added to the images
            predicted_noise = model(noised_images, t)
            
            # Calculate the loss between the predicted noise and the actual noise
            # Using Mean Squared Error (MSE) ensures that the model learns to accurately predict the noise
            # This provides a stable target for the model to learn from
            loss = mse_loss(predicted_noise, noise)
            
            # Backpropagate the loss and update the model parameters
            loss.backward()
            optimizer.step()
        
            # Log the loss for monitoring training progress
            logger.debug(f"Train - Epoch {epoch+1}, Loss: {loss.item()}")

        # Print the loss at the end of each epoch
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item()}")

    # Ensure the directory exists before saving the model
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    
    # Save the trained model's state dictionary
    torch.save(model.state_dict(), save_path)

    # Verify if the file was saved
    if os.path.exists(save_path):
        logger.info(f"Verified: Model file exists at {save_path}")
    else:
        logger.error(f"Error: Model file does not exist at {save_path}")

    return model

@torch.no_grad()
def evaluate_model(model, text, font_name, font_size, image_size, is_arabic, device):
    # Render the input text as an image
    image = render_text_image(text, image_size, font_name, font_size, 'bottom-right', is_arabic)
    image = transforms.ToTensor()(image).unsqueeze(0).to(device)
    t = torch.ones(1, 1, 1, 1).to(device)
    logger.debug(f"evaluate_model: input image shape: {image.shape}, t shape: {t.shape}")
    with torch.no_grad():
        model.to(device)  # Ensure the model is on the device
        model.eval()  # Set the model to evaluation mode
        # Forward pass through the model
        output = model(image, t)
        logger.debug(f"evaluate_model: output shape: {output.shape}")
    return output