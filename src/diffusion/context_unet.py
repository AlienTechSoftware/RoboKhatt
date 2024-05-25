# -*- coding: utf-8 -*-
# src/diffusion/context_unet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from .model_blocks import ResidualConvBlock, UnetDown, UnetUp, EmbedFC

logger = logging.getLogger(__name__)

class ContextUnet(nn.Module):
    """
    @brief A neural network model designed for image processing tasks, following a U-Net architecture.

    @param in_channels The number of input channels in the images.
    @param n_feat The number of features for the convolution layers.
    @param n_cfeat The number of context features.
    @param height The height of the input images.

    The class consists of several layers including:
    - init_conv: An initial convolution block to process the input image.
    - down1, down2, down3: Downsampling blocks that reduce the spatial dimensions of the image while increasing the number of channels.
    - to_vec: A bottleneck layer that transforms the downsampled image into a vector.
    - timeembed1, timeembed2: Embedding layers for the time steps.
    - contextembed1, contextembed2: Embedding layers for the context information.
    - up0, up1, up2, up3: Upsampling blocks that restore the spatial dimensions of the image while combining information from the downsampling path.
    - out: A final convolution block that produces the output image.
    """
    def __init__(self, in_channels, n_feat=64, n_cfeat=10, height=128):
        super().__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.h = height

        # Initial convolution block
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        # Downsampling layers
        self.down1 = UnetDown(n_feat, n_feat * 2)
        self.down2 = UnetDown(n_feat * 2, n_feat * 4)
        self.down3 = UnetDown(n_feat * 4, n_feat * 8)

        # Vector transformation for the bottleneck
        self.to_vec = nn.Sequential(nn.AvgPool2d(kernel_size=4), nn.GELU())

        # Time embedding layers
        self.timeembed1 = EmbedFC(1, n_feat * 8)
        self.timeembed2 = EmbedFC(1, n_feat * 4)

        # Context embedding layers
        self.contextembed1 = EmbedFC(n_cfeat, n_feat * 8)
        self.contextembed2 = EmbedFC(n_cfeat, n_feat * 4)

        # Upsampling layers
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(n_feat * 8, n_feat * 8, kernel_size=(height // 32), stride=(height // 32)),
            nn.GroupNorm(8, n_feat * 8),
            nn.ReLU(),
        )
        self.up1 = UnetUp(n_feat * 8, n_feat * 4)  # Ensure consistent channel sizes
        self.up2 = UnetUp(n_feat * 4, n_feat * 2)  # Ensure consistent channel sizes
        self.up3 = UnetUp(n_feat * 2, n_feat)  # Ensure consistent channel sizes

        # Output convolution block
        self.out = nn.Sequential(
            nn.Conv2d(n_feat * 2, n_feat, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, t, c=None):
        """
        @brief Forward pass of the ContextUnet model.

        @param x The input image tensor.
        @param t The time step tensor.
        @param c The optional context tensor.

        @return The output image tensor after processing through the U-Net architecture.

        The forward pass includes the following steps:
        - Initial Convolution: Process the input image through the initial convolution block.
        - Downsampling: Reduce the spatial dimensions of the image through three downsampling blocks.
        - Bottleneck: Transform the downsampled image into a vector.
        - Embedding Generation: Generate context and time embeddings.
        - Upsampling: Restore the spatial dimensions of the image while combining information from the downsampling path.
        - Output Generation: Produce the final output image through the output convolution block.
        """

        # Log the shapes of the input tensors
        logger.debug(f"ContextUnet: input x shape: {x.shape}, t shape: {t.shape}, c shape: {c.shape if c is not None else 'None'}")

        # Initial convolution
        x = self.init_conv(x)
        logger.debug(f"ContextUnet: after init_conv - x shape: {x.shape}")

        # Downsampling
        down1 = self.down1(x)
        logger.debug(f"ContextUnet: after down1 - down1 shape: {down1.shape}")
        down2 = self.down2(down1)
        logger.debug(f"ContextUnet: after down2 - down2 shape: {down2.shape}")
        down3 = self.down3(down2)
        logger.debug(f"ContextUnet: after down3 - down3 shape: {down3.shape}")

        # Bottleneck transformation
        hiddenvec = self.to_vec(down3)
        logger.debug(f"ContextUnet: hiddenvec shape: {hiddenvec.shape}")

        # Handle context tensor if not provided
        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat).to(x.device)
        
        # Generate context and time embeddings
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 8, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 8, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat * 4, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat * 4, 1, 1)
        
        # Log the shapes of the embeddings
        logger.debug(f"ContextUnet: cemb1 shape: {cemb1.shape}, temb1 shape: {temb1.shape}")
        logger.debug(f"ContextUnet: cemb2 shape: {cemb2.shape}, temb2 shape: {temb2.shape}")

        # Upsampling
        up1 = self.up0(hiddenvec)
        logger.debug(f"ContextUnet: up1 shape: {up1.shape}")
        up2 = self.up1(cemb1 * up1 + temb1, down3)
        logger.debug(f"ContextUnet: up2 shape: {up2.shape}")
        up3 = self.up2(cemb2 * up2 + temb2, down2)
        logger.debug(f"ContextUnet: up3 shape: {up3.shape}")
        up4 = self.up3(up3, down1)
        logger.debug(f"ContextUnet: up4 shape: {up4.shape}")

        # Final output layer
        out = self.out(torch.cat((up4, x), 1))
        logger.debug(f"ContextUnet: concatenated out shape: {torch.cat((up4, x), 1).shape}")
        logger.debug(f"ContextUnet: out shape: {out.shape}")

        return out
