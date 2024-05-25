# -*- coding: utf-8 -*-
# src/diffusion/model_blocks.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_res=False):
        super().__init__()
        self.is_res = is_res
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if is_res and in_channels != out_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.res_conv = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.is_res:
            if self.res_conv is not None:
                identity = self.res_conv(identity)
            out += identity
        out = self.relu(out)
        return out

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
    """
    @brief An upsampling block used in the ContextUnet class to restore the spatial dimensions of the image.

    @param in_channels The number of input channels for the upsampling layer.
    @param out_channels The number of output channels for the upsampling layer.

    The class consists of:
    - up: A transposed convolution layer that increases the spatial dimensions of the input.
    - conv: A sequence of two residual convolution blocks that refine the upsampled image.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            ResidualConvBlock(out_channels * 2, out_channels, is_res=True),
            ResidualConvBlock(out_channels, out_channels, is_res=True)
        )

    def forward(self, x, skip):
        """
        @brief Forward pass of the UnetUp block.

        @param x The upsampled image tensor.
        @param skip The corresponding skip connection tensor from the downsampling path.

        @return The refined upsampled image tensor.

        The forward pass includes the following steps:
        - Upsampling: Increase the spatial dimensions of the input tensor using the transposed convolution layer.
        - Padding: If necessary, pad the upsampled image to match the spatial dimensions of the skip connection.
        - Concatenation: Concatenate the upsampled image and the skip connection along the channel dimension.
        - Convolution: Process the concatenated image through the residual convolution blocks to produce the final output.
        """

        logger.debug(f"UnetUp: before up - x shape: {x.shape}, skip shape: {skip.shape}")
        x = self.up(x)
        logger.debug(f"UnetUp: after up - x shape: {x.shape}")
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        logger.debug(f"UnetUp: padding - diffY: {diffY}, diffX: {diffX}")
        x = F.pad(x, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        logger.debug(f"UnetUp: after padding - x shape: {x.shape}")
        x = torch.cat((skip, x), dim=1)
        logger.debug(f"UnetUp: after cat - x shape: {x.shape}")
        x = self.conv(x)
        logger.debug(f"UnetUp: after conv - x shape: {x.shape}")
        return x


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
