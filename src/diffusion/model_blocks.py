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
        logger.debug(f"ResidualConvBlock: input x shape: {x.shape}")
        identity = x

        out = self.conv1(x)
        logger.debug(f"ResidualConvBlock: after conv1 - out shape: {out.shape}")

        out = self.bn1(out)
        logger.debug(f"ResidualConvBlock: after bn1 - out shape: {out.shape}")

        out = self.relu(out)
        logger.debug(f"ResidualConvBlock: after relu - out shape: {out.shape}")

        out = self.conv2(out)
        logger.debug(f"ResidualConvBlock: after conv2 - out shape: {out.shape}")

        out = self.bn2(out)
        logger.debug(f"ResidualConvBlock: after bn2 - out shape: {out.shape}")

        if self.is_res:
            if self.res_conv is not None:
                identity = self.res_conv(identity)
                logger.debug(f"ResidualConvBlock: after res_conv - identity shape: {identity.shape}")
            out += identity
            logger.debug(f"ResidualConvBlock: after adding identity - out shape: {out.shape}")

        out = self.relu(out)
        logger.debug(f"ResidualConvBlock: final output shape: {out.shape}")

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
    - match_channels: A convolution layer to match the channels of the skip connection.
    - conv: A sequence of two residual convolution blocks that refine the upsampled image.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # Adjust the number of channels from out_channels to in_channels to match the skip connection
        self.match_channels = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        logger.debug(f"UnetUp: init out_channels : {out_channels}")
        logger.debug(f"UnetUp: init in_channels : {in_channels}")
        logger.debug(f"UnetUp: init match_channels : {self.match_channels}")
        self.conv = nn.Sequential(
            ResidualConvBlock(out_channels * 2, out_channels, is_res=True),
            ResidualConvBlock(out_channels, out_channels, is_res=True)
            # nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
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
        - Channel Matching: Adjust the number of channels in the upsampled image to match the skip connection if necessary.
        - Concatenation: Concatenate the upsampled image and the skip connection along the channel dimension.
        - Convolution: Process the concatenated image through the residual convolution blocks to produce the final output.
        """
        # x = F.interpolate(x, scale_factor=2, mode="nearest")
        # x = torch.cat((x, skip_input), dim=1)
        
        # device = x.device  # Ensure the device is the same as the input tensor
        # x = self.conv(x.to(device))  # Explicitly move x to the correct device
        logger.debug(f"UnetUp: before up - x shape: {x.shape}, skip shape: {skip.shape}")
        x = self.up(x)
        logger.debug(f"UnetUp: after up - x shape: {x.shape}")

        # Correctly pad x if necessary
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        logger.debug(f"UnetUp: padding - diffY: {diffY}, diffX: {diffX}")
        x = F.pad(x, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        logger.debug(f"UnetUp: after padding - x shape: {x.shape}")

        # Match channels if necessary
        if x.size(1) != skip.size(1):
            logger.debug(f"UnetUp: before matching channels - x shape: {x.shape}, skip shape: {skip.shape}")
            x = self.match_channels(x)  # Adjust channels to match the skip connection
            logger.debug(f"UnetUp: after matching channels - x shape: {x.shape}")

        # Concatenate and convolve
        x = torch.cat((skip, x), dim=1)
        logger.debug(f"UnetUp: after cat - x shape: {x.shape}")

        # Adjusting conv block to handle increased number of channels
        if x.size(1) != self.conv[0].conv1.in_channels:
            logger.debug(f"UnetUp: before conv - x shape: {x.shape}, expected: {self.conv[0].conv1.in_channels}")
            self.conv[0] = ResidualConvBlock(x.size(1), self.conv[0].conv1.out_channels, is_res=True)
            logger.debug(f"UnetUp: updated conv block to handle {x.size(1)} channels")

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