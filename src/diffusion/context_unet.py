# -*- coding: utf-8 -*-
# .\src\diffusion\context_unet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from .model_blocks import ResidualConvBlock, UnetDown, UnetUp, EmbedFC

logger = logging.getLogger(__name__)

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
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, kernel_size=(height // 16), stride=(height // 16)),
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
        logger.debug(f"ContextUnet: after init_conv - x shape: {x.shape}")
        down1 = self.down1(x)
        logger.debug(f"ContextUnet: after down1 - down1 shape: {down1.shape}")
        down2 = self.down2(down1)
        logger.debug(f"ContextUnet: after down2 - down2 shape: {down2.shape}")
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
