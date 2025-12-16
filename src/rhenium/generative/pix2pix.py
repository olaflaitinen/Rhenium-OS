"""Pix2Pix paired image-to-image translation."""

from __future__ import annotations
import torch
import torch.nn as nn
from rhenium.core.registry import registry


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, normalize: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: bool = False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


@registry.register("generator", "pix2pix", version="1.0.0")
class Pix2PixGenerator(nn.Module):
    """U-Net generator for Pix2Pix."""

    def __init__(self, in_ch: int = 1, out_ch: int = 1, features: int = 64):
        super().__init__()
        self.down1 = DownBlock(in_ch, features, normalize=False)
        self.down2 = DownBlock(features, features * 2)
        self.down3 = DownBlock(features * 2, features * 4)
        self.down4 = DownBlock(features * 4, features * 8)
        self.down5 = DownBlock(features * 8, features * 8)

        self.up1 = UpBlock(features * 8, features * 8, dropout=True)
        self.up2 = UpBlock(features * 16, features * 4)
        self.up3 = UpBlock(features * 8, features * 2)
        self.up4 = UpBlock(features * 4, features)
        self.out = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_ch, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        u1 = torch.cat([self.up1(d5), d4], dim=1)
        u2 = torch.cat([self.up2(u1), d3], dim=1)
        u3 = torch.cat([self.up3(u2), d2], dim=1)
        u4 = torch.cat([self.up4(u3), d1], dim=1)
        return self.out(u4)


@registry.register("discriminator", "patchgan", version="1.0.0")
class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator."""

    def __init__(self, in_ch: int = 2, features: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, features, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features, features * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features * 2, features * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features * 4, 1, 4, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
