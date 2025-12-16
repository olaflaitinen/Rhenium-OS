"""SRGAN/ESRGAN super-resolution models."""

from __future__ import annotations
import torch
import torch.nn as nn
from rhenium.core.registry import registry


class DenseBlock(nn.Module):
    def __init__(self, features: int, growth: int = 32):
        super().__init__()
        self.conv = nn.Conv2d(features, growth, 3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, self.lrelu(self.conv(x))], dim=1)


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block."""

    def __init__(self, features: int, growth: int = 32, scale: float = 0.2):
        super().__init__()
        self.dense1 = DenseBlock(features, growth)
        self.dense2 = DenseBlock(features + growth, growth)
        self.dense3 = DenseBlock(features + 2 * growth, growth)
        self.conv = nn.Conv2d(features + 3 * growth, features, 1)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.dense1(x)
        d2 = self.dense2(d1)
        d3 = self.dense3(d2)
        return x + self.scale * self.conv(d3)


@registry.register("generator", "srgan", version="1.0.0")
class SRGenerator(nn.Module):
    """ESRGAN-style super-resolution generator."""

    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        features: int = 64,
        n_rrdb: int = 16,
        upscale: int = 4,
    ):
        super().__init__()
        self.conv_first = nn.Conv2d(in_ch, features, 3, padding=1)
        self.body = nn.Sequential(*[RRDB(features) for _ in range(n_rrdb)])
        self.conv_body = nn.Conv2d(features, features, 3, padding=1)

        # Upsampling
        up_layers = []
        for _ in range(upscale // 2):
            up_layers += [
                nn.Conv2d(features, features * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2),
            ]
        self.upsampler = nn.Sequential(*up_layers)
        self.conv_last = nn.Sequential(
            nn.Conv2d(features, features, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features, out_ch, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        body = self.conv_body(self.body(feat))
        feat = feat + body
        return self.conv_last(self.upsampler(feat))


@registry.register("discriminator", "vgg_discriminator", version="1.0.0")
class VGGDiscriminator(nn.Module):
    """VGG-style discriminator for SRGAN."""

    def __init__(self, in_ch: int = 1, features: int = 64):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, features, 3, padding=1),
            nn.LeakyReLU(0.2),
        ]
        ch = features
        for _ in range(4):
            layers += [
                nn.Conv2d(ch, ch * 2, 4, stride=2, padding=1),
                nn.BatchNorm2d(ch * 2),
                nn.LeakyReLU(0.2),
            ]
            ch = ch * 2
        layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(ch, 1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
