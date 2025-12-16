"""
Rhenium OS X-ray Processing and Enhancement.
"""

from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from rhenium.core.registry import registry
from rhenium.reconstruction.base import BaseReconstructor


class ResBlock(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(features)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + x)


@registry.register("reconstructor", "xray_enhancer", version="1.0.0")
class XRayEnhancer(BaseReconstructor):
    """X-ray enhancement network."""

    def __init__(self, features: int = 64, num_blocks: int = 8):
        super().__init__()
        self.conv_in = nn.Conv2d(1, features, 3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(features) for _ in range(num_blocks)])
        self.conv_out = nn.Conv2d(features, 1, 3, padding=1)

    def forward(self, image: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if image.dim() == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.dim() == 3:
            image = image.unsqueeze(1)
        feat = F.relu(self.conv_in(image))
        feat = self.blocks(feat)
        return (self.conv_out(feat) + image).squeeze(1)


@registry.register("reconstructor", "xray_bone_suppressor", version="1.0.0")
class BoneSuppressor(BaseReconstructor):
    """Bone suppression for chest X-rays."""

    def __init__(self, features: int = 32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, features, 3, padding=1), nn.ReLU(),
            nn.Conv2d(features, features * 2, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(features * 2, features * 4, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(features * 4, features * 2, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(features * 2, features, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(features, 1, 3, padding=1),
        )

    def forward(self, image: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if image.dim() == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.dim() == 3:
            image = image.unsqueeze(1)
        return self.dec(self.enc(image)).squeeze(1)
