"""
Rhenium OS 3D Detection Models.

This module provides detection architectures for localizing lesions,
nodules, and other findings in medical images.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from rhenium.core.registry import registry


@dataclass
class Detection:
    """Detection result."""

    center: tuple[float, float, float]  # (z, y, x)
    size: tuple[float, float, float]    # (d, h, w)
    score: float
    class_id: int = 0
    class_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "center": self.center,
            "size": self.size,
            "score": self.score,
            "class_id": self.class_id,
            "class_name": self.class_name,
        }


class ConvBNReLU3D(nn.Module):
    """3D Conv + BatchNorm + ReLU block."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class DetectionBackbone(nn.Module):
    """Simple backbone for detection."""

    def __init__(self, in_channels: int = 1, features: list[int] | None = None):
        super().__init__()

        if features is None:
            features = [32, 64, 128, 256]

        layers = []
        in_ch = in_channels
        for i, feat in enumerate(features):
            layers.append(ConvBNReLU3D(in_ch, feat, stride=2 if i > 0 else 1))
            layers.append(ConvBNReLU3D(feat, feat))
            in_ch = feat

        self.layers = nn.Sequential(*layers)
        self.out_channels = features[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


@registry.register("model", "centernet3d", version="1.0.0", description="CenterNet-style 3D detection")
class CenterNet3D(nn.Module):
    """
    CenterNet-style anchor-free 3D detection.

    Predicts center heatmaps, size, and offset for each detection.

    Args:
        in_channels: Number of input channels
        num_classes: Number of detection classes
        features: Backbone feature sizes
        head_channels: Number of channels in detection heads
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        features: list[int] | None = None,
        head_channels: int = 64,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        # Backbone
        self.backbone = DetectionBackbone(in_channels, features)
        backbone_out = self.backbone.out_channels

        # Shared intermediate layer
        self.shared = nn.Sequential(
            ConvBNReLU3D(backbone_out, head_channels),
            ConvBNReLU3D(head_channels, head_channels),
        )

        # Heatmap head (class probabilities at each location)
        self.heatmap_head = nn.Sequential(
            nn.Conv3d(head_channels, head_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(head_channels, num_classes, kernel_size=1),
        )

        # Size head (predicted box dimensions)
        self.size_head = nn.Sequential(
            nn.Conv3d(head_channels, head_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(head_channels, 3, kernel_size=1),  # d, h, w
        )

        # Offset head (sub-voxel offset)
        self.offset_head = nn.Sequential(
            nn.Conv3d(head_channels, head_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(head_channels, 3, kernel_size=1),  # dz, dy, dx
        )

        self._init_heads()

    def _init_heads(self) -> None:
        """Initialize head biases."""
        # Initialize heatmap head bias for better initial predictions
        self.heatmap_head[-1].bias.data.fill_(-2.19)  # log(0.1 / 0.9)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, D, H, W)

        Returns:
            Dictionary with 'heatmap', 'size', 'offset' tensors
        """
        # Backbone
        features = self.backbone(x)

        # Shared layers
        shared = self.shared(features)

        # Heads
        heatmap = self.heatmap_head(shared).sigmoid()
        size = self.size_head(shared)
        offset = self.offset_head(shared)

        return {
            "heatmap": heatmap,
            "size": size,
            "offset": offset,
        }

    def decode_detections(
        self,
        outputs: dict[str, torch.Tensor],
        threshold: float = 0.1,
        max_detections: int = 100,
        input_shape: tuple[int, int, int] | None = None,
    ) -> list[list[Detection]]:
        """
        Decode network outputs to detections.

        Args:
            outputs: Output dictionary from forward()
            threshold: Score threshold
            max_detections: Maximum detections per sample
            input_shape: Original input shape for coordinate scaling

        Returns:
            List of detection lists (one per batch sample)
        """
        heatmap = outputs["heatmap"]
        size = outputs["size"]
        offset = outputs["offset"]

        B, C, D, H, W = heatmap.shape
        feature_shape = (D, H, W)

        all_detections = []

        for b in range(B):
            detections = []

            for c in range(C):
                heat = heatmap[b, c]

                # Find local maxima
                heat_max = F.max_pool3d(
                    heat.unsqueeze(0).unsqueeze(0),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )[0, 0]

                peaks = (heat == heat_max) & (heat > threshold)
                peak_coords = torch.nonzero(peaks, as_tuple=False)

                for coord in peak_coords:
                    z, y, x = coord.tolist()
                    score = heat[z, y, x].item()

                    # Get size and offset
                    sz = size[b, :, z, y, x].tolist()
                    off = offset[b, :, z, y, x].tolist()

                    # Apply offset
                    center = (
                        z + off[0],
                        y + off[1],
                        x + off[2],
                    )

                    # Scale to input coordinates if shape provided
                    if input_shape is not None:
                        scale = tuple(
                            inp / feat for inp, feat in zip(input_shape, feature_shape)
                        )
                        center = tuple(c * s for c, s in zip(center, scale))
                        sz = [s * sc for s, sc in zip(sz, scale)]

                    detections.append(Detection(
                        center=center,
                        size=tuple(abs(s) for s in sz),
                        score=score,
                        class_id=c,
                    ))

            # Sort by score and limit
            detections.sort(key=lambda x: x.score, reverse=True)
            detections = detections[:max_detections]

            all_detections.append(detections)

        return all_detections


def load_detection_model(
    name: str,
    checkpoint: Path | str | None = None,
    device: str = "cuda",
    **kwargs: Any,
) -> nn.Module:
    """
    Load a detection model from registry.

    Args:
        name: Model name
        checkpoint: Path to checkpoint
        device: Target device
        **kwargs: Model configuration

    Returns:
        Loaded model
    """
    model_cls = registry.get("model", name)
    model = model_cls(**kwargs)

    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        if checkpoint.exists():
            state_dict = torch.load(checkpoint, map_location=device)
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            model.load_state_dict(state_dict)

    model = model.to(device)
    return model
