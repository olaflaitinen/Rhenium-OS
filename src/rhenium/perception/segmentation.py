"""
Rhenium OS 3D Segmentation Models.

This module provides segmentation architectures including 3D U-Net,
Attention U-Net, and UNETR for volumetric medical image segmentation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from rhenium.core.registry import registry, ComponentType


class ConvBlock3D(nn.Module):
    """3D convolution block with BatchNorm and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class EncoderBlock3D(nn.Module):
    """Encoder block with convolution and max pooling."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock3D(in_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.conv(x)
        pooled = self.pool(features)
        return features, pooled


class DecoderBlock3D(nn.Module):
    """Decoder block with upsampling and skip connection."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_attention: bool = False,
    ):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2,
        )
        self.conv = ConvBlock3D(in_channels // 2 + skip_channels, out_channels)
        self.use_attention = use_attention

        if use_attention:
            self.attention = AttentionGate(skip_channels, in_channels // 2, skip_channels // 2)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        x = self.upsample(x)

        # Handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)

        if self.use_attention:
            skip = self.attention(skip, x)

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class AttentionGate(nn.Module):
    """Attention gate for focusing on relevant features."""

    def __init__(self, skip_channels: int, gate_channels: int, inter_channels: int):
        super().__init__()
        self.W_skip = nn.Conv3d(skip_channels, inter_channels, kernel_size=1)
        self.W_gate = nn.Conv3d(gate_channels, inter_channels, kernel_size=1)
        self.psi = nn.Conv3d(inter_channels, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        skip_proj = self.W_skip(skip)
        gate_proj = self.W_gate(gate)

        if skip_proj.shape[2:] != gate_proj.shape[2:]:
            gate_proj = F.interpolate(
                gate_proj, size=skip_proj.shape[2:],
                mode='trilinear', align_corners=False
            )

        attention = self.sigmoid(self.psi(self.relu(skip_proj + gate_proj)))
        return skip * attention


@registry.register("model", "unet3d", version="1.0.0", description="3D U-Net for volumetric segmentation")
class UNet3D(nn.Module):
    """
    3D U-Net for volumetric medical image segmentation.

    Architecture follows the original U-Net design extended to 3D,
    with optional attention gates.

    Args:
        in_channels: Number of input channels (e.g., 1 for grayscale)
        out_channels: Number of output classes
        features: List of feature sizes for each encoder level
        use_attention: Whether to use attention gates
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        features: list[int] | None = None,
        use_attention: bool = False,
    ):
        super().__init__()

        if features is None:
            features = [32, 64, 128, 256]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.use_attention = use_attention

        # Encoder
        self.encoders = nn.ModuleList()
        in_ch = in_channels
        for feat in features:
            self.encoders.append(EncoderBlock3D(in_ch, feat))
            in_ch = feat

        # Bottleneck
        self.bottleneck = ConvBlock3D(features[-1], features[-1] * 2)

        # Decoder
        self.decoders = nn.ModuleList()
        in_ch = features[-1] * 2
        for feat in reversed(features):
            self.decoders.append(DecoderBlock3D(in_ch, feat, feat, use_attention))
            in_ch = feat

        # Output
        self.head = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, D, H, W)

        Returns:
            Logits tensor of shape (B, num_classes, D, H, W)
        """
        # Encoder path
        skip_connections = []
        for encoder in self.encoders:
            skip, x = encoder(x)
            skip_connections.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        for decoder, skip in zip(self.decoders, reversed(skip_connections)):
            x = decoder(x, skip)

        # Output
        return self.head(x)

    def predict(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Make prediction with threshold.

        Args:
            x: Input tensor
            threshold: Probability threshold for binary segmentation

        Returns:
            Integer mask
        """
        with torch.no_grad():
            logits = self.forward(x)
            if self.out_channels == 2:
                probs = F.softmax(logits, dim=1)
                return (probs[:, 1] > threshold).long()
            else:
                return logits.argmax(dim=1)


@registry.register("model", "unetr", version="1.0.0", description="UNETR transformer-based segmentation")
class UNETR(nn.Module):
    """
    UNETR: Transformers for 3D Medical Image Segmentation.

    Uses a Vision Transformer encoder with a U-Net style decoder.
    Simplified implementation for demonstration.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output classes
        img_size: Input image size (D, H, W)
        patch_size: Patch size for tokenization
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        img_size: tuple[int, int, int] = (96, 96, 96),
        patch_size: int = 16,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        features: list[int] | None = None,
    ):
        super().__init__()

        if features is None:
            features = [32, 64, 128, 256]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Calculate number of patches
        self.num_patches = (
            (img_size[0] // patch_size) *
            (img_size[1] // patch_size) *
            (img_size[2] // patch_size)
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Decoder (simplified CNN decoder)
        self.decoder = nn.Sequential(
            nn.Conv3d(embed_dim, features[-1], kernel_size=3, padding=1),
            nn.BatchNorm3d(features[-1]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(features[-1], features[-2], kernel_size=3, padding=1),
            nn.BatchNorm3d(features[-2]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(features[-2], features[-3], kernel_size=3, padding=1),
            nn.BatchNorm3d(features[-3]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(features[-3], features[-4], kernel_size=3, padding=1),
            nn.BatchNorm3d(features[-4]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
        )

        self.head = nn.Conv3d(features[-4], out_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, D, H, W)

        Returns:
            Logits tensor of shape (B, num_classes, D, H, W)
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, D', H', W')
        spatial_shape = x.shape[2:]

        # Reshape to sequence
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed[:, :x.shape[1], :]

        # Transformer
        x = self.transformer(x)

        # Reshape back to spatial
        x = x.transpose(1, 2).view(B, self.embed_dim, *spatial_shape)

        # Decode
        x = self.decoder(x)

        # Final upsampling to match input size if needed
        if x.shape[2:] != self.img_size:
            x = F.interpolate(x, size=self.img_size, mode='trilinear', align_corners=False)

        return self.head(x)


def load_segmentation_model(
    name: str,
    checkpoint: Path | str | None = None,
    device: str = "cuda",
    **kwargs: Any,
) -> nn.Module:
    """
    Load a segmentation model from registry.

    Args:
        name: Model name (e.g., "unet3d", "unetr")
        checkpoint: Path to checkpoint file
        device: Target device
        **kwargs: Model configuration overrides

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
