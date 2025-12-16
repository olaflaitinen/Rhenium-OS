"""
Rhenium OS Ultrasound Reconstruction and Processing.

This module provides ultrasound-specific processing including
beamforming, speckle reduction, and enhancement.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from rhenium.core.registry import registry
from rhenium.reconstruction.base import BaseReconstructor


@registry.register("reconstructor", "us_beamformer", version="1.0.0")
class USReconstructor(BaseReconstructor):
    """
    Ultrasound beamforming and reconstruction.

    Implements delay-and-sum beamforming for channel data.

    Args:
        speed_of_sound: Speed of sound in tissue (m/s)
        sampling_freq: RF sampling frequency (Hz)
    """

    def __init__(
        self,
        speed_of_sound: float = 1540.0,
        sampling_freq: float = 40e6,
    ):
        super().__init__()
        self.speed_of_sound = speed_of_sound
        self.sampling_freq = sampling_freq

    def forward(
        self,
        channel_data: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Beamform ultrasound channel data.

        Args:
            channel_data: Raw RF channel data (batch, channels, samples)

        Returns:
            Beamformed image
        """
        if channel_data.dim() == 2:
            channel_data = channel_data.unsqueeze(0)

        B, num_channels, num_samples = channel_data.shape

        # Simple delay-and-sum (DAS) beamforming
        # In practice, this would compute proper delays based on geometry
        beamformed = channel_data.mean(dim=1)

        # Envelope detection (Hilbert transform approximation)
        envelope = self._envelope_detection(beamformed)

        # Log compression
        compressed = self._log_compress(envelope)

        # Reshape to 2D image (simplified)
        sqrt_samples = int(num_samples ** 0.5)
        image = compressed[:, :sqrt_samples * sqrt_samples].view(B, sqrt_samples, sqrt_samples)

        return image

    def _envelope_detection(self, signal: torch.Tensor) -> torch.Tensor:
        """Compute envelope using magnitude of analytic signal."""
        # FFT
        spectrum = torch.fft.fft(signal, dim=-1)
        n = spectrum.shape[-1]

        # Create analytic signal (set negative frequencies to zero)
        h = torch.zeros(n, device=signal.device)
        h[0] = 1
        h[1:(n + 1) // 2] = 2
        if n % 2 == 0:
            h[n // 2] = 1

        analytic = torch.fft.ifft(spectrum * h.unsqueeze(0), dim=-1)
        return analytic.abs()

    def _log_compress(
        self,
        envelope: torch.Tensor,
        dynamic_range: float = 60.0,
    ) -> torch.Tensor:
        """Apply log compression for display."""
        # Normalize
        max_val = envelope.max()
        if max_val > 0:
            envelope = envelope / max_val

        # Log compression
        compressed = 20 * torch.log10(envelope.clamp(min=1e-10))

        # Map to 0-1 range
        compressed = (compressed + dynamic_range) / dynamic_range
        compressed = compressed.clamp(0, 1)

        return compressed


@registry.register("reconstructor", "us_enhancer", version="1.0.0")
class USEnhancer(BaseReconstructor):
    """
    Deep learning ultrasound enhancement.

    Reduces speckle noise and improves image quality.

    Args:
        features: Number of CNN features
        num_layers: Number of convolutional layers
    """

    def __init__(
        self,
        features: int = 64,
        num_layers: int = 8,
    ):
        super().__init__()

        layers = [
            nn.Conv2d(1, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]

        for _ in range(num_layers - 2):
            layers.extend([
                nn.Conv2d(features, features, kernel_size=3, padding=1),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True),
            ])

        layers.append(nn.Conv2d(features, 1, kernel_size=3, padding=1))

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        image: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Enhance ultrasound image.

        Args:
            image: Input ultrasound image

        Returns:
            Enhanced image
        """
        if image.dim() == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.dim() == 3:
            image = image.unsqueeze(1)

        # Residual learning
        enhanced = self.net(image) + image

        return enhanced.squeeze(1) if image.dim() == 4 else enhanced


@registry.register("reconstructor", "us_speckle_reducer", version="1.0.0")
class SpeckleReducer(nn.Module):
    """
    Speckle reduction network for ultrasound.

    Based on deep convolutional neural networks trained
    for speckle noise removal.
    """

    def __init__(self, features: int = 64):
        super().__init__()

        # Encoder
        self.enc1 = self._make_block(1, features)
        self.enc2 = self._make_block(features, features * 2)
        self.enc3 = self._make_block(features * 2, features * 4)

        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.up3 = nn.ConvTranspose2d(features * 4, features * 2, 2, stride=2)
        self.dec3 = self._make_block(features * 4, features * 2)

        self.up2 = nn.ConvTranspose2d(features * 2, features, 2, stride=2)
        self.dec2 = self._make_block(features * 2, features)

        self.out = nn.Conv2d(features, 1, 1)

    def _make_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce speckle noise."""
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Decoder
        d3 = self.dec3(torch.cat([self.up3(e3), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))

        return self.out(d2) + x
