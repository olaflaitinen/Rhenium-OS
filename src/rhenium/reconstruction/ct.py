"""
Rhenium OS CT Reconstruction.

This module provides CT reconstruction methods including filtered
backprojection (FBP) and iterative/learned methods.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rhenium.core.registry import registry
from rhenium.reconstruction.base import BaseReconstructor


def create_ramp_filter(size: int, filter_type: str = "ramp") -> np.ndarray:
    """
    Create frequency domain filter for FBP.

    Args:
        size: Filter size
        filter_type: Filter type (ramp, shepp-logan, cosine, hamming)

    Returns:
        Filter array
    """
    n = np.arange(size) - size // 2

    # Ramp filter
    ramp = np.abs(n) / (size // 2)

    if filter_type == "ramp":
        return ramp
    elif filter_type == "shepp-logan":
        # Shepp-Logan windowing
        window = np.sinc(n / (2 * size))
        return ramp * window
    elif filter_type == "cosine":
        window = np.cos(np.pi * n / size)
        return ramp * window
    elif filter_type == "hamming":
        window = 0.54 + 0.46 * np.cos(2 * np.pi * n / size)
        return ramp * window
    else:
        return ramp


@registry.register("reconstructor", "ct_fbp", version="1.0.0")
class FBPReconstructor(BaseReconstructor):
    """
    Filtered Backprojection for CT reconstruction.

    Args:
        filter_type: Type of frequency filter to use
        interpolation: Interpolation mode for backprojection
    """

    def __init__(
        self,
        filter_type: Literal["ramp", "shepp-logan", "cosine", "hamming"] = "ramp",
        interpolation: str = "bilinear",
    ):
        super().__init__()
        self.filter_type = filter_type
        self.interpolation = interpolation
        self._filter_cache: dict[int, torch.Tensor] = {}

    def _get_filter(self, size: int, device: torch.device) -> torch.Tensor:
        """Get or create filter of given size."""
        if size not in self._filter_cache:
            filt = create_ramp_filter(size, self.filter_type)
            self._filter_cache[size] = torch.from_numpy(filt).float()
        return self._filter_cache[size].to(device)

    def forward(
        self,
        sinogram: torch.Tensor,
        angles: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Reconstruct image from sinogram.

        Args:
            sinogram: Sinogram data (batch, angles, detectors)
            angles: Projection angles in radians

        Returns:
            Reconstructed image
        """
        if sinogram.dim() == 2:
            sinogram = sinogram.unsqueeze(0)

        B, num_angles, num_detectors = sinogram.shape
        device = sinogram.device

        # Filtering
        filt = self._get_filter(num_detectors, device)
        sinogram_fft = torch.fft.fft(sinogram, dim=-1)
        filtered = torch.fft.ifft(sinogram_fft * filt.unsqueeze(0).unsqueeze(0), dim=-1).real

        # Backprojection
        output_size = num_detectors
        image = torch.zeros(B, output_size, output_size, device=device)

        # Create coordinate grids
        coords = torch.linspace(-1, 1, output_size, device=device)
        y, x = torch.meshgrid(coords, coords, indexing='ij')

        for i, theta in enumerate(angles):
            # Rotate coordinates
            t = x * torch.cos(theta) + y * torch.sin(theta)

            # Map to detector indices
            t_idx = (t + 1) / 2 * (num_detectors - 1)

            # Sample from filtered projection
            t_idx_norm = (t_idx / (num_detectors - 1)) * 2 - 1
            grid = t_idx_norm.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, 1)
            proj = filtered[:, i:i+1, :].unsqueeze(1)

            sampled = F.grid_sample(
                proj,
                torch.cat([grid, torch.zeros_like(grid)], dim=-1),
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True,
            )

            image += sampled.squeeze(1).squeeze(1)

        # Normalize by number of angles
        image = image * np.pi / num_angles

        return image


@registry.register("reconstructor", "ct_sirt", version="1.0.0")
class SIRTReconstructor(BaseReconstructor):
    """
    Simultaneous Iterative Reconstruction Technique (SIRT) for CT.

    Args:
        num_iterations: Number of iterations
        relaxation: Relaxation parameter
    """

    def __init__(
        self,
        num_iterations: int = 50,
        relaxation: float = 0.1,
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.relaxation = relaxation

    def forward(
        self,
        sinogram: torch.Tensor,
        angles: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        SIRT reconstruction.

        Args:
            sinogram: Sinogram data
            angles: Projection angles

        Returns:
            Reconstructed image
        """
        if sinogram.dim() == 2:
            sinogram = sinogram.unsqueeze(0)

        B, num_angles, num_detectors = sinogram.shape
        device = sinogram.device
        output_size = num_detectors

        # Initialize with zeros
        image = torch.zeros(B, output_size, output_size, device=device)

        # Create coordinate grids
        coords = torch.linspace(-1, 1, output_size, device=device)
        y, x = torch.meshgrid(coords, coords, indexing='ij')

        for _ in range(self.num_iterations):
            # Forward projection
            forward_proj = self._forward_project(image, angles, num_detectors)

            # Compute residual
            residual = sinogram - forward_proj

            # Backproject residual
            backproj = self._backproject(residual, angles, output_size)

            # Update
            image = image + self.relaxation * backproj

            # Enforce non-negativity
            image = torch.clamp(image, min=0)

        return image

    def _forward_project(
        self,
        image: torch.Tensor,
        angles: torch.Tensor,
        num_detectors: int,
    ) -> torch.Tensor:
        """Forward projection (Radon transform)."""
        B = image.shape[0]
        device = image.device
        sinogram = torch.zeros(B, len(angles), num_detectors, device=device)

        for i, theta in enumerate(angles):
            # Rotate image
            rotated = self._rotate_image(image, theta)
            # Sum along columns
            sinogram[:, i, :] = rotated.sum(dim=-2)

        return sinogram

    def _backproject(
        self,
        sinogram: torch.Tensor,
        angles: torch.Tensor,
        output_size: int,
    ) -> torch.Tensor:
        """Backprojection."""
        B = sinogram.shape[0]
        device = sinogram.device
        image = torch.zeros(B, output_size, output_size, device=device)

        for i, theta in enumerate(angles):
            proj = sinogram[:, i, :]
            # Smear projection
            smeared = proj.unsqueeze(-2).expand(-1, output_size, -1)
            # Rotate back
            rotated = self._rotate_image(smeared, -theta)
            image += rotated

        return image / len(angles)

    def _rotate_image(self, image: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """Rotate image by angle."""
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)

        # Rotation matrix
        theta = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
        ], device=image.device).unsqueeze(0).expand(image.shape[0], -1, -1)

        grid = F.affine_grid(theta, image.unsqueeze(1).shape, align_corners=False)
        rotated = F.grid_sample(
            image.unsqueeze(1),
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )

        return rotated.squeeze(1)


@registry.register("reconstructor", "ct_learned", version="1.0.0")
class CTReconstructor(BaseReconstructor):
    """
    Learned CT reconstruction with dual-domain processing.

    Combines sinogram domain filtering with image domain refinement.

    Args:
        features: Number of CNN features
        num_blocks: Number of residual blocks
    """

    def __init__(
        self,
        features: int = 64,
        num_blocks: int = 8,
    ):
        super().__init__()

        # Sinogram domain network
        self.sino_net = nn.Sequential(
            nn.Conv1d(1, features, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(features, features, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(features, 1, kernel_size=5, padding=2),
        )

        # FBP
        self.fbp = FBPReconstructor()

        # Image domain network
        image_layers = [
            nn.Conv2d(1, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_blocks):
            image_layers.extend([
                nn.Conv2d(features, features, kernel_size=3, padding=1),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True),
            ])
        image_layers.append(nn.Conv2d(features, 1, kernel_size=3, padding=1))

        self.image_net = nn.Sequential(*image_layers)

    def forward(
        self,
        sinogram: torch.Tensor,
        angles: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Learned reconstruction.

        Args:
            sinogram: Sinogram data
            angles: Projection angles

        Returns:
            Reconstructed image
        """
        if sinogram.dim() == 2:
            sinogram = sinogram.unsqueeze(0)

        B, num_angles, num_detectors = sinogram.shape

        # Sinogram domain processing
        sino_in = sinogram.unsqueeze(2).view(B * num_angles, 1, num_detectors)
        sino_refined = self.sino_net(sino_in) + sino_in
        sino_refined = sino_refined.view(B, num_angles, num_detectors)

        # FBP reconstruction
        fbp_image = self.fbp(sino_refined, angles)

        # Image domain refinement
        image_refined = self.image_net(fbp_image.unsqueeze(1)) + fbp_image.unsqueeze(1)

        return image_refined.squeeze(1)
