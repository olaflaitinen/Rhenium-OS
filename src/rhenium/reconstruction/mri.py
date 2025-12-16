"""
Rhenium OS MRI Reconstruction.

This module provides MRI reconstruction methods including zero-filled,
SENSE-based, and deep learning unrolled networks.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from rhenium.core.registry import registry
from rhenium.reconstruction.base import (
    BaseReconstructor,
    DataConsistencyLayer,
    fft2c,
    ifft2c,
)


class ResBlock(nn.Module):
    """Residual block for MRI reconstruction networks."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class MRIDataConsistency(nn.Module):
    """MRI-specific data consistency layer."""

    def __init__(self, lambda_dc: float = 1.0):
        super().__init__()
        self.lambda_dc = nn.Parameter(torch.tensor(lambda_dc))

    def forward(
        self,
        x: torch.Tensor,
        kspace: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply k-space data consistency.

        Args:
            x: Current image estimate (real-valued)
            kspace: Measured k-space (complex)
            mask: Sampling mask

        Returns:
            Data-consistent image
        """
        # Convert to complex if needed
        if x.is_complex():
            x_complex = x
        else:
            x_complex = torch.complex(x, torch.zeros_like(x))

        # Forward to k-space
        pred_kspace = fft2c(x_complex)

        # Apply data consistency
        dc_kspace = mask * kspace + (1 - mask) * pred_kspace

        # Back to image
        dc_image = ifft2c(dc_kspace)

        return dc_image.abs() if not x.is_complex() else dc_image


@registry.register("reconstructor", "mri_zerofilled", version="1.0.0")
class MRIReconstructor(BaseReconstructor):
    """
    Zero-filled MRI reconstruction.

    Simply applies inverse FFT to k-space data.
    """

    def forward(
        self,
        kspace: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Reconstruct from k-space.

        Args:
            kspace: K-space data (complex tensor)

        Returns:
            Reconstructed magnitude image
        """
        image = ifft2c(kspace)
        return image.abs()


@registry.register("reconstructor", "mri_unrolled", version="1.0.0")
class UnrolledMRIRecon(BaseReconstructor):
    """
    Unrolled MRI reconstruction network.

    Alternates between CNN denoising and data consistency.

    Args:
        in_channels: Number of input channels
        features: Number of CNN features
        num_cascades: Number of unrolled iterations
        num_resblocks: Number of res blocks per cascade
    """

    def __init__(
        self,
        in_channels: int = 2,  # Real + Imag
        features: int = 64,
        num_cascades: int = 10,
        num_resblocks: int = 5,
    ):
        super().__init__()

        self.num_cascades = num_cascades

        # CNN blocks for each cascade
        self.cnn_blocks = nn.ModuleList()
        self.dc_blocks = nn.ModuleList()

        for _ in range(num_cascades):
            # CNN block
            layers = [
                nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ]
            for _ in range(num_resblocks):
                layers.append(ResBlock(features))
            layers.append(nn.Conv2d(features, in_channels, kernel_size=3, padding=1))

            self.cnn_blocks.append(nn.Sequential(*layers))
            self.dc_blocks.append(MRIDataConsistency())

    def forward(
        self,
        kspace: torch.Tensor,
        mask: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Unrolled reconstruction.

        Args:
            kspace: Undersampled k-space (complex)
            mask: Sampling mask

        Returns:
            Reconstructed image
        """
        # Initial estimate
        x = ifft2c(kspace)  # Complex image

        # Convert to real channels [real, imag]
        x_real = torch.stack([x.real, x.imag], dim=1)

        for cnn, dc in zip(self.cnn_blocks, self.dc_blocks):
            # CNN refinement
            x_refined = cnn(x_real) + x_real

            # Convert back to complex for DC
            x_complex = torch.complex(x_refined[:, 0], x_refined[:, 1])

            # Data consistency
            x_dc = dc(x_complex, kspace, mask)

            # Back to real channels
            x_real = torch.stack([x_dc.real, x_dc.imag], dim=1)

        # Final complex image
        x_final = torch.complex(x_real[:, 0], x_real[:, 1])

        return x_final.abs()


@registry.register("reconstructor", "mri_varnet", version="1.0.0")
class VarNetMRI(BaseReconstructor):
    """
    Variational Network for MRI reconstruction.

    End-to-end learnable reconstruction with sensitivity maps.

    Args:
        num_cascades: Number of unrolled iterations
        sense_chans: Sensitivity estimation channels
        model_chans: Main model channels
    """

    def __init__(
        self,
        num_cascades: int = 12,
        sense_chans: int = 8,
        model_chans: int = 18,
    ):
        super().__init__()
        self.num_cascades = num_cascades

        # Sensitivity estimation network
        self.sens_net = SensitivityNet(sense_chans)

        # Cascades
        self.cascades = nn.ModuleList([
            VarNetCascade(model_chans) for _ in range(num_cascades)
        ])

    def forward(
        self,
        kspace: torch.Tensor,
        mask: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        VarNet reconstruction.

        Args:
            kspace: Multi-coil k-space data
            mask: Sampling mask

        Returns:
            Reconstructed image
        """
        # Estimate sensitivity maps
        sens_maps = self.sens_net(kspace)

        # Initial estimate (SENSE-combined)
        x = self._sense_combine(kspace, sens_maps)

        # Cascaded refinement
        for cascade in self.cascades:
            x = cascade(x, kspace, mask, sens_maps)

        return x.abs()

    def _sense_combine(
        self,
        kspace: torch.Tensor,
        sens_maps: torch.Tensor,
    ) -> torch.Tensor:
        """Combine multi-coil images using sensitivity maps."""
        # Per-coil images
        coil_images = ifft2c(kspace)
        # Weighted combination
        combined = (coil_images * sens_maps.conj()).sum(dim=1)
        return combined


class SensitivityNet(nn.Module):
    """Network to estimate coil sensitivity maps."""

    def __init__(self, chans: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, chans, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(chans, chans, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(chans, 2, kernel_size=3, padding=1),
        )

    def forward(self, kspace: torch.Tensor) -> torch.Tensor:
        """Estimate sensitivity maps from k-space."""
        # Low-res images
        images = ifft2c(kspace)

        # Stack real/imag
        x = torch.stack([images.real, images.imag], dim=2)
        B, C, _, H, W = x.shape
        x = x.view(B * C, 2, H, W)

        # Estimate
        sens = self.net(x)
        sens = sens.view(B, C, 2, H, W)

        return torch.complex(sens[:, :, 0], sens[:, :, 1])


class VarNetCascade(nn.Module):
    """Single cascade of VarNet."""

    def __init__(self, chans: int = 18):
        super().__init__()
        self.refinement = nn.Sequential(
            nn.Conv2d(2, chans, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResBlock(chans),
            ResBlock(chans),
            nn.Conv2d(chans, 2, kernel_size=3, padding=1),
        )
        self.dc_weight = nn.Parameter(torch.ones(1))

    def forward(
        self,
        x: torch.Tensor,
        kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
    ) -> torch.Tensor:
        """Apply cascade."""
        # CNN refinement
        x_stack = torch.stack([x.real, x.imag], dim=1)
        refined = self.refinement(x_stack)
        x_refined = torch.complex(refined[:, 0], refined[:, 1]) + x

        # Data consistency (simplified)
        pred_kspace = fft2c(x_refined.unsqueeze(1) * sens_maps).sum(dim=1)
        dc_kspace = mask * kspace.sum(dim=1) + (1 - mask) * pred_kspace
        x_dc = ifft2c(dc_kspace)

        return x_dc
