"""
Rhenium OS Reconstruction Base Classes.

This module defines base protocols and classes for image reconstruction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol

import torch
import torch.nn as nn


class Reconstructor(Protocol):
    """Protocol for reconstruction methods."""

    def __call__(
        self,
        raw_data: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Reconstruct image from raw acquisition data.

        Args:
            raw_data: Raw acquisition data (k-space, sinogram, etc.)
            **kwargs: Additional reconstruction parameters

        Returns:
            Reconstructed image tensor
        """
        ...


class BaseReconstructor(nn.Module, ABC):
    """Abstract base class for reconstructors."""

    @abstractmethod
    def forward(
        self,
        raw_data: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Perform reconstruction."""
        pass

    def __call__(
        self,
        raw_data: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Call forward method."""
        return self.forward(raw_data, **kwargs)


class DataConsistencyLayer(nn.Module):
    """
    Data consistency layer for enforcing measurement constraints.

    Used in unrolled optimization networks to enforce fidelity
    to acquired data.
    """

    def __init__(self, lambda_dc: float = 1.0, learnable: bool = False):
        """
        Initialize data consistency layer.

        Args:
            lambda_dc: Data consistency weight
            learnable: Whether lambda is learnable
        """
        super().__init__()

        if learnable:
            self.lambda_dc = nn.Parameter(torch.tensor(lambda_dc))
        else:
            self.register_buffer("lambda_dc", torch.tensor(lambda_dc))

    def forward(
        self,
        x: torch.Tensor,
        x0: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply data consistency.

        Args:
            x: Current estimate
            x0: Initial estimate from zero-filled recon
            mask: Sampling mask

        Returns:
            Data-consistent estimate
        """
        # Soft DC: weighted average in sampled regions
        dc = mask * x0 + (1 - self.lambda_dc * mask) * x
        return dc


def fft2c(x: torch.Tensor) -> torch.Tensor:
    """Centered 2D FFT."""
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1))), dim=(-2, -1))


def ifft2c(x: torch.Tensor) -> torch.Tensor:
    """Centered 2D IFFT."""
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-2, -1))), dim=(-2, -1))


def fft3c(x: torch.Tensor) -> torch.Tensor:
    """Centered 3D FFT."""
    return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(x, dim=(-3, -2, -1)), dim=(-3, -2, -1)), dim=(-3, -2, -1))


def ifft3c(x: torch.Tensor) -> torch.Tensor:
    """Centered 3D IFFT."""
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(x, dim=(-3, -2, -1)), dim=(-3, -2, -1)), dim=(-3, -2, -1))
