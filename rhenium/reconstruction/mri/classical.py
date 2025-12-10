# Copyright (c) 2025 Skolyn LLC. All rights reserved.
# SPDX-License-Identifier: EUPL-1.1

"""
Rhenium OS MRI Classical Reconstruction
========================================

Classical MRI reconstruction methods including FFT-based reconstruction,
parallel imaging (GRAPPA, SENSE), and coil combination strategies.

Skolyn: Early. Accurate. Trusted.

Last Updated: December 2025
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class CoilCombinationMethod(Enum):
    """Methods for combining multi-coil data."""
    ROOT_SUM_OF_SQUARES = "rss"
    SUM_OF_SQUARES = "sos"
    ADAPTIVE = "adaptive"
    SENSITIVITY_WEIGHTED = "sensitivity"


@dataclass
class ReconstructionConfig:
    """Configuration for classical MRI reconstruction."""
    coil_combination: CoilCombinationMethod = CoilCombinationMethod.ROOT_SUM_OF_SQUARES
    apply_hamming_filter: bool = False
    zero_pad_factor: float = 1.0  # 1.0 = no padding, 2.0 = double
    partial_fourier_correction: bool = True
    normalize_output: bool = True


class BaseMRIReconstructor(ABC):
    """
    Abstract base class for MRI reconstruction.
    
    All MRI reconstructors in Rhenium OS implement this interface,
    enabling consistent pipeline integration and model swapping.
    """
    
    @abstractmethod
    def reconstruct(
        self,
        kspace: np.ndarray,
        sensitivity_maps: np.ndarray | None = None,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Reconstruct image from k-space data.
        
        Args:
            kspace: Complex k-space data, shape (coils, kx, ky) or (kx, ky)
            sensitivity_maps: Coil sensitivity maps if available
            mask: Sampling mask for undersampled data
        
        Returns:
            Reconstructed image magnitude, shape (x, y)
        """
        pass


class FFTReconstructor(BaseMRIReconstructor):
    """
    FFT-based MRI reconstruction for fully-sampled Cartesian data.
    
    The basic MRI reconstruction process:
    1. Apply inverse FFT from k-space to image space
    2. Combine multi-coil data using specified method
    3. Extract magnitude image
    
    MRI Signal Model:
    The relationship between k-space S(k) and image I(r) is:
    
    S(k) = integral{ I(r) * exp(-i * 2 * pi * k * r) dr }
    I(r) = IFFT{ S(k) }
    """
    
    def __init__(self, config: ReconstructionConfig | None = None):
        """Initialize reconstructor with configuration."""
        self.config = config or ReconstructionConfig()
    
    def reconstruct(
        self,
        kspace: np.ndarray,
        sensitivity_maps: np.ndarray | None = None,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Reconstruct image from k-space using inverse FFT.
        
        Supports:
        - Single-coil data: shape (kx, ky) or (kx, ky, kz)
        - Multi-coil data: shape (coils, kx, ky) or (coils, kx, ky, kz)
        """
        # Handle single vs multi-coil
        if kspace.ndim == 2:
            coil_images = np.array([self._ifft2c(kspace)])
        elif kspace.ndim == 3:
            if kspace.shape[0] < 64:  # Likely multi-coil
                coil_images = np.array([self._ifft2c(k) for k in kspace])
            else:  # 3D single coil
                coil_images = np.array([self._ifft3c(kspace)])
        elif kspace.ndim == 4:
            # Multi-coil 3D
            coil_images = np.array([self._ifft3c(k) for k in kspace])
        else:
            raise ValueError(f"Unsupported k-space dimensions: {kspace.ndim}")
        
        # Combine coils
        combined = self._combine_coils(coil_images, sensitivity_maps)
        
        # Extract magnitude
        magnitude = np.abs(combined)
        
        # Normalize if configured
        if self.config.normalize_output:
            magnitude = self._normalize(magnitude)
        
        return magnitude
    
    def _ifft2c(self, kspace: np.ndarray) -> np.ndarray:
        """Centered 2D inverse FFT."""
        return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace)))
    
    def _ifft3c(self, kspace: np.ndarray) -> np.ndarray:
        """Centered 3D inverse FFT."""
        return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(kspace)))
    
    def _combine_coils(
        self,
        coil_images: np.ndarray,
        sensitivity_maps: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Combine multi-coil images.
        
        Root-sum-of-squares (RSS) combination:
        I_combined = sqrt( sum_c |I_c|^2 )
        """
        if coil_images.shape[0] == 1:
            return coil_images[0]
        
        if self.config.coil_combination == CoilCombinationMethod.ROOT_SUM_OF_SQUARES:
            return np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=0))
        elif self.config.coil_combination == CoilCombinationMethod.SUM_OF_SQUARES:
            return np.sum(np.abs(coil_images) ** 2, axis=0)
        elif self.config.coil_combination == CoilCombinationMethod.SENSITIVITY_WEIGHTED:
            if sensitivity_maps is None:
                return np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=0))
            # Weighted combination using sensitivity maps
            num = np.sum(np.conj(sensitivity_maps) * coil_images, axis=0)
            den = np.sum(np.abs(sensitivity_maps) ** 2, axis=0) + 1e-8
            return num / den
        else:
            return np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=0))
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range."""
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val > min_val:
            return (image - min_val) / (max_val - min_val)
        return image


class GRAPPAReconstructor(BaseMRIReconstructor):
    """
    GRAPPA (GeneRalized Autocalibrating Partially Parallel Acquisitions).
    
    Parallel imaging reconstruction that uses autocalibration signal (ACS)
    to estimate missing k-space lines from neighboring acquired lines.
    
    This is a conceptual implementation. Production systems should use
    BART or SigPy for optimized GRAPPA reconstruction.
    """
    
    def __init__(
        self,
        acceleration_factor: int = 2,
        acs_lines: int = 24,
        kernel_size: tuple[int, int] = (5, 4),
    ):
        """
        Initialize GRAPPA reconstructor.
        
        Args:
            acceleration_factor: Undersampling factor (R)
            acs_lines: Number of autocalibration signal lines
            kernel_size: GRAPPA kernel size (kx, ky)
        """
        self.acceleration_factor = acceleration_factor
        self.acs_lines = acs_lines
        self.kernel_size = kernel_size
    
    def reconstruct(
        self,
        kspace: np.ndarray,
        sensitivity_maps: np.ndarray | None = None,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Reconstruct undersampled k-space using GRAPPA.
        
        Note: This is a conceptual placeholder. Full GRAPPA implementation
        requires kernel calibration and application, typically using
        optimized libraries like BART, SigPy, or vendor-specific tools.
        """
        # Placeholder: falls back to zero-filled reconstruction
        # TODO: Implement full GRAPPA kernel calibration and application
        filled_kspace = kspace.copy()
        
        # Basic zero-filling for demonstration
        if mask is not None:
            filled_kspace = kspace * mask
        
        # Reconstruct using FFT
        fft_recon = FFTReconstructor()
        return fft_recon.reconstruct(filled_kspace, sensitivity_maps)


class SENSEReconstructor(BaseMRIReconstructor):
    """
    SENSE (SENSitivity Encoding) parallel imaging reconstruction.
    
    Uses coil sensitivity maps to unfold aliased images from
    undersampled k-space data.
    
    SENSE equation:
    I = (S^H * Psi^-1 * S)^-1 * S^H * Psi^-1 * m
    
    Where:
    - I = unfolded image
    - S = sensitivity encoding matrix
    - Psi = noise covariance matrix
    - m = aliased coil images
    """
    
    def __init__(
        self,
        acceleration_factor: int = 2,
        regularization: float = 0.001,
    ):
        """
        Initialize SENSE reconstructor.
        
        Args:
            acceleration_factor: Undersampling factor
            regularization: Tikhonov regularization parameter
        """
        self.acceleration_factor = acceleration_factor
        self.regularization = regularization
    
    def reconstruct(
        self,
        kspace: np.ndarray,
        sensitivity_maps: np.ndarray | None = None,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Reconstruct using SENSE.
        
        Requires coil sensitivity maps for proper SENSE reconstruction.
        Falls back to RSS combination if sensitivity maps unavailable.
        """
        if sensitivity_maps is None:
            # Fall back to RSS if no sensitivity maps
            fft_recon = FFTReconstructor()
            return fft_recon.reconstruct(kspace)
        
        # Placeholder: Full SENSE requires solving the encoding equation
        # TODO: Implement full SENSE inversion with regularization
        fft_recon = FFTReconstructor(
            ReconstructionConfig(
                coil_combination=CoilCombinationMethod.SENSITIVITY_WEIGHTED
            )
        )
        return fft_recon.reconstruct(kspace, sensitivity_maps)


def estimate_sensitivity_maps(
    kspace: np.ndarray,
    method: str = "espirit",
    acs_size: tuple[int, int] = (24, 24),
) -> np.ndarray:
    """
    Estimate coil sensitivity maps from k-space data.
    
    Methods:
    - espirit: ESPIRiT (Eigenvalue-based approach)
    - walsh: Walsh method
    - sos: Sum-of-squares normalization
    
    Args:
        kspace: Multi-coil k-space, shape (coils, kx, ky)
        method: Estimation method
        acs_size: Size of autocalibration region
    
    Returns:
        Sensitivity maps, shape (coils, x, y)
    
    Note: This is a simplified implementation. Production systems should
    use BART's ecalib or SigPy's ESPIRiT implementation.
    """
    # Simple SOS-based estimation
    # Extract center (ACS region)
    n_coils, nx, ny = kspace.shape
    cx, cy = nx // 2, ny // 2
    ax, ay = acs_size[0] // 2, acs_size[1] // 2
    
    acs = kspace[:, cx-ax:cx+ax, cy-ay:cy+ay]
    
    # Low-resolution coil images from ACS
    coil_images = np.array([
        np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(acs[c])))
        for c in range(n_coils)
    ])
    
    # SOS normalization
    sos = np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=0)) + 1e-8
    sensitivity_maps = coil_images / sos
    
    # Resize to full image
    from scipy.ndimage import zoom
    scale_x = nx / acs_size[0]
    scale_y = ny / acs_size[1]
    
    full_maps = np.array([
        zoom(np.abs(sensitivity_maps[c]), (scale_x, scale_y)) *
        np.exp(1j * zoom(np.angle(sensitivity_maps[c]), (scale_x, scale_y)))
        for c in range(n_coils)
    ])
    
    return full_maps
