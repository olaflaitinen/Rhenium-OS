# Copyright (c) 2025 Skolyn LLC. All rights reserved.
# SPDX-License-Identifier: EUPL-1.1

"""
Rhenium OS CT Classical Reconstruction
=======================================

Classical CT reconstruction methods including Filtered Backprojection (FBP),
iterative reconstruction, and convolution kernels.

Skolyn: Early. Accurate. Trusted.

Last Updated: December 2025
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class ReconFilter(Enum):
    """Reconstruction filters for FBP."""
    RAM_LAK = "ram_lak"
    SHEPP_LOGAN = "shepp_logan"
    COSINE = "cosine"
    HAMMING = "hamming"
    HANN = "hann"


class ConvolutionKernel(Enum):
    """Post-reconstruction convolution kernels."""
    SOFT = "soft"
    STANDARD = "standard"
    LUNG = "lung"
    BONE = "bone"
    ULTRA_SHARP = "ultra_sharp"


@dataclass
class CTReconConfig:
    """Configuration for CT reconstruction."""
    # FBP settings
    filter_type: ReconFilter = ReconFilter.RAM_LAK
    cutoff_frequency: float = 1.0  # Normalized frequency
    
    # Kernel
    kernel: ConvolutionKernel = ConvolutionKernel.STANDARD
    
    # Geometry
    num_projections: int = 720
    detector_count: int = 512
    source_to_detector_mm: float = 1000.0
    source_to_isocenter_mm: float = 500.0
    
    # Output
    output_size: tuple[int, int] = (512, 512)
    pixel_size_mm: float = 0.5


class BaseCTReconstructor(ABC):
    """
    Abstract base class for CT reconstruction.
    
    All CT reconstructors in Rhenium OS implement this interface.
    """
    
    @abstractmethod
    def reconstruct(
        self,
        sinogram: np.ndarray,
        angles: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Reconstruct image from sinogram.
        
        Args:
            sinogram: 2D sinogram (angles x detectors) or 3D (slices x angles x detectors)
            angles: Projection angles in radians
        
        Returns:
            Reconstructed image volume
        """
        pass


class FBPReconstructor(BaseCTReconstructor):
    """
    Filtered Backprojection (FBP) CT reconstruction.
    
    The standard analytical CT reconstruction algorithm:
    1. Apply ramp filter (or variant) in frequency domain
    2. Backproject filtered projections
    
    FBP Formula:
    f(x,y) = integral_0^pi { p_filtered(theta, x*cos(theta) + y*sin(theta)) } d_theta
    
    where p_filtered = IFFT{ FFT{p} * H(omega) }
    and H(omega) = |omega| * W(omega) is the ramp filter with apodization window.
    """
    
    def __init__(self, config: CTReconConfig | None = None):
        """Initialize FBP reconstructor."""
        self.config = config or CTReconConfig()
    
    def reconstruct(
        self,
        sinogram: np.ndarray,
        angles: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Reconstruct using FBP.
        
        Args:
            sinogram: 2D (angles x detectors) or 3D (slices x angles x detectors)
            angles: Projection angles in radians, defaults to [0, pi)
        
        Returns:
            Reconstructed image(s)
        """
        if sinogram.ndim == 2:
            return self._reconstruct_slice(sinogram, angles)
        elif sinogram.ndim == 3:
            # Process each slice
            slices = []
            for i in range(sinogram.shape[0]):
                slices.append(self._reconstruct_slice(sinogram[i], angles))
            return np.stack(slices, axis=0)
        else:
            raise ValueError(f"Invalid sinogram dimensions: {sinogram.ndim}")
    
    def _reconstruct_slice(
        self,
        sinogram: np.ndarray,
        angles: np.ndarray | None = None,
    ) -> np.ndarray:
        """Reconstruct a single slice."""
        num_angles, num_detectors = sinogram.shape
        
        if angles is None:
            angles = np.linspace(0, np.pi, num_angles, endpoint=False)
        
        # Step 1: Apply ramp filter
        filtered = self._apply_filter(sinogram)
        
        # Step 2: Backprojection
        recon = self._backproject(filtered, angles)
        
        return recon
    
    def _apply_filter(self, sinogram: np.ndarray) -> np.ndarray:
        """Apply ramp filter in frequency domain."""
        num_angles, num_detectors = sinogram.shape
        
        # Zero-pad for FFT
        padded_size = int(2 ** np.ceil(np.log2(2 * num_detectors)))
        padded = np.zeros((num_angles, padded_size))
        padded[:, :num_detectors] = sinogram
        
        # Create ramp filter
        freq = np.fft.fftfreq(padded_size)
        ramp = np.abs(freq)
        
        # Apply window based on filter type
        window = self._get_filter_window(freq)
        filt = ramp * window
        
        # Apply filter in frequency domain
        sino_fft = np.fft.fft(padded, axis=1)
        filtered_fft = sino_fft * filt
        filtered = np.real(np.fft.ifft(filtered_fft, axis=1))
        
        return filtered[:, :num_detectors]
    
    def _get_filter_window(self, freq: np.ndarray) -> np.ndarray:
        """Get apodization window for filter."""
        if self.config.filter_type == ReconFilter.RAM_LAK:
            return np.ones_like(freq)
        elif self.config.filter_type == ReconFilter.SHEPP_LOGAN:
            # sinc function
            eps = 1e-10
            return np.sinc(freq / (2 * self.config.cutoff_frequency + eps))
        elif self.config.filter_type == ReconFilter.COSINE:
            return np.cos(np.pi * freq / (2 * self.config.cutoff_frequency))
        elif self.config.filter_type == ReconFilter.HAMMING:
            return 0.54 + 0.46 * np.cos(np.pi * freq / self.config.cutoff_frequency)
        elif self.config.filter_type == ReconFilter.HANN:
            return 0.5 * (1 + np.cos(np.pi * freq / self.config.cutoff_frequency))
        return np.ones_like(freq)
    
    def _backproject(
        self,
        filtered_sinogram: np.ndarray,
        angles: np.ndarray,
    ) -> np.ndarray:
        """Backproject filtered sinogram."""
        num_angles, num_detectors = filtered_sinogram.shape
        size = self.config.output_size[0]
        
        # Create output grid
        recon = np.zeros((size, size))
        
        # Coordinate grids
        x = np.linspace(-size/2, size/2, size)
        y = np.linspace(-size/2, size/2, size)
        X, Y = np.meshgrid(x, y)
        
        # Detector positions
        detector_pos = np.linspace(-num_detectors/2, num_detectors/2, num_detectors)
        
        for i, theta in enumerate(angles):
            # Calculate projection coordinate for each pixel
            t = X * np.cos(theta) + Y * np.sin(theta)
            
            # Interpolate projection values
            proj_interp = np.interp(t.flatten(), detector_pos, filtered_sinogram[i])
            recon += proj_interp.reshape(size, size)
        
        return recon * np.pi / num_angles


class IterativeReconstructor(BaseCTReconstructor):
    """
    Iterative CT reconstruction.
    
    Implements algebraic reconstruction techniques with regularization:
    - SIRT (Simultaneous Iterative Reconstruction Technique)
    - Conjugate gradient with TV regularization (conceptual)
    
    Minimizes: ||Ax - b||^2 + lambda * R(x)
    
    Where:
    - A = System matrix (forward projection)
    - x = Image to reconstruct
    - b = Measured projections
    - R = Regularization term
    """
    
    def __init__(
        self,
        num_iterations: int = 50,
        regularization: float = 0.01,
    ):
        """Initialize iterative reconstructor."""
        self.num_iterations = num_iterations
        self.regularization = regularization
    
    def reconstruct(
        self,
        sinogram: np.ndarray,
        angles: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Reconstruct using iterative method.
        
        Note: This is a simplified conceptual implementation.
        Production systems should use ASTRA Toolbox or similar.
        """
        # Placeholder: falls back to FBP for now
        # TODO: Implement full iterative reconstruction
        fbp = FBPReconstructor()
        return fbp.reconstruct(sinogram, angles)


def apply_kernel(
    image: np.ndarray,
    kernel: ConvolutionKernel,
) -> np.ndarray:
    """
    Apply post-reconstruction convolution kernel.
    
    Simulates the effect of different CT reconstruction kernels
    (soft tissue, lung, bone) through filtering.
    
    Args:
        image: Reconstructed CT image
        kernel: Kernel type
    
    Returns:
        Filtered image
    """
    from scipy.ndimage import gaussian_filter, laplace
    
    if kernel == ConvolutionKernel.SOFT:
        # Smooth kernel - Gaussian blur
        return gaussian_filter(image, sigma=1.0)
    elif kernel == ConvolutionKernel.STANDARD:
        return gaussian_filter(image, sigma=0.5)
    elif kernel == ConvolutionKernel.LUNG:
        # Edge-enhancing
        enhanced = image + 0.3 * laplace(image)
        return enhanced
    elif kernel == ConvolutionKernel.BONE:
        # Strong edge enhancement
        enhanced = image + 0.5 * laplace(image)
        return enhanced
    elif kernel == ConvolutionKernel.ULTRA_SHARP:
        enhanced = image + 0.7 * laplace(image)
        return enhanced
    
    return image
