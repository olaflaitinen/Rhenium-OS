# Copyright (c) 2025 Skolyn LLC. All rights reserved.
# SPDX-License-Identifier: EUPL-1.1

"""
Denoising Models
================

Deep learning models for image denoising in medical imaging.
Supports GAN-based and diffusion-based architectures.

CLINICAL DISCLAIMER:
Denoising may remove fine structures that could be clinically relevant.
Always compare denoised outputs with original images and exercise caution
when small features are critical for diagnosis.

Last Updated: December 2025
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

from rhenium.core.logging import get_perception_logger
from rhenium.core.registry import registry

logger = get_perception_logger()


@dataclass
class DenoisingConfig:
    """Configuration for denoising models."""
    noise_level_estimation: bool = True
    preserve_edges: bool = True
    max_iterations: int = 1000  # For diffusion models
    step_size: float = 0.01


class BaseDenoiser(ABC):
    """Abstract base class for denoising models."""
    
    name: str = "base_denoiser"
    version: str = "1.0.0"
    
    def __init__(self, config: DenoisingConfig | None = None):
        self.config = config or DenoisingConfig()
        self._loaded = False
        
    @abstractmethod
    def load(self) -> None:
        """Load model weights."""
        pass
    
    @abstractmethod
    def denoise(self, image: np.ndarray, noise_level: float | None = None) -> np.ndarray:
        """
        Denoise image.
        
        Args:
            image: Noisy input image
            noise_level: Optional noise level estimate
            
        Returns:
            Denoised image
        """
        pass
    
    def estimate_noise_level(self, image: np.ndarray) -> float:
        """Estimate noise standard deviation using MAD estimator."""
        # Median absolute deviation of Laplacian
        from scipy.ndimage import laplace
        lap = laplace(image)
        mad = np.median(np.abs(lap - np.median(lap)))
        sigma = mad / 0.6745
        return float(sigma)


class GANDenoiser(BaseDenoiser):
    """
    GAN-based denoiser.
    
    Uses adversarial training to produce realistic denoised images
    while preserving texture and fine structures.
    """
    
    name = "gan_denoiser"
    version = "1.0.0"
    
    def load(self) -> None:
        logger.info("Loading GAN denoiser")
        self._loaded = True
        
    def denoise(self, image: np.ndarray, noise_level: float | None = None) -> np.ndarray:
        if not self._loaded:
            self.load()
            
        if noise_level is None and self.config.noise_level_estimation:
            noise_level = self.estimate_noise_level(image)
            
        logger.debug("GAN denoising", estimated_noise=noise_level)
        
        # Placeholder: adaptive smoothing
        sigma = max(0.5, min(noise_level / 10, 2.0)) if noise_level else 1.0
        return gaussian_filter(image, sigma=sigma)


class DiffusionDenoiser(BaseDenoiser):
    """
    Diffusion-based denoiser.
    
    Uses score-based diffusion models to iteratively remove noise
    while staying on the manifold of natural images.
    """
    
    name = "diffusion_denoiser"
    version = "1.0.0"
    
    def load(self) -> None:
        logger.info("Loading diffusion denoiser")
        self._loaded = True
        
    def denoise(self, image: np.ndarray, noise_level: float | None = None) -> np.ndarray:
        if not self._loaded:
            self.load()
            
        if noise_level is None and self.config.noise_level_estimation:
            noise_level = self.estimate_noise_level(image)
            
        logger.debug("Diffusion denoising", 
                    iterations=self.config.max_iterations,
                    estimated_noise=noise_level)
        
        # Placeholder: iterative smoothing simulating diffusion steps
        result = image.copy()
        num_steps = min(10, self.config.max_iterations)
        
        for _ in range(num_steps):
            result = gaussian_filter(result, sigma=0.3)
            # Mix with original to preserve structure
            result = 0.9 * result + 0.1 * image
            
        return result


class NonLocalMeansDenoiser(BaseDenoiser):
    """
    Non-local means denoising.
    
    Classical patch-based denoising that preserves edges and textures.
    """
    
    name = "nlm_denoiser"
    version = "1.0.0"
    
    def load(self) -> None:
        self._loaded = True
        
    def denoise(self, image: np.ndarray, noise_level: float | None = None) -> np.ndarray:
        # Simplified: median filter as placeholder
        from scipy.ndimage import median_filter
        return median_filter(image, size=3)


# Register denoisers
registry.register_pipeline(
    "gan_denoiser",
    GANDenoiser,
    version="1.0.0",
    description="GAN-based image denoiser",
    tags=["generative", "denoising", "gan"],
)

registry.register_pipeline(
    "diffusion_denoiser",
    DiffusionDenoiser,
    version="1.0.0",
    description="Diffusion-based denoiser",
    tags=["generative", "denoising", "diffusion"],
)
