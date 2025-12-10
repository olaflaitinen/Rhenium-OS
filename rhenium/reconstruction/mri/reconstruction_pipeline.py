# Copyright (c) 2025 Skolyn LLC. All rights reserved.
# SPDX-License-Identifier: EUPL-1.1

"""
MRI Reconstruction Pipeline
===========================

Rhenium Reconstruction Engine interface for MRI reconstruction from k-space.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rhenium.core.logging import get_reconstruction_logger
from rhenium.core.registry import registry, ComponentType
from rhenium.data.raw_io import KSpaceData
from rhenium.data.dicom_io import ImageVolume, Modality

logger = get_reconstruction_logger()


@dataclass
class ReconstructionConfig:
    """Configuration for MRI reconstruction."""
    method: str = "ifft"
    denoise: bool = True
    denoise_strength: float = 0.1
    coil_combination: str = "sos"


class BaseReconstructor(ABC):
    """Abstract base class for reconstruction methods."""

    @abstractmethod
    def reconstruct(self, kspace: KSpaceData) -> np.ndarray:
        """Reconstruct image from k-space."""
        pass


class IFFTReconstructor(BaseReconstructor):
    """Simple IFFT-based reconstruction."""

    def reconstruct(self, kspace: KSpaceData) -> np.ndarray:
        from scipy.fft import ifft2, ifftshift, fftshift

        data = kspace.data
        if data.ndim == 2:
            return np.abs(fftshift(ifft2(ifftshift(data))))
        elif data.ndim >= 3:
            images = []
            for coil in range(data.shape[0]):
                img = np.abs(fftshift(ifft2(ifftshift(data[coil]))))
                images.append(img ** 2)
            return np.sqrt(np.sum(images, axis=0))
        return data


class RheniumDeepReconstructor(BaseReconstructor):
    """
    Rhenium Reconstruction Engine deep learning reconstructor.

    This is a placeholder for the actual Rhenium DL reconstruction integration.
    The real implementation would load trained DL models for
    accelerated MRI reconstruction.
    """

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        self.model = None
        logger.info("Rhenium deep reconstructor initialized")

    def load_model(self) -> None:
        """Load reconstruction model weights."""
        # Placeholder for model loading
        logger.info("Loading Rhenium reconstruction model", path=self.model_path)

    def reconstruct(self, kspace: KSpaceData) -> np.ndarray:
        """Reconstruct using Rhenium deep learning."""
        if self.model is None:
            # Fall back to IFFT if model not loaded
            logger.warning("Model not loaded, falling back to IFFT")
            return IFFTReconstructor().reconstruct(kspace)

        # Placeholder for DL reconstruction
        return IFFTReconstructor().reconstruct(kspace)


@dataclass
class MRIReconstructionPipeline:
    """
    Full MRI reconstruction pipeline.

    Integrates:
    - K-space preprocessing
    - Reconstruction (IFFT or Rhenium Deep Learning)
    - Post-processing (denoising, artifact removal)
    """
    config: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    reconstructor: BaseReconstructor = field(default_factory=IFFTReconstructor)

    def run(self, kspace: KSpaceData) -> ImageVolume:
        """Execute reconstruction pipeline."""
        logger.info("Running MRI reconstruction", method=self.config.method)

        # Reconstruct
        image = self.reconstructor.reconstruct(kspace)

        # Apply denoising if configured
        if self.config.denoise:
            image = self._denoise(image)

        # Create ImageVolume
        spacing = (1.0, 1.0, 1.0)  # Would be extracted from metadata
        volume = ImageVolume(
            array=image if image.ndim == 3 else image[np.newaxis, ...],
            spacing=spacing,
            modality=Modality.MR,
            series_uid=kspace.metadata.get("series_uid", ""),
        )

        logger.info("Reconstruction complete", shape=volume.shape)
        return volume

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply denoising."""
        from scipy.ndimage import gaussian_filter
        sigma = self.config.denoise_strength * 2
        return gaussian_filter(image, sigma=sigma)


# Register with component registry
registry.register(
    "mri_reconstruction",
    ComponentType.RECONSTRUCTION,
    MRIReconstructionPipeline,
    version="1.0.0",
    description="Rhenium MRI reconstruction pipeline with deep learning support",
)
