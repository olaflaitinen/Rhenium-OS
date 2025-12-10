# Copyright (c) 2025 Skolyn LLC. All rights reserved.
# SPDX-License-Identifier: EUPL-1.1

"""
CT Reconstruction Pipeline
==========================

Sinogram-to-image reconstruction for CT with DL enhancement hooks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from rhenium.core.logging import get_reconstruction_logger
from rhenium.core.registry import registry, ComponentType
from rhenium.data.raw_io import SinogramData
from rhenium.data.dicom_io import ImageVolume, Modality

logger = get_reconstruction_logger()


@dataclass
class CTReconstructionConfig:
    """CT reconstruction configuration."""
    method: str = "fbp"
    filter_type: str = "ram-lak"
    metal_artifact_reduction: bool = False
    beam_hardening_correction: bool = True


class BaseCTReconstructor(ABC):
    """Abstract CT reconstructor."""

    @abstractmethod
    def reconstruct(self, sinogram: SinogramData) -> np.ndarray:
        pass


class FBPReconstructor(BaseCTReconstructor):
    """Filtered Back-Projection reconstructor."""

    def __init__(self, filter_type: str = "ram-lak"):
        self.filter_type = filter_type

    def reconstruct(self, sinogram: SinogramData) -> np.ndarray:
        """
        Reconstruct using filtered back-projection.

        This is a simplified implementation. Production would use
        optimized libraries like ASTRA or TomoPy.
        """
        from scipy.ndimage import rotate
        from scipy.fft import fft, ifft, fftfreq

        data = sinogram.data
        num_detectors = sinogram.num_detectors
        num_projections = sinogram.num_projections

        # Generate angles if not provided
        if sinogram.projection_angles is not None:
            angles = sinogram.projection_angles
        else:
            angles = np.linspace(0, np.pi, num_projections, endpoint=False)

        # Apply Ram-Lak filter
        freq = fftfreq(num_detectors)
        ram_lak = np.abs(freq)
        filtered = np.real(ifft(fft(data, axis=0) * ram_lak[:, np.newaxis], axis=0))

        # Back-project
        output_size = num_detectors
        reconstruction = np.zeros((output_size, output_size))

        for i, angle in enumerate(angles):
            projection = filtered[:, i]
            # Create 2D projection
            proj_2d = np.tile(projection, (output_size, 1)).T
            rotated = rotate(proj_2d, np.degrees(angle), reshape=False)
            reconstruction += rotated

        reconstruction *= np.pi / num_projections
        return reconstruction


@dataclass
class CTReconstructionPipeline:
    """
    CT reconstruction pipeline.

    Integrates:
    - Sinogram preprocessing
    - Reconstruction (FBP or DL-based)
    - Artifact correction
    """
    config: CTReconstructionConfig = field(default_factory=CTReconstructionConfig)
    reconstructor: BaseCTReconstructor = field(default_factory=FBPReconstructor)

    def run(self, sinogram: SinogramData) -> ImageVolume:
        """Execute CT reconstruction."""
        logger.info("Running CT reconstruction", method=self.config.method)

        image = self.reconstructor.reconstruct(sinogram)

        if self.config.beam_hardening_correction:
            image = self._beam_hardening_correction(image)

        if self.config.metal_artifact_reduction:
            image = self._metal_artifact_reduction(image)

        volume = ImageVolume(
            array=image[np.newaxis, ...] if image.ndim == 2 else image,
            spacing=(1.0, 1.0, 1.0),
            modality=Modality.CT,
        )

        logger.info("CT reconstruction complete", shape=volume.shape)
        return volume

    def _beam_hardening_correction(self, image: np.ndarray) -> np.ndarray:
        """Apply beam hardening correction (placeholder)."""
        return image

    def _metal_artifact_reduction(self, image: np.ndarray) -> np.ndarray:
        """Apply metal artifact reduction (placeholder)."""
        return image


registry.register(
    "ct_reconstruction",
    ComponentType.RECONSTRUCTION,
    CTReconstructionPipeline,
    version="1.0.0",
    description="CT reconstruction pipeline with FBP and DL hooks",
)
