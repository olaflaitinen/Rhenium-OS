# Copyright (c) 2025 Skolyn LLC. All rights reserved.
# SPDX-License-Identifier: EUPL-1.1

"""
Generative Models Module
========================

GAN-based and diffusion-based models for super-resolution, denoising,
and anomaly detection in medical imaging.

IMPORTANT ETHICAL CONSIDERATION:
Generative models must be used with extreme caution in clinical contexts.
Synthetic image content may introduce artifacts that could be mistaken for
pathology or obscure real findings. All outputs must be clearly labeled
and subject to human review before clinical use.

Last Updated: December 2025
"""

from rhenium.generative.super_resolution import (
    BaseSuperResolution,
    MRISuperResolution,
    CTSuperResolution,
)
from rhenium.generative.denoising import (
    BaseDenoiser,
    GANDenoiser,
    DiffusionDenoiser,
)
from rhenium.generative.anomaly_detection import (
    BaseAnomalyDetector,
    ReconstructionAnomalyDetector,
)

__all__ = [
    "BaseSuperResolution",
    "MRISuperResolution",
    "CTSuperResolution",
    "BaseDenoiser",
    "GANDenoiser",
    "DiffusionDenoiser",
    "BaseAnomalyDetector",
    "ReconstructionAnomalyDetector",
]
