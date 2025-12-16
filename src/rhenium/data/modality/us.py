"""Ultrasound-specific data handling."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from rhenium.data.volume import ImageVolume, Modality


@dataclass
class USMetadata:
    """Ultrasound-specific metadata."""
    frequency: float | None = None  # MHz
    depth: float | None = None      # cm
    gain: float | None = None       # dB
    dynamic_range: float | None = None


def despeckle(image: np.ndarray, filter_type: str = "median", size: int = 3) -> np.ndarray:
    """Apply speckle reduction filter."""
    from scipy import ndimage

    if filter_type == "median":
        return ndimage.median_filter(image, size=size)
    elif filter_type == "lee":
        return _lee_filter(image, size)
    else:
        return image


def _lee_filter(image: np.ndarray, size: int = 5) -> np.ndarray:
    """Lee speckle filter."""
    from scipy.ndimage import uniform_filter
    mean = uniform_filter(image, size)
    sqr_mean = uniform_filter(image ** 2, size)
    var = sqr_mean - mean ** 2

    overall_var = image.var()
    weight = var / (var + overall_var + 1e-8)

    return mean + weight * (image - mean)


def log_compress(image: np.ndarray, dynamic_range: float = 60) -> np.ndarray:
    """Apply log compression for display."""
    image = np.clip(image, 1e-10, None)
    compressed = 20 * np.log10(image / image.max())
    compressed = (compressed + dynamic_range) / dynamic_range
    return np.clip(compressed, 0, 1)
