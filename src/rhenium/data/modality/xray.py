"""X-ray specific data handling."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from rhenium.data.volume import ImageVolume


@dataclass
class XRayMetadata:
    """X-ray acquisition metadata."""
    kvp: float | None = None
    mas: float | None = None
    exposure_time: float | None = None
    body_part: str = ""
    view_position: str = ""  # AP, PA, LAT


def enhance_contrast(image: np.ndarray, clip_limit: float = 0.03) -> np.ndarray:
    """Apply CLAHE-like contrast enhancement."""
    from skimage import exposure
    return exposure.equalize_adapthist(image, clip_limit=clip_limit)


def invert_if_needed(image: np.ndarray, expected_polarity: str = "standard") -> np.ndarray:
    """Invert image if polarity is inverted."""
    # Standard: air is dark, bone is bright
    mean_border = np.mean([image[0, :], image[-1, :], image[:, 0], image[:, -1]])
    mean_center = np.mean(image[image.shape[0]//3:2*image.shape[0]//3,
                                 image.shape[1]//3:2*image.shape[1]//3])

    if expected_polarity == "standard" and mean_border > mean_center:
        return 1 - image
    return image


def detect_orientation(image: np.ndarray) -> str:
    """Detect if image is rotated."""
    h, w = image.shape[:2]
    if h > w * 1.2:
        return "portrait"
    elif w > h * 1.2:
        return "landscape"
    return "square"
