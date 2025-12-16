"""CT-specific data handling and preprocessing."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np
from rhenium.data.volume import ImageVolume, Modality, CTWindow, CT_WINDOW_PRESETS


@dataclass
class CTMetadata:
    """CT-specific acquisition metadata."""
    kvp: float | None = None           # kVp
    mas: float | None = None           # mAs
    slice_thickness: float | None = None
    convolution_kernel: str = ""
    scanner_type: str = ""


def hounsfield_to_mu(hu: np.ndarray, mu_water: float = 0.019) -> np.ndarray:
    """Convert Hounsfield Units to linear attenuation coefficient."""
    return mu_water * (hu / 1000 + 1)


def mu_to_hounsfield(mu: np.ndarray, mu_water: float = 0.019) -> np.ndarray:
    """Convert linear attenuation to Hounsfield Units."""
    return (mu / mu_water - 1) * 1000


def apply_window(
    volume: ImageVolume,
    preset: CTWindow | None = None,
    center: float | None = None,
    width: float | None = None,
) -> ImageVolume:
    """Apply CT windowing."""
    if preset:
        center, width = CT_WINDOW_PRESETS[preset]
    elif center is None or width is None:
        raise ValueError("Must provide either preset or center/width")

    return volume.apply_window(center, width)


def segment_body(volume: ImageVolume, threshold: float = -200) -> np.ndarray:
    """Segment body from air in CT."""
    from scipy import ndimage

    mask = volume.array > threshold
    mask = ndimage.binary_fill_holes(mask)

    # Keep largest connected component
    labeled, num = ndimage.label(mask)
    if num > 0:
        sizes = ndimage.sum(mask, labeled, range(1, num + 1))
        largest = np.argmax(sizes) + 1
        mask = labeled == largest

    return mask.astype(np.uint8)


def segment_lungs(volume: ImageVolume) -> np.ndarray:
    """Segment lungs from chest CT."""
    from scipy import ndimage

    # Air threshold
    air_mask = volume.array < -400

    # Remove background air
    body_mask = segment_body(volume)
    lung_candidates = air_mask & body_mask

    # Morphological cleanup
    lung_candidates = ndimage.binary_opening(lung_candidates, iterations=2)

    return lung_candidates.astype(np.uint8)


def calculate_lung_density(volume: ImageVolume, lung_mask: np.ndarray) -> dict[str, float]:
    """Calculate lung density statistics."""
    lung_values = volume.array[lung_mask > 0]

    if len(lung_values) == 0:
        return {"mean_hu": 0, "std_hu": 0, "volume_ml": 0}

    voxel_vol = np.prod(volume.spacing)

    return {
        "mean_hu": float(lung_values.mean()),
        "std_hu": float(lung_values.std()),
        "volume_ml": float(lung_mask.sum() * voxel_vol / 1000),
    }
