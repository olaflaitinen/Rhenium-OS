"""MRI-specific data handling and preprocessing."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from rhenium.data.volume import ImageVolume, Modality, MRISequence


@dataclass
class MRIMetadata:
    """MRI-specific acquisition metadata."""
    sequence: MRISequence = MRISequence.UNKNOWN
    repetition_time: float | None = None  # TR in ms
    echo_time: float | None = None        # TE in ms
    flip_angle: float | None = None       # degrees
    field_strength: float | None = None   # Tesla
    slice_thickness: float | None = None  # mm
    pixel_bandwidth: float | None = None


def classify_mri_sequence(volume: ImageVolume) -> MRISequence:
    """Classify MRI sequence type from metadata."""
    meta = volume.metadata
    if meta.sequence_type:
        return meta.sequence_type

    # Heuristic based on common TR/TE patterns
    tr = meta.repetition_time
    te = meta.echo_time

    if tr and te:
        if tr < 800 and te < 30:
            return MRISequence.T1
        elif tr > 2000 and te > 80:
            return MRISequence.T2
        elif tr > 6000 and te > 100:
            return MRISequence.FLAIR

    return MRISequence.UNKNOWN


def apply_bias_field_correction(volume: ImageVolume, iterations: int = 50) -> ImageVolume:
    """Apply N4 bias field correction (simplified)."""
    # Simplified bias correction using polynomial fit
    data = volume.array.astype(np.float64)
    shape = data.shape

    # Create coordinate grids
    coords = [np.linspace(-1, 1, s) for s in shape]
    grids = np.meshgrid(*coords, indexing='ij')

    # Estimate low-frequency bias field
    bias = np.ones_like(data)
    for grid in grids:
        bias += 0.1 * grid ** 2

    # Correct
    corrected = data / bias
    corrected = (corrected - corrected.min()) / (corrected.max() - corrected.min())

    return ImageVolume(
        array=corrected.astype(np.float32),
        spacing=volume.spacing,
        origin=volume.origin,
        orientation=volume.orientation,
        modality=Modality.MRI,
        metadata=volume.metadata,
    )


def skull_strip(volume: ImageVolume, threshold: float = 0.1) -> tuple[ImageVolume, np.ndarray]:
    """Simple skull stripping based on intensity thresholding."""
    from scipy import ndimage

    data = volume.array
    mask = data > threshold

    # Morphological operations to clean up
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_erosion(mask, iterations=2)
    mask = ndimage.binary_dilation(mask, iterations=2)

    stripped = data * mask

    return ImageVolume(
        array=stripped,
        spacing=volume.spacing,
        origin=volume.origin,
        orientation=volume.orientation,
        modality=Modality.MRI,
        metadata=volume.metadata,
    ), mask
