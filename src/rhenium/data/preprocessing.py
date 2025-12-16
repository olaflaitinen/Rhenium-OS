"""
Rhenium OS Preprocessing Pipeline.

This module provides preprocessing operations for medical imaging data
including resampling, normalization, and cropping/padding.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np
from scipy import ndimage

from rhenium.data.volume import ImageVolume


class Preprocessor(Protocol):
    """Protocol for preprocessing operations."""

    def __call__(self, volume: ImageVolume, **kwargs: Any) -> ImageVolume:
        """Apply preprocessing to volume."""
        ...


@dataclass
class Resample:
    """
    Resample volume to target spacing.

    Attributes:
        target_spacing: Target voxel spacing (z, y, x) in mm
        order: Interpolation order (0=nearest, 1=linear, 3=cubic)
    """

    target_spacing: tuple[float, float, float]
    order: int = 1

    def __call__(self, volume: ImageVolume, **kwargs: Any) -> ImageVolume:
        """Apply resampling."""
        return volume.resample(self.target_spacing, self.order)


@dataclass
class Normalize:
    """
    Normalize intensity values.

    Attributes:
        method: Normalization method ("minmax", "zscore", "percentile")
        clip_percentile: Optional percentile clipping range
    """

    method: Literal["minmax", "zscore", "percentile"] = "minmax"
    clip_percentile: tuple[float, float] | None = None

    def __call__(self, volume: ImageVolume, **kwargs: Any) -> ImageVolume:
        """Apply normalization."""
        return volume.normalize(self.method, self.clip_percentile)


@dataclass
class CropOrPad:
    """
    Crop or pad volume to target size.

    Attributes:
        target_size: Target size (D, H, W)
        mode: "center" for center crop/pad, "random" for random crop
        pad_value: Value to use for padding
    """

    target_size: tuple[int, int, int]
    mode: Literal["center", "random"] = "center"
    pad_value: float = 0.0

    def __call__(self, volume: ImageVolume, **kwargs: Any) -> ImageVolume:
        """Apply crop or pad."""
        data = volume.array
        current_size = data.shape[:3]
        target = self.target_size

        # Calculate differences
        diff = [t - c for t, c in zip(target, current_size)]

        # Handle each dimension
        result = data
        new_origin = list(volume.origin)

        for axis, d in enumerate(diff):
            if d > 0:
                # Pad
                pad_before = d // 2
                pad_after = d - pad_before
                pad_width = [(0, 0)] * result.ndim
                pad_width[axis] = (pad_before, pad_after)
                result = np.pad(result, pad_width, constant_values=self.pad_value)
                # Adjust origin
                new_origin[axis] -= pad_before * volume.spacing[axis]
            elif d < 0:
                # Crop
                crop_size = -d
                if self.mode == "center":
                    start = crop_size // 2
                else:  # random
                    start = np.random.randint(0, crop_size + 1)
                end = start + target[axis]
                slices = [slice(None)] * result.ndim
                slices[axis] = slice(start, end)
                result = result[tuple(slices)]
                # Adjust origin
                new_origin[axis] += start * volume.spacing[axis]

        return ImageVolume(
            array=result,
            spacing=volume.spacing,
            origin=tuple(new_origin),
            orientation=volume.orientation.copy(),
            modality=volume.modality,
            metadata=volume.metadata,
        )


@dataclass
class WindowLevel:
    """
    Apply CT window/level.

    Attributes:
        window_center: Window center (level)
        window_width: Window width
    """

    window_center: float
    window_width: float

    def __call__(self, volume: ImageVolume, **kwargs: Any) -> ImageVolume:
        """Apply windowing."""
        return volume.apply_window(self.window_center, self.window_width)


@dataclass
class GaussianSmooth:
    """
    Apply Gaussian smoothing.

    Attributes:
        sigma: Standard deviation of Gaussian kernel
    """

    sigma: float = 1.0

    def __call__(self, volume: ImageVolume, **kwargs: Any) -> ImageVolume:
        """Apply Gaussian smoothing."""
        smoothed = ndimage.gaussian_filter(volume.array, sigma=self.sigma)

        return ImageVolume(
            array=smoothed,
            spacing=volume.spacing,
            origin=volume.origin,
            orientation=volume.orientation.copy(),
            modality=volume.modality,
            metadata=volume.metadata,
        )


@dataclass
class IntensityClip:
    """
    Clip intensity values to range.

    Attributes:
        min_value: Minimum value
        max_value: Maximum value
    """

    min_value: float | None = None
    max_value: float | None = None

    def __call__(self, volume: ImageVolume, **kwargs: Any) -> ImageVolume:
        """Apply intensity clipping."""
        data = volume.array.copy()

        if self.min_value is not None:
            data = np.maximum(data, self.min_value)
        if self.max_value is not None:
            data = np.minimum(data, self.max_value)

        return ImageVolume(
            array=data,
            spacing=volume.spacing,
            origin=volume.origin,
            orientation=volume.orientation.copy(),
            modality=volume.modality,
            metadata=volume.metadata,
        )


@dataclass
class PreprocessingPipeline:
    """
    Composable preprocessing pipeline.

    Attributes:
        steps: List of preprocessors to apply in order
    """

    steps: list[Preprocessor] = field(default_factory=list)

    def __call__(self, volume: ImageVolume, **kwargs: Any) -> ImageVolume:
        """Apply all preprocessing steps."""
        result = volume
        for step in self.steps:
            result = step(result, **kwargs)
        return result

    def add(self, step: Preprocessor) -> "PreprocessingPipeline":
        """Add a step to the pipeline."""
        self.steps.append(step)
        return self

    @classmethod
    def for_mri(
        cls,
        target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
        target_size: tuple[int, int, int] | None = None,
    ) -> "PreprocessingPipeline":
        """Create standard MRI preprocessing pipeline."""
        steps: list[Preprocessor] = [
            Resample(target_spacing=target_spacing),
            Normalize(method="percentile", clip_percentile=(0.5, 99.5)),
        ]
        if target_size:
            steps.append(CropOrPad(target_size=target_size))
        return cls(steps=steps)

    @classmethod
    def for_ct(
        cls,
        target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
        target_size: tuple[int, int, int] | None = None,
        window_center: float = 40,
        window_width: float = 400,
    ) -> "PreprocessingPipeline":
        """Create standard CT preprocessing pipeline."""
        steps: list[Preprocessor] = [
            Resample(target_spacing=target_spacing),
            WindowLevel(window_center=window_center, window_width=window_width),
        ]
        if target_size:
            steps.append(CropOrPad(target_size=target_size))
        return cls(steps=steps)
