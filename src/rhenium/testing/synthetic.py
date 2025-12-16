"""
Rhenium OS Synthetic Data Generator.

This module provides synthetic data generation for testing without
requiring real patient data. Generates realistic-looking medical imaging
data including volumes, k-space, sinograms, and DICOM-like structures.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from rhenium.data.volume import ImageVolume, Modality, MRISequence, VolumeMetadata


@dataclass
class SyntheticStudy:
    """Synthetic DICOM-like study structure."""

    study_uid: str
    patient_id: str
    study_date: str
    series: list["SyntheticSeries"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "study_uid": self.study_uid,
            "patient_id": self.patient_id,
            "study_date": self.study_date,
            "series": [s.to_dict() for s in self.series],
        }


@dataclass
class SyntheticSeries:
    """Synthetic series with volume data."""

    series_uid: str
    modality: str
    body_part: str
    volume: np.ndarray
    spacing: tuple[float, float, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "series_uid": self.series_uid,
            "modality": self.modality,
            "body_part": self.body_part,
            "shape": list(self.volume.shape),
            "spacing": list(self.spacing),
        }


class SyntheticDataGenerator:
    """
    Generate synthetic medical imaging data for testing.

    This generator creates realistic-looking synthetic data without
    using any real patient data, suitable for CI/CD testing.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize generator with random seed.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)
        self._uid_counter = 0

    def _generate_uid(self) -> str:
        """Generate a unique identifier."""
        self._uid_counter += 1
        return f"1.2.826.0.1.{self._uid_counter}.{uuid.uuid4().int % 1000000}"

    def generate_volume(
        self,
        shape: tuple[int, int, int] = (64, 128, 128),
        modality: str | Modality = "MRI",
        add_lesion: bool = False,
        noise_level: float = 0.1,
        spacing: tuple[float, float, float] = (3.0, 1.0, 1.0),
    ) -> ImageVolume:
        """
        Generate a synthetic 3D volume.

        Args:
            shape: Volume shape (D, H, W)
            modality: Imaging modality
            add_lesion: Whether to add a synthetic lesion
            noise_level: Amount of Gaussian noise to add
            spacing: Voxel spacing in mm

        Returns:
            ImageVolume with synthetic data
        """
        if isinstance(modality, str):
            modality = Modality(modality) if modality in [m.value for m in Modality] else Modality.UNKNOWN

        # Generate base anatomy
        volume = self._generate_anatomy(shape)

        # Add lesion if requested
        if add_lesion:
            volume = self._add_lesion(volume)

        # Add noise
        volume = volume + self.rng.normal(0, noise_level, shape)
        volume = np.clip(volume, 0, 1).astype(np.float32)

        return ImageVolume(
            array=volume,
            spacing=spacing,
            modality=modality,
            metadata=VolumeMetadata(
                study_uid=self._generate_uid(),
                series_uid=self._generate_uid(),
                modality=modality,
            ),
        )

    def _generate_anatomy(self, shape: tuple[int, int, int]) -> np.ndarray:
        """Generate synthetic anatomy with ellipsoids."""
        D, H, W = shape
        volume = np.zeros(shape, dtype=np.float32)

        # Create coordinate grids
        z, y, x = np.meshgrid(
            np.linspace(-1, 1, D),
            np.linspace(-1, 1, H),
            np.linspace(-1, 1, W),
            indexing='ij',
        )

        # Add main body ellipsoid
        body = (x**2 / 0.7**2 + y**2 / 0.8**2 + z**2 / 0.9**2) < 1.0
        volume[body] = 0.3 + self.rng.random() * 0.1

        # Add internal structure (smaller ellipsoid)
        inner = (x**2 / 0.4**2 + y**2 / 0.5**2 + z**2 / 0.6**2) < 1.0
        volume[inner] = 0.5 + self.rng.random() * 0.1

        # Add some heterogeneity
        texture = self.rng.random(shape) * 0.1
        volume = volume + texture * body

        return volume

    def _add_lesion(
        self,
        volume: np.ndarray,
        center: tuple[float, float, float] | None = None,
        radius: float | None = None,
    ) -> np.ndarray:
        """Add a synthetic lesion to the volume."""
        D, H, W = volume.shape

        if center is None:
            center = (
                0.3 + self.rng.random() * 0.4,
                0.3 + self.rng.random() * 0.4,
                0.3 + self.rng.random() * 0.4,
            )
        if radius is None:
            radius = 0.05 + self.rng.random() * 0.1

        z, y, x = np.meshgrid(
            np.linspace(0, 1, D),
            np.linspace(0, 1, H),
            np.linspace(0, 1, W),
            indexing='ij',
        )

        dist = np.sqrt(
            (z - center[0])**2 +
            (y - center[1])**2 +
            (x - center[2])**2
        )

        lesion_mask = dist < radius
        lesion_value = 0.8 + self.rng.random() * 0.2

        result = volume.copy()
        result[lesion_mask] = lesion_value

        return result

    def generate_segmentation_mask(
        self,
        shape: tuple[int, int, int] = (64, 128, 128),
        num_classes: int = 3,
    ) -> np.ndarray:
        """
        Generate synthetic segmentation mask.

        Args:
            shape: Mask shape (D, H, W)
            num_classes: Number of segmentation classes

        Returns:
            Integer mask with class labels
        """
        mask = np.zeros(shape, dtype=np.int64)
        D, H, W = shape
        center = np.array(shape) // 2

        # Add spherical regions for each class
        for c in range(1, num_classes):
            offset = self.rng.integers(-10, 10, size=3)
            radius = 8 + self.rng.integers(0, 8)

            z, y, x = np.ogrid[:D, :H, :W]
            dist = np.sqrt(
                (z - center[0] - offset[0])**2 +
                (y - center[1] - offset[1])**2 +
                (x - center[2] - offset[2])**2
            )
            mask[dist < radius] = c

        return mask

    def generate_kspace(
        self,
        image_shape: tuple[int, int] = (256, 256),
        acceleration: int = 4,
        center_fraction: float = 0.08,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic k-space data with undersampling.

        Args:
            image_shape: Image dimensions (H, W)
            acceleration: Undersampling acceleration factor
            center_fraction: Fraction of center k-space to fully sample

        Returns:
            Tuple of (undersampled k-space, sampling mask)
        """
        # Generate phantom image
        phantom = self._generate_2d_phantom(image_shape)

        # Compute full k-space
        kspace = np.fft.fft2(phantom)
        kspace = np.fft.fftshift(kspace)

        # Generate undersampling mask
        mask = self._generate_undersampling_mask(
            image_shape,
            acceleration,
            center_fraction,
        )

        return kspace * mask, mask

    def _generate_2d_phantom(self, shape: tuple[int, int]) -> np.ndarray:
        """Generate 2D phantom image (Shepp-Logan style)."""
        H, W = shape
        phantom = np.zeros(shape, dtype=np.float32)

        y, x = np.meshgrid(
            np.linspace(-1, 1, H),
            np.linspace(-1, 1, W),
            indexing='ij',
        )

        # Main ellipse
        e1 = ((x / 0.69)**2 + (y / 0.92)**2) < 1.0
        phantom[e1] = 1.0

        # Inner ellipse
        e2 = ((x / 0.6)**2 + (y / 0.5)**2) < 1.0
        phantom[e2] = 0.8

        # Small circles for features
        for _ in range(3):
            cx = self.rng.uniform(-0.3, 0.3)
            cy = self.rng.uniform(-0.3, 0.3)
            r = self.rng.uniform(0.05, 0.12)
            circle = ((x - cx)**2 + (y - cy)**2) < r**2
            phantom[circle] = self.rng.uniform(0.5, 0.9)

        return phantom

    def _generate_undersampling_mask(
        self,
        shape: tuple[int, int],
        acceleration: int,
        center_fraction: float,
    ) -> np.ndarray:
        """Generate random undersampling mask for k-space."""
        H, W = shape
        mask = np.zeros(shape, dtype=np.float32)

        # Fully sample center
        center_lines = int(W * center_fraction)
        center_start = W // 2 - center_lines // 2
        center_end = center_start + center_lines
        mask[:, center_start:center_end] = 1.0

        # Random sample remaining lines
        remaining_lines = list(range(0, center_start)) + list(range(center_end, W))
        num_to_sample = len(remaining_lines) // acceleration

        sampled = self.rng.choice(remaining_lines, size=num_to_sample, replace=False)
        mask[:, sampled] = 1.0

        return mask

    def generate_sinogram(
        self,
        image_shape: tuple[int, int] = (256, 256),
        num_angles: int = 180,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic sinogram from phantom.

        Args:
            image_shape: Phantom image dimensions
            num_angles: Number of projection angles

        Returns:
            Tuple of (sinogram, angles in radians)
        """
        # Generate phantom
        phantom = self._generate_2d_phantom(image_shape)

        # Projection angles
        angles = np.linspace(0, np.pi, num_angles, endpoint=False)

        # Simple forward projection (Radon transform approximation)
        H, W = image_shape
        num_detectors = int(np.ceil(np.sqrt(H**2 + W**2)))
        sinogram = np.zeros((num_angles, num_detectors), dtype=np.float32)

        for i, theta in enumerate(angles):
            # Rotate and sum
            from scipy import ndimage
            rotated = ndimage.rotate(phantom, np.degrees(theta), reshape=False)
            projection = np.sum(rotated, axis=0)

            # Resize to detector count
            if len(projection) != num_detectors:
                from scipy.ndimage import zoom
                projection = zoom(projection, num_detectors / len(projection))

            sinogram[i, :len(projection)] = projection[:num_detectors]

        return sinogram, angles

    def generate_study(
        self,
        num_series: int = 2,
        modalities: list[str] | None = None,
        shape: tuple[int, int, int] = (32, 128, 128),
    ) -> SyntheticStudy:
        """
        Generate a complete synthetic study with multiple series.

        Args:
            num_series: Number of series to generate
            modalities: List of modalities (random if None)
            shape: Volume shape for each series

        Returns:
            SyntheticStudy with series
        """
        if modalities is None:
            all_modalities = ["MRI", "CT", "US", "XRAY"]
            modalities = [self.rng.choice(all_modalities) for _ in range(num_series)]

        study = SyntheticStudy(
            study_uid=self._generate_uid(),
            patient_id=f"SYNTH_{self.rng.integers(1000, 9999)}",
            study_date=datetime.now().strftime("%Y%m%d"),
        )

        for i, mod in enumerate(modalities[:num_series]):
            volume = self.generate_volume(shape=shape, modality=mod)

            series = SyntheticSeries(
                series_uid=self._generate_uid(),
                modality=mod,
                body_part="SYNTHETIC",
                volume=volume.array,
                spacing=volume.spacing,
            )
            study.series.append(series)

        return study

    def generate_paired_data(
        self,
        shape: tuple[int, int, int] = (32, 128, 128),
        noise_level: float = 0.2,
    ) -> tuple[ImageVolume, ImageVolume]:
        """
        Generate paired data for image-to-image translation.

        Args:
            shape: Volume shape
            noise_level: Noise level for degraded version

        Returns:
            Tuple of (clean volume, noisy volume)
        """
        clean = self.generate_volume(shape=shape, noise_level=0.05)

        noisy_array = clean.array + self.rng.normal(0, noise_level, shape)
        noisy_array = np.clip(noisy_array, 0, 1).astype(np.float32)

        noisy = ImageVolume(
            array=noisy_array,
            spacing=clean.spacing,
            modality=clean.modality,
            metadata=clean.metadata,
        )

        return clean, noisy

    def generate_low_high_res_pair(
        self,
        high_res_shape: tuple[int, int, int] = (64, 256, 256),
        scale_factor: int = 4,
    ) -> tuple[ImageVolume, ImageVolume]:
        """
        Generate low/high resolution pair for super-resolution.

        Args:
            high_res_shape: High resolution shape
            scale_factor: Downsampling factor

        Returns:
            Tuple of (low-res volume, high-res volume)
        """
        from scipy.ndimage import zoom

        high_res = self.generate_volume(shape=high_res_shape, noise_level=0.05)

        # Downsample
        low_res_array = zoom(
            high_res.array,
            1 / scale_factor,
            order=1,
        )

        low_res = ImageVolume(
            array=low_res_array,
            spacing=tuple(s * scale_factor for s in high_res.spacing),
            modality=high_res.modality,
            metadata=high_res.metadata,
        )

        return low_res, high_res
