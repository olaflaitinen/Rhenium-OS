# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
FastMRI Dataset Adapter
=======================

Adapter for fastMRI-style datasets to Rhenium OS internal representations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from rhenium.core.logging import get_reconstruction_logger
from rhenium.data.raw_io import KSpaceData, SamplingPattern

logger = get_reconstruction_logger()


@dataclass
class FastMRISlice:
    """Single slice from fastMRI dataset."""
    kspace: np.ndarray
    target: np.ndarray | None = None
    mask: np.ndarray | None = None
    attrs: dict | None = None


class FastMRIAdapter:
    """
    Adapter for fastMRI HDF5 files.

    Maps fastMRI data structure to Rhenium OS internal representations.
    """

    def __init__(self, data_path: str | Path):
        self.data_path = Path(data_path)
        logger.info("FastMRI adapter initialized", path=str(self.data_path))

    def load_file(self, filename: str) -> KSpaceData:
        """Load a single fastMRI HDF5 file."""
        import h5py

        filepath = self.data_path / filename
        with h5py.File(filepath, "r") as f:
            kspace = np.array(f["kspace"])
            mask = np.array(f["mask"]) if "mask" in f else None

            attrs = dict(f.attrs) if f.attrs else {}

        return KSpaceData(
            data=kspace,
            sampling_mask=mask,
            sampling_pattern=SamplingPattern.CARTESIAN,
            metadata=attrs,
            source_path=filepath,
        )

    def iterate_files(self) -> Iterator[KSpaceData]:
        """Iterate over all HDF5 files in directory."""
        for filepath in sorted(self.data_path.glob("*.h5")):
            yield self.load_file(filepath.name)

    @staticmethod
    def to_image(kspace: np.ndarray) -> np.ndarray:
        """Convert k-space to image domain via inverse FFT."""
        from scipy.fft import ifft2, ifftshift, fftshift

        if kspace.ndim == 2:
            return np.abs(fftshift(ifft2(ifftshift(kspace))))
        elif kspace.ndim == 3:
            # Multi-coil: SoS combination
            images = []
            for coil in range(kspace.shape[0]):
                img = np.abs(fftshift(ifft2(ifftshift(kspace[coil]))))
                images.append(img ** 2)
            return np.sqrt(np.sum(images, axis=0))
        return kspace
