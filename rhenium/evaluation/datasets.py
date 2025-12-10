# Copyright (c) 2025 Skolyn LLC. All rights reserved.
# SPDX-License-Identifier: EUPL-1.1

"""Dataset abstractions for evaluation."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Any
import numpy as np


@dataclass
class EvaluationSample:
    """Single evaluation sample."""
    sample_id: str
    image: np.ndarray
    ground_truth: np.ndarray | None = None
    metadata: dict[str, Any] | None = None


class BaseDataset(ABC):
    """Abstract dataset interface."""

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> EvaluationSample:
        pass

    def __iter__(self) -> Iterator[EvaluationSample]:
        for i in range(len(self)):
            yield self[i]
