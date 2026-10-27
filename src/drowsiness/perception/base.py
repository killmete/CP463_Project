"""The LandmarkDetector interface, so MediaPipe can be swapped (or faked in tests)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class LandmarkResult:
    """One face: landmarks normalised to [0, 1] of the image (x, y) plus relative depth."""

    landmarks: np.ndarray
    """(478, 3) float32, normalised x, y and MediaPipe z (same scale as x)."""
    blendshapes: Mapping[str, float] | None = None


class LandmarkDetector(Protocol):
    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> LandmarkResult | None:
        """Landmarks of the driver's face in an RGB uint8 image, or None if no face.

        `timestamp_ms` must be strictly increasing across calls (video tracking).
        """
        ...

    def close(self) -> None: ...
