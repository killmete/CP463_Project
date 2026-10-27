"""Perception: one frame -> FrameFeatures (EAR, MAR, head pose, blendshapes).

Importing this package does not import mediapipe; `build_perception` does.
"""

from __future__ import annotations

from drowsiness.perception.base import LandmarkDetector, LandmarkResult
from drowsiness.perception.pipeline import (
    ModelNotFoundError,
    Perception,
    PerceptionSetupError,
    build_perception,
    resolve_model_path,
)

__all__ = [
    "LandmarkDetector",
    "LandmarkResult",
    "ModelNotFoundError",
    "PerceptionSetupError",
    "Perception",
    "build_perception",
    "resolve_model_path",
]
