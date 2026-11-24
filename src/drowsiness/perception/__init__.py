"""Perception: one frame -> FrameFeatures (EAR, MAR, head pose, blendshapes).

Importing this package does not import mediapipe; `build_perception` does.
"""

from __future__ import annotations

# รวม class/ฟังก์ชันที่ส่วนอื่นใช้ไว้ที่เดียว
# import แพ็กเกจนี้ยังไม่โหลด mediapipe (โหลดตอน build_perception เท่านั้น)
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
