"""MediaPipe Face Landmarker (Tasks API, VIDEO mode) behind the LandmarkDetector protocol.

This is the only module that imports mediapipe; it is imported lazily by
`build_perception`, so the capture-only loop never loads it.
"""

from __future__ import annotations

from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from drowsiness.config import PerceptionConfig
from drowsiness.perception.base import LandmarkResult


class MediaPipeDetector:
    def __init__(self, cfg: PerceptionConfig, model_path: Path) -> None:
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=cfg.num_faces,
            min_face_detection_confidence=cfg.min_face_detection_confidence,
            min_face_presence_confidence=cfg.min_face_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
            output_face_blendshapes=cfg.output_blendshapes,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> LandmarkResult | None:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.face_landmarks:
            return None
        faces = [
            np.array([(p.x, p.y, p.z) for p in face], dtype=np.float32)
            for face in result.face_landmarks
        ]
        best = max(range(len(faces)), key=lambda i: _bbox_area(faces[i]))
        blendshapes = None
        if result.face_blendshapes:
            blendshapes = {c.category_name: float(c.score) for c in result.face_blendshapes[best]}
        return LandmarkResult(faces[best], blendshapes)

    def close(self) -> None:
        self._landmarker.close()


def _bbox_area(landmarks: np.ndarray) -> float:
    span = landmarks[:, :2].max(axis=0) - landmarks[:, :2].min(axis=0)
    return float(span[0] * span[1])
