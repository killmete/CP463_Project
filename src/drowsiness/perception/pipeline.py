"""One FramePacket in, one PerceptionOutput (FrameFeatures + landmarks) out.

Stateless per frame apart from the detector's own video tracking and the
timestamp guard that keeps MediaPipe's timestamps strictly increasing.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Self

import cv2
import numpy as np

from drowsiness.config import Config, PerceptionConfig, PreprocessConfig
from drowsiness.perception import features as F
from drowsiness.perception.base import LandmarkDetector
from drowsiness.perception.preprocess import Clahe
from drowsiness.types import FrameFeatures, FramePacket, PerceptionOutput

FETCH_COMMAND = "python -m drowsiness.perception.fetch_model"
INSTALL_COMMAND = "pip install --no-deps -r requirements-mediapipe.txt"


class PerceptionSetupError(RuntimeError):
    """Perception is enabled but cannot start (missing mediapipe or model, bad model)."""


class ModelNotFoundError(PerceptionSetupError, FileNotFoundError):
    """Perception is enabled but the landmark model file is missing."""


class Perception:
    def __init__(self, detector: LandmarkDetector, preprocess: PreprocessConfig) -> None:
        self._detector = detector
        self._clahe = (
            Clahe(preprocess.clahe_clip, preprocess.clahe_tile) if preprocess.clahe else None
        )
        self._last_ts_ms: int | None = None

    def process(self, packet: FramePacket) -> PerceptionOutput:
        start = time.perf_counter()
        image = packet.image
        if self._clahe is not None:
            image = self._clahe(image)  # a copy; the overlay draws on the original
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self._detector.detect(rgb, self._next_ts_ms(packet.timestamp))

        if result is None:
            features = FrameFeatures(timestamp=packet.timestamp, face_found=False)
            landmarks = None
        else:
            height, width = packet.image.shape[:2]
            landmarks = result.landmarks.astype(np.float32, copy=True)
            landmarks *= np.array([width, height, width], dtype=np.float32)
            pose = F.head_pose(landmarks, width, height)
            pitch, yaw, roll = pose if pose is not None else (None, None, None)
            features = FrameFeatures(
                timestamp=packet.timestamp,
                face_found=True,
                ear_left=F.ear_left(landmarks),
                ear_right=F.ear_right(landmarks),
                mar=F.mouth_aspect_ratio(landmarks),
                pitch=pitch,
                yaw=yaw,
                roll=roll,
                blendshapes=result.blendshapes,
            )
        perception_ms = (time.perf_counter() - start) * 1000.0
        return PerceptionOutput(features, landmarks, perception_ms)

    def _next_ts_ms(self, timestamp_s: float) -> int:
        ts = round(timestamp_s * 1000.0)
        if self._last_ts_ms is not None and ts <= self._last_ts_ms:
            ts = self._last_ts_ms + 1
        self._last_ts_ms = ts
        return ts

    def close(self) -> None:
        self._detector.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def resolve_model_path(cfg: PerceptionConfig) -> Path:
    path = Path(cfg.model_path).expanduser().resolve()
    if not path.is_file():
        raise ModelNotFoundError(
            f"landmark model not found: {path}\n"
            f"  download it with: {FETCH_COMMAND}  (or set perception.model_path)"
        )
    return path


def build_perception(cfg: Config) -> Perception:
    """Create the MediaPipe-backed perception stage (imports mediapipe here, lazily)."""
    model_path = resolve_model_path(cfg.perception)
    try:
        from drowsiness.perception.mediapipe_detector import MediaPipeDetector
    except ImportError as e:
        raise PerceptionSetupError(
            f"cannot import mediapipe ({e})\n"
            f"  install it with: {INSTALL_COMMAND}  (or run with --no-perception)"
        ) from e
    try:
        detector = MediaPipeDetector(cfg.perception, model_path)
    except (RuntimeError, ValueError) as e:
        raise PerceptionSetupError(
            f"cannot load landmark model {model_path}: {e}\n  re-download it with: {FETCH_COMMAND}"
        ) from e
    return Perception(detector, cfg.preprocess)
