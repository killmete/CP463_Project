"""Replay a recorded video file with deterministic timestamps."""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from drowsiness.capture.base import FrameSource
from drowsiness.types import FramePacket

DEFAULT_FPS = 30.0


class VideoFileSource(FrameSource):
    """Reads every frame in order (no dropping). timestamp = index / fps."""

    def __init__(self, path: str | Path, fps_override: float | None = None) -> None:
        self.path = Path(path)
        self._fps_override = fps_override
        self._fps: float | None = None
        self._cap: cv2.VideoCapture | None = None
        self._index = 0

    def open(self) -> None:
        if not self.path.is_file():
            raise FileNotFoundError(f"video file not found: {self.path}")
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise OSError(f"cannot open video file: {self.path}")
        self._cap = cap
        self._fps = self._fps_override or cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
        self._index = 0

    def read(self) -> FramePacket | None:
        if self._cap is None:
            raise RuntimeError("source is not open")
        ok, image = self._cap.read()
        grabbed_at = time.perf_counter()
        if not ok:
            return None
        index = self._index
        self._index += 1
        return FramePacket(image, index / self._fps, index, grabbed_at)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def fps(self) -> float | None:
        return self._fps
