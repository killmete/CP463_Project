from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest


@dataclass(frozen=True)
class ClipSpec:
    frames: int = 30
    fps: float = 20.0
    width: int = 64
    height: int = 48


@pytest.fixture
def clip_spec() -> ClipSpec:
    return ClipSpec()


@pytest.fixture
def synthetic_clip(tmp_path: Path, clip_spec: ClipSpec) -> Path:
    """An MJPG clip described by `clip_spec`; frame i is a flat gray image of level 8*i."""
    path = tmp_path / "clip.avi"
    size = (clip_spec.width, clip_spec.height)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), clip_spec.fps, size)
    assert writer.isOpened(), "OpenCV could not create an MJPG writer"
    for i in range(clip_spec.frames):
        writer.write(np.full((clip_spec.height, clip_spec.width, 3), 8 * i, dtype=np.uint8))
    writer.release()
    return path
