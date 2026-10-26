from pathlib import Path

import cv2
import numpy as np
import pytest

CLIP_FRAMES = 30
CLIP_FPS = 20.0
CLIP_SIZE = (64, 48)  # width, height


@pytest.fixture
def synthetic_clip(tmp_path: Path) -> Path:
    """A 30-frame, 20 FPS MJPG clip; frame i is a flat gray image of level 8*i."""
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), CLIP_FPS, CLIP_SIZE)
    assert writer.isOpened(), "OpenCV could not create an MJPG writer"
    width, height = CLIP_SIZE
    for i in range(CLIP_FRAMES):
        writer.write(np.full((height, width, 3), 8 * i, dtype=np.uint8))
    writer.release()
    return path
