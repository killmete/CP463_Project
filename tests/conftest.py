# fixture ที่เทสต์หลายไฟล์ใช้ร่วมกัน (pytest โหลดไฟล์นี้อัตโนมัติ)
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest


# ขนาดของวิดีโอปลอมที่ใช้เทสต์ (จำนวนเฟรม, fps, กว้าง, สูง)
@dataclass(frozen=True)
class ClipSpec:
    frames: int = 30
    fps: float = 20.0
    width: int = 64
    height: int = 48


# ค่าเริ่มต้นของวิดีโอปลอม; เทสต์เปลี่ยนได้โดยประกาศ fixture ชื่อเดียวกัน
@pytest.fixture
def clip_spec() -> ClipSpec:
    return ClipSpec()


# สร้างวิดีโอปลอม: เฟรมที่ i เป็นสีเทาระดับ 8*i จึงรู้ได้ว่าอ่านได้เฟรมไหน (ไม่มีหน้าคนจริง)
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


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "face_landmarker.task"


# path ของโมเดล MediaPipe; ถ้าไม่มีไฟล์ให้ข้ามเทสต์ (แต่ใน CI ให้ fail)
@pytest.fixture
def model_path() -> Path:
    """The Face Landmarker model; skipped if absent unless DROWSINESS_REQUIRE_MODEL=1 (CI)."""
    if not MODEL_PATH.is_file():
        message = f"model not found at {MODEL_PATH} (python -m drowsiness.perception.fetch_model)"
        if os.environ.get("DROWSINESS_REQUIRE_MODEL"):
            pytest.fail(message)
        pytest.skip(message)
    return MODEL_PATH
