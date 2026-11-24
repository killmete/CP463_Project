"""MediaPipe Face Landmarker (Tasks API, VIDEO mode) behind the LandmarkDetector protocol.

This is the only module that imports mediapipe; it is imported lazily by
`build_perception`, so the capture-only loop never loads it.
"""

from __future__ import annotations

import sys
from pathlib import Path

from drowsiness.config import PerceptionConfig
from drowsiness.perception.base import LandmarkResult


# กันปัญหาบน Linux ไม่มีระบบเสียง: mediapipe import sounddevice แล้ว crash
# จึงบอก Python ว่าไม่มี sounddevice ให้ mediapipe ข้ามไป
def _guard_audio_import() -> None:
    """`import mediapipe` also imports its audio tasks, which import sounddevice.

    mediapipe only tolerates sounddevice failing with ImportError/OSError, but on Linux
    without an audio server PortAudio raises PortAudioError at import time. Mark the
    module as unavailable so mediapipe skips it (we never use its audio tasks).
    """
    try:
        import sounddevice  # noqa: F401
    except (ImportError, OSError):
        pass  # mediapipe handles these itself
    except Exception:  # PortAudioError: no usable audio backend
        sys.modules["sounddevice"] = None  # type: ignore[assignment]


_guard_audio_import()

import mediapipe as mp  # noqa: E402
import numpy as np  # noqa: E402
from mediapipe.tasks.python import BaseOptions, vision  # noqa: E402


# ตัวตรวจจับจริง: ห่อ MediaPipe Face Landmarker ให้ตรงสัญญา LandmarkDetector
class MediaPipeDetector:
    def __init__(self, cfg: PerceptionConfig, model_path: Path) -> None:
        # ตั้งค่า: path โมเดล, โหมด VIDEO (ติดตามหน้าข้ามเฟรม เร็วกว่า IMAGE),
        # จำนวนหน้า, ค่าความมั่นใจขั้นต่ำ 3 ค่า, และเปิด blendshapes
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

    # หาหน้าในภาพ RGB 1 ภาพ คืนหน้าคนขับ หรือ None ถ้าไม่เจอ
    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> LandmarkResult | None:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.face_landmarks:
            return None
        # แปลงทุกหน้าที่เจอเป็น array (478, 3)
        faces = [
            np.array([(p.x, p.y, p.z) for p in face], dtype=np.float32)
            for face in result.face_landmarks
        ]
        # หลายหน้า (เช่นมีผู้โดยสาร) -> เลือกหน้าที่กรอบใหญ่สุด = ใกล้กล้องสุด = คนขับ
        best = max(range(len(faces)), key=lambda i: _bbox_area(faces[i]))
        # blendshapes ของหน้าเดียวกัน แปลงเป็น dict ชื่อ -> คะแนน
        blendshapes = None
        if result.face_blendshapes:
            blendshapes = {c.category_name: float(c.score) for c in result.face_blendshapes[best]}
        return LandmarkResult(faces[best], blendshapes)

    def close(self) -> None:
        self._landmarker.close()


# พื้นที่กรอบสี่เหลี่ยมที่ครอบจุดทั้งหมดของหน้า
def _bbox_area(landmarks: np.ndarray) -> float:
    span = landmarks[:, :2].max(axis=0) - landmarks[:, :2].min(axis=0)
    return float(span[0] * span[1])
