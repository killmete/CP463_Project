"""Replay a recorded video file with deterministic timestamps."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2

from drowsiness.capture.base import FrameSource
from drowsiness.types import FramePacket

log = logging.getLogger(__name__)

# Containers sometimes report a time base (e.g. 1000 or 90000) instead of a frame rate,
# or nothing. Timestamps drive the timing rules, so refuse to guess outside this range.
# fps ที่ยอมรับได้ 1-240: บางไฟล์รายงานค่าแปลก (1000, 90000) เวลาทั้งระบบจะผิด
MIN_PLAUSIBLE_FPS = 1.0
MAX_PLAUSIBLE_FPS = 240.0


# เล่นไฟล์วิดีโอ: อ่านทุกเฟรมตามลำดับ ไม่ใช้ thread ไม่ทิ้งเฟรม
class VideoFileSource(FrameSource):
    """Reads every frame in order (no dropping). timestamp = index / fps.

    Assumes constant frame rate. For variable-frame-rate clips (e.g. phone recordings),
    re-encode to a constant rate first or set `fps_override`.
    """

    def __init__(self, path: str | Path, fps_override: float | None = None) -> None:
        self.path = Path(path)
        self._fps_override = fps_override
        self._fps: float | None = None
        self._cap: cv2.VideoCapture | None = None
        self._index = 0

    # เปิดไฟล์และหา fps: ใช้ fps_override ถ้ามี ไม่งั้นอ่านจากไฟล์แล้วตรวจว่าสมเหตุสมผล
    def open(self) -> None:
        if not self.path.is_file():
            raise FileNotFoundError(f"video file not found: {self.path}")
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise OSError(f"cannot open video file: {self.path}")
        if self._fps_override is not None:
            fps = self._fps_override
        else:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not MIN_PLAUSIBLE_FPS <= fps <= MAX_PLAUSIBLE_FPS:
                cap.release()
                raise OSError(
                    f"{self.path} reports {fps:g} fps, outside {MIN_PLAUSIBLE_FPS:g}-"
                    f"{MAX_PLAUSIBLE_FPS:g}; set capture.fps_override to the real frame rate"
                )
        log.info("video %s: %.2f fps%s", self.path, fps, " (override)" * bool(self._fps_override))
        self._cap = cap
        self._fps = fps
        self._index = 0

    # อ่านเฟรมถัดไป: timestamp = index / fps (เช่นเฟรม 30 ที่ 30 fps = 1.0 วินาที)
    # ไม่ขึ้นกับความเร็วเครื่อง รันซ้ำได้ผลเท่าเดิมทุกครั้ง (deterministic)
    def read(self) -> FramePacket | None:
        if self._cap is None:
            raise RuntimeError("source is not open")
        ok, image = self._cap.read()
        # เวลาจริง ใช้วัด latency เท่านั้น ไม่ใช้ในการตัดสิน
        grabbed_at = time.perf_counter()
        if not ok:
            return None
        index = self._index
        self._index += 1
        return FramePacket(image, index / self._fps, index, grabbed_at)

    # ปล่อยไฟล์ (เรียกซ้ำได้ปลอดภัย)
    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def fps(self) -> float | None:
        return self._fps
