"""Frame sources: live camera or recorded video."""

from __future__ import annotations

from drowsiness.capture.base import FrameSource
from drowsiness.capture.camera import CameraError, CameraSource
from drowsiness.capture.video_file import VideoFileSource
from drowsiness.config import CaptureConfig

__all__ = ["CameraError", "CameraSource", "FrameSource", "VideoFileSource", "open_source"]


# เลือกแหล่งภาพจาก config: ตัวเลข = กล้อง, อย่างอื่น = path ไฟล์วิดีโอ
def open_source(cfg: CaptureConfig) -> FrameSource:
    """Build (but don't open) a source: an int or all-digit string is a camera index,
    anything else a video file path."""
    source = cfg.source
    # "0" จาก command line เป็นสตริง แปลงเป็น int ก่อน
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    # เป็นเลขกล้อง -> สร้าง CameraSource พร้อมค่าจาก config
    if isinstance(source, int):
        return CameraSource(
            index=source,
            backend=cfg.backend,
            width=cfg.width,
            height=cfg.height,
            fps=cfg.fps,
            buffer_size=cfg.buffer_size,
            read_timeout_s=cfg.read_timeout_s,
        )
    # ไม่ใช่ตัวเลข -> ถือเป็นไฟล์วิดีโอ
    return VideoFileSource(source, fps_override=cfg.fps_override)
