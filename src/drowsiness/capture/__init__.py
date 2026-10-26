"""Frame sources: live camera or recorded video."""

from __future__ import annotations

from drowsiness.capture.base import FrameSource
from drowsiness.capture.camera import CameraError, CameraSource
from drowsiness.capture.video_file import VideoFileSource
from drowsiness.config import CaptureConfig

__all__ = ["CameraError", "CameraSource", "FrameSource", "VideoFileSource", "open_source"]


def open_source(cfg: CaptureConfig) -> FrameSource:
    """Build (but don't open) a source: an int or all-digit string is a camera index,
    anything else a video file path."""
    source = cfg.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)
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
    return VideoFileSource(source)
