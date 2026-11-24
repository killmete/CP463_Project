"""The FrameSource interface every input (camera, video file, ...) implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

from drowsiness.types import FramePacket


# สัญญากลางของแหล่งภาพทุกชนิด (กล้อง, ไฟล์วิดีโอ)
# ส่วนอื่นของโปรแกรมเรียกแค่ open/read/close จึงไม่ต้องรู้ว่าภาพมาจากไหน
class FrameSource(ABC):
    """A stream of frames. Use as a context manager: `with source: ...`."""

    @abstractmethod
    # เปิดแหล่งภาพ (คลาสลูกต้องเขียนเอง)
    def open(self) -> None: ...

    @abstractmethod
    # อ่านเฟรมถัดไป คืน None เมื่อภาพหมด (เช่น วิดีโอจบ)
    def read(self) -> FramePacket | None:
        """Return the next frame, or None when the stream has ended."""

    @abstractmethod
    # ปิดและคืนทรัพยากร (กล้อง, ไฟล์)
    def close(self) -> None: ...

    @property
    @abstractmethod
    # จำนวนเฟรมต่อวินาทีที่แหล่งภาพบอก (None = ไม่รู้)
    def fps(self) -> float | None:
        """Nominal frames per second, if known."""

    @property
    # จำนวนเฟรมที่ถูกทิ้ง ค่าเริ่มต้น 0 (ไฟล์วิดีโอไม่ทิ้งเฟรม)
    def dropped_frames(self) -> int:
        return 0

    # ทำให้ใช้ `with source:` ได้: เปิดตอนเข้า ปิดตอนออกเสมอแม้มี error
    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
