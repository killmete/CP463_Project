"""The FrameSource interface every input (camera, video file, ...) implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

from drowsiness.types import FramePacket


class FrameSource(ABC):
    """A stream of frames. Use as a context manager: `with source: ...`."""

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def read(self) -> FramePacket | None:
        """Return the next frame, or None when the stream has ended."""

    @abstractmethod
    def close(self) -> None: ...

    @property
    @abstractmethod
    def fps(self) -> float | None:
        """Nominal frames per second, if known."""

    @property
    def dropped_frames(self) -> int:
        return 0

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
