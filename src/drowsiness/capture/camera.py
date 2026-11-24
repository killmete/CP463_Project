"""Live camera input with a background grabber thread.

The grabber keeps only the newest frame, so processing always works on the latest
image and latency never builds up when processing is slower than the camera.
The grabber thread owns the VideoCapture and is the only code that reads or
releases it, so `close()` can never race a blocked native read.
"""

from __future__ import annotations

import logging
import sys
import threading
import time

import cv2
import numpy as np

from drowsiness.capture.base import FrameSource
from drowsiness.types import FramePacket

log = logging.getLogger(__name__)

# ชื่อ backend -> ค่าคงที่ของ OpenCV (MSMF/DSHOW ใช้บน Windows)
_BACKENDS = {"msmf": cv2.CAP_MSMF, "dshow": cv2.CAP_DSHOW, "any": cv2.CAP_ANY}
# อ่านพลาดชั่วคราว: รอ 5 ms แล้วลองใหม่
_RETRY_SLEEP_S = 0.005
# ตอนปิด รอ thread หยุดได้สูงสุด 2 วินาที
_JOIN_TIMEOUT_S = 2.0


# error ของกล้อง สืบจาก OSError เพื่อให้ main จับรวมแล้วจบด้วย exit code 1
class CameraError(OSError):
    """The camera could not be opened, or stopped delivering frames."""


# ลำดับ backend ที่จะลอง: "auto" บน Windows ลอง msmf ก่อน แล้วค่อย dshow
def backend_order(backend: str) -> list[str]:
    """Backends to try, in order. 'auto' tries MSMF then DSHOW on Windows."""
    if backend != "auto":
        return [backend]
    return ["msmf", "dshow"] if sys.platform == "win32" else ["any"]


# กล้องสด: มี thread แยกคอยอ่านภาพ และเก็บไว้แค่ภาพล่าสุดภาพเดียว
class CameraSource(FrameSource):
    def __init__(
        self,
        index: int = 0,
        backend: str = "auto",
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
        buffer_size: int = 1,
        read_timeout_s: float = 1.0,
    ) -> None:
        self.index = index
        self.backend = backend
        self.width = width
        self.height = height
        self.requested_fps = fps
        self.buffer_size = buffer_size
        self.read_timeout_s = read_timeout_s

        # ---- สถานะภายใน ----
        # _stop = ธงสั่งหยุด thread, _cond = lock + ปลุกคนที่รอภาพ
        # _latest_seq = เลขภาพล่าสุดที่ grab, _read_seq = เลขภาพล่าสุดที่ส่งให้ read() แล้ว
        self.backend_used: str | None = None
        self._fps: float | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._cond = threading.Condition()
        self._latest: tuple[np.ndarray, float] | None = None
        self._latest_seq = -1  # sequence number of the newest grabbed frame
        self._read_seq = -1  # sequence number of the last frame handed to read()
        self._ended = False
        self._error: str | None = None
        self._dropped = 0

    # เปิดกล้อง: ลองทีละ backend จนสำเร็จ แล้วเริ่ม thread grabber
    def open(self) -> None:
        if self._thread is not None:
            raise RuntimeError("camera source is already open (or its grabber is still stopping)")
        # for ... else: ส่วน else ทำงานเมื่อวนครบโดยไม่ break = เปิดไม่ได้เลย
        for name in backend_order(self.backend):
            cap = self._try_open(name)
            if cap is not None:
                self.backend_used = name
                break
        else:
            raise CameraError(f"cannot open camera {self.index} (backend={self.backend})")

        # อ่านขนาดและ fps ที่กล้องให้จริง (อาจไม่ตรงที่ขอ)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = cap.get(cv2.CAP_PROP_FPS) or self.requested_fps
        log.info(
            "camera %d opened with %s: %dx%d @ %.1f fps",
            self.index,
            self.backend_used,
            actual_w,
            actual_h,
            self._fps,
        )

        # รีเซ็ตสถานะทั้งหมด (เผื่อเปิดใหม่หลังปิด)
        self._stop.clear()
        self._latest, self._latest_seq, self._read_seq = None, -1, -1
        self._ended, self._error, self._dropped = False, None, 0
        # thread แบบ daemon: โปรแกรมปิดได้แม้ thread ค้าง; thread เป็นเจ้าของ cap คนเดียว
        self._thread = threading.Thread(
            target=self._grab_loop, args=(cap,), name="camera-grabber", daemon=True
        )
        self._thread.start()

    # ลองเปิดกล้องด้วย backend เดียว คืน None ถ้าไม่สำเร็จ
    def _try_open(self, name: str) -> cv2.VideoCapture | None:
        cap = cv2.VideoCapture(self.index, _BACKENDS[name])
        if not cap.isOpened():
            log.info("camera %d: backend %s failed to open", self.index, name)
            cap.release()
            return None
        # ขอขนาด, fps และ buffer เล็ก (1) เพื่อไม่ให้มีภาพเก่าค้างใน driver
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.requested_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        # Some Windows backends open but only return empty frames; check one.
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            log.info("camera %d: backend %s returned no frame", self.index, name)
            cap.release()
            return None
        return cap

    # ทำงานใน thread แยก: อ่านภาพจากกล้องวนไปเรื่อยๆ เก็บแค่ภาพล่าสุด
    def _grab_loop(self, cap: cv2.VideoCapture) -> None:
        # เวลาที่อ่านภาพสำเร็จล่าสุด ใช้ตรวจว่ากล้องหายนานเกินไหม
        last_good = time.perf_counter()
        try:
            while not self._stop.is_set():
                try:
                    ok, frame = cap.read()
                except cv2.error as e:
                    self._error = f"camera {self.index} read failed: {e}"
                    break
                # Time the frame was handed to us after decode (excludes sensor/driver delay).
                grabbed_at = time.perf_counter()
                # อ่านไม่ได้: ถ้านานเกิน read_timeout_s = กล้องหลุด (ตั้ง _error แล้วออก)
                # ถ้ายังไม่นาน ถือว่ากระตุกชั่วคราว รอแล้วลองใหม่
                if not ok or frame is None:
                    if grabbed_at - last_good > self.read_timeout_s:
                        self._error = (
                            f"camera {self.index} gave no frame for {self.read_timeout_s:.1f} s"
                        )
                        break
                    time.sleep(_RETRY_SLEEP_S)  # brief glitch: back off and retry
                    continue
                last_good = grabbed_at
                # ส่วนที่ต้องล็อก: ถ้าภาพก่อนหน้ายังไม่มีใครอ่าน = ถูกทิ้ง (นับ dropped)
                # แล้วเก็บภาพใหม่ทับ เพิ่มเลขลำดับ และปลุก read() ที่กำลังรอ
                with self._cond:
                    if self._latest_seq > self._read_seq:
                        self._dropped += 1  # previous frame was never read
                    self._latest = (frame, grabbed_at)
                    self._latest_seq += 1
                    self._cond.notify_all()
        # จบ thread เสมอ: ปล่อยกล้อง ตั้ง _ended แล้วปลุกคนที่รอ
        finally:
            cap.release()
            with self._cond:
                self._ended = True
                self._cond.notify_all()

    # เรียกจากลูปหลัก: รอภาพใหม่ที่ยังไม่เคยส่ง แล้วคืนเป็น FramePacket
    def read(self) -> FramePacket | None:
        """Return the newest frame not yet returned.

        Returns None once the source has been closed. Raises CameraError if the
        camera stops delivering frames, so a lost camera is never mistaken for a
        normal end of stream.
        """
        if self._thread is None and not self._ended:
            raise RuntimeError("source is not open")
        if self._stop.is_set():
            return None  # closed
        with self._cond:
            # รอจนมีภาพใหม่หรือ thread จบ แต่ไม่เกิน read_timeout_s
            self._cond.wait_for(
                lambda: self._latest_seq > self._read_seq or self._ended,
                timeout=self.read_timeout_s,
            )
            # มีภาพใหม่: จำว่าอ่านแล้ว และหยิบภาพออกมา
            if self._latest_seq > self._read_seq:
                self._read_seq = self._latest_seq
                image, grabbed_at = self._latest
                seq = self._latest_seq
            # ไม่มีภาพและ thread เจอ error -> แจ้ง CameraError (ไม่เงียบ)
            elif self._error is not None:
                raise CameraError(self._error)
            # thread จบปกติ (ถูกปิด) -> None
            elif self._ended:
                return None
            else:
                raise CameraError(
                    f"camera {self.index} gave no frame for {self.read_timeout_s:.1f} s"
                )
        # กล้อง: timestamp = เวลาที่ grab ภาพ (นาฬิกาจริง)
        return FramePacket(image, grabbed_at, seq, grabbed_at)

    # สั่งหยุด thread แล้วรอสูงสุด 2 วินาที
    # ถ้า thread ค้างอยู่ใน native read ไม่ปล่อยกล้องจากที่นี่ (thread ปล่อยเองตอนอ่านเสร็จ)
    def close(self) -> None:
        self._stop.set()
        if self._thread is None:
            return
        self._thread.join(timeout=_JOIN_TIMEOUT_S)
        if self._thread.is_alive():
            # Blocked in a native read; the thread releases the camera when it returns.
            log.warning("camera %d grabber still blocked; it will release on return", self.index)
        else:
            self._thread = None

    @property
    def fps(self) -> float | None:
        return self._fps

    @property
    def dropped_frames(self) -> int:
        return self._dropped
