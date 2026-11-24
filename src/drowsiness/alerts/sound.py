"""Alert sounds: tones generated with numpy (no audio files), played off the frame loop.

SoundSink owns a worker thread; `emit` only queues a command. sounddevice is imported
lazily in the worker. If audio is unavailable (no device, no PortAudio, no audio
server, a headset unplugged) the sink logs a warning, `available` turns False (the
banner shows "SOUND OFF") and it retries every `retry_s` when there is something to
play, so sound comes back on its own.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from drowsiness.alerts.manager import AlertDecision, SoundCue
from drowsiness.config import AlertsConfig

log = logging.getLogger(__name__)

# ค่อยๆ เพิ่ม/ลดเสียง 5 ms ต้นและท้ายบี๊บ กันเสียงคลิก
_FADE_S = 0.005  # fade in/out of each beep, avoids clicks


# สัญญาของตัวเล่นเสียง (play, stop) เทสต์ใช้ตัวปลอมได้
class AudioBackend(Protocol):
    def play(self, data: np.ndarray, samplerate: int) -> Any: ...

    def stop(self) -> Any: ...


# สร้างเสียงบี๊บด้วยคลื่น sin ที่ความถี่ freq (ไม่ต้องมีไฟล์เสียง)
def _beep(freq: float, seconds: float, rate: int, volume: float) -> np.ndarray:
    t = np.arange(round(seconds * rate)) / rate
    tone = np.sin(2 * np.pi * freq * t)
    fade = min(round(_FADE_S * rate), len(t) // 2)
    if fade:
        ramp = np.linspace(0.0, 1.0, fade)
        tone[:fade] *= ramp
        tone[-fade:] *= ramp[::-1]
    return (volume * tone).astype(np.float32)


# ช่วงเงียบ = array ศูนย์
def _silence(seconds: float, rate: int) -> np.ndarray:
    return np.zeros(round(seconds * rate), dtype=np.float32)


# WARNING = บี๊บ 880 Hz สองครั้ง; ALARM = สลับ 1000/1500 Hz 4 ครั้ง (~1 วินาที)
def make_patterns(rate: int, volume: float) -> dict[SoundCue, np.ndarray]:
    """WARNING: two 880 Hz beeps. ALARM: ~1 s of alternating 1000/1500 Hz beeps."""
    warning = [_beep(880, 0.15, rate, volume), _silence(0.1, rate), _beep(880, 0.15, rate, volume)]
    alarm: list[np.ndarray] = []
    for i in range(4):
        alarm += [_beep(1500 if i % 2 else 1000, 0.2, rate, volume), _silence(0.05, rate)]
    return {SoundCue.WARNING: np.concatenate(warning), SoundCue.ALARM: np.concatenate(alarm)}


# โหลด sounddevice ตอนใช้ และเช็คว่ามีลำโพงจริง
def _sounddevice() -> AudioBackend:
    import sounddevice  # lazy: fails without PortAudio or an audio server

    sounddevice.query_devices(kind="output")  # raises if there is no output device
    return sounddevice


# เล่นเสียงใน thread แยก: emit แค่ใส่คำสั่งลง queue แล้วคืนทันที
# ถ้าไม่มีลำโพง: available=False (แสดง SOUND OFF) และลองใหม่ทุก 10 วินาที
class SoundSink:
    def __init__(
        self,
        cfg: AlertsConfig,
        backend_factory: Callable[[], AudioBackend] = _sounddevice,
        maxsize: int = 4,
        retry_s: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg = cfg
        self.available = True
        self.retry_s = retry_s
        self._clock = clock
        self._failed_at: float | None = None
        self._warned = False
        self._patterns = make_patterns(cfg.sample_rate, cfg.volume)
        self._factory = backend_factory
        self._backend: AudioBackend | None = None
        self._queue: queue.Queue[SoundCue | None] = queue.Queue(maxsize)
        self._thread = threading.Thread(target=self._run, name="alert-sound", daemon=True)
        self._thread.start()

    # มีคำสั่งเสียง -> ใส่ queue
    def emit(self, decision: AlertDecision) -> None:
        if decision.sound != SoundCue.NONE:
            self._put(decision.sound)

    def close(self) -> None:
        self._put(None)
        self._thread.join(timeout=1.0)

    # ไม่บล็อก: queue เต็มทิ้งคำสั่งเก่าสุด (เสียงล่าสุดสำคัญกว่า)
    def _put(self, item: SoundCue | None) -> None:
        """Never blocks: when the queue is full the oldest command is dropped."""
        while True:
            try:
                self._queue.put_nowait(item)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass

    # -- worker thread ---------------------------------------------------------------

    # thread เสียง: รับคำสั่งจาก queue แล้วเล่นหรือหยุด
    def _run(self) -> None:
        self._open()
        while (cue := self._queue.get()) is not None:
            # ยังไม่มีลำโพง: ข้ามจนถึงเวลาลองใหม่
            if self._backend is None:
                if cue == SoundCue.STOP or not self._retry_due():
                    continue
                self._open()
                if self._backend is None:
                    continue
            try:
                if cue == SoundCue.STOP:
                    self._backend.stop()
                else:
                    self._backend.play(self._patterns[cue], self.cfg.sample_rate)  # non-blocking
            except Exception as exc:
                self._fail(exc)
        if self._backend is not None:
            try:
                self._backend.stop()
            except Exception:
                pass

    # ผ่านไป retry_s (10 วินาที) นับจากพังครั้งล่าสุดหรือยัง
    def _retry_due(self) -> bool:
        return self._failed_at is None or self._clock() - self._failed_at >= self.retry_s

    # ลองเปิดตัวเล่นเสียง สำเร็จ -> available=True
    def _open(self) -> None:
        try:
            self._backend = self._factory()
        except Exception as exc:  # ImportError, OSError, PortAudioError, ...
            self._fail(exc)
            return
        if not self.available:
            log.warning("audio available again")
        self.available, self._failed_at = True, None

    # เสียงใช้ไม่ได้: ทิ้ง backend, จำเวลา, เตือนใน log ครั้งเดียว, available=False
    def _fail(self, exc: BaseException) -> None:
        self._backend, self._failed_at = None, self._clock()
        if self.available or not self._warned:
            log.warning("audio unavailable, alerts are visual only (%s)", exc)
            self._warned = True
        self.available = False
