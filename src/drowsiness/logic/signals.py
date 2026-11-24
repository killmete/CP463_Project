"""Time-based signal helpers: smoothing, held/carried conditions and sliding windows.

Everything is keyed by FrameFeatures timestamps (seconds), never frame counts, so
20-30 FPS, dropped frames and replays all give the same timing.
"""

from __future__ import annotations

import statistics
from collections import deque


# ทำค่าให้เรียบ: median ของค่าในช่วง window_s วินาทีล่าสุด (ตัดสัญญาณกระตุก)
class MedianSmoother:
    """Median of the values seen in the last `window_s` seconds (None values are skipped)."""

    def __init__(self, window_s: float) -> None:
        self.window_s = window_s
        self._samples: deque[tuple[float, float]] = deque()

    def update(self, t: float, value: float | None) -> float | None:
        # ลบค่าที่เก่ากว่า window ออก
        while self._samples and self._samples[0][0] < t - self.window_s - 1e-9:
            self._samples.popleft()
        if value is None:
            return None
        self._samples.append((t, value))
        values = [v for _, v in self._samples]
        # ทำให้จำนวนเป็นเลขคี่ -> median เป็นค่าจริงตัวหนึ่ง และหน่วงขึ้น/ลงเท่ากัน
        if len(values) % 2 == 0:
            values = values[1:]  # odd count: onset and release are delayed equally
        return statistics.median(values)

    def reset(self) -> None:
        self._samples.clear()


# เงื่อนไขจริง/เท็จตามเวลา พร้อมนับว่าจริงต่อเนื่องมากี่วินาที
# ข้อมูลหายสั้นๆ (<= max_gap_s): ใช้ค่าเดิมต่อ
# หายนานและเป็นเงื่อนไขอันตรายที่เป็นจริงมานานพอ: ถือว่ายังจริง (carry)
class Condition:
    """A boolean condition over time with gap tolerance.

    A known value (True/False) is taken as is. An unknown value (None) holds the last
    known value for up to `max_gap_s`. After that, an `adverse` condition carries
    (stays True) only if it was established, i.e. True for at least `carry_min_s`
    before the data went missing (so a blink at the moment the face is lost does not
    carry), and for at most `max_carry_s`; anything else becomes unknown.
    `carrying` tells whether the last value returned was inferred by carrying.
    `duration(t)` is how long the condition has been continuously True.
    """

    def __init__(
        self,
        max_gap_s: float,
        adverse: bool,
        carry_min_s: float = 0.0,
        max_carry_s: float = float("inf"),
    ) -> None:
        self.max_gap_s = max_gap_s
        self.adverse = adverse
        self.carry_min_s = carry_min_s
        self.max_carry_s = max_carry_s
        self.reset()

    def reset(self) -> None:
        self._last_value: bool | None = None
        self._last_known_t: float | None = None
        self._since: float | None = None
        self.carrying = False

    # รับค่าเฟรมนี้ (True/False/None=ไม่รู้) คืนค่าที่ใช้จริง
    def update(self, t: float, value: bool | None) -> bool | None:
        """Feed one frame; returns the effective value (True/False/None)."""
        self.carrying = False
        # รู้ค่า: จำไว้ และจดเวลาที่เริ่มเป็นจริง (_since)
        if value is not None:
            self._last_value, self._last_known_t = value, t
            if value and self._since is None:
                self._since = t
            elif not value:
                self._since = None
            return value
        if self._last_value is None or self._last_known_t is None:
            return None
        # ไม่รู้ค่า: ช่องว่างสั้น -> ใช้ค่าเดิม
        gap = t - self._last_known_t
        if gap <= self.max_gap_s:
            return self._last_value  # short gap: hold
        # ช่องว่างยาว: carry เฉพาะเงื่อนไขอันตรายที่จริงมานาน >= carry_min_s
        # (กันการกระพริบตาตอนหน้าหายถูกนับเป็นหลับ)
        established = (
            self._since is not None and self._last_known_t - self._since >= self.carry_min_s
        )
        if self.adverse and self._last_value and established and gap <= self.max_carry_s:
            self.carrying = True
            return True  # carry: an established adverse condition is not silently cleared
        self._last_value, self._since = None, None
        return None

    # เป็นจริงต่อเนื่องมากี่วินาที (เช่น ตาปิดมานานเท่าไร)
    def duration(self, t: float) -> float:
        return t - self._since if self._since is not None else 0.0


# นับวินาทีที่เงื่อนไขจริง เทียบกับวินาทีที่รู้ค่า ในช่วง window_s ล่าสุด
# ใช้คำนวณ PERCLOS = เวลาตาปิด / เวลาที่รู้ค่า; ช่วงที่ไม่รู้ค่าไม่นับทั้งสองฝั่ง
class TimeWindow:
    """Seconds a condition was True vs known over the last `window_s` seconds.

    Intervals with an unknown value are left out of both totals, so missing data
    neither dilutes nor resets the window.
    """

    def __init__(self, window_s: float) -> None:
        self.window_s = window_s
        self._segments: deque[tuple[float, float, bool]] = deque()

    # เพิ่มช่วงเวลา [t0, t1) พร้อมค่า; ช่วงติดกันที่ค่าเดียวกันรวมเป็นช่วงเดียว
    def add(self, t0: float, t1: float, value: bool | None) -> None:
        if value is None or t1 <= t0:
            return
        if self._segments and self._segments[-1][2] == value and self._segments[-1][1] == t0:
            self._segments[-1] = (self._segments[-1][0], t1, value)  # merge runs
        else:
            self._segments.append((t0, t1, value))

    # ตัดช่วงที่หลุด window แล้วรวมเวลาจริงและเวลาที่รู้ค่า
    def totals(self, now: float) -> tuple[float, float]:
        """(true_s, known_s) inside [now - window_s, now]."""
        start = now - self.window_s
        while self._segments and self._segments[0][1] <= start:
            self._segments.popleft()
        true_s = known_s = 0.0
        for t0, t1, value in self._segments:
            span = min(t1, now) - max(t0, start)
            if span > 0:
                known_s += span
                if value:
                    true_s += span
        return true_s, known_s

    def reset(self) -> None:
        self._segments.clear()


# นับจำนวนเหตุการณ์ (เช่น หาว) ใน window_s วินาทีล่าสุด
class EventTimes:
    """Times of discrete events (e.g. yawns) within the last `window_s` seconds."""

    def __init__(self, window_s: float) -> None:
        self.window_s = window_s
        self._times: deque[float] = deque()

    def add(self, t: float) -> None:
        self._times.append(t)

    def count(self, now: float) -> int:
        while self._times and self._times[0] < now - self.window_s:
            self._times.popleft()
        return len(self._times)

    def reset(self) -> None:
        self._times.clear()
