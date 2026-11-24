"""Stage 2a: replay extracted features through the real logic layer (DriverMonitor).

Features are loaded from the feature CSV once and cached as a compact .npz next to it,
so tuning can replay thousands of clip/config pairs quickly. The logic code is the same
code the app runs: nothing is re-implemented here.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from drowsiness.config import Config
from drowsiness.logic import REASONS, DriverMonitor
from drowsiness.types import AlertLevel, DriverState, FrameFeatures

# คอลัมน์จาก CSV ที่ต้องใช้ replay
COLUMNS = (
    "timestamp",
    "frame_index",
    "face_found",
    "ear_left",
    "ear_right",
    "mar",
    "pitch",
    "yaw",
    "roll",
)
EXTRA = ("bs_eyeBlinkLeft", "bs_eyeBlinkRight", "bs_jawOpen")  # kept for later analysis
# รหัสตัวเลขของแต่ละสถานะ และเหตุผลที่นับเป็นการง่วง
STATES = tuple(DriverState)
_STATE_CODE = {s: i for i, s in enumerate(STATES)}
DROWSY_REASONS = ("sleep", "perclos_alarm", "microsleep", "perclos_warning", "head_nod", "yawning")


# อ่าน CSV เป็น array (ค่าว่าง = NaN) แล้ว cache เป็น .npz ให้ครั้งต่อไปเร็วขึ้น
# ตรวจว่าแถว i = เฟรม i ไม่งั้น label จะไม่ตรงเฟรม
def load_features(csv_path: Path) -> dict[str, np.ndarray]:
    """Columns as float arrays (NaN = missing); cached in <clip>.npz.

    Row i must be video frame i, because labels are indexed by frame: a CSV whose
    frame_index column is not 0, 1, 2, ... raises ValueError.
    """
    cache = csv_path.with_suffix(".npz")
    if cache.is_file() and cache.stat().st_mtime >= csv_path.stat().st_mtime:
        with np.load(cache) as data:
            return {k: data[k] for k in data.files}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        wanted = [c for c in (*COLUMNS, *EXTRA) if c in header]
        idx = [header.index(c) for c in wanted]
        rows = [[row[i] for i in idx] for row in reader]
    arrays = {}
    for j, name in enumerate(wanted):
        arrays[name] = np.array([float(r[j]) if r[j] else np.nan for r in rows], dtype=np.float64)
    if "frame_index" in arrays and not np.array_equal(
        arrays["frame_index"], np.arange(len(arrays["frame_index"]))
    ):
        raise ValueError(f"{csv_path}: rows are not consecutive frames 0..n-1")
    partial = cache.with_suffix(".partial.npz")  # a killed writer never leaves a bad cache
    np.savez(partial, **arrays)
    os.replace(partial, cache)
    return arrays


# แปลง array กลับเป็น FrameFeatures ทีละเฟรม (NaN -> None)
def to_frames(arrays: dict[str, np.ndarray]) -> list[FrameFeatures]:
    def opt(name: str, i: int) -> float | None:
        v = arrays[name][i]
        return None if np.isnan(v) else float(v)

    frames = []
    for i in range(len(arrays["timestamp"])):
        face = arrays["face_found"][i] == 1.0  # NaN (empty cell) is no face
        if not face:
            frames.append(FrameFeatures(timestamp=float(arrays["timestamp"][i]), face_found=False))
            continue
        frames.append(
            FrameFeatures(
                timestamp=float(arrays["timestamp"][i]),
                face_found=True,
                ear_left=opt("ear_left", i),
                ear_right=opt("ear_right", i),
                mar=opt("mar", i),
                pitch=opt("pitch", i),
                yaw=opt("yaw", i),
                roll=opt("roll", i),
            )
        )
    return frames


@dataclass
# สิ่งที่ logic ตัดสินในแต่ละเฟรม: สถานะ, เงื่อนไขต่างๆ, PERCLOS, เหตุผล, การเตือน
class Trace:
    """What logic decided, frame by frame (index i = frame i of the clip)."""

    t: np.ndarray
    state: np.ndarray  # int8 codes into STATES
    closed: np.ndarray  # eyes-closed condition active
    off_road: np.ndarray  # head-off-road condition active
    nod: np.ndarray  # head-dropped condition active
    drowsy_reason: np.ndarray  # any drowsiness reason active
    perclos: np.ndarray  # NaN until enough data
    reasons: np.ndarray  # int16 bitmask: bit k = REASONS[k] active
    yawn_times: list[float] = field(default_factory=list)  # when each yawn was counted
    alerts: list[tuple[float, str, str, int]] = field(default_factory=list)  # t, reason, state, lvl
    calibrated_t: float | None = None
    fallback: bool = False

    def in_state(self, *states: DriverState) -> np.ndarray:
        return np.isin(self.state, [_STATE_CODE[s] for s in states])

    def reason(self, name: str) -> np.ndarray:
        return (self.reasons & (1 << REASONS.index(name))) != 0


# ป้อน features ผ่าน DriverMonitor ตัวจริง (โค้ดเดียวกับในแอป) แล้วเก็บผลทุกเฟรม
def replay(frames: list[FrameFeatures], cfg: Config) -> Trace:
    n = len(frames)
    monitor = DriverMonitor(cfg)
    state = np.zeros(n, dtype=np.int8)
    closed, off_road, nod, drowsy = (np.zeros(n, dtype=bool) for _ in range(4))
    perclos = np.full(n, np.nan)
    reasons = np.zeros(n, dtype=np.int16)
    bit = {r: 1 << k for k, r in enumerate(REASONS)}
    trace = Trace(
        np.array([f.timestamp for f in frames]),
        state,
        closed,
        off_road,
        nod,
        drowsy,
        perclos,
        reasons,
    )
    durations = {name: np.zeros(n) for name in ("closed_s", "off_road_s", "nod_s")}
    yawns = 0
    # ทีละเฟรม: เก็บสถานะ, ตัวจับเวลา, เหตุผล (เป็น bitmask), PERCLOS, เวลาที่หาว, การเตือน
    for i, f in enumerate(frames):
        out = monitor.update(f)
        m = out.metrics
        state[i] = _STATE_CODE[out.state]
        for name, d in durations.items():
            d[i] = m.get(name, 0.0)
        drowsy[i] = any(r in DROWSY_REASONS for r in out.reasons)
        reasons[i] = sum(bit[r] for r in out.reasons)
        perclos[i] = m.get("perclos", np.nan)
        count = int(m.get("yawn_count", 0))
        if count > yawns:
            trace.yawn_times.append(f.timestamp)
        yawns = count
        for e in out.events:
            if e.level >= AlertLevel.WARNING:
                trace.alerts.append((e.timestamp, e.reason, e.state.value, int(e.level)))
            elif e.reason in ("calibrated", "calibration_fallback") and trace.calibrated_t is None:
                trace.calibrated_t = e.timestamp
                trace.fallback = e.reason == "calibration_fallback"
    for flag, name in ((closed, "closed_s"), (off_road, "off_road_s"), (nod, "nod_s")):
        _condition_flags(flag, durations[name], trace.t)
    return trace


# หาเฟรมที่เงื่อนไขเป็นจริงจากตัวจับเวลา และย้อนไปติดเฟรมแรกของแต่ละช่วงด้วย
def _condition_flags(flag: np.ndarray, duration: np.ndarray, t: np.ndarray) -> None:
    """Frames where a condition holds, from its live timer (0 on the frame it starts).

    The timer is t - since, so it is still 0 on the first frame of each run. Each run
    is traced back to `since` so that frame is flagged as well.
    """
    flag[:] = duration > 0
    starts = np.flatnonzero(flag & ~np.concatenate(([False], flag[:-1])))
    for i in starts:
        first = int(np.searchsorted(t, t[i] - duration[i] - 1e-9))
        flag[first:i] = True
