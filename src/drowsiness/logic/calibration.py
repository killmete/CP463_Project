"""Per-driver baseline: open EAR per eye and neutral head pose (lit. review §2A, §6).

The driver looks ahead with eyes open for `calibration.duration_s`. Fixed EAR/MAR
thresholds do not generalise across people [M1], so eye closure is judged relative
to this baseline, and pose relative to the neutral angles (an off-axis dashcam or
laptop camera gives a nonzero neutral pitch/yaw).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from drowsiness.config import CalibrationConfig
from drowsiness.types import FrameFeatures


@dataclass(frozen=True, slots=True)
class Calibration:
    ear_open_left: float | None
    """Median open EAR of the driver's left eye; None in fallback mode."""
    ear_open_right: float | None
    neutral_pitch: float
    neutral_yaw: float
    neutral_roll: float
    valid_ratio: float
    fallback: bool = False
    """True when calibration failed: absolute EAR threshold, zero pose offsets."""

    @classmethod
    def fallback_values(
        cls, valid_ratio: float = 0.0, pose: tuple[float, float, float] | None = None
    ) -> Calibration:
        """Absolute EAR threshold; the measured neutral pose if there was one, else zero."""
        pitch, yaw, roll = pose if pose is not None else (0.0, 0.0, 0.0)
        return cls(None, None, pitch, yaw, roll, valid_ratio, fallback=True)


class Calibrator:
    """Collects one attempt at a time; retries up to `max_attempts`, then falls back."""

    def __init__(self, cfg: CalibrationConfig) -> None:
        self.cfg = cfg
        self.attempts = 0
        self._start: float | None = None
        self._frames = 0
        self._valid: list[tuple[float, float, float, float, float]] = []
        self._poses: list[tuple[float, float, float]] = []
        self.last_valid_ratio = 0.0
        self.last_pose: tuple[float, float, float] | None = None

    def update(self, f: FrameFeatures) -> Calibration | None:
        """Feed one frame. Returns a Calibration when finished (success or fallback)."""
        if self._start is None:
            self._start = f.timestamp
        self._frames += 1
        values = (f.ear_left, f.ear_right, f.pitch, f.yaw, f.roll)
        if f.face_found and all(v is not None for v in values):
            self._valid.append(values)  # type: ignore[arg-type]
        pose = (f.pitch, f.yaw, f.roll)
        if f.face_found and all(v is not None for v in pose):
            self._poses.append(pose)  # type: ignore[arg-type]
        if f.timestamp - self._start < self.cfg.duration_s:
            return None

        self.attempts += 1
        result = self._finish()
        if result is not None:
            return result
        if self.attempts >= self.cfg.max_attempts:
            return Calibration.fallback_values(self.last_valid_ratio, self.last_pose)
        self._start, self._frames, self._valid, self._poses = None, 0, [], []  # retry
        return None

    def _finish(self) -> Calibration | None:
        self.last_valid_ratio = len(self._valid) / self._frames if self._frames else 0.0
        if self._frames and len(self._poses) / self._frames >= self.cfg.min_valid_frames_ratio:
            # Pose can be fine even when the eyes failed (e.g. closed): keep it for fallback.
            pitch, yaw, roll = (statistics.median(c) for c in zip(*self._poses, strict=True))
            self.last_pose = (pitch, yaw, roll)
        if self.last_valid_ratio < self.cfg.min_valid_frames_ratio:
            return None
        left, right, pitch, yaw, roll = (
            statistics.median(col) for col in zip(*self._valid, strict=True)
        )
        if min(left, right) < self.cfg.min_open_ear:
            return None  # eyes were not open (or landmarks failed) during calibration
        return Calibration(left, right, pitch, yaw, roll, self.last_valid_ratio)
