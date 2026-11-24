"""Shared data contracts between pipeline stages.

capture -> FramePacket -> perception -> FrameFeatures -> logic -> DriverState / AlertEvent -> alerts

Owned by system-architect. Changing a field is a breaking change: update every caller.
All timestamps are seconds on a monotonic clock.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum

import numpy as np


@dataclass(frozen=True, slots=True)
# ภาพ 1 เฟรมจาก capture: ภาพ, เวลา (ใช้ตัดสิน), เลขเฟรม, เวลาจริงตอน grab (วัด latency)
# frozen=True = แก้ค่าไม่ได้หลังสร้าง กันส่วนอื่นแอบเปลี่ยนข้อมูล
class FramePacket:
    """One captured frame."""

    image: np.ndarray
    """H x W x 3, uint8, BGR."""
    timestamp: float
    """Time the logic layer uses. Camera: perf_counter() at grab. Video file: index / fps,
    so replays are deterministic."""
    index: int
    """Frame number in the source. For a camera, gaps mean frames were dropped."""
    grabbed_at: float
    """perf_counter() at grab, always real time. Only used to measure latency."""


@dataclass(frozen=True, slots=True)
# ค่าที่ perception วัดได้ใน 1 เฟรม (ส่งให้ logic); ค่าที่หาไม่ได้เป็น None
class FrameFeatures:
    """Per-frame measurements from the perception layer. Missing values are None.

    Left/right are the driver's anatomical sides (frames are never mirrored). Angles
    are in the camera frame and uncalibrated: logic removes the neutral offset.
    """

    timestamp: float
    face_found: bool
    ear_left: float | None = None
    ear_right: float | None = None
    mar: float | None = None
    pitch: float | None = None
    """Degrees; positive = head up."""
    yaw: float | None = None
    """Degrees; positive = head turned to the driver's left."""
    roll: float | None = None
    """Degrees; positive = head tilted toward the driver's left shoulder."""
    gaze: tuple[float, float] | None = None
    """Normalised (horizontal, vertical) iris offset in [-1, 1]."""
    blendshapes: Mapping[str, float] | None = None


@dataclass(frozen=True, slots=True)
# ผลของ perception: features ไป logic, landmarks ไปวาด overlay, เวลาที่ใช้
class PerceptionOutput:
    """What perception hands to the app: features for logic, landmarks for the overlay.

    Only `features` crosses into logic, so logic can be replayed from a feature CSV.
    """

    features: FrameFeatures
    landmarks: np.ndarray | None
    """(478, 3) float32: x, y in pixels of the original frame; z is MediaPipe depth x width.
    None when no face was found."""
    perception_ms: float
    """Wall time spent in perception for this frame."""


# สถานะคนขับ 6 แบบ
class DriverState(StrEnum):
    CALIBRATING = "CALIBRATING"
    NORMAL = "NORMAL"
    DROWSY_WARNING = "DROWSY_WARNING"
    DROWSY_ALARM = "DROWSY_ALARM"
    DISTRACTED = "DISTRACTED"
    NO_FACE = "NO_FACE"


# ระดับการเตือน เป็นตัวเลขจึงเทียบมาก/น้อยได้ (INFO < WARNING < ALARM)
class AlertLevel(IntEnum):
    INFO = 0
    WARNING = 1
    ALARM = 2


@dataclass(frozen=True, slots=True)
# event จาก logic เมื่อสถานะเปลี่ยน (เริ่ม/รุนแรงขึ้น/จบ) ไม่ใช่ทุกเฟรม
class AlertEvent:
    """Emitted by the logic layer when the driver state starts, escalates or ends."""

    timestamp: float
    state: DriverState
    reason: str
    """Machine-readable cause, e.g. "microsleep", "long_distraction"."""
    level: AlertLevel
    details: Mapping[str, float | str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
# สิ่งที่ logic คืนทุกเฟรม: สถานะ, events, เหตุผลที่เกิดขึ้น, ค่าสดสำหรับแสดงผล
class LogicOutput:
    """What logic hands to the app each frame. Alerts consume `events`; ui shows the rest."""

    state: DriverState
    events: tuple[AlertEvent, ...] = ()
    """Emitted only on state transitions (and calibration results), never every frame."""
    reasons: tuple[str, ...] = ()
    """Active conditions, highest priority first (e.g. "microsleep", "no_face")."""
    metrics: Mapping[str, float] = field(default_factory=dict)
    """Live values for the overlay and logs: closed_s, perclos, off_road_s, ..."""
