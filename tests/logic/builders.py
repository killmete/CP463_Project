# ตัวช่วยสร้างลำดับ FrameFeatures ปลอมสำหรับเทสต์ logic (ไม่ใช้ภาพหรือหน้าคนจริง)
"""Synthetic FrameFeatures sequences for logic tests (no images, no face recordings).

frames = Seq(fps=30).calibrate().closed(1.6).open(2).frames
"""

from __future__ import annotations

import random
from typing import Self

from drowsiness.config import Config, load_config
from drowsiness.logic import DriverMonitor
from drowsiness.types import FrameFeatures, LogicOutput

EAR_OPEN = 0.30
EAR_CLOSED = 0.03
MAR_CLOSED = 0.05
MAR_YAWN = 0.9
NEUTRAL_PITCH = -6.0  # laptop camera below eye level: nonzero neutral pose
NEUTRAL_YAW = 3.0


# สร้างลำดับเฟรมตามเวลา: เช่น Seq().calibrate().closed(2.0) = calibrate แล้วหลับตา 2 วินาที
# jitter = เวลาเฟรมไม่สม่ำเสมอ, drop = ทิ้งเฟรมแบบสุ่ม (seed คงที่ ผลจึงเหมือนเดิมทุกครั้ง)
class Seq:
    def __init__(
        self, fps: float = 30.0, start: float = 0.0, jitter: float = 0.0, drop: float = 0.0,
        seed: int = 0,
    ) -> None:  # fmt: skip
        self.fps, self.t, self.jitter, self.drop = fps, start, jitter, drop
        self.rng = random.Random(seed)
        self.frames: list[FrameFeatures] = []

    # เพิ่มเฟรมเป็นเวลา seconds วินาที ด้วยค่าที่กำหนด (ค่าอื่นเป็นตาเปิด หน้าตรง)
    def hold(self, seconds: float, face: bool = True, **values: float | None) -> Self:
        """Append frames for `seconds` with the given feature values (others: neutral open)."""
        n = round(seconds * self.fps)
        base = {
            "ear_left": EAR_OPEN, "ear_right": EAR_OPEN, "mar": MAR_CLOSED,
            "pitch": NEUTRAL_PITCH, "yaw": NEUTRAL_YAW, "roll": 0.0,
        }  # fmt: skip
        base.update(values)
        for i in range(n):
            t = self.t + i / self.fps
            if i and self.jitter:
                t += self.rng.uniform(-self.jitter, self.jitter) / self.fps
            if self.drop and i and self.rng.random() < self.drop:
                continue
            if face:
                self.frames.append(FrameFeatures(timestamp=t, face_found=True, **base))
            else:
                self.frames.append(FrameFeatures(timestamp=t, face_found=False))
        self.t += n / self.fps
        return self

    # ตาเปิด หน้าตรง นานพอให้ calibrate เสร็จ
    def calibrate(self, seconds: float = 5.5) -> Self:
        return self.hold(seconds)

    # ตาเปิดตามปกติ
    def open(self, seconds: float) -> Self:
        return self.hold(seconds)

    # ตาปิด (EAR ต่ำ)
    def closed(self, seconds: float, **values: float | None) -> Self:
        return self.hold(seconds, **{"ear_left": EAR_CLOSED, "ear_right": EAR_CLOSED, **values})

    # อ้าปากกว้าง (MAR สูง) = หาว
    def yawn(self, seconds: float) -> Self:
        return self.hold(seconds, mar=MAR_YAWN)

    # ไม่เจอหน้า
    def no_face(self, seconds: float) -> Self:
        return self.hold(seconds, face=False)

    # หันหัว: pitch/yaw นับจากท่าปกติของคนขับ
    def pose(self, seconds: float, pitch: float = 0.0, yaw: float = 0.0, **values) -> Self:
        """Head offset from neutral by (pitch, yaw) degrees."""
        return self.hold(seconds, pitch=NEUTRAL_PITCH + pitch, yaw=NEUTRAL_YAW + yaw, **values)


# โหลด config พร้อมค่าที่ทับ
def config(**overrides) -> Config:
    return load_config(overrides=overrides)


# ป้อนทุกเฟรมเข้า DriverMonitor แล้วคืนผลทุกเฟรม
def run(frames: list[FrameFeatures], cfg: Config | None = None) -> list[LogicOutput]:
    monitor = DriverMonitor(cfg or load_config())
    return [monitor.update(f) for f in frames]


# รวม event ของทุกเฟรม
def events(outputs: list[LogicOutput]):
    return [e for out in outputs for e in out.events]


# เหตุผลทุกตัวที่เคยเกิดขึ้น
def reasons_seen(outputs: list[LogicOutput]) -> set[str]:
    return {r for out in outputs for r in out.reasons}
