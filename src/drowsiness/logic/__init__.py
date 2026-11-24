"""Logic: a time-ordered stream of FrameFeatures -> DriverState + AlertEvents.

Pure and deterministic: no I/O, OpenCV or clocks; only FrameFeatures.timestamp is
used, so a feature CSV replays exactly. See docs/architecture.md (Logic).
"""

from __future__ import annotations

from drowsiness.config import Config
from drowsiness.logic.calibration import Calibration, Calibrator
from drowsiness.logic.rules import REASON_STATE, REASONS, RuleEngine
from drowsiness.logic.state_machine import LEVEL, StateMachine
from drowsiness.types import AlertEvent, AlertLevel, DriverState, FrameFeatures, LogicOutput

__all__ = ["LEVEL", "REASONS", "REASON_STATE", "Calibration", "DriverMonitor"]


# หัวใจของ logic: รับ FrameFeatures ทีละเฟรม คืนสถานะคนขับ (LogicOutput)
# ขั้นตอน: calibrate -> ใช้กฎ (rules) -> รวมเป็นสถานะเดียว (state machine)
class DriverMonitor:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        # กฎตรวจจับทั้งหมด, ตัวรวมสถานะ, และตัว calibrate (เริ่ม calibrate ทันที)
        self._rules = RuleEngine(cfg)
        self._machine = StateMachine(cfg.state_machine.clear_s)
        self._calibrator: Calibrator | None = Calibrator(cfg.calibration)
        self.calibration: Calibration | None = None
        self._last_t: float | None = None
        # event ที่รอส่งในเฟรมถัดไป (เช่นแจ้งว่ากำลัง recalibrate)
        self._pending: list[AlertEvent] = []

    @property
    def state(self) -> DriverState:
        return self._machine.state

    # เริ่ม calibrate ใหม่ (กดปุ่ม c) แต่ไม่ลบค่าเดิม กฎจึงยังทำงานต่อระหว่างรอ
    def recalibrate(self) -> None:
        """Start a new calibration (e.g. a new driver).

        The current calibration keeps the rules running until the new one succeeds, so
        recalibrating never pauses detection or silences an active alert.
        """
        self._calibrator = Calibrator(self.cfg.calibration)
        if self._last_t is not None:
            self._pending.append(self._info(self._last_t, "recalibrating", {}))

    # เรียกทุกเฟรมจาก app.py
    def update(self, f: FrameFeatures) -> LogicOutput:
        t = f.timestamp
        # เวลาต้องเพิ่มขึ้นเสมอ ไม่งั้นตัวจับเวลาทุกตัวพัง -> ValueError (app ข้ามเฟรมนี้)
        if self._last_t is not None and t <= self._last_t:
            raise ValueError(f"timestamps must strictly increase ({t} after {self._last_t})")
        self._last_t = t
        events, self._pending = self._pending, []

        # กำลัง calibrate: ส่งเฟรมให้ Calibrator ถ้าครบเวลาจะได้ผลกลับมา
        if self._calibrator is not None:
            result = self._calibrator.update(f)
            if result is not None:
                events.append(self._finish_calibration(t, result))
        # ยังไม่มีค่าฐาน: ใช้แค่กฎไม่เห็นหน้า (กฎอื่นต้องรู้ค่าปกติของคนขับก่อน)
        # มีค่าฐานแล้ว: ใช้กฎครบทุกข้อ
        if self.calibration is None:
            reasons, metrics = self._rules.update_no_face_only(f)
            base = DriverState.CALIBRATING
        else:
            reasons, metrics = self._rules.update(f, self.calibration)
            base = DriverState.NORMAL
        # กำลัง recalibrate: บอก UI ให้แสดงข้อความ calibrating
        if self._calibrator is not None:
            metrics["calibrating"] = 1.0
        # ส่งเหตุผลที่เกิดขึ้นให้ state machine เลือกสถานะเดียว แล้วคืนผลรวม
        events += self._machine.update(t, reasons, base, self._details(metrics))
        return LogicOutput(self._machine.state, tuple(events), tuple(reasons), metrics)

    # สร้าง event แบบ INFO (แจ้งเฉยๆ ไม่เปลี่ยนสถานะ)
    def _info(self, t: float, reason: str, details: dict[str, float | str]) -> AlertEvent:
        """An informational event that does not change the state."""
        state = self._machine.state
        extra = {"previous_state": state.value, "reasons": reason, **details}
        return AlertEvent(t, state, reason, AlertLevel.INFO, extra)

    # จัดการตอน calibrate จบ (สำเร็จหรือล้มเหลว)
    def _finish_calibration(self, t: float, cal: Calibration) -> AlertEvent:
        attempts = float(self._calibrator.attempts if self._calibrator else 0)
        self._calibrator = None
        previous = self.calibration
        # recalibrate ล้มเหลวแต่มีค่าเก่าที่ดีอยู่ -> เก็บค่าเก่าไว้
        if cal.fallback and previous is not None:
            # A failed recalibration keeps the previous (good) baseline.
            return self._info(t, "recalibration_failed", {"attempts": attempts})
        self.calibration = cal
        # ค่าฐานเปลี่ยน -> รีเซ็ตตัวจับเวลาของกฎทั้งหมด
        if previous is not None:
            self._rules.reset()  # new baseline: restart timers and windows
        details: dict[str, float | str] = {
            "neutral_pitch": cal.neutral_pitch,
            "neutral_yaw": cal.neutral_yaw,
            "neutral_roll": cal.neutral_roll,
            "valid_ratio": cal.valid_ratio,
        }
        # ล้มเหลวครั้งแรก -> ใช้เกณฑ์ EAR ตายตัวแทน (calibration_fallback)
        # สำเร็จ -> เกณฑ์ตาปิด = 0.2 + 0.2 x 0.8 = 0.36 เท่าของ EAR ตอนลืมตา
        if cal.fallback:
            reason = "calibration_fallback"
            details["attempts"] = attempts
        else:
            reason = "calibrated"
            frac = self.cfg.eyes.closed_ear_fraction
            details["ear_open_left"] = cal.ear_open_left
            details["ear_open_right"] = cal.ear_open_right
            details["ear_closed_threshold"] = frac + self.cfg.eyes.closed_ratio * (1.0 - frac)
        # ยังอยู่ CALIBRATING -> เปลี่ยนเป็น NORMAL ทันที; สถานะอื่น (เช่น NO_FACE) แค่แจ้ง INFO
        if self._machine.state == DriverState.CALIBRATING:
            return self._machine.force(t, DriverState.NORMAL, reason, extra=details)
        return self._info(t, reason, details)  # e.g. NO_FACE stays, and keeps its clear timer

    # เลือกค่าที่เกี่ยวกับแต่ละเหตุผลใส่ใน event (ใช้ดูใน log)
    def _details(self, m: dict[str, float]) -> dict[str, dict[str, float]]:
        cfg = self.cfg

        def pick(*keys: str) -> dict[str, float]:
            return {k: m[k] for k in keys if k in m}

        return {
            "sleep": pick("closed_s"),
            "microsleep": pick("closed_s"),
            "perclos_alarm": {**pick("perclos"), "window_s": cfg.perclos.window_s},
            "perclos_warning": {**pick("perclos"), "window_s": cfg.perclos.window_s},
            "yawning": {**pick("yawn_count"), "window_s": cfg.yawn.count_window_s},
            "head_nod": {**pick("pitch_delta_deg"), "duration_s": m.get("nod_s", 0.0)},
            "long_distraction": pick("off_road_s", "yaw_delta_deg", "pitch_delta_deg"),
            "short_distraction": {
                **pick("off_road_total_s"),
                "window_s": cfg.distraction.short_window_s,
            },
            "no_face": pick("no_face_s"),
        }
