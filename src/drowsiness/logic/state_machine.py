"""Combine active reasons into one DriverState with priority and hysteresis.

Priority: DROWSY_ALARM > NO_FACE > DISTRACTED > DROWSY_WARNING > NORMAL/CALIBRATING.
Escalation is immediate; a state is left only after its condition has been clear
for `state_machine.clear_s`. An AlertEvent is emitted per transition, never per frame.
"""

from __future__ import annotations

from collections.abc import Mapping

from drowsiness.logic.rules import REASON_STATE
from drowsiness.types import AlertEvent, AlertLevel, DriverState

# ลำดับความรุนแรง: ALARM > NO_FACE > DISTRACTED > WARNING > NORMAL
RANK = {
    DriverState.CALIBRATING: 0,
    DriverState.NORMAL: 0,
    DriverState.DROWSY_WARNING: 1,
    DriverState.DISTRACTED: 2,
    DriverState.NO_FACE: 3,
    DriverState.DROWSY_ALARM: 4,
}
# ระดับการเตือนของแต่ละสถานะ (INFO / WARNING / ALARM)
LEVEL = {
    DriverState.CALIBRATING: AlertLevel.INFO,
    DriverState.NORMAL: AlertLevel.INFO,
    DriverState.DROWSY_WARNING: AlertLevel.WARNING,
    DriverState.DISTRACTED: AlertLevel.WARNING,
    DriverState.NO_FACE: AlertLevel.WARNING,
    DriverState.DROWSY_ALARM: AlertLevel.ALARM,
}


# รวมเหตุผลหลายข้อเป็นสถานะเดียว: ขึ้นสถานะรุนแรงทันที
# แต่ลงได้ต่อเมื่อปลอดภัยต่อเนื่อง clear_s วินาที (hysteresis) กันสถานะกระพริบ
class StateMachine:
    def __init__(self, clear_s: float) -> None:
        self.clear_s = clear_s
        self.state = DriverState.CALIBRATING
        self._entered_at: float | None = None
        self._clear_since: float | None = None

    def update(
        self,
        t: float,
        reasons: list[str],
        base: DriverState,
        details: Mapping[str, Mapping[str, float]],
    ) -> list[AlertEvent]:
        """`base` is the state with no active reason (CALIBRATING or NORMAL).

        `details[reason]` holds the metrics reported with that reason's event.
        """
        if self._entered_at is None:
            self._entered_at = t
        # สถานะที่ควรเป็น = สถานะของเหตุผลที่รุนแรงที่สุด (ตัวแรก) หรือ base ถ้าไม่มีเหตุผล
        candidate = REASON_STATE[reasons[0]] if reasons else base
        # สถานะเดิม -> ไม่มี event
        if candidate == self.state:
            self._clear_since = None
            return []
        # จะลดความรุนแรง -> ต้องรอให้ครบ clear_s ก่อน
        if RANK[candidate] < RANK[self.state]:
            if self._clear_since is None:
                self._clear_since = t
            if t - self._clear_since < self.clear_s:
                return []
        # เปลี่ยนสถานะ: สร้าง event 1 ครั้งต่อการเปลี่ยน (ไม่ใช่ทุกเฟรม)
        reason = reasons[0] if reasons else "cleared"
        extra = dict(details.get(reason, {}))
        if not reasons:
            extra["duration_s"] = t - self._entered_at
        return [self.force(t, candidate, reason, reasons, extra)]

    # เปลี่ยนสถานะทันทีและคืน event ที่อธิบายการเปลี่ยน
    def force(
        self,
        t: float,
        state: DriverState,
        reason: str,
        reasons: list[str] | None = None,
        extra: Mapping[str, float | str] | None = None,
    ) -> AlertEvent:
        """Switch to `state` now and return the event describing the transition."""
        details: dict[str, float | str] = {
            "previous_state": self.state.value,
            "reasons": ",".join(reasons or [reason]),
            **(extra or {}),
        }
        self.state, self._entered_at, self._clear_since = state, t, None
        return AlertEvent(t, state, reason, LEVEL[state], details)
