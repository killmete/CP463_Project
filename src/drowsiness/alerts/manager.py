"""AlertManager: turns logic's output into what the driver sees and hears.

It never decides anything about the driver: state and events come from logic.
It only decides presentation timing (cooldown, repeats, escalation), using the
frame timestamps, never the wall clock, so a replayed clip gives the same alerts.
See docs/architecture.md (Alerts).
"""

from __future__ import annotations

import enum
import logging
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from drowsiness.alerts import messages
from drowsiness.config import AlertsConfig
from drowsiness.logic import LEVEL, REASON_STATE, REASONS
from drowsiness.types import AlertEvent, AlertLevel, DriverState, LogicOutput

log = logging.getLogger(__name__)


# คำสั่งเสียง: เงียบ, เสียงเตือน, เสียงสัญญาณภัย, หยุดเสียง
class SoundCue(enum.Enum):
    NONE = "none"
    WARNING = "warning"
    ALARM = "alarm"
    STOP = "stop"


@dataclass(frozen=True, slots=True)
# ผลการตัดสินของ 1 เฟรม: จะแสดงข้อความอะไร สีอะไร กระพริบไหม เล่นเสียงอะไร
# UI และ sink ทุกตัวทำตามสิ่งนี้อย่างเดียว ไม่ตัดสินเอง
class AlertDecision:
    """What to present for one frame."""

    timestamp: float
    state: DriverState
    level: AlertLevel  # presented level (an escalated warning is presented as ALARM)
    reason: str | None
    message: str | None  # banner text; None = no banner (NORMAL)
    flash: bool  # the ALARM banner flashes
    sound: SoundCue
    escalated: bool = False
    toast: str | None = None  # short INFO message (e.g. "Calibrated - monitoring")
    events: tuple[AlertEvent, ...] = ()
    suppressed: bool = False  # an announcement held back by the cooldown (logged)

    @property
    # True ถ้าเฟรมนี้มีเสียงเตือนจริง
    def sounded(self) -> bool:
        return self.sound in (SoundCue.WARNING, SoundCue.ALARM)


# สัญญาของปลายทางแจ้งเตือน (log, เสียง, webhook): emit ต้องคืนทันที ไม่บล็อกลูป
class AlertSink(Protocol):
    def emit(self, decision: AlertDecision) -> None:
        """Must return immediately; anything slow runs on the sink's own thread."""

    def close(self) -> None: ...


_ALERT_LEVELS = (AlertLevel.WARNING, AlertLevel.ALARM)
# ลำดับความรุนแรงของเหตุผล (เลขน้อย = รุนแรงกว่า)
_PRIORITY = {reason: i for i, reason in enumerate(REASONS)}  # lower = more serious
_NO_TOAST = {"recalibrating"}  # the calibrating bar already says it


# reason รุนแรงกว่า than ไหม
def _more_serious(reason: str | None, than: str | None) -> bool:
    if reason is None or reason not in _PRIORITY:
        return False
    return than is None or _PRIORITY[reason] < _PRIORITY.get(than, len(_PRIORITY))


# ตัดสินแค่ "จังหวะ" การเตือน: cooldown, เตือนซ้ำ, escalation
# ใช้เวลาของเฟรม ไม่ใช้นาฬิกาเครื่อง จึง replay วิดีโอแล้วได้การเตือนเหมือนเดิม
class AlertManager:
    def __init__(self, cfg: AlertsConfig, sinks: Sequence[AlertSink] = ()) -> None:
        self.cfg = cfg
        self.sinks = list(sinks)
        # ---- สถานะที่จำไว้ข้ามเฟรม ----
        # _last_sounded = เวลาที่เสียงดังล่าสุดของแต่ละเหตุผล (ใช้ทำ cooldown)
        # _drowsy_entries = เวลาที่เข้าสถานะง่วงแต่ละครั้ง (ใช้นับ escalation)
        self._prev_state: DriverState | None = None
        self._prev_level = AlertLevel.INFO
        self._entry_reason: str | None = None
        self._announced: str | None = None  # reason last announced in this episode
        self._eased_since: float | None = None  # a less serious reason took over at
        self._last_sounded: dict[str, float] = {}
        self._next_repeat: float | None = None
        self._drowsy_entries: deque[float] = deque()
        self._escalated = False
        self._toast: str | None = None
        self._toast_until = float("-inf")

    @property
    # มี sink ไหนเล่นเสียงได้ไหม (ไม่ได้ -> banner แสดง SOUND OFF)
    def sound_available(self) -> bool:
        """False when no sink can currently play sound (muted, or no audio device)."""
        return any(getattr(sink, "available", False) for sink in self.sinks)

    # เรียกทุกเฟรม: ตัดสินใจครั้งเดียว แล้วส่งให้ทุก sink
    # sink ตัวไหนพังแค่ log ไว้ ระบบเฝ้าระวังต้องทำงานต่อ
    def handle(self, out: LogicOutput, now: float) -> AlertDecision:
        """Decide what to present for this frame and pass it to every sink."""
        decision = self._decide(out, now)
        for sink in self.sinks:
            try:
                sink.emit(decision)
            except Exception:  # a broken sink must never stop monitoring
                log.exception("alert sink %s failed", type(sink).__name__)
        return decision

    def close(self) -> None:
        for sink in self.sinks:
            try:
                sink.close()
            except Exception:
                log.exception("closing alert sink %s failed", type(sink).__name__)

    def __enter__(self) -> AlertManager:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- policy (deterministic: depends only on the outputs and timestamps given) -----

    # นโยบายการเตือนทั้งหมดอยู่ที่นี่
    def _decide(self, out: LogicOutput, now: float) -> AlertDecision:
        state = out.state
        level = LEVEL[state]
        # event INFO (เช่น calibrated) -> แสดง toast สั้นๆ; event เข้าสถานะ -> จำเหตุผลไว้
        for event in out.events:
            if event.level == AlertLevel.INFO:
                text = messages.MESSAGES.get(event.reason)
                if text and event.reason not in _NO_TOAST:
                    self._toast, self._toast_until = text, now + self.cfg.info_display_s
            elif event.state == state:
                self._entry_reason = event.reason
        # เหตุผลที่รุนแรงสุดที่ตรงกับสถานะตอนนี้; ถ้าไม่มี (กำลังรอ clear_s) ใช้เหตุผลตอนเข้า
        reason = next((r for r in out.reasons if REASON_STATE.get(r) == state), None)
        active = reason is not None
        reason = reason or self._entry_reason  # state held during clear_s

        sound, suppressed = SoundCue.NONE, False
        if level in _ALERT_LEVELS:
            # เข้าสถานะเตือนใหม่
            if state != self._prev_state:
                sound, suppressed = self._enter(state, level, reason, now)
            # สถานะเดิมแต่เหตุผลรุนแรงขึ้น -> ประกาศใหม่
            elif active and _more_serious(reason, self._announced):
                # A more serious reason within the same state (e.g. yawning -> microsleep):
                # announce it like a new episode.
                sound, suppressed = self._announce(state, level, reason, now)
            # ถึงเวลาเตือนซ้ำ (alarm ถี่กว่า warning)
            elif self._next_repeat is not None and now >= self._next_repeat:
                sound = SoundCue.ALARM if level == AlertLevel.ALARM else SoundCue.WARNING
                self._next_repeat = now + self._repeat_s(level)
            self._track_easing(reason if active else None, now)
            if sound in (SoundCue.WARNING, SoundCue.ALARM) and reason is not None:
                self._last_sounded[reason] = now
        else:
            # กลับมาปกติ: หยุดเสียง แล้วล้างสถานะของรอบนี้ (cooldown/escalation ยังจำไว้)
            if self._prev_level in _ALERT_LEVELS:
                sound = SoundCue.STOP
            self._escalated = False
            self._next_repeat = self._entry_reason = self._announced = self._eased_since = None
        self._prev_state, self._prev_level = state, level

        # เลือกข้อความบน banner
        if level in _ALERT_LEVELS:
            text = messages.ESCALATED if self._escalated else messages.message_for(reason, level)
        elif state == DriverState.CALIBRATING or out.metrics.get("calibrating"):
            text = messages.CALIBRATING
        else:
            text = None
        # สร้างผลการตัดสินของเฟรมนี้
        return AlertDecision(
            timestamp=now,
            state=state,
            level=AlertLevel.ALARM if self._escalated else level,
            reason=reason if level in _ALERT_LEVELS else None,
            message=text,
            flash=level == AlertLevel.ALARM,
            sound=sound,
            escalated=self._escalated,
            toast=self._toast if now < self._toast_until else None,
            events=tuple(out.events),
            suppressed=suppressed,
        )

    # นับว่าเหตุผลเบาลงมานานแค่ไหน: ต้องเบาต่อเนื่อง rearm_s ถึงจะนับว่าจบจริง
    # กันค่าที่กระพริบรอบเกณฑ์ (เช่น PERCLOS) ทำให้เตือนใหม่ทุกเฟรม
    def _track_easing(self, reason: str | None, now: float) -> None:
        """Re-arm the announced reason once a less serious one has held for rearm_s.

        Logic's reasons have no hysteresis of their own (e.g. PERCLOS hovering at its
        threshold flips every frame), so a recurrence only counts as new after the
        serious reason has really gone, never on a single-frame flicker.
        """
        if reason is None or not _more_serious(self._announced, reason):
            self._eased_since = None
        elif self._eased_since is None:
            self._eased_since = now
        elif now - self._eased_since >= self.cfg.rearm_s:
            self._announced, self._eased_since = reason, None

    # เข้าสถานะเตือนใหม่: ถ้าลดจาก ALARM -> ไม่ดังใหม่ (STOP) หรือเตือนต่อถ้า NO_FACE
    def _enter(
        self, state: DriverState, level: AlertLevel, reason: str | None, now: float
    ) -> tuple[SoundCue, bool]:
        """A new alert state."""
        self._escalated = False
        if level < self._prev_level:  # de-escalation from ALARM
            self._announced = reason
            self._next_repeat = now + self._repeat_s(level)
            if state == DriverState.DROWSY_WARNING:
                return SoundCue.STOP, False  # same drowsy episode, recovering: no new chime
            # e.g. asleep and slumped out of view -> NO_FACE: keep sounding, no gap.
            return SoundCue.WARNING, False
        return self._announce(state, level, reason, now)

    # ประกาศการเตือน: ALARM ดังเสมอ, ง่วงซ้ำหลายครั้ง -> ดังเป็น ALARM,
    # นอกนั้นดังถ้าพ้น cooldown ไม่งั้นแสดงแค่ข้อความ (suppressed)
    def _announce(
        self, state: DriverState, level: AlertLevel, reason: str | None, now: float
    ) -> tuple[SoundCue, bool]:
        self._announced, self._eased_since = reason, None
        self._next_repeat = now + self._repeat_s(level)
        if level == AlertLevel.ALARM:
            return SoundCue.ALARM, False  # never held back by the cooldown
        if state == DriverState.DROWSY_WARNING and self._escalate(now):
            return SoundCue.ALARM, False  # escalation bypasses the cooldown; plays once
        if self._cooled_down(reason, now):
            return SoundCue.WARNING, False
        return SoundCue.NONE, True

    # ระยะเวลาเตือนซ้ำตามระดับ
    def _repeat_s(self, level: AlertLevel) -> float:
        return self.cfg.alarm_repeat_s if level == AlertLevel.ALARM else self.cfg.warning_repeat_s

    # เหตุผลนี้ไม่ได้ดังมาอย่างน้อย cooldown_s (30 วินาที) แล้วหรือยัง
    def _cooled_down(self, reason: str | None, now: float) -> bool:
        last = self._last_sounded.get(reason) if reason is not None else None
        return last is None or now - last >= self.cfg.cooldown_s

    # นับการง่วงใน escalation_window_s ล่าสุด (sliding window)
    # ครบ escalation_count (3 ครั้งใน 300 วินาที) -> escalate
    def _escalate(self, now: float) -> bool:
        """Count a drowsiness warning (sounded or not); True if it is an escalating one."""
        window = self._drowsy_entries
        window.append(now)
        while window and window[0] < now - self.cfg.escalation_window_s:
            window.popleft()
        self._escalated = len(window) >= self.cfg.escalation_count
        return self._escalated
