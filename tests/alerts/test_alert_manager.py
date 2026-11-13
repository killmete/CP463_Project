"""AlertManager policy: cooldown, repeats, de-escalation, escalation, INFO toasts.

Driven by hand-built LogicOutput sequences and timestamps (no sleeping, no audio).
"""

from __future__ import annotations

import dataclasses

import pytest

from drowsiness.alerts import AlertManager, SoundCue
from drowsiness.alerts.messages import CALIBRATING, ESCALATED, MAX_CHARS, MESSAGES
from drowsiness.config import load_config
from drowsiness.logic import REASON_STATE, REASONS
from drowsiness.types import AlertEvent, AlertLevel, DriverState, LogicOutput

S = DriverState
CFG = load_config().alerts
LEVEL = {
    S.DROWSY_ALARM: AlertLevel.ALARM,
    S.DROWSY_WARNING: AlertLevel.WARNING,
    S.DISTRACTED: AlertLevel.WARNING,
    S.NO_FACE: AlertLevel.WARNING,
    S.NORMAL: AlertLevel.INFO,
    S.CALIBRATING: AlertLevel.INFO,
}


class Recorder:
    def __init__(self):
        self.decisions = []
        self.closed = False

    def emit(self, decision):
        self.decisions.append(decision)

    def close(self):
        self.closed = True


class Driver:
    """Feeds the manager a state timeline at a fixed frame rate, like logic would."""

    def __init__(self, cfg=CFG, fps: float = 10.0):
        self.sink = Recorder()
        self.manager = AlertManager(cfg, [self.sink])
        self.fps, self.n, self.state = fps, 0, S.NORMAL
        self.decisions = []

    def hold(self, seconds: float, state: DriverState, reason: str | None = None, **kw):
        reasons = (reason,) if reason else ()
        for i in range(round(seconds * self.fps)):
            self.t = self.n / self.fps  # exact frame times, no accumulated rounding
            events = kw.pop("events", ())
            if i == 0 and state != self.state:
                cause = reason or "cleared"
                events = (AlertEvent(self.t, state, cause, LEVEL[state], {}), *events)
            out = LogicOutput(state, tuple(events), reasons, kw.get("metrics", {}))
            self.decisions.append(self.manager.handle(out, self.t))
            self.n += 1
        self.state = state
        return self

    def sounds(self, since: int = 0):
        return [(round(d.timestamp, 1), d.sound) for d in self.decisions[since:] if d.sound
                != SoundCue.NONE]  # fmt: skip


def test_normal_shows_no_banner_and_no_sound():
    d = Driver().hold(5, S.NORMAL)
    assert all(x.message is None and x.sound == SoundCue.NONE for x in d.decisions)


def test_alarm_sounds_at_once_repeats_and_stops_on_clear():
    d = Driver().hold(1, S.NORMAL).hold(5, S.DROWSY_ALARM, "sleep").hold(1, S.NORMAL)
    assert d.sounds() == [
        (1.0, SoundCue.ALARM), (3.0, SoundCue.ALARM), (5.0, SoundCue.ALARM), (6.0, SoundCue.STOP),
    ]  # fmt: skip
    alarm = d.decisions[10]
    assert alarm.flash and alarm.level == AlertLevel.ALARM
    assert alarm.message == MESSAGES["sleep"]


def test_warning_chimes_then_repeats_while_it_persists():
    d = Driver().hold(1, S.NORMAL).hold(21, S.DISTRACTED, "long_distraction")
    assert d.sounds() == [(1.0, SoundCue.WARNING), (11.0, SoundCue.WARNING),
                          (21.0, SoundCue.WARNING)]  # fmt: skip
    assert not d.decisions[-1].flash


@pytest.mark.parametrize("gap_s, sounded", [(27.9, False), (28.1, True)])  # 30 s after t=1
def test_same_warning_within_cooldown_is_not_sounded_again(gap_s, sounded):
    # First episode sounded at t=1, cleared at t=3; it comes back after gap_s.
    d = Driver(fps=100).hold(1, S.NORMAL).hold(2, S.DISTRACTED, "long_distraction")
    d.hold(gap_s, S.NORMAL)
    mark = len(d.decisions)
    d.hold(1, S.DISTRACTED, "long_distraction")
    assert (SoundCue.WARNING in [s for _, s in d.sounds(mark)]) is sounded
    assert d.decisions[-1].message == MESSAGES["long_distraction"]  # banner shows anyway


def test_cooldown_is_per_reason():
    d = Driver().hold(1, S.NORMAL).hold(2, S.DISTRACTED, "long_distraction").hold(2, S.NORMAL)
    mark = len(d.decisions)
    d.hold(2, S.DROWSY_WARNING, "microsleep")
    assert d.sounds(mark) == [(5.0, SoundCue.WARNING)]


def test_alarm_ignores_the_cooldown():
    d = Driver().hold(1, S.NORMAL).hold(1, S.DROWSY_ALARM, "sleep").hold(1, S.NORMAL)
    mark = len(d.decisions)
    d.hold(1, S.DROWSY_ALARM, "sleep")
    assert d.sounds(mark)[0] == (3.0, SoundCue.ALARM)


def test_de_escalation_stops_the_alarm_without_a_new_chime():
    d = Driver().hold(1, S.NORMAL).hold(2, S.DROWSY_ALARM, "sleep")
    mark = len(d.decisions)
    d.hold(5, S.DROWSY_WARNING, "microsleep")
    assert d.sounds(mark) == [(3.0, SoundCue.STOP)]
    assert d.decisions[-1].level == AlertLevel.WARNING and not d.decisions[-1].flash


def test_repeated_drowsiness_warnings_escalate():
    d = Driver().hold(1, S.NORMAL)
    sounds = []
    for reason in ("microsleep", "head_nod", "yawning"):
        mark = len(d.decisions)
        d.hold(2, S.DROWSY_WARNING, reason).hold(5, S.NORMAL)
        sounds.append(d.sounds(mark)[0][1])
        last_warning = d.decisions[mark]  # different reasons: the cooldown doesn't apply
    assert sounds == [SoundCue.WARNING, SoundCue.WARNING, SoundCue.ALARM]
    assert last_warning.escalated and last_warning.level == AlertLevel.ALARM
    assert last_warning.message == ESCALATED and not last_warning.flash
    assert d.decisions[-1].escalated is False  # cleared


def test_escalation_window_expires():
    cfg = dataclasses.replace(CFG, escalation_window_s=10.0)
    d = Driver(cfg).hold(1, S.NORMAL)
    for reason in ("microsleep", "head_nod"):
        d.hold(2, S.DROWSY_WARNING, reason).hold(5, S.NORMAL)
    mark = len(d.decisions)
    d.hold(2, S.DROWSY_WARNING, "yawning")  # first warning is now > 10 s old
    assert d.sounds(mark)[0][1] == SoundCue.WARNING and not d.decisions[mark].escalated


def test_distraction_warnings_do_not_escalate():
    d = Driver().hold(1, S.NORMAL)
    for reason in ("long_distraction", "short_distraction", "no_face"):
        state = REASON_STATE[reason]
        mark = len(d.decisions)
        d.hold(2, state, reason).hold(3, S.NORMAL)
        assert not d.decisions[mark].escalated


def test_info_events_show_a_toast_and_are_never_sounded():
    info = AlertEvent(0.0, S.NORMAL, "calibrated", AlertLevel.INFO, {})
    d = Driver().hold(0.1, S.CALIBRATING).hold(5, S.NORMAL, events=(info,))
    assert d.sounds() == []
    toasts = [x.toast for x in d.decisions]
    assert toasts[1] == MESSAGES["calibrated"] and toasts[-1] is None
    assert abs(sum(t is not None for t in toasts) - CFG.info_display_s * 10) <= 1


def test_calibrating_shows_the_calibrating_message():
    d = Driver().hold(1, S.CALIBRATING).hold(1, S.NORMAL, metrics={"calibrating": 1.0})
    assert {x.message for x in d.decisions} == {CALIBRATING}


def test_banner_keeps_the_entry_reason_while_the_state_is_held():
    # During clear_s the state is held but its reason is no longer active.
    d = Driver().hold(1, S.NORMAL).hold(0.5, S.DROWSY_WARNING, "microsleep")
    d.hold(0.5, S.DROWSY_WARNING)  # no active reasons
    assert d.decisions[-1].message == MESSAGES["microsleep"]


def test_unknown_reason_gets_a_generic_message():
    d = Driver().hold(1, S.NORMAL).hold(1, S.DISTRACTED, "new_rule")
    assert d.decisions[-1].message == "Attention - watch the road"


def test_every_reason_has_a_short_ascii_message():
    texts = [*MESSAGES.values(), ESCALATED, CALIBRATING]
    assert set(REASONS) <= set(MESSAGES)
    assert all(len(t) <= MAX_CHARS and t.isascii() for t in texts)


def test_same_inputs_give_the_same_decisions():
    def scenario():
        d = Driver().hold(1, S.NORMAL).hold(3, S.DROWSY_ALARM, "sleep")
        return d.hold(2, S.DROWSY_WARNING, "microsleep").hold(2, S.NORMAL).decisions

    assert scenario() == scenario()


def test_a_failing_sink_does_not_stop_the_others():
    class Broken:
        def emit(self, decision):
            raise RuntimeError("boom")

        def close(self):
            raise RuntimeError("boom")

    good = Recorder()
    with AlertManager(CFG, [Broken(), good]) as manager:
        manager.handle(LogicOutput(S.NORMAL), 0.0)
    assert len(good.decisions) == 1 and good.closed
