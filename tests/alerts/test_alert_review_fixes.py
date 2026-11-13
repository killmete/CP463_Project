"""Regression tests for the milestone 4 review: no silent gaps, escalation, sound recovery."""

from __future__ import annotations

import sys
import threading

import numpy as np
from test_alert_manager import CFG, Driver, S
from test_sound_sink import FakeBackend, decision

from drowsiness import app as app_module
from drowsiness.alerts import AlertDecision, AlertManager, LogSink, SoundCue
from drowsiness.alerts.messages import MESSAGES
from drowsiness.alerts.sound import SoundSink, _sounddevice
from drowsiness.app import main
from drowsiness.types import AlertEvent, AlertLevel, LogicOutput
from drowsiness.ui import draw_banner


def test_more_serious_reason_within_a_state_is_announced():
    d = Driver().hold(1, S.NORMAL).hold(3, S.DROWSY_WARNING, "yawning")
    mark = len(d.decisions)
    d.hold(2, S.DROWSY_WARNING, "microsleep")  # no state change, no logic event
    assert d.sounds(mark) == [(4.0, SoundCue.WARNING)]
    assert d.decisions[-1].message == MESSAGES["microsleep"]


def test_repeated_microsleeps_inside_a_long_yawning_state_escalate():
    d = Driver().hold(1, S.NORMAL).hold(2, S.DROWSY_WARNING, "yawning")
    for _ in range(2):  # yawning for longer than alerts.rearm_s between them
        d.hold(1.6, S.DROWSY_WARNING, "microsleep").hold(4, S.DROWSY_WARNING, "yawning")
    mark = len(d.decisions)
    d.hold(1.6, S.DROWSY_WARNING, "microsleep")
    assert d.decisions[mark].escalated and d.sounds(mark)[0][1] == SoundCue.ALARM


def test_a_reason_flickering_every_frame_is_announced_once():
    # PERCLOS hovering at its threshold: the reason flips every frame (30 fps, 20 s).
    d = Driver(fps=30).hold(1, S.NORMAL).hold(2, S.DROWSY_WARNING, "yawning")
    mark = len(d.decisions)
    for i in range(600):
        d.hold(1 / 30, S.DROWSY_WARNING, "perclos_warning" if i % 2 else "yawning")
    later = d.decisions[mark:]
    assert not any(x.escalated for x in later)
    assert [s for _, s in d.sounds(mark)].count(SoundCue.ALARM) == 0
    assert [s for _, s in d.sounds(mark)].count(SoundCue.WARNING) <= 3  # 1 + 10 s repeats


def test_a_reason_that_eased_briefly_is_not_re_announced():
    d = Driver().hold(1, S.NORMAL).hold(2, S.DROWSY_WARNING, "yawning")
    d.hold(1.6, S.DROWSY_WARNING, "microsleep").hold(2, S.DROWSY_WARNING, "yawning")
    mark = len(d.decisions)
    d.hold(1.6, S.DROWSY_WARNING, "microsleep")  # back within rearm_s (3 s)
    assert d.sounds(mark) == []


def test_warnings_held_back_by_the_cooldown_still_count_for_escalation():
    d = Driver().hold(1, S.NORMAL)
    for _ in range(2):
        d.hold(1.6, S.DROWSY_WARNING, "microsleep").hold(8, S.NORMAL)  # every ~10 s
    assert d.decisions[-1].sound != SoundCue.ALARM
    mark = len(d.decisions)
    d.hold(1.6, S.DROWSY_WARNING, "microsleep")  # 3rd within 300 s; cooldown is 30 s
    assert d.decisions[mark].escalated and d.sounds(mark)[0][1] == SoundCue.ALARM


def test_cooldown_suppression_is_flagged_for_the_log():
    d = Driver().hold(1, S.NORMAL).hold(1, S.DISTRACTED, "long_distraction").hold(1, S.NORMAL)
    mark = len(d.decisions)
    d.hold(1, S.DISTRACTED, "long_distraction")
    assert d.decisions[mark].suppressed and d.decisions[mark].sound == SoundCue.NONE


def test_alarm_dropping_to_no_face_keeps_sounding():
    d = Driver().hold(1, S.NORMAL).hold(3, S.DROWSY_ALARM, "sleep")
    mark = len(d.decisions)
    d.hold(11, S.NO_FACE, "no_face")
    assert d.sounds(mark) == [(4.0, SoundCue.WARNING), (14.0, SoundCue.WARNING)]


def test_alarm_dropping_to_a_drowsy_warning_just_stops():
    d = Driver().hold(1, S.NORMAL).hold(3, S.DROWSY_ALARM, "sleep")
    mark = len(d.decisions)
    d.hold(2, S.DROWSY_WARNING, "microsleep")
    assert d.sounds(mark) == [(4.0, SoundCue.STOP)]


def test_escalated_warning_stops_when_it_clears():
    d = Driver().hold(1, S.NORMAL)
    for reason in ("microsleep", "head_nod", "yawning"):
        d.hold(2, S.DROWSY_WARNING, reason).hold(2, S.NORMAL)
    assert d.decisions[-20].escalated is False  # cleared frame
    assert d.sounds()[-1][1] == SoundCue.STOP


def test_cleared_event_does_not_wipe_a_toast():
    calibrated = AlertEvent(0.0, S.NORMAL, "calibrated", AlertLevel.INFO, {})
    cleared = AlertEvent(0.2, S.NORMAL, "cleared", AlertLevel.INFO, {})
    m = AlertManager(CFG)
    m.handle(LogicOutput(S.NORMAL, (calibrated,)), 0.0)
    assert m.handle(LogicOutput(S.NORMAL, (cleared,)), 0.2).toast == MESSAGES["calibrated"]


def test_recalibrating_shows_the_bar_not_a_second_toast():
    event = AlertEvent(0.0, S.NORMAL, "recalibrating", AlertLevel.INFO, {})
    out = LogicOutput(S.NORMAL, (event,), (), {"calibrating": 1.0})
    decision = AlertManager(CFG).handle(out, 0.0)
    assert decision.toast is None and decision.message == "Calibrating - look ahead"


def test_log_sink_logs_suppressed_announcements(caplog):
    d = AlertDecision(1.0, S.DISTRACTED, AlertLevel.WARNING, "long_distraction", "x", False,
                      SoundCue.NONE, suppressed=True)  # fmt: skip
    with caplog.at_level("DEBUG", logger="drowsiness"):
        LogSink().emit(d)
    assert "not sounded (cooldown)" in caplog.text


# -- sound recovery ---------------------------------------------------------------


def test_sound_comes_back_after_the_device_returns(caplog):
    now = [0.0]
    attempts = []
    played = threading.Event()

    class Recovering(FakeBackend):
        def play(self, data, samplerate):
            super().play(data, samplerate)
            played.set()

    def factory():
        attempts.append(now[0])
        if len(attempts) == 1:
            raise OSError("no device")
        return Recovering()

    with caplog.at_level("WARNING"):
        sink = SoundSink(CFG, factory, retry_s=10.0, clock=lambda: now[0])
        sink.emit(decision(SoundCue.ALARM))  # too early to retry: dropped
        now[0] = 11.0
        sink.emit(decision(SoundCue.ALARM))
        assert played.wait(2.0)
        sink.close()
    assert attempts == [0.0, 11.0] and sink.available
    assert "audio available again" in caplog.text


def test_missing_sounddevice_module_is_a_silent_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)  # what the mediapipe guard does
    sink = SoundSink(CFG, _sounddevice)
    sink.emit(decision(SoundCue.WARNING))
    sink.close()
    assert not sink.available


def test_full_queue_keeps_the_latest_stop():
    gate = threading.Event()
    backend = FakeBackend(gate)
    sink = SoundSink(CFG, lambda: backend, maxsize=2)
    sink.emit(decision(SoundCue.ALARM))  # the worker blocks inside play()
    for _ in range(5):
        sink.emit(decision(SoundCue.ALARM))
    sink.emit(decision(SoundCue.STOP))
    gate.set()
    sink.close()
    assert backend.calls[-2:] == ["stop", "stop"]  # the STOP cue, then close()


def test_manager_reports_sound_availability():
    class Fake:
        available = False

        def emit(self, d):
            pass

        def close(self):
            pass

    assert AlertManager(CFG, [LogSink()]).sound_available is False  # muted
    assert AlertManager(CFG, [Fake()]).sound_available is False
    Fake.available = True
    assert AlertManager(CFG, [Fake()]).sound_available is True


def test_banner_shows_sound_off():
    with_tag, without = (np.zeros((480, 640, 3), np.uint8) for _ in range(2))
    d = AlertDecision(0.0, S.NORMAL, AlertLevel.INFO, None, None, False, SoundCue.NONE)
    draw_banner(with_tag, d, 0.0, 2.0, sound_off=True)
    draw_banner(without, d, 0.0, 2.0)
    assert with_tag[400:450, 520:].any() and not without[400:450, 520:].any()


def test_headless_file_replay_is_muted(monkeypatch, synthetic_clip):
    built = []
    monkeypatch.setattr(app_module, "build_perception", lambda cfg: _no_face_perception())
    monkeypatch.setattr(app_module, "build_alerts", lambda cfg: built.append(cfg) or
                        AlertManager(cfg.alerts))  # fmt: skip
    assert main(["--source", str(synthetic_clip), "--no-display"]) == 0
    assert built[0].alerts.sound is False


def _no_face_perception():
    from drowsiness.config import load_config
    from drowsiness.perception import Perception

    class NeverFace:
        def detect(self, rgb, timestamp_ms):
            return None

        def close(self):
            pass

    return Perception(NeverFace(), load_config().preprocess)
