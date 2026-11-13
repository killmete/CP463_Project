"""SoundSink with fake audio backends: no real audio in tests or CI."""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from drowsiness.alerts import AlertDecision, SoundCue, WebhookSink
from drowsiness.alerts.sound import SoundSink, make_patterns
from drowsiness.config import load_config
from drowsiness.types import AlertLevel, DriverState

CFG = load_config().alerts


def decision(sound: SoundCue, t: float = 0.0) -> AlertDecision:
    return AlertDecision(t, DriverState.DROWSY_ALARM, AlertLevel.ALARM, "sleep", "x", True, sound)


class FakeBackend:
    def __init__(self, gate: threading.Event | None = None):
        self.calls: list[str] = []
        self.gate = gate

    def play(self, data, samplerate):
        if self.gate is not None:
            self.gate.wait(2.0)  # simulate a slow audio device
        self.calls.append(f"play {len(data)}@{samplerate}")

    def stop(self):
        self.calls.append("stop")


def test_patterns_are_short_float32_and_within_volume():
    patterns = make_patterns(CFG.sample_rate, CFG.volume)
    for cue, data in patterns.items():
        assert data.dtype == np.float32
        assert np.abs(data).max() <= CFG.volume + 1e-6
        assert 0.2 < len(data) / CFG.sample_rate < 1.5, cue
        assert abs(data[0]) < 1e-3 and abs(data[-1]) < 1e-3  # faded: no clicks


def test_cues_reach_the_backend_in_order():
    backend = FakeBackend()
    sink = SoundSink(CFG, lambda: backend)
    for cue in (SoundCue.WARNING, SoundCue.NONE, SoundCue.ALARM, SoundCue.STOP):
        sink.emit(decision(cue))
    sink.close()
    assert [c.split()[0] for c in backend.calls] == ["play", "play", "stop", "stop"]
    assert sink.available


def test_emit_never_blocks_on_a_slow_device():
    gate = threading.Event()
    backend = FakeBackend(gate)
    sink = SoundSink(CFG, lambda: backend, maxsize=2)
    start = time.perf_counter()
    for _ in range(50):
        sink.emit(decision(SoundCue.ALARM))
    assert time.perf_counter() - start < 0.1
    gate.set()
    sink.close()
    assert len(backend.calls) <= 4  # the backlog was dropped, not queued up


def test_no_audio_device_falls_back_to_silent(caplog):
    def broken():
        raise OSError("PortAudio library not found")

    with caplog.at_level(logging.WARNING):
        sink = SoundSink(CFG, broken)
        sink.emit(decision(SoundCue.ALARM))
        sink.close()
    assert not sink.available
    assert caplog.text.count("audio unavailable") == 1


def test_playback_error_disables_sound_once(caplog):
    class Failing(FakeBackend):
        def play(self, data, samplerate):
            raise RuntimeError("device unplugged")

    with caplog.at_level(logging.WARNING):
        sink = SoundSink(CFG, Failing)
        for _ in range(3):
            sink.emit(decision(SoundCue.WARNING))
        sink.close()
    assert not sink.available and caplog.text.count("audio unavailable") == 1


def test_webhook_posts_only_sounded_alerts(monkeypatch):
    sent = []
    monkeypatch.setattr(WebhookSink, "post", lambda self, p: sent.append(p))
    sink = WebhookSink("http://127.0.0.1:9/alerts", timeout_s=0.5)
    sink.emit(decision(SoundCue.NONE))
    sink.emit(decision(SoundCue.ALARM, t=2.0))
    sink.close()
    assert sent == [{"timestamp": 2.0, "state": "DROWSY_ALARM", "reason": "sleep",
                     "level": "ALARM", "escalated": False}]  # fmt: skip


def test_webhook_network_error_is_logged_not_raised(caplog):
    sink = WebhookSink("http://127.0.0.1:9/alerts", timeout_s=0.5)  # nothing listens there
    with caplog.at_level(logging.WARNING):
        sink.post({"timestamp": 0.0})
    sink.close()
    assert "webhook POST failed" in caplog.text
