import os
import threading
import time

import cv2
import numpy as np
import pytest

from drowsiness.capture import camera as camera_module
from drowsiness.capture.camera import CameraError, CameraSource, backend_order


class FakeCapture:
    """Stands in for cv2.VideoCapture: produces numbered frames every `period_s`."""

    instances: list["FakeCapture"] = []

    def __init__(self, *args, opens=True, period_s=0.002, fail_after=None):
        self.args = args
        self._opens = opens
        self.period_s = period_s
        self.fail_after = fail_after
        self.count = 0
        self.released = False
        FakeCapture.instances.append(self)

    def isOpened(self):  # noqa: N802 - mirrors the OpenCV API
        return self._opens

    def set(self, prop, value):
        return True

    def get(self, prop):
        return {cv2.CAP_PROP_FRAME_WIDTH: 64, cv2.CAP_PROP_FRAME_HEIGHT: 48}.get(prop, 0.0)

    def read(self):
        if self.released or (self.fail_after is not None and self.count >= self.fail_after):
            return False, None
        time.sleep(self.period_s)
        self.count += 1
        return True, np.full((48, 64, 3), self.count % 256, dtype=np.uint8)

    def release(self):
        self.released = True


@pytest.fixture
def fake_cv2(monkeypatch):
    FakeCapture.instances = []

    def install(**kwargs):
        monkeypatch.setattr(camera_module.cv2, "VideoCapture", lambda *a: FakeCapture(*a, **kwargs))

    install()
    return install


def test_backend_order():
    assert backend_order("dshow") == ["dshow"]
    assert backend_order("auto") in (["msmf", "dshow"], ["any"])


def test_returns_only_newest_frames_and_counts_drops(fake_cv2):
    with CameraSource(index=0, read_timeout_s=1.0) as source:
        indices = []
        for _ in range(10):
            packet = source.read()
            assert packet is not None
            indices.append(packet.index)
            time.sleep(0.02)  # process slower than the camera (2 ms/frame)
    assert all(b > a for a, b in zip(indices, indices[1:], strict=False)), "a frame repeated"
    assert any(b - a > 1 for a, b in zip(indices, indices[1:], strict=False)), "nothing skipped"
    assert source.dropped_frames > 0


def test_timestamps_never_decrease(fake_cv2):
    with CameraSource() as source:
        stamps = [source.read().timestamp for _ in range(20)]
    assert stamps == sorted(stamps)


def test_close_stops_thread_and_releases(fake_cv2):
    source = CameraSource()
    source.open()
    source.read()
    source.close()
    assert not any(t.name == "camera-grabber" and t.is_alive() for t in threading.enumerate())
    assert all(c.released for c in FakeCapture.instances)


def test_open_failure_raises(fake_cv2):
    fake_cv2(opens=False)
    with pytest.raises(CameraError):
        CameraSource(backend="any").open()


def test_lost_camera_raises_instead_of_ending_quietly(fake_cv2):
    fake_cv2(fail_after=5)
    start = time.perf_counter()
    with CameraSource(read_timeout_s=0.3) as source:
        packets = []
        with pytest.raises(CameraError, match="no frame"):
            while True:
                packets.append(source.read())
    assert len(packets) <= 5
    assert time.perf_counter() - start < 2.0
    assert all(c.released for c in FakeCapture.instances)


def test_brief_glitch_does_not_end_stream(fake_cv2, monkeypatch):
    calls = {"n": 0}
    original_read = FakeCapture.read

    def glitchy_read(self):
        calls["n"] += 1
        if 5 <= calls["n"] < 15:  # ten failed reads in a row, then recovers
            return False, None
        return original_read(self)

    monkeypatch.setattr(FakeCapture, "read", glitchy_read)
    with CameraSource(read_timeout_s=1.0) as source:
        for _ in range(10):
            assert source.read() is not None


def test_read_error_in_grabber_raises(fake_cv2, monkeypatch):
    def broken_read(self):
        raise cv2.error("device unplugged")

    with CameraSource(read_timeout_s=1.0) as source:
        monkeypatch.setattr(FakeCapture, "read", broken_read)
        with pytest.raises(CameraError, match="read failed"):
            for _ in range(100):
                source.read()


def test_read_after_close_returns_none(fake_cv2):
    source = CameraSource()
    source.open()
    source.close()
    assert source.read() is None


def test_read_before_open_raises():
    with pytest.raises(RuntimeError):
        CameraSource().read()


def test_open_twice_raises(fake_cv2):
    with CameraSource() as source, pytest.raises(RuntimeError, match="already open"):
        source.open()


def test_close_with_blocked_grabber_never_releases_from_caller(fake_cv2, monkeypatch):
    monkeypatch.setattr(camera_module, "_JOIN_TIMEOUT_S", 0.1)
    source = CameraSource()
    source.open()
    capture = FakeCapture.instances[-1]
    capture.period_s = 0.5  # next native read blocks longer than the join timeout
    time.sleep(0.05)
    source.close()
    assert not capture.released, "released while the grabber was still inside read()"
    time.sleep(0.8)
    assert capture.released, "grabber should release once its read returns"


@pytest.mark.camera
@pytest.mark.skipif(not os.environ.get("DROWSINESS_CAMERA_TESTS"), reason="needs a webcam")
def test_real_camera_delivers_frames():
    with CameraSource(index=0) as source:
        packet = source.read()
    assert packet is not None and packet.image.ndim == 3
