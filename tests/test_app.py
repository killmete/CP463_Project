import sys
from pathlib import Path

from conftest import CLIP_FRAMES
from drowsiness import app as app_module
from drowsiness.app import FpsMeter, main, run_loop
from drowsiness.capture import CameraError, FrameSource, VideoFileSource
from drowsiness.config import load_config


def test_run_loop_processes_frames_headless(synthetic_clip: Path):
    with VideoFileSource(synthetic_clip) as source:
        stats = run_loop(source, load_config(), display=False, max_frames=10)
    assert stats.frames == 10
    assert stats.mean_fps > 0
    assert 0 <= stats.latency_p50_ms <= stats.latency_p95_ms


def test_run_loop_stops_at_end_of_stream(synthetic_clip: Path):
    with VideoFileSource(synthetic_clip) as source:
        stats = run_loop(source, load_config(), display=False)
    assert stats.frames == CLIP_FRAMES


def test_main_smoke(synthetic_clip: Path):
    assert main(["--source", str(synthetic_clip), "--no-display", "--max-frames", "10"]) == 0


def test_main_missing_file_fails(tmp_path: Path):
    assert main(["--source", str(tmp_path / "missing.avi"), "--no-display"]) == 1


def test_main_bad_config_fails(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("capture:\n  colour: red\n", encoding="utf-8")
    assert main(["--config", str(bad), "--no-display"]) == 2


def test_main_fails_when_live_camera_is_lost(monkeypatch):
    class LostCamera(FrameSource):
        def open(self):
            pass

        def read(self):
            raise CameraError("camera 0 gave no frame for 1.0 s")

        def close(self):
            pass

        @property
        def fps(self):
            return 30.0

    monkeypatch.setattr(app_module, "open_source", lambda cfg: LostCamera())
    assert main(["--source", "0", "--no-display"]) == 1


def test_fps_meter():
    meter = FpsMeter(window_s=1.0)
    fps = 0.0
    for i in range(31):
        fps = meter.tick(i / 30)
    assert abs(fps - 30) < 1e-6


def test_milestone1_does_not_import_mediapipe(synthetic_clip: Path):
    main(["--source", str(synthetic_clip), "--no-display", "--max-frames", "2"])
    assert "mediapipe" not in sys.modules
