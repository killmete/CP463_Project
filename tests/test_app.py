import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from drowsiness import app as app_module
from drowsiness.app import FpsMeter, main, run_loop
from drowsiness.capture import CameraError, FrameSource, VideoFileSource
from drowsiness.config import load_config
from drowsiness.feature_log import FeatureCsvWriter, read_feature_csv
from drowsiness.perception import LandmarkResult, Perception

CAPTURE_ONLY = ["--no-display", "--no-perception"]


def test_run_loop_processes_frames_headless(synthetic_clip: Path):
    with VideoFileSource(synthetic_clip) as source:
        stats = run_loop(source, load_config(), display=False, max_frames=10)
    assert stats.frames == 10
    assert stats.mean_fps > 0
    assert 0 <= stats.latency_p50_ms <= stats.latency_p95_ms


def test_run_loop_stops_at_end_of_stream(synthetic_clip: Path, clip_spec):
    with VideoFileSource(synthetic_clip) as source:
        stats = run_loop(source, load_config(), display=False)
    assert stats.frames == clip_spec.frames


def test_main_smoke(synthetic_clip: Path):
    assert main(["--source", str(synthetic_clip), *CAPTURE_ONLY, "--max-frames", "10"]) == 0


def test_main_missing_file_fails(tmp_path: Path):
    assert main(["--source", str(tmp_path / "missing.avi"), *CAPTURE_ONLY]) == 1


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
    assert main(["--source", "0", *CAPTURE_ONLY]) == 1


def test_ctrl_c_stops_cleanly_with_stats(synthetic_clip: Path):
    class InterruptAfterFive(VideoFileSource):
        def read(self):
            if self._index == 5:
                raise KeyboardInterrupt
            return super().read()

    with InterruptAfterFive(synthetic_clip) as source:
        stats = run_loop(source, load_config(), display=False)
    assert stats.interrupted
    assert stats.frames == 5


def test_fps_meter():
    meter = FpsMeter(window_s=1.0)
    fps = 0.0
    for i in range(31):
        fps = meter.tick(i / 30)
    assert abs(fps - 30) < 1e-6


def test_capture_only_does_not_import_mediapipe(synthetic_clip: Path):
    # Fresh interpreter: other tests in this process may already have imported mediapipe.
    args = ["--source", str(synthetic_clip), *CAPTURE_ONLY, "--max-frames", "2"]
    code = (
        "import sys; from drowsiness.app import main; "
        f"assert main({args!r}) == 0; "
        "assert 'mediapipe' not in sys.modules, 'mediapipe was imported'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr


class AlwaysFace:
    def __init__(self, landmarks):
        self.landmarks = landmarks

    def detect(self, rgb, timestamp_ms):
        return LandmarkResult(self.landmarks, {"jawOpen": 0.5})

    def close(self):
        pass


def test_run_loop_logs_one_feature_row_per_frame(synthetic_clip: Path, tmp_path: Path, clip_spec):
    cfg = load_config()
    landmarks = np.random.default_rng(0).uniform(0.3, 0.7, (478, 3)).astype(np.float32)
    path = tmp_path / "features.csv"
    with (
        VideoFileSource(synthetic_clip) as source,
        Perception(AlwaysFace(landmarks), cfg.preprocess) as perception,
        FeatureCsvWriter(path) as log,
    ):
        stats = run_loop(source, cfg, display=False, perception=perception, feature_log=log)
    rows = read_feature_csv(path)
    assert len(rows) == stats.frames == clip_spec.frames
    assert [r.timestamp for r in rows] == [i / clip_spec.fps for i in range(clip_spec.frames)]
    assert all(r.face_found and r.blendshapes == {"jawOpen": 0.5} for r in rows)
    assert stats.perception_p95_ms is not None


def test_missing_model_exits_2(synthetic_clip: Path, tmp_path: Path):
    config = tmp_path / "cfg.yaml"
    missing = (tmp_path / "missing.task").as_posix()
    config.write_text(f"perception:\n  model_path: {missing}\n", encoding="utf-8")
    assert main(["--source", str(synthetic_clip), "--no-display", "--config", str(config)]) == 2


def test_log_features_needs_perception(synthetic_clip: Path, tmp_path: Path):
    args = ["--source", str(synthetic_clip), *CAPTURE_ONLY, "--log-features", "x.csv"]
    assert main(args) == 2


@pytest.mark.model
def test_main_with_model_writes_feature_csv(
    synthetic_clip: Path, tmp_path: Path, model_path: Path, clip_spec, monkeypatch
):
    monkeypatch.chdir(model_path.parents[1])  # default perception.model_path is cwd-relative
    out = tmp_path / "features.csv"
    args = ["--source", str(synthetic_clip), "--no-display", "--log-features", str(out)]
    assert main(args) == 0
    rows = read_feature_csv(out)
    assert len(rows) == clip_spec.frames
    assert not any(r.face_found for r in rows)  # flat gray frames have no face


def test_main_closes_perception_when_source_fails(monkeypatch, tmp_path: Path):
    closed = []

    class FakePerception(Perception):
        def close(self):
            closed.append(True)

    cfg_preprocess = load_config().preprocess
    monkeypatch.setattr(
        app_module, "build_perception", lambda cfg: FakePerception(AlwaysFace(None), cfg_preprocess)
    )
    csv_path = tmp_path / "f.csv"
    args = [
        "--source",
        str(tmp_path / "missing.avi"),
        "--no-display",
        "--log-features",
        str(csv_path),
    ]
    assert main(args) == 1
    assert closed == [True]
    assert not csv_path.exists()  # no stub CSV for a source that never opened
