from pathlib import Path

import pytest

from drowsiness.config import ConfigError, load_config


def test_defaults_load():
    cfg = load_config()
    assert cfg.capture.source == 0
    assert cfg.capture.backend == "auto"
    assert cfg.runtime.max_frames is None


def test_defaults_match_literature_review():
    cfg = load_config()
    assert cfg.eyes.microsleep_s == 1.5
    assert cfg.eyes.sleep_s == 3.0
    assert cfg.eyes.closed_ratio == 0.2
    assert cfg.perclos.window_s == 60.0
    assert cfg.distraction.long_s == 3.0
    assert cfg.distraction.short_total_s == 10.0
    assert cfg.distraction.short_window_s == 30.0


def test_ints_are_accepted_for_floats():
    cfg = load_config(overrides={"capture": {"fps": 25}})
    assert cfg.capture.fps == 25.0
    assert isinstance(cfg.capture.fps, float)


def test_user_file_is_deep_merged(tmp_path: Path):
    user = tmp_path / "user.yaml"
    user.write_text("capture:\n  width: 1280\nyawn:\n  warning_count: 5\n", encoding="utf-8")
    cfg = load_config(user)
    assert cfg.capture.width == 1280
    assert cfg.capture.height == 480  # untouched default
    assert cfg.yawn.warning_count == 5


def test_overrides_win_over_file(tmp_path: Path):
    user = tmp_path / "user.yaml"
    user.write_text("capture:\n  source: 1\n", encoding="utf-8")
    cfg = load_config(user, overrides={"capture": {"source": "clip.mp4"}})
    assert cfg.capture.source == "clip.mp4"


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"capture": {"widht": 640}}, "unknown key"),
        ({"bogus": {}}, "unknown key"),
        ({"capture": {"width": "wide"}}, "expected int"),
        ({"runtime": {"display": 1}}, "expected bool"),
        ({"eyes": {"sleep_s": True}}, "expected float"),
        ({"capture": {"backend": "v4l9"}}, "backend"),
        ({"runtime": {"max_frames": 0}}, "max_frames"),
        ({"eyes": 3}, "expected a mapping"),
        ({"eyes": {"sleep_s": float("nan")}}, "finite"),
        ({"capture": {"fps": float("inf")}}, "finite"),
        ({"runtime": {"fps_window_s": -1}}, "must be > 0"),
        ({"capture": {"read_timeout_s": 0}}, "must be > 0"),
        ({"capture": {"width": 0}}, "must be > 0"),
        ({"runtime": {"log_level": "info"}}, "log_level"),
        ({"perclos": {"alarm": 1.5}}, "ratio"),
        ({"eyes": {"microsleep_s": 4.0}}, "microsleep_s < eyes.sleep_s"),
        ({"perclos": {"warning": 0.3}}, "warning < perclos.alarm"),
        ({"distraction": {"short_total_s": 40.0}}, "short_total_s"),
        ({"perception": {"min_tracking_confidence": 1.2}}, "min_tracking_confidence"),
        ({"perception": {"num_faces": 0}}, "num_faces"),
        ({"perception": {"model_path": None}}, "expected str"),
        ({"overlay": {"values": "yes"}}, "expected bool"),
    ],
)
def test_invalid_config_raises(overrides, message):
    with pytest.raises(ConfigError, match=message):
        load_config(overrides=overrides)


def test_camera_index_zero_is_valid():
    assert load_config(overrides={"capture": {"source": 0}}).capture.source == 0


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")
