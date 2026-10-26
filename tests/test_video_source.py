from pathlib import Path

import pytest

from conftest import CLIP_FPS, CLIP_FRAMES, CLIP_SIZE
from drowsiness.capture import VideoFileSource, open_source
from drowsiness.config import load_config


def _read_all(path: Path) -> list:
    with VideoFileSource(path) as source:
        packets = []
        while (packet := source.read()) is not None:
            packets.append(packet)
    return packets


def test_reads_every_frame_with_deterministic_timestamps(synthetic_clip: Path):
    packets = _read_all(synthetic_clip)
    assert [p.index for p in packets] == list(range(CLIP_FRAMES))
    for p in packets:
        assert p.timestamp == pytest.approx(p.index / CLIP_FPS, abs=1e-9)
        assert p.image.shape == (CLIP_SIZE[1], CLIP_SIZE[0], 3)


def test_replay_is_repeatable(synthetic_clip: Path):
    first = [p.timestamp for p in _read_all(synthetic_clip)]
    second = [p.timestamp for p in _read_all(synthetic_clip)]
    assert first == second


def test_frames_arrive_in_order(synthetic_clip: Path):
    # Frame i was written with gray level 8*i; MJPG is lossy, so allow some error.
    levels = [float(p.image.mean()) for p in _read_all(synthetic_clip)]
    for i, level in enumerate(levels):
        assert level == pytest.approx(8 * i, abs=3)


def test_returns_none_after_end(synthetic_clip: Path):
    with VideoFileSource(synthetic_clip) as source:
        while source.read() is not None:
            pass
        assert source.read() is None


def test_fps_override(synthetic_clip: Path):
    with VideoFileSource(synthetic_clip, fps_override=10.0) as source:
        source.read()
        packet = source.read()
    assert packet.timestamp == pytest.approx(0.1)


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        VideoFileSource(tmp_path / "missing.mp4").open()


def test_open_source_picks_type(synthetic_clip: Path):
    assert isinstance(
        open_source(load_config(overrides={"capture": {"source": str(synthetic_clip)}}).capture),
        VideoFileSource,
    )
    camera = open_source(load_config(overrides={"capture": {"source": "1"}}).capture)
    assert type(camera).__name__ == "CameraSource"
    assert camera.index == 1
