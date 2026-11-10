"""Replay: synthetic feature CSV (format v1) -> DriverMonitor -> golden event sequence."""

import pytest
from builders import Seq, events, run

from drowsiness.feature_log import FeatureCsvWriter, read_feature_csv
from drowsiness.types import PerceptionOutput

GOLDEN = [
    # (reason, state, approx time s)
    ("calibrated", "NORMAL", 5.0),
    ("microsleep", "DROWSY_WARNING", 9.0),
    ("cleared", "NORMAL", 10.1),
    ("microsleep", "DROWSY_WARNING", 12.6),
    ("sleep", "DROWSY_ALARM", 14.1),
    ("cleared", "NORMAL", 15.4),
    ("long_distraction", "DISTRACTED", 19.4),
    ("cleared", "NORMAL", 20.9),
    ("no_face", "NO_FACE", 23.9),
    ("cleared", "NORMAL", 25.7),  # face confirmed for max_gap_s (0.3 s), then clear_s
]


def scenario(**seq_args):
    return (
        Seq(**seq_args).calibrate().open(2).closed(1.6).open(2).closed(3.3).open(2)
        .pose(3.5, yaw=40).open(2).no_face(2.5).open(2).frames
    )  # fmt: skip


def replay_csv(frames, path):
    with FeatureCsvWriter(path) as writer:
        for i, f in enumerate(frames):
            writer.write(PerceptionOutput(f, None, 0.0), frame_index=i)
    return read_feature_csv(path)


@pytest.mark.parametrize(
    "seq_args",
    [{"fps": 30}, {"fps": 20}, {"fps": 25, "jitter": 0.3}, {"fps": 30, "drop": 0.1, "seed": 3}],
    ids=["30fps", "20fps", "25fps-jitter", "30fps-drops"],
)
def test_replay_matches_golden_events(tmp_path, seq_args):
    outs = run(replay_csv(scenario(**seq_args), tmp_path / "features.csv"))
    got = [(e.reason, e.state.value, e.timestamp) for e in events(outs)]
    assert [g[:2] for g in got] == [g[:2] for g in GOLDEN]
    for (_, _, t), (_, _, expected) in zip(got, GOLDEN, strict=True):
        assert t == pytest.approx(expected, abs=0.2)


def test_replay_is_deterministic(tmp_path):
    frames = replay_csv(scenario(fps=30, drop=0.1, seed=7), tmp_path / "f.csv")
    assert run(frames) == run(frames)
