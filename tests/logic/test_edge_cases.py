"""Regression tests for gaps, blinks and dropouts around the rules (milestone 3 review)."""

from builders import Seq, events, reasons_seen, run

from drowsiness.types import DriverState


def test_blink_while_looking_away_does_not_reset_the_distraction_timer():
    seq = Seq().calibrate().open(1).pose(2.0, yaw=40)
    seq.pose(0.15, yaw=40, ear_left=0.03, ear_right=0.03).pose(1.5, yaw=40)
    assert "long_distraction" in reasons_seen(run(seq.frames))


def test_blink_then_face_lost_is_no_face_not_sleep():
    outs = run(Seq().calibrate().open(1).closed(0.2).no_face(3.5).frames)
    assert not reasons_seen(outs) & {"sleep", "microsleep"}
    assert outs[-1].state == DriverState.NO_FACE


def test_carried_closure_clears_once_open_eyes_are_seen():
    seq = Seq().calibrate().open(1).closed(1.0).no_face(3.0)
    back = seq.t
    outs = run(seq.open(3).frames)
    assert "sleep" in reasons_seen(outs)  # closure carried through the face loss
    # face confirmation (0.3 s) + smoothing + clear_s (1.0 s)
    final = [o.state for o, f in zip(outs, seq.frames, strict=True) if f.timestamp >= back + 1.6]
    assert final and set(final) == {DriverState.NORMAL}


def test_carry_ends_after_max_carry_s():
    outs = run(Seq().calibrate().open(1).closed(1.0).no_face(32).frames)
    assert "sleep" in reasons_seen(outs)
    assert outs[-1].reasons == ("no_face",) and outs[-1].state == DriverState.NO_FACE


def test_one_spurious_face_frame_does_not_reset_no_face():
    outs = run(Seq().calibrate().open(1).no_face(1.5).open(1 / 30).no_face(1.0).frames)
    assert "no_face" in reasons_seen(outs)


def test_yawn_interrupted_by_a_dropout_counts_once():
    outs = run(Seq().calibrate().open(1).yawn(1.2).no_face(0.5).yawn(1.2).open(1).frames)
    assert max(o.metrics.get("yawn_count", 0) for o in outs) == 1


def test_short_distraction_clears_and_restarts_after_driver_looks_back():
    seq = Seq().calibrate().open(1)
    for _ in range(4):
        seq.pose(2.6, yaw=40).open(0.4)
    seq.open(3)
    first = len(seq.frames)
    seq.pose(2.6, yaw=40).open(2)
    outs = run(seq.frames)
    assert [e.reason for e in events(outs[:first])][-2:] == ["short_distraction", "cleared"]
    assert "short_distraction" not in reasons_seen(outs[first:])  # window restarted


def _flaky(seq: Seq, seconds: float, miss_every: int) -> Seq:
    """Face present, but the detector misses one frame in `miss_every`."""
    for i in range(round(seconds * seq.fps)):
        seq.hold(1 / seq.fps, face=bool(i % miss_every))
    return seq


def test_unreliable_detection_of_a_visible_face_is_not_no_face():
    seq = _flaky(Seq(fps=20).calibrate().open(1), 10, miss_every=5)  # 80 % detected
    outs = run(seq.frames)
    assert "no_face" not in reasons_seen(outs)
    assert max(o.metrics["no_face_s"] for o in outs) == 0.0


def test_face_returning_with_unreliable_detection_ends_no_face():
    seq = Seq(fps=20).calibrate().open(1).no_face(3)
    outs = run(_flaky(seq, 4, miss_every=5).frames)
    assert "no_face" in reasons_seen(outs)
    assert outs[-1].state == DriverState.NORMAL
