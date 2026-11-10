"""Table-driven boundary tests: each rule just below and just above its threshold."""

import pytest
from builders import EAR_CLOSED, Seq, config, events, reasons_seen, run

from drowsiness.types import AlertLevel, DriverState


@pytest.mark.parametrize(
    "seconds, expected",
    [(1.4, False), (1.6, True)],
)
def test_microsleep_boundary(seconds, expected):
    outs = run(Seq().calibrate().open(2).closed(seconds).open(3).frames)
    assert ("microsleep" in reasons_seen(outs)) is expected
    assert "sleep" not in reasons_seen(outs)


@pytest.mark.parametrize("seconds, expected", [(2.9, False), (3.15, True)])
def test_sleep_boundary(seconds, expected):
    outs = run(Seq().calibrate().open(2).closed(seconds).open(3).frames)
    assert ("sleep" in reasons_seen(outs)) is expected
    alarms = [e for e in events(outs) if e.level == AlertLevel.ALARM]
    assert bool(alarms) is expected


def test_alert_is_raised_within_half_a_second_of_the_rule():
    seq = Seq().calibrate().open(2)
    closure_start = seq.t
    outs = run(seq.closed(2.0).open(2).frames)
    (event,) = [e for e in events(outs) if e.reason == "microsleep"]
    assert event.state == DriverState.DROWSY_WARNING
    assert 0 <= event.timestamp - (closure_start + 1.5) <= 0.5


def blinking(seq, seconds, fraction, blink_s=0.5):
    """Short blinks (< microsleep) so that `fraction` of the time the eyes are closed."""
    period = blink_s / fraction
    for _ in range(round(seconds / period)):
        seq.closed(blink_s).open(period - blink_s)
    return seq


@pytest.mark.parametrize(
    "fraction, warning, alarm",
    [(0.12, False, False), (0.18, True, False), (0.22, True, False), (0.29, True, True)],
)
def test_perclos_thresholds(fraction, warning, alarm):
    outs = run(blinking(Seq().calibrate(), 70, fraction).frames)
    seen = reasons_seen(outs)
    assert ("perclos_warning" in seen) is warning
    assert ("perclos_alarm" in seen) is alarm
    assert "microsleep" not in seen


def test_perclos_waits_for_enough_valid_data():
    outs = run(blinking(Seq().calibrate(), 25, 0.5).frames)  # < 30 s of the 60 s window
    assert not {"perclos_warning", "perclos_alarm"} & reasons_seen(outs)
    assert "perclos" not in outs[-1].metrics


@pytest.mark.parametrize("seconds, counted", [(0.9, False), (1.1, True)])
def test_yawn_duration_boundary(seconds, counted):
    outs = run(Seq().calibrate().yawn(seconds).open(1).frames)
    assert outs[-1].metrics["yawn_count"] == (1.0 if counted else 0.0)


@pytest.mark.parametrize("yawns, expected", [(2, False), (3, True)])
def test_yawning_needs_three_yawns(yawns, expected):
    seq = Seq().calibrate()
    for _ in range(yawns):
        seq.yawn(1.5).open(20)
    outs = run(seq.frames)
    assert ("yawning" in reasons_seen(outs)) is expected


def test_yawns_outside_the_window_do_not_count():
    cfg = config(yawn={"count_window_s": 30.0})
    seq = Seq().calibrate()
    for _ in range(3):
        seq.yawn(1.5).open(20)  # 3 yawns ~21.5 s apart: never 3 within 30 s
    assert "yawning" not in reasons_seen(run(seq.frames, cfg))


@pytest.mark.parametrize(
    "drop_deg, seconds, expected",
    [(14, 1.5, False), (17, 1.5, True), (17, 0.9, False), (17, 1.2, True)],
)
def test_head_nod_boundaries(drop_deg, seconds, expected):
    # Eyes closed while the head drops: a nod. Closure stays short of microsleep.
    outs = run(Seq().calibrate().pose(seconds, pitch=-drop_deg, ear_left=EAR_CLOSED,
                                      ear_right=EAR_CLOSED).open(2).frames)  # fmt: skip
    assert ("head_nod" in reasons_seen(outs)) is expected


def test_head_down_with_open_eyes_is_not_a_nod():
    outs = run(Seq().calibrate().pose(2.0, pitch=-20).open(2).frames)
    assert "head_nod" not in reasons_seen(outs)


def test_looking_down_at_a_phone_is_a_distraction():
    outs = run(Seq().calibrate().pose(3.5, pitch=-30).open(2).frames)
    seen = reasons_seen(outs)
    assert "long_distraction" in seen and "head_nod" not in seen


@pytest.mark.parametrize("seconds, expected", [(2.9, False), (3.1, True)])
@pytest.mark.parametrize("pose", [{"yaw": 40}, {"yaw": -40}, {"pitch": 30}])
def test_long_distraction_boundary(seconds, expected, pose):
    outs = run(Seq().calibrate().pose(seconds, **pose).open(3).frames)
    assert ("long_distraction" in reasons_seen(outs)) is expected


def test_neutral_pose_offsets_are_removed():
    # 28 deg absolute yaw is only 25 deg from the calibrated neutral (3 deg): not off-road.
    outs = run(Seq().calibrate().pose(4, yaw=25).frames)
    assert "long_distraction" not in reasons_seen(outs)


@pytest.mark.parametrize("glances, expected", [(4, False), (6, True)])
def test_short_distraction_total_in_window(glances, expected):
    # 2 s glances (each below the 3 s long rule), 2 s apart: 8 s vs 12 s within 30 s.
    seq = Seq().calibrate()
    for _ in range(glances):
        seq.pose(2.0, yaw=40).open(2.0)
    outs = run(seq.frames)
    seen = reasons_seen(outs)
    assert ("short_distraction" in seen) is expected
    assert "long_distraction" not in seen


@pytest.mark.parametrize("seconds, expected", [(1.9, False), (2.1, True)])
def test_no_face_boundary(seconds, expected):
    outs = run(Seq().calibrate().no_face(seconds).open(2).frames)
    assert ("no_face" in reasons_seen(outs)) is expected


def test_no_face_is_reported_during_calibration():
    outs = run(Seq().no_face(3).frames)
    assert outs[-1].state == DriverState.NO_FACE


def test_closure_then_face_lost_still_reaches_the_sleep_alarm():
    # Head drops out of view while the eyes are closed: must not reset to NORMAL.
    outs = run(Seq().calibrate().open(2).closed(1.0).no_face(3.0).frames)
    assert "sleep" in reasons_seen(outs)
    assert outs[-1].state == DriverState.DROWSY_ALARM
    assert DriverState.NORMAL not in {o.state for o in outs[-60:]}


def test_short_gap_does_not_reset_the_closure_timer():
    # 0.25 s of missing landmarks inside a 2 s closure (EAR None, face still found).
    seq = Seq().calibrate().open(2).closed(1.0).hold(0.25, ear_left=None, ear_right=None)
    outs = run(seq.closed(0.75).open(2).frames)
    assert "microsleep" in reasons_seen(outs)


def test_open_eyes_are_not_carried_through_long_gaps():
    seq = Seq().calibrate().open(2).hold(1.0, ear_left=None, ear_right=None)
    outs = run(seq.open(1).frames)
    assert outs[-1].state == DriverState.NORMAL
    assert not reasons_seen(outs) - {"no_face"}


def test_one_eye_occluded_uses_the_other():
    outs = run(Seq().calibrate().closed(2.0, ear_left=None).open(2).frames)
    assert "microsleep" in reasons_seen(outs)


def test_single_frame_spikes_are_smoothed_out():
    seq = Seq().calibrate()
    for _ in range(60):  # one closed frame every 0.5 s for 30 s
        seq.closed(1 / 30).open(0.5 - 1 / 30)
    outs = run(seq.frames)
    assert outs[-1].metrics.get("perclos", 0.0) < 0.01
