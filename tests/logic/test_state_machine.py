import pytest
from builders import Seq, config, events, run

from drowsiness.logic import DriverMonitor
from drowsiness.logic.state_machine import StateMachine
from drowsiness.types import AlertLevel, DriverState, FrameFeatures


def test_calibration_then_normal():
    outs = run(Seq().calibrate().open(1).frames)
    assert outs[0].state == DriverState.CALIBRATING
    (event,) = events(outs)
    assert event.reason == "calibrated" and event.state == DriverState.NORMAL
    assert event.level == AlertLevel.INFO
    assert event.details["neutral_pitch"] == pytest.approx(-6.0)
    assert outs[-1].state == DriverState.NORMAL


def test_events_only_on_transitions():
    outs = run(Seq().calibrate().closed(2.5).frames)
    assert [e.reason for e in events(outs)] == ["calibrated", "microsleep"]
    assert sum(1 for o in outs if o.events) == 2


def test_escalation_is_immediate_and_carries_details():
    outs = run(Seq().calibrate().closed(3.5).frames)
    microsleep, sleep = events(outs)[1:]
    assert (microsleep.reason, sleep.reason) == ("microsleep", "sleep")
    assert sleep.level == AlertLevel.ALARM and sleep.details["previous_state"] == "DROWSY_WARNING"
    assert sleep.details["closed_s"] > 3.0
    assert sleep.details["reasons"] == "sleep,microsleep"


@pytest.mark.parametrize("open_s, cleared", [(0.9, False), (1.1, True)])
def test_hysteresis_before_clearing(open_s, cleared):
    outs = run(Seq().calibrate().closed(1.8).open(open_s).frames)
    assert (outs[-1].state == DriverState.NORMAL) is cleared
    assert ("cleared" in [e.reason for e in events(outs)]) is cleared


def test_flickering_reason_does_not_flood_events():
    machine = StateMachine(clear_s=1.0)
    machine.state = DriverState.NORMAL
    emitted = []
    for i in range(300):  # reason on 0.5 s, off 0.5 s, for 10 s at 30 fps
        t = i / 30
        reasons = ["perclos_warning"] if (t % 1.0) < 0.5 else []
        emitted += machine.update(t, reasons, DriverState.NORMAL, {})
    assert [e.reason for e in emitted] == ["perclos_warning"]
    assert machine.state == DriverState.DROWSY_WARNING


def test_priority_alarm_beats_no_face_beats_distraction():
    outs = run(Seq().calibrate().pose(3.5, yaw=45).no_face(2.5).frames)
    states = [e.state for e in events(outs)]
    assert states[-2:] == [DriverState.DISTRACTED, DriverState.NO_FACE]
    outs = run(Seq().calibrate().closed(1.0).no_face(3.0).frames)
    assert outs[-1].state == DriverState.DROWSY_ALARM


def test_calibration_retries_then_falls_back():
    cfg = config(calibration={"max_attempts": 2})
    outs = run(Seq().closed(5.5).closed(5.5).open(2).frames, cfg)  # eyes closed twice
    (event,) = [e for e in events(outs) if e.reason.startswith("calibrat")]
    assert event.reason == "calibration_fallback" and event.details["attempts"] == 2.0
    assert event.level == AlertLevel.INFO
    assert outs[-1].state == DriverState.NORMAL


def test_calibration_retry_can_succeed():
    outs = run(Seq().no_face(5.5).calibrate().open(1).frames)
    reasons = [e.reason for e in events(outs)]
    assert "calibrated" in reasons and "calibration_fallback" not in reasons


def test_fallback_uses_absolute_ear_threshold():
    cfg = config(calibration={"max_attempts": 1})
    outs = run(Seq().no_face(5.5).open(1).closed(1.8).frames, cfg)
    assert outs[-1].state == DriverState.DROWSY_WARNING


def test_recalibrate_keeps_detecting_until_the_new_baseline_is_ready():
    monitor = DriverMonitor(config())
    seq = Seq().calibrate().open(1)
    for f in seq.frames:
        monitor.update(f)
    monitor.recalibrate()
    outs = [monitor.update(f) for f in Seq(start=seq.t).open(5.5).frames]
    reasons = [(e.reason, e.state, e.level) for e in events(outs)]
    assert reasons == [
        ("recalibrating", DriverState.NORMAL, AlertLevel.INFO),
        ("calibrated", DriverState.NORMAL, AlertLevel.INFO),
    ]
    assert all(o.state == DriverState.NORMAL for o in outs)
    assert outs[0].metrics["calibrating"] == 1.0 and "calibrating" not in outs[-1].metrics


def test_recalibrate_during_an_alarm_does_not_silence_it():
    monitor = DriverMonitor(config(calibration={"max_attempts": 1}))
    seq = Seq().calibrate().closed(3.5)
    for f in seq.frames:
        out = monitor.update(f)
    assert out.state == DriverState.DROWSY_ALARM
    monitor.recalibrate()
    # Eyes stay closed: detection continues, and the failed recalibration keeps the old baseline.
    outs = [monitor.update(f) for f in Seq(start=seq.t).closed(5.5).frames]
    assert all(o.state == DriverState.DROWSY_ALARM for o in outs)
    assert [e.reason for e in events(outs)] == ["recalibrating", "recalibration_failed"]
    assert monitor.calibration is not None and not monitor.calibration.fallback


def test_calibration_finishing_during_no_face_keeps_no_face():
    cfg = config(calibration={"min_valid_frames_ratio": 0.5})
    outs = run(Seq().calibrate(2.8).no_face(4).open(3).frames, cfg)
    (cal,) = [e for e in events(outs) if e.reason == "calibrated"]
    assert cal.state == DriverState.NO_FACE and cal.level == AlertLevel.INFO
    assert [e.reason for e in events(outs)] == ["no_face", "calibrated", "cleared"]
    assert outs[-1].state == DriverState.NORMAL


def test_fallback_keeps_a_valid_neutral_pose():
    # Eyes never open enough to calibrate EAR, but the head pose was measured fine.
    outs = run(Seq().closed(16).frames, config(calibration={"max_attempts": 3}))
    (fallback,) = [e for e in events(outs) if e.reason == "calibration_fallback"]
    assert fallback.details["neutral_pitch"] == pytest.approx(-6.0)
    assert fallback.details["neutral_yaw"] == pytest.approx(3.0)


def test_timestamps_must_increase():
    monitor = DriverMonitor(config())
    monitor.update(FrameFeatures(timestamp=1.0, face_found=False))
    with pytest.raises(ValueError):
        monitor.update(FrameFeatures(timestamp=1.0, face_found=False))


def test_logic_is_pure():
    import ast
    from pathlib import Path

    import drowsiness.logic as logic_pkg

    banned = {"cv2", "time", "os", "io", "pathlib", "logging", "random", "numpy"}
    for path in Path(logic_pkg.__file__).parent.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                names = {(node.module or "").split(".")[0]}
            else:
                continue
            assert not names & banned, f"{path.name} imports {names & banned}"
