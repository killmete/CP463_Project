import dataclasses

import numpy as np
import pytest

from drowsiness.types import (
    AlertEvent,
    AlertLevel,
    DriverState,
    FrameFeatures,
    FramePacket,
)


def test_contracts_are_frozen():
    packet = FramePacket(np.zeros((2, 2, 3), np.uint8), 0.0, 0, 0.0)
    features = FrameFeatures(timestamp=0.0, face_found=False)
    event = AlertEvent(0.0, DriverState.NO_FACE, "no_face", AlertLevel.WARNING)
    for obj, name in [(packet, "timestamp"), (features, "mar"), (event, "reason")]:
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(obj, name, 1)


def test_frame_features_missing_values_default_to_none():
    features = FrameFeatures(timestamp=1.5, face_found=False)
    assert features.ear_left is None
    assert features.yaw is None
    assert features.gaze is None
    assert features.blendshapes is None


def test_driver_state_values_are_strings():
    assert DriverState.DROWSY_ALARM == "DROWSY_ALARM"
    assert {s.value for s in DriverState} == {s.name for s in DriverState}


def test_alert_levels_are_ordered():
    assert AlertLevel.INFO < AlertLevel.WARNING < AlertLevel.ALARM


def test_alert_event_details_default_is_empty_and_not_shared():
    a = AlertEvent(0.0, DriverState.NORMAL, "x", AlertLevel.INFO)
    b = AlertEvent(0.0, DriverState.NORMAL, "x", AlertLevel.INFO)
    assert a.details == {}
    assert a.details is not b.details
