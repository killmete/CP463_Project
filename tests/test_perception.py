"""Perception pipeline with a fake detector (no MediaPipe, no model file)."""

import sys

import cv2
import numpy as np
import pytest

from drowsiness.config import load_config
from drowsiness.perception import (
    LandmarkResult,
    ModelNotFoundError,
    Perception,
    PerceptionSetupError,
    build_perception,
)
from drowsiness.perception import features as features_module
from drowsiness.perception.landmarks import NUM_LANDMARKS, RIGHT_EYE
from drowsiness.types import FrameFeatures, FramePacket


class FakeDetector:
    def __init__(self, results):
        self._results = iter(results)
        self.calls: list[tuple[np.ndarray, int]] = []
        self.closed = False

    def detect(self, rgb, timestamp_ms):
        self.calls.append((rgb, timestamp_ms))
        return next(self._results)

    def close(self):
        self.closed = True


def packet(t=0.0, index=0, width=640, height=480, value=60):
    image = np.full((height, width, 3), value, np.uint8)
    image[:, : width // 2] = 20  # some contrast for CLAHE
    return FramePacket(image, t, index, 0.0)


def preprocess(clahe=False):
    return load_config(overrides={"preprocess": {"clahe": clahe}}).preprocess


def normalised_face():
    """A plausible face in normalised coordinates: open eyes, closed mouth, frontal."""
    lm = np.zeros((NUM_LANDMARKS, 3), np.float32)
    rng = np.random.default_rng(0)
    lm[:, :2] = 0.5 + rng.uniform(-0.1, 0.1, (NUM_LANDMARKS, 2))
    return lm


def test_no_face_gives_all_none():
    with Perception(FakeDetector([None]), preprocess()) as perception:
        out = perception.process(packet(t=1.25))
    assert out.features == FrameFeatures(timestamp=1.25, face_found=False)
    assert out.landmarks is None
    assert out.perception_ms >= 0


def test_face_fills_every_m2_feature_and_passes_blendshapes():
    blendshapes = {"eyeBlinkLeft": 0.1, "jawOpen": 0.7}
    detector = FakeDetector([LandmarkResult(normalised_face(), blendshapes)])
    out = Perception(detector, preprocess()).process(packet(t=2.0))
    f = out.features
    assert f.face_found and f.timestamp == 2.0
    for value in (f.ear_left, f.ear_right, f.mar, f.pitch, f.yaw, f.roll):
        assert isinstance(value, float)
    assert f.gaze is None  # later milestone
    assert f.blendshapes == blendshapes


def test_landmarks_are_converted_to_pixels_on_non_square_frames():
    lm = np.zeros((NUM_LANDMARKS, 3), np.float32)
    p1, p2, p3, p4, p5, p6 = RIGHT_EYE
    # Normalised: corners 0.1 apart in x, lids 0.1 apart in y. On a 400x100 frame this
    # is 40 px wide and 10 px tall, so EAR must be 0.25 (not 1.0 from raw coordinates).
    lm[[p1, p4], :2] = [(0.40, 0.5), (0.50, 0.5)]
    lm[[p2, p3], :2] = [(0.425, 0.45), (0.475, 0.45)]
    lm[[p6, p5], :2] = [(0.425, 0.55), (0.475, 0.55)]
    detector = FakeDetector([LandmarkResult(lm)])
    out = Perception(detector, preprocess()).process(packet(width=400, height=100))
    assert out.features.ear_right == pytest.approx(0.25)
    assert out.landmarks[p4, 0] == pytest.approx(200.0)
    assert out.landmarks[p4, 1] == pytest.approx(50.0)


def test_detector_timestamps_strictly_increase():
    times = [0.0, 0.0, 0.0331, 0.0334, 0.0334, 1.0, 0.5]  # repeats, sub-ms, backwards
    detector = FakeDetector([None] * len(times))
    perception = Perception(detector, preprocess())
    outs = [perception.process(packet(t)) for t in times]
    sent = [ts for _, ts in detector.calls]
    assert all(b > a for a, b in zip(sent, sent[1:], strict=False))
    assert sent[0] == 0 and sent[5] == 1000
    # FrameFeatures keeps the original timestamps for logic.
    assert [o.features.timestamp for o in outs] == times


def test_detector_receives_rgb():
    image_packet = packet()
    image_packet.image[0, 0] = (255, 0, 0)  # BGR blue
    detector = FakeDetector([None])
    Perception(detector, preprocess()).process(image_packet)
    rgb, _ = detector.calls[0]
    assert tuple(rgb[0, 0]) == (0, 0, 255)


def test_clahe_never_modifies_the_displayed_frame():
    frame = packet()
    original = frame.image.copy()
    detector = FakeDetector([None])
    Perception(detector, preprocess(clahe=True)).process(frame)
    np.testing.assert_array_equal(frame.image, original)
    rgb, _ = detector.calls[0]
    assert not np.array_equal(rgb, original[:, :, ::-1])  # the detector saw the enhanced copy


def test_close_closes_detector():
    detector = FakeDetector([])
    with Perception(detector, preprocess()):
        pass
    assert detector.closed


def test_missing_model_raises_with_fetch_hint(tmp_path):
    cfg = load_config(overrides={"perception": {"model_path": str(tmp_path / "none.task")}})
    with pytest.raises(ModelNotFoundError, match="fetch_model"):
        build_perception(cfg)


def test_missing_mediapipe_gives_install_hint(tmp_path, monkeypatch):
    model = tmp_path / "m.task"
    model.write_bytes(b"x")
    monkeypatch.setitem(sys.modules, "drowsiness.perception.mediapipe_detector", None)
    cfg = load_config(overrides={"perception": {"model_path": str(model)}})
    with pytest.raises(PerceptionSetupError, match="requirements-mediapipe.txt"):
        build_perception(cfg)


def test_pose_failure_keeps_eye_and_mouth_features(monkeypatch):
    def broken_solvepnp(*args, **kwargs):
        raise cv2.error("degenerate")

    monkeypatch.setattr(features_module.cv2, "solvePnP", broken_solvepnp)
    detector = FakeDetector([LandmarkResult(normalised_face())])
    f = Perception(detector, preprocess()).process(packet()).features
    assert f.face_found and f.ear_left is not None and f.mar is not None
    assert (f.pitch, f.yaw, f.roll) == (None, None, None)
