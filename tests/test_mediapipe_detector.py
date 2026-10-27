"""Real MediaPipe Face Landmarker tests (need the model file; see the `model_path` fixture).

The face speed check needs frames with a face, and no face images are kept in the
repo. Point DROWSINESS_PERF_SOURCE at an image or clip of your own, or set
DROWSINESS_CAMERA_TESTS=1 to use the webcam. Frames stay in memory only.
"""

import os
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from drowsiness.config import load_config
from drowsiness.perception import build_perception
from drowsiness.types import FrameFeatures, FramePacket

pytestmark = pytest.mark.model

BUDGET_MS = 25.0  # vision-engineer budget at 640x480 on a laptop CPU
WARMUP, TIMED = 10, 100


def build(model_path: Path, **perception):
    overrides = {"perception": {"model_path": str(model_path), **perception}}
    return build_perception(load_config(overrides=overrides))


def test_blank_frame_gives_no_face(model_path):
    with build(model_path) as perception:
        for i in range(3):
            image = np.zeros((480, 640, 3), np.uint8)
            out = perception.process(FramePacket(image, i / 30, i, 0.0))
            assert out.features == FrameFeatures(timestamp=i / 30, face_found=False)
            assert out.landmarks is None


def _face_frames(n: int) -> list[np.ndarray] | None:
    source = os.environ.get("DROWSINESS_PERF_SOURCE")
    frames: list[np.ndarray] = []
    if source:
        image = cv2.imread(source)
        if image is not None:
            return [image] * n
        cap = cv2.VideoCapture(source)
    elif os.environ.get("DROWSINESS_CAMERA_TESTS"):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    else:
        return None
    try:
        while len(frames) < n:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames or None


def _resize(image: np.ndarray) -> np.ndarray:
    return cv2.resize(image, (640, 480)) if image.shape[:2] != (480, 640) else image


@pytest.fixture(scope="module")
def face_frames() -> list[np.ndarray]:
    """Captured once, so every variant below sees exactly the same frames."""
    frames = _face_frames(WARMUP + TIMED)
    if frames is None:
        pytest.skip("set DROWSINESS_PERF_SOURCE=<image|clip> or DROWSINESS_CAMERA_TESTS=1")
    return [_resize(f) for f in frames]


def _run(model_path: Path, frames: list[np.ndarray], clahe: bool):
    overrides = {"perception": {"model_path": str(model_path)}, "preprocess": {"clahe": clahe}}
    times, outs = [], []
    with build_perception(load_config(overrides=overrides)) as perception:
        for i in range(WARMUP + TIMED):
            packet = FramePacket(frames[i % len(frames)], i / 30, i, 0.0)
            start = time.perf_counter()
            out = perception.process(packet)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if i >= WARMUP:
                times.append(elapsed_ms)
                outs.append(out)
    return times, outs


@pytest.mark.perf
@pytest.mark.parametrize("clahe", [False, True], ids=["plain", "clahe"])
def test_perception_with_face_is_within_budget(model_path, face_frames, clahe):
    times, outs = _run(model_path, face_frames, clahe)
    found = sum(o.features.face_found for o in outs)
    p50, p95 = np.percentile(times, [50, 95])
    name = "clahe" if clahe else "plain"
    print(f"\nperception ({name}) p50 {p50:.1f} ms, p95 {p95:.1f} ms, face {found}/{TIMED}")
    assert found >= 0.5 * TIMED, f"face found in only {found}/{TIMED} frames: not a timing test"
    assert p95 <= BUDGET_MS


def test_real_face_landmark_layout(model_path, face_frames):
    """Not circular: checks MediaPipe's actual index layout, which the formulas assume."""
    _, outs = _run(model_path, face_frames, clahe=False)
    faces = [o for o in outs if o.features.face_found]
    assert len(faces) >= 0.9 * TIMED, f"face found in only {len(faces)}/{TIMED} frames"
    for out in faces:
        lm = out.landmarks
        # Driver's left eye / mouth corner has the larger image x (unmirrored frame).
        assert lm[263, 0] > lm[33, 0] and lm[291, 0] > lm[61, 0]
        assert lm[1, 1] < lm[152, 1]  # nose tip above chin
        f = out.features
        assert 0.0 <= f.ear_left < 0.6 and 0.0 <= f.ear_right < 0.6  # blinks reach ~0
        assert 0.0 <= f.mar < 1.5
        assert abs(f.pitch) < 60 and abs(f.yaw) < 60 and abs(f.roll) < 60


@pytest.mark.perf
def test_no_face_frames_are_fast(model_path):
    times = []
    with build(model_path) as perception:
        for i in range(WARMUP + TIMED):
            image = np.full((480, 640, 3), 90, np.uint8)
            start = time.perf_counter()
            perception.process(FramePacket(image, i / 30, i, 0.0))
            times.append((time.perf_counter() - start) * 1000.0)
    assert np.percentile(times[WARMUP:], 95) <= BUDGET_MS
