import numpy as np
import pytest

from drowsiness.config import load_config
from drowsiness.types import FrameFeatures, PerceptionOutput
from drowsiness.ui import draw_perception, draw_status

CFG = load_config().overlay


def blank():
    return np.zeros((480, 640, 3), np.uint8)


def face_output():
    rng = np.random.default_rng(1)
    landmarks = rng.uniform((200, 150, 0), (440, 350, 1), (478, 3)).astype(np.float32)
    features = FrameFeatures(
        timestamp=0.0, face_found=True, ear_left=0.3, ear_right=0.28, mar=0.1,
        pitch=5.0, yaw=-10.0, roll=2.0,
    )  # fmt: skip
    return PerceptionOutput(features, landmarks, 7.0)


@pytest.mark.parametrize(
    "out",
    [face_output(), PerceptionOutput(FrameFeatures(timestamp=0.0, face_found=False), None, 1.0)],
    ids=["face", "no_face"],
)
def test_draw_perception_draws_in_place(out):
    image = blank()
    draw_perception(image, out, CFG)
    assert image.shape == (480, 640, 3) and image.dtype == np.uint8
    assert image.any()


def test_no_face_label_is_red():
    image = blank()
    out = PerceptionOutput(FrameFeatures(timestamp=0.0, face_found=False), None, 1.0)
    draw_perception(image, out, CFG)
    red = (image[:, :, 2] > 200) & (image[:, :, 1] < 50)
    assert red[400:].any()


def test_overlay_can_be_turned_off():
    image = blank()
    off = load_config(overrides={"overlay": {"landmarks": False, "values": False}}).overlay
    draw_perception(image, face_output(), off)
    assert not image.any()


def test_draw_status_turns_red_over_budget():
    ok, slow = blank(), blank()
    draw_status(ok, 30.0, 20.0, 50.0)
    draw_status(slow, 30.0, 80.0, 50.0)
    assert ok[:, :, 1].max() > 150 and ok[:, :, 2].max() == 0
    assert slow[:, :, 2].max() > 200
