"""Driver-facing banner: drawn for every state, fits the frame, flashes for ALARM."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from drowsiness.alerts import AlertDecision, SoundCue
from drowsiness.alerts.messages import ESCALATED, MESSAGES
from drowsiness.types import AlertLevel, DriverState
from drowsiness.ui import banner_height, draw_banner


def blank(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), np.uint8)


def make(level=AlertLevel.WARNING, message="Look at the road", flash=False, toast=None):
    state = DriverState.DROWSY_ALARM if level == AlertLevel.ALARM else DriverState.DISTRACTED
    return AlertDecision(0.0, state, level, None, message, flash, SoundCue.NONE, toast=toast)


@pytest.mark.parametrize("message", [*MESSAGES.values(), ESCALATED])
@pytest.mark.parametrize("level", list(AlertLevel))
def test_every_message_draws_inside_the_banner(level, message):
    image = blank()
    draw_banner(image, make(level, message), 0.0, 2.0)
    assert image.shape == (480, 640, 3) and image.dtype == np.uint8
    bar = image[480 - banner_height(image) :]
    assert bar.any()
    # Text is centred and leaves a margin on both sides (it was scaled to fit).
    text = cv2.absdiff(bar, np.full_like(bar, bar[2, 2]))
    cols = np.flatnonzero(text[:, 10:-10].any(axis=(0, 2)))
    assert cols.size and cols.min() > 0 and cols.max() < 640 - 21


def test_normal_shows_only_a_small_monitoring_tag():
    image = blank()
    draw_banner(image, make(AlertLevel.INFO, None), 0.0, 2.0)
    ys, xs = np.nonzero(image.any(axis=2))
    assert ys.min() > 440 and xs.min() > 500  # bottom-right corner only


def test_warning_and_alarm_draw_a_border():
    image = blank()
    draw_banner(image, make(AlertLevel.WARNING), 0.0, 2.0)
    assert image[0, 320].any() and image[240, 0].any() and image[240, 639].any()
    info = blank()
    draw_banner(info, make(AlertLevel.INFO, "Calibrating - look ahead"), 0.0, 2.0)
    assert not info[0, 320].any()


def test_alarm_flashes_from_the_timestamp():
    on, off = blank(), blank()
    draw_banner(on, make(AlertLevel.ALARM, "WAKE UP", flash=True), 0.0, 2.0)
    draw_banner(off, make(AlertLevel.ALARM, "WAKE UP", flash=True), 0.3, 2.0)  # 2 Hz half period
    assert on[475, 5, 2] > off[475, 5, 2] > 0


def test_escalated_alarm_is_steady():
    a, b = blank(), blank()
    draw_banner(a, make(AlertLevel.ALARM, ESCALATED, flash=False), 0.0, 2.0)
    draw_banner(b, make(AlertLevel.ALARM, ESCALATED, flash=False), 0.3, 2.0)
    assert np.array_equal(a, b)


def test_toast_is_drawn_below_the_debug_panels():
    image = blank()
    draw_banner(image, make(AlertLevel.INFO, None, toast="Calibrated - monitoring"), 0.0, 2.0)
    assert image[100:140].any() and not image[:100].any()


def test_small_frames_still_work():
    image = blank(120, 160)
    draw_banner(image, make(AlertLevel.ALARM, MESSAGES["perclos_alarm"], flash=True), 0.0, 2.0)
    assert image.shape == (120, 160, 3)
