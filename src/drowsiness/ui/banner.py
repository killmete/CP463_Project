"""Driver-facing banner: what the driver sees, drawn whenever there is a window.

Drawing only; the AlertDecision says what to show. Draws in place on a BGR uint8
image and never changes its shape or dtype. Text is ASCII (Hershey fonts).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from drowsiness.alerts import AlertDecision
from drowsiness.types import AlertLevel

_FONT = cv2.FONT_HERSHEY_DUPLEX
_SMALL = cv2.FONT_HERSHEY_SIMPLEX
_RED, _DARK_RED, _AMBER, _GREY, _GREEN = (
    (0, 0, 230),
    (0, 0, 90),
    (0, 170, 255),
    (150, 150, 150),
    (0, 200, 0),
)  # BGR
_BLACK, _WHITE = (0, 0, 0), (255, 255, 255)
_BORDER_PX = 8
_BANNER_RATIO = 0.12
_TOAST_Y = 110  # below the status bar and the debug logic panel (top right)


def banner_height(image: np.ndarray) -> int:
    """Height of the bottom banner; debug panels stay above it."""
    return max(28, round(image.shape[0] * _BANNER_RATIO))


def _fit_scale(text: str, font: int, max_w: int, max_h: int, thickness: int) -> float:
    (w, h), _ = cv2.getTextSize(text, font, 1.0, thickness)
    return max(0.3, min(max_w / max(w, 1), max_h / max(h, 1), 1.4))


def _flash_on(now: float, hz: float) -> bool:
    return math.floor(now * hz * 2) % 2 == 0  # on for the first half of each period


def draw_banner(
    image: np.ndarray,
    decision: AlertDecision,
    now: float,
    flash_hz: float,
    sound_off: bool = False,
) -> None:
    """Bottom banner (+ frame border for warnings/alarms), INFO toast, SOUND OFF tag."""
    height, width = image.shape[:2]
    level = decision.level
    if decision.message is None:
        cv2.putText(image, "Monitoring", (width - 110, height - 10), _SMALL, 0.5, _GREEN, 1,
                    cv2.LINE_AA)  # fmt: skip
    else:
        if level == AlertLevel.ALARM:
            on = not decision.flash or _flash_on(now, flash_hz)
            fill, text_color = (_RED, _WHITE) if on else (_DARK_RED, _WHITE)
        elif level == AlertLevel.WARNING:
            fill, text_color = _AMBER, _BLACK
        else:
            fill, text_color = _GREY, _BLACK
        if level in (AlertLevel.WARNING, AlertLevel.ALARM):
            cv2.rectangle(image, (0, 0), (width - 1, height - 1), fill, _BORDER_PX * 2)
        bar = banner_height(image)
        cv2.rectangle(image, (0, height - bar), (width, height), fill, thickness=-1)
        thick = 2 if level == AlertLevel.ALARM else 1
        scale = _fit_scale(decision.message, _FONT, width - 24, int(bar * 0.55), thick)
        (w, h), _ = cv2.getTextSize(decision.message, _FONT, scale, thick)
        org = ((width - w) // 2, height - (bar - h) // 2)
        cv2.putText(image, decision.message, org, _FONT, scale, text_color, thick, cv2.LINE_AA)
    if sound_off:  # the driver must know that alerts are visual only
        bar = banner_height(image) if decision.message is not None else 28
        y = height - bar - 10
        cv2.rectangle(image, (width - 120, y - 20), (width - 12, y + 6), _BLACK, thickness=-1)
        cv2.putText(image, "SOUND OFF", (width - 112, y), _SMALL, 0.55, _AMBER, 1, cv2.LINE_AA)
    if decision.toast:
        (w, h), _ = cv2.getTextSize(decision.toast, _SMALL, 0.6, 1)
        x, y = max((width - w) // 2, 0), _TOAST_Y
        cv2.rectangle(image, (x - 8, y - 8), (x + w + 8, y + h + 8), _GREY, thickness=-1)
        cv2.putText(image, decision.toast, (x, y + h), _SMALL, 0.6, _BLACK, 1, cv2.LINE_AA)
