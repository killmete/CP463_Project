"""Debug overlay drawn on the displayed frame: status bar, landmarks and live values.

Drawing only; nothing here decides anything. All functions draw in place on a BGR
uint8 image and never change its shape or dtype.
"""

from __future__ import annotations

import cv2
import numpy as np

from drowsiness.config import OverlayConfig
from drowsiness.perception.landmarks import (
    INNER_LIPS_CONTOUR,
    LEFT_EYE_CONTOUR,
    POSE_POINTS,
    RIGHT_EYE_CONTOUR,
)
from drowsiness.types import PerceptionOutput

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_GREEN, _RED, _YELLOW, _CYAN, _WHITE, _BLACK = (
    (0, 200, 0),
    (0, 0, 255),
    (0, 220, 255),
    (255, 220, 0),
    (255, 255, 255),
    (0, 0, 0),
)  # BGR
_NOSE_LINE_PX = 60.0


def draw_status(image: np.ndarray, fps: float, latency_ms: float, budget_ms: float) -> None:
    """Top bar: FPS and end-to-end latency, red when over the budget."""
    color = _RED if latency_ms > budget_ms else _GREEN
    text = f"FPS {fps:5.1f} | latency {latency_ms:5.1f} ms"
    cv2.rectangle(image, (0, 0), (300, 28), _BLACK, thickness=-1)
    cv2.putText(image, text, (8, 20), _FONT, 0.55, color, 1, cv2.LINE_AA)


def _fmt(value: float | None, spec: str) -> str:
    return "  --" if value is None else format(value, spec)


def draw_perception(image: np.ndarray, out: PerceptionOutput, cfg: OverlayConfig) -> None:
    """Eye/mouth contours, head direction and EAR/MAR/pose readouts; NO FACE if missing."""
    f = out.features
    if cfg.landmarks and out.landmarks is not None:
        pts = out.landmarks[:, :2]
        for contour, color in (
            (LEFT_EYE_CONTOUR, _CYAN),
            (RIGHT_EYE_CONTOUR, _CYAN),
            (INNER_LIPS_CONTOUR, _YELLOW),
        ):
            poly = np.round(pts[list(contour)]).astype(np.int32)
            cv2.polylines(image, [poly], isClosed=True, color=color, thickness=1)
        if f.pitch is not None and f.yaw is not None:
            # Nose direction: a short line in the direction the head points.
            nose = pts[POSE_POINTS[0]]
            dx = np.sin(np.radians(f.yaw)) * _NOSE_LINE_PX
            dy = -np.sin(np.radians(f.pitch)) * _NOSE_LINE_PX
            start = tuple(int(v) for v in np.round(nose))
            end = (int(round(nose[0] + dx)), int(round(nose[1] + dy)))
            cv2.line(image, start, end, _GREEN, 2, cv2.LINE_AA)

    if not cfg.values:
        return
    height = image.shape[0]
    if not f.face_found:
        cv2.putText(image, "NO FACE", (8, height - 12), _FONT, 0.9, _RED, 2, cv2.LINE_AA)
        return
    lines = [
        f"EAR L {_fmt(f.ear_left, '.3f')}  R {_fmt(f.ear_right, '.3f')}",
        f"MAR {_fmt(f.mar, '.3f')}",
        f"pitch {_fmt(f.pitch, '+6.1f')}  yaw {_fmt(f.yaw, '+6.1f')}  roll {_fmt(f.roll, '+6.1f')}",
        f"perception {out.perception_ms:5.1f} ms",
    ]
    top = height - 10 - 22 * len(lines)
    cv2.rectangle(image, (0, top - 6), (400, height), _BLACK, thickness=-1)
    for i, text in enumerate(lines):
        cv2.putText(image, text, (8, top + 16 + 22 * i), _FONT, 0.55, _WHITE, 1, cv2.LINE_AA)
