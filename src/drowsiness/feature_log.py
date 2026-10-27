"""Feature CSV v1 (`--log-features`): one row per processed frame, replayable by logic tests.

Format (docs/architecture.md): header row, then columns
timestamp, frame_index, face_found, ear_left, ear_right, mar, pitch, yaw, roll,
gaze_h, gaze_v, perception_ms, and (if blendshapes are on) one bs_<name> column per
MediaPipe blendshape in BLENDSHAPE_NAMES order. face_found is 0/1, None is an empty
cell, floats use repr() so they round-trip exactly.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Self

from drowsiness.types import FrameFeatures, PerceptionOutput

# MediaPipe Face Landmarker blendshape categories, in model output order.
BLENDSHAPE_NAMES = (
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft",
    "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft",
    "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
    "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
)  # fmt: skip

BASE_COLUMNS = (
    "timestamp", "frame_index", "face_found", "ear_left", "ear_right", "mar",
    "pitch", "yaw", "roll", "gaze_h", "gaze_v", "perception_ms",
)  # fmt: skip
_FLOAT_FIELDS = ("ear_left", "ear_right", "mar", "pitch", "yaw", "roll")
FLUSH_INTERVAL_S = 1.0


def _fmt(value: float | None) -> str:
    return "" if value is None else repr(float(value))


def _parse(cell: str) -> float | None:
    return None if cell == "" else float(cell)


class FeatureCsvWriter:
    """Buffered writer; flushes about once a second and on close (never per frame)."""

    def __init__(self, path: str | Path, blendshapes: bool = True) -> None:
        self.path = Path(path)
        self._blendshapes = blendshapes
        self._file = None
        self._writer = None
        self._last_flush = 0.0

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8", newline="")
        self._writer = csv.writer(self._file)
        bs_columns = [f"bs_{name}" for name in BLENDSHAPE_NAMES] if self._blendshapes else []
        self._writer.writerow([*BASE_COLUMNS, *bs_columns])
        self._last_flush = time.perf_counter()

    def write(self, out: PerceptionOutput, frame_index: int) -> None:
        if self._writer is None:
            raise RuntimeError("feature log is not open")
        f = out.features
        gaze_h, gaze_v = f.gaze if f.gaze is not None else (None, None)
        row = [
            repr(float(f.timestamp)),
            str(frame_index),
            "1" if f.face_found else "0",
            *(_fmt(getattr(f, name)) for name in _FLOAT_FIELDS),
            _fmt(gaze_h),
            _fmt(gaze_v),
            f"{out.perception_ms:.3f}",
        ]
        if self._blendshapes:
            bs = f.blendshapes or {}
            row.extend(_fmt(bs.get(name)) for name in BLENDSHAPE_NAMES)
        self._writer.writerow(row)
        now = time.perf_counter()
        if now - self._last_flush >= FLUSH_INTERVAL_S:
            self._file.flush()
            self._last_flush = now

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_feature_csv(path: str | Path) -> list[FrameFeatures]:
    """Rebuild the FrameFeatures stream from a v1 CSV (frame_index/perception_ms ignored)."""
    features = []
    with Path(path).open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = set(BASE_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: not a feature CSV v1, missing columns {sorted(missing)}")
        bs_columns = [c for c in reader.fieldnames if c.startswith("bs_")]
        for row in reader:
            gaze_h, gaze_v = _parse(row["gaze_h"]), _parse(row["gaze_v"])
            bs = {c[3:]: float(row[c]) for c in bs_columns if row[c] != ""}
            features.append(
                FrameFeatures(
                    timestamp=float(row["timestamp"]),
                    face_found=row["face_found"] == "1",
                    **{name: _parse(row[name]) for name in _FLOAT_FIELDS},
                    gaze=None if gaze_h is None or gaze_v is None else (gaze_h, gaze_v),
                    blendshapes=bs or None,
                )
            )
    return features
