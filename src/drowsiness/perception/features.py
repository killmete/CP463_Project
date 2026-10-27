"""Pure per-frame feature functions on a (N, 2+) landmark array in pixel coordinates.

All outputs are scale-invariant (ratios, angles). Geometry must be in pixels
(normalised x * width, y * height), not normalised coordinates, or the ratios
would depend on the frame's aspect ratio.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from drowsiness.perception.landmarks import (
    LEFT_EYE,
    MOUTH_CORNERS,
    MOUTH_VERTICAL,
    POSE_POINTS,
    RIGHT_EYE,
)

# Generic 3D face (mm) for POSE_POINTS, in OpenCV camera axes for a face looking straight
# at the camera: x = image right, y = image down, z = away from the camera (the nose is
# the closest point). Proportions from the common 6-point solvePnP model (literature
# review [I1]); only the shape matters, since we return angles, not distance.
HEAD_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),  # nose tip
        (0.0, 330.0, 65.0),  # chin
        (-225.0, -170.0, 135.0),  # right eye outer corner (image left)
        (225.0, -170.0, 135.0),  # left eye outer corner (image right)
        (-150.0, 150.0, 125.0),  # right mouth corner
        (150.0, 150.0, 125.0),  # left mouth corner
    ],
    dtype=np.float64,
)

_EPS = 1e-6


def _dist(points: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(points[a, :2] - points[b, :2]))


def eye_aspect_ratio(points: np.ndarray, eye: tuple[int, ...]) -> float | None:
    """EAR = (|p2-p6| + |p3-p5|) / (2 |p1-p4|); None if the eye corners coincide."""
    p1, p2, p3, p4, p5, p6 = eye
    width = _dist(points, p1, p4)
    if width < _EPS:
        return None
    return (_dist(points, p2, p6) + _dist(points, p3, p5)) / (2.0 * width)


def ear_left(points: np.ndarray) -> float | None:
    return eye_aspect_ratio(points, LEFT_EYE)


def ear_right(points: np.ndarray) -> float | None:
    return eye_aspect_ratio(points, RIGHT_EYE)


def mouth_aspect_ratio(points: np.ndarray) -> float | None:
    """MAR = mean inner-lip vertical gap / inner mouth width; None if the width is 0."""
    width = _dist(points, *MOUTH_CORNERS)
    if width < _EPS:
        return None
    gaps = [_dist(points, upper, lower) for upper, lower in MOUTH_VERTICAL]
    return sum(gaps) / len(gaps) / width


def camera_matrix(width: int, height: int) -> np.ndarray:
    """Pinhole camera with focal length = image width and the centre as principal point."""
    f = float(width)
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])


def head_pose(points: np.ndarray, width: int, height: int) -> tuple[float, float, float] | None:
    """(pitch, yaw, roll) in degrees from solvePnP, or None if it fails.

    pitch + = head up, yaw + = turned to the driver's left (image right),
    roll + = tilted toward the driver's left shoulder. Uncalibrated camera frame.
    """
    image_points = np.ascontiguousarray(points[list(POSE_POINTS), :2], dtype=np.float64)
    if not np.isfinite(image_points).all():
        return None
    try:
        ok, rvec, _ = cv2.solvePnP(
            HEAD_MODEL_POINTS,
            image_points,
            camera_matrix(width, height),
            np.zeros(4),
            flags=cv2.SOLVEPNP_SQPNP,
        )
    except cv2.error:  # degenerate input; a bad frame must not stop monitoring
        return None
    if not ok:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    return rotation_to_angles(rotation)


def rotation_to_angles(rotation: np.ndarray) -> tuple[float, float, float]:
    """Decompose R = Rz(rz) @ Ry(ry) @ Rx(rx) (camera axes) into (pitch, yaw, roll) degrees.

    The face looks along -z, so a positive rotation about x tips the nose down and a
    positive rotation about y turns it to image left: pitch = -rx, yaw = -ry, roll = rz.
    """
    r = rotation
    rx = math.atan2(r[2, 1], r[2, 2])
    ry = math.atan2(-r[2, 0], math.hypot(r[2, 1], r[2, 2]))
    rz = math.atan2(r[1, 0], r[0, 0])
    return -math.degrees(rx), -math.degrees(ry), math.degrees(rz)


def angles_to_rotation(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """Inverse of `rotation_to_angles` (degrees in, 3x3 matrix out)."""
    rx, ry, rz = math.radians(-pitch), math.radians(-yaw), math.radians(roll)
    cx, sx, cy, sy, cz, sz = (
        math.cos(rx),
        math.sin(rx),
        math.cos(ry),
        math.sin(ry),
        math.cos(rz),
        math.sin(rz),
    )
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rot_z @ rot_y @ rot_x
