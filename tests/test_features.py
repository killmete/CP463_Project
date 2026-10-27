"""Feature formulas on synthetic landmark geometry with known answers."""

import math

import cv2
import numpy as np
import pytest

from drowsiness.perception import features as F
from drowsiness.perception.landmarks import (
    LEFT_EYE,
    MOUTH_CORNERS,
    MOUTH_VERTICAL,
    NUM_LANDMARKS,
    POSE_POINTS,
    RIGHT_EYE,
)


def place_eye(points, eye, cx, cy, width, height):
    """Eye corners `width` apart, both lid pairs `height` apart: EAR = height / width."""
    p1, p2, p3, p4, p5, p6 = eye
    half = width / 2
    points[p1, :2] = (cx - half, cy)
    points[p4, :2] = (cx + half, cy)
    for top, bottom, x in ((p2, p6, cx - half / 2), (p3, p5, cx + half / 2)):
        points[top, :2] = (x, cy - height / 2)
        points[bottom, :2] = (x, cy + height / 2)


def face_points(right_height=9.0, left_height=9.0):
    points = np.zeros((NUM_LANDMARKS, 3), np.float32)
    place_eye(points, RIGHT_EYE, 100, 100, 30, right_height)
    place_eye(points, LEFT_EYE, 160, 100, 30, left_height)
    return points


def test_ear_of_known_geometry():
    assert F.ear_right(face_points()) == pytest.approx(9 / 30)
    assert F.ear_left(face_points(left_height=6)) == pytest.approx(6 / 30)


def test_ear_textbook_example():
    points = np.zeros((NUM_LANDMARKS, 3))
    p1, p2, p3, p4, p5, p6 = RIGHT_EYE
    points[[p1, p2, p3, p4, p5, p6], :2] = [(0, 0), (1, 1), (3, 1), (4, 0), (3, -1), (1, -1)]
    assert F.ear_right(points) == pytest.approx((2 + 2) / (2 * 4))


def test_closed_eye_gives_zero_ear():
    assert F.ear_right(face_points(right_height=0)) == pytest.approx(0.0)


def test_degenerate_eye_gives_none():
    assert F.ear_right(np.zeros((NUM_LANDMARKS, 3))) is None


def test_ear_is_scale_invariant():
    points = face_points()
    assert F.ear_left(points * 3.7) == pytest.approx(F.ear_left(points))


def test_only_the_closed_eye_changes():
    points = face_points(left_height=0.0)  # driver's left eye closed
    assert F.ear_left(points) == pytest.approx(0.0)
    assert F.ear_right(points) == pytest.approx(9 / 30)


def test_eye_index_sets_are_distinct_and_in_range():
    assert not set(LEFT_EYE) & set(RIGHT_EYE)
    assert all(0 <= i < NUM_LANDMARKS for i in LEFT_EYE + RIGHT_EYE)


def mouth_points(gaps, width=40.0):
    points = np.zeros((NUM_LANDMARKS, 3))
    left, right = MOUTH_CORNERS
    points[left, :2] = (0, 0)
    points[right, :2] = (width, 0)
    for (upper, lower), gap, x in zip(MOUTH_VERTICAL, gaps, (10, 20, 30), strict=True):
        points[upper, :2] = (x, -gap / 2)
        points[lower, :2] = (x, gap / 2)
    return points


def test_mar_of_known_geometry():
    assert F.mouth_aspect_ratio(mouth_points((8, 16, 24))) == pytest.approx(16 / 40)


def test_closed_mouth_gives_zero_mar():
    assert F.mouth_aspect_ratio(mouth_points((0, 0, 0))) == pytest.approx(0.0)


def test_mar_is_scale_invariant():
    points = mouth_points((8, 16, 24))
    assert F.mouth_aspect_ratio(points * 0.25) == pytest.approx(0.4)


def test_degenerate_mouth_gives_none():
    assert F.mouth_aspect_ratio(np.zeros((NUM_LANDMARKS, 3))) is None


W, H = 640, 480


def projected_face(pitch, yaw, roll, distance_mm=1500.0):
    """Landmarks (pixels) of the generic head model posed at the given angles."""
    rvec, _ = cv2.Rodrigues(F.angles_to_rotation(pitch, yaw, roll))
    image_points, _ = cv2.projectPoints(
        F.HEAD_MODEL_POINTS,
        rvec,
        np.array([0.0, 0.0, distance_mm]),
        F.camera_matrix(W, H),
        np.zeros(4),
    )
    points = np.zeros((NUM_LANDMARKS, 3))
    points[list(POSE_POINTS), :2] = image_points.reshape(-1, 2)
    return points


@pytest.mark.parametrize(
    "angles",
    [(0, 0, 0), (20, 0, 0), (-20, 0, 0), (0, 30, 0), (0, -30, 0), (0, 0, 15), (0, 0, -15),
     (10, -25, 8), (-15, 35, -10)],
)  # fmt: skip
def test_head_pose_recovers_known_angles(angles):
    pose = F.head_pose(projected_face(*angles), W, H)
    assert pose is not None
    assert pose == pytest.approx(angles, abs=1.0)


def test_rotation_angle_round_trip():
    angles = (12.0, -33.0, 7.5)
    assert F.rotation_to_angles(F.angles_to_rotation(*angles)) == pytest.approx(angles)


def _nose_offset(points):
    """Nose tip relative to the centre of the eye and mouth corners (pixels)."""
    nose = points[POSE_POINTS[0], :2]
    centre = points[list(POSE_POINTS[2:]), :2].mean(axis=0)
    return nose - centre


def test_pose_signs_match_the_contract():
    # Contract (types.py): pitch + = head up, yaw + = driver's left (= image right
    # in an unmirrored front camera), roll + = toward the driver's left shoulder.
    assert _nose_offset(projected_face(20, 0, 0))[1] < 0  # nose moves up in the image
    assert _nose_offset(projected_face(0, 30, 0))[0] > 0  # nose moves image right
    tilted = projected_face(0, 0, 15)
    right_eye, left_eye = tilted[POSE_POINTS[2], :2], tilted[POSE_POINTS[3], :2]
    # Head tipped toward the left shoulder (image right): the left eye drops lower.
    assert left_eye[1] > right_eye[1]
    assert math.isclose(F.head_pose(tilted, W, H)[2], 15, abs_tol=1.0)
