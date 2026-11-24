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
# โมเดลหน้า 3 มิติแบบทั่วไป (หน่วย mm) จับคู่กับ POSE_POINTS 6 จุด
# solvePnP หาการหมุนที่ทำให้จุด 3D นี้ตรงกับจุด 2D ในภาพ -> ได้มุมหัว
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

# ค่าเล็กมาก ใช้กันการหารด้วยศูนย์
_EPS = 1e-6


# ระยะห่างระหว่างจุด a กับ b (ใช้แค่ x, y)
def _dist(points: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(points[a, :2] - points[b, :2]))


# EAR = (ความสูงตา 2 จุดรวมกัน) / (2 x ความกว้างตา)
# ตาเปิด ~0.25-0.35, ตาปิดเข้าใกล้ 0; เป็นอัตราส่วนจึงไม่ขึ้นกับระยะห่างจากกล้อง
def eye_aspect_ratio(points: np.ndarray, eye: tuple[int, ...]) -> float | None:
    """EAR = (|p2-p6| + |p3-p5|) / (2 |p1-p4|); None if the eye corners coincide."""
    p1, p2, p3, p4, p5, p6 = eye
    width = _dist(points, p1, p4)
    if width < _EPS:
        return None
    return (_dist(points, p2, p6) + _dist(points, p3, p5)) / (2.0 * width)


# EAR ของตาซ้ายคนขับ
def ear_left(points: np.ndarray) -> float | None:
    return eye_aspect_ratio(points, LEFT_EYE)


# EAR ของตาขวาคนขับ
def ear_right(points: np.ndarray) -> float | None:
    return eye_aspect_ratio(points, RIGHT_EYE)


# MAR = ค่าเฉลี่ยความสูงช่องปาก 3 จุด / ความกว้างปาก (หาว = ค่าสูง)
def mouth_aspect_ratio(points: np.ndarray) -> float | None:
    """MAR = mean inner-lip vertical gap / inner mouth width; None if the width is 0."""
    width = _dist(points, *MOUTH_CORNERS)
    if width < _EPS:
        return None
    gaps = [_dist(points, upper, lower) for upper, lower in MOUTH_VERTICAL]
    return sum(gaps) / len(gaps) / width


# กล้องจำลองแบบง่าย: focal length = ความกว้างภาพ, จุดกลางภาพเป็นจุดศูนย์กลาง
# (ไม่ต้อง calibrate กล้องจริง เพราะเราต้องการแค่มุม)
def camera_matrix(width: int, height: int) -> np.ndarray:
    """Pinhole camera with focal length = image width and the centre as principal point."""
    f = float(width)
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])


# หามุมหัว (pitch ก้ม/เงย, yaw หันซ้าย/ขวา, roll เอียง) เป็นองศา
def head_pose(points: np.ndarray, width: int, height: int) -> tuple[float, float, float] | None:
    """(pitch, yaw, roll) in degrees from solvePnP, or None if it fails.

    pitch + = head up, yaw + = turned to the driver's left (image right),
    roll + = tilted toward the driver's left shoulder. Uncalibrated camera frame.
    """
    # ดึง 6 จุดจากภาพ; ถ้ามีค่า NaN/inf คืน None
    image_points = np.ascontiguousarray(points[list(POSE_POINTS), :2], dtype=np.float64)
    if not np.isfinite(image_points).all():
        return None
    try:
        # solvePnP วิธี SQPNP: หาเวกเตอร์การหมุน (rvec) ของหัวเทียบกับกล้อง
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
    # แปลงเวกเตอร์การหมุนเป็นเมทริกซ์ 3x3 แล้วแยกเป็นมุม
    rotation, _ = cv2.Rodrigues(rvec)
    return rotation_to_angles(rotation)


# แยกเมทริกซ์การหมุนเป็นมุม 3 แกน แล้วปรับเครื่องหมายให้ pitch + = เงย, yaw + = หันซ้าย
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


# ฟังก์ชันกลับด้าน (มุม -> เมทริกซ์) ใช้ในเทสต์และการลบมุมกลางตอน calibrate
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
