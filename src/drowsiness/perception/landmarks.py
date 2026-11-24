"""MediaPipe Face Landmarker (478-point) index sets used by the feature functions.

Left/right are the driver's anatomical sides. In an unmirrored front camera the
driver's left eye has the larger image x. Indices follow the canonical face mesh
(https://github.com/google-ai-edge/mediapipe, face_geometry canonical model).
"""

from __future__ import annotations

# MediaPipe ให้ 478 จุด: 468 จุดบนหน้า + 10 จุดม่านตา
NUM_LANDMARKS = 478

# EAR order p1..p6 (Soukupova & Cech 2016): p1/p4 = eye corners, p2/p3 upper lid,
# p6/p5 the lower-lid points below p2/p3, so EAR = (|p2-p6| + |p3-p5|) / (2 |p1-p4|).
# เลขจุดรอบตาขวา/ซ้ายของคนขับ ใช้คำนวณ EAR (ความเปิดของตา)
RIGHT_EYE = (33, 160, 158, 133, 153, 144)  # outer corner, ..., inner corner, ...
LEFT_EYE = (362, 385, 387, 263, 373, 380)  # inner corner, ..., outer corner, ...

# MAR: mean of three inner-lip vertical gaps over the inner mouth width.
# คู่จุดริมฝีปากบน-ล่าง 3 คู่ และมุมปาก 2 ข้าง ใช้คำนวณ MAR (ความอ้าปาก)
MOUTH_VERTICAL = ((81, 178), (13, 14), (311, 402))  # (upper lip, lower lip)
MOUTH_CORNERS = (78, 308)

# Six points for head pose (solvePnP), matched to HEAD_MODEL_POINTS in features.py.
# 6 จุดที่ใช้หามุมหัว (head pose) ด้วย solvePnP
POSE_POINTS = (
    1,  # nose tip
    152,  # chin
    33,  # right eye outer corner
    263,  # left eye outer corner
    61,  # right mouth corner
    291,  # left mouth corner
)

# Contours drawn by the debug overlay.
# เส้นรอบตาและปาก ใช้วาด overlay บนจอเท่านั้น
RIGHT_EYE_CONTOUR = (33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7)
LEFT_EYE_CONTOUR = (
    362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382,
)  # fmt: skip
INNER_LIPS_CONTOUR = (
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
)  # fmt: skip
# จุดม่านตา เตรียมไว้ทำ gaze (ยังไม่ได้ใช้)
IRISES = tuple(range(468, 478))  # two irises, 5 points each (gaze, later milestone)
