"""Optional low-light preprocessing (literature review §2A: CLAHE helps in low light [M1])."""

from __future__ import annotations

import cv2
import numpy as np


class Clahe:
    """CLAHE on the L channel of LAB. Returns a new image; the input is not modified."""

    def __init__(self, clip_limit: float, tile: int) -> None:
        self._clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
