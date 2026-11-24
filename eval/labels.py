"""Ground truth from each dataset's own annotations, as per-frame boolean tracks.

Every label is mapped to one of these names, so metrics don't care where it came from:

    eyes_closed   NTHU eye=1 (sleepy eyes), DMD eyes_state/close
    yawn          NTHU mouth=1, DMD yawning/*
    talking       NTHU mouth=2, DMD talking/talking
    nod           NTHU head=1
    look_aside    NTHU head=2, DMD gaze_on_road/not_looking_road
    drowsy        NTHU drowsiness=1

YawDD and UTA-RLDD only have clip-level labels (`Clip.kind`): YawDD Normal / Talking /
Yawning / TalkingYawning / mixed, UTA-RLDD 0 (alert) / 5 (low vigilance) / 10 (drowsy).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from eval.catalog import Clip

# ชื่อ label กลาง 6 ตัว: label ของทุกชุดข้อมูลแปลงมาเป็นชื่อเหล่านี้
LABELS = ("eyes_closed", "yawn", "talking", "nod", "look_aside", "drowsy")

# NTHU: ตัวเลขในไฟล์ label -> ชื่อ label กลาง
_NTHU = {  # file kind -> {digit: label}
    "eye": {"1": "eyes_closed"},
    "mouth": {"1": "yawn", "2": "talking"},
    "head": {"1": "nod", "2": "look_aside"},
    "drowsiness": {"1": "drowsy"},
}
# DMD: ชื่อ action ใน JSON -> ชื่อ label กลาง
_DMD = {
    "eyes_state/close": "eyes_closed",
    "yawning/Yawning with hand": "yawn",
    "yawning/Yawning without hand": "yawn",
    "talking/talking": "talking",
    "gaze_on_road/not_looking_road": "look_aside",
}
# สถานะที่ชัดเจนของ DMD (เช่น ตาเปิด/ปิด) ช่วงอื่น (กำลังปิด, ไม่ระบุ) ไม่นับ
_DMD_KNOWN = {"eyes_state": ("open", "close"), "gaze_on_road": ("looking_road", "not_looking_road")}


@dataclass
# label ต่อเฟรมของ 1 คลิป: tracks = เฟรมที่เป็นจริง, annotated = เฟรมที่มี label
class Truth:
    """Per-frame labels for one clip (frame i of the feature CSV = index i here)."""

    n_frames: int
    tracks: dict[str, np.ndarray] = field(default_factory=dict)  # label -> bool[n_frames]
    annotated: dict[str, np.ndarray] = field(default_factory=dict)  # label -> bool, frames labelled

    def has(self, label: str) -> bool:
        return label in self.tracks


# array bool ว่าง n เฟรม
def _track(n: int) -> np.ndarray:
    return np.zeros(n, dtype=bool)


# อ่าน label NTHU: ไฟล์เป็นตัวเลข 1 ตัวต่อเฟรม; ถ้าไฟล์สั้นกว่าวิดีโอ ส่วนที่เหลือถือว่าไม่รู้
def nthu_truth(clip: Clip, n_frames: int) -> Truth:
    truth = Truth(n_frames)
    for kind, mapping in _NTHU.items():
        path = clip.labels.get(kind)
        if path is None:
            continue
        digits = "".join(
            ch for ch in path.read_text(encoding="ascii", errors="ignore") if ch.isdigit()
        )
        values = np.frombuffer(digits.encode()[:n_frames], dtype=np.uint8) - ord("0")
        covered = _track(n_frames)
        covered[: len(values)] = True  # a label file shorter than the video: rest unknown
        for digit, label in mapping.items():
            track = _track(n_frames)
            track[: len(values)] = values == int(digit)
            truth.tracks[label] = track
            truth.annotated[label] = covered
    return truth


# อ่าน label DMD จาก JSON (OpenLABEL): แต่ละ action มีช่วงเฟรม (frame_end รวมด้วย)
def dmd_truth(clip: Clip, n_frames: int) -> Truth:
    truth = Truth(n_frames)
    for path in clip.labels.values():
        ol = json.loads(Path(path).read_text(encoding="utf-8"))["openlabel"]
        span = ol.get("frame_intervals") or [{"frame_start": 0, "frame_end": n_frames - 1}]
        covered = _track(n_frames)
        for iv in span:
            covered[iv["frame_start"] : iv["frame_end"] + 1] = True
        states: dict[str, np.ndarray] = {}  # every action type -> its frames
        for action in ol.get("actions", {}).values():
            track = states.setdefault(action.get("type", ""), _track(n_frames))
            for iv in action.get("frame_intervals", []):
                track[iv["frame_start"] : iv["frame_end"] + 1] = True  # frame_end is inclusive
        families = {t.split("/")[0] for t in states if t}
        for dmd_type, label in _DMD.items():
            family = dmd_type.split("/")[0]
            if family not in families:
                continue
            track = truth.tracks.setdefault(label, _track(n_frames))
            track |= states.get(dmd_type, _track(n_frames))
            if family in _DMD_KNOWN:
                # Only frames with a definite state count: eyes "undefined", "opening" and
                # "closing" (or unlabelled gaze) are neither positives nor negatives.
                known = _track(n_frames)
                for state in _DMD_KNOWN[family]:
                    known |= states.get(f"{family}/{state}", _track(n_frames))
                truth.annotated[label] = truth.annotated.get(label, _track(n_frames)) | known
            else:  # event-only families (yawning, talking): the whole span is known
                truth.annotated[label] = truth.annotated.get(label, _track(n_frames)) | covered
    return truth


# เลือกตัวอ่าน label ตามชุดข้อมูล; YawDD/UTA มีแค่ label ระดับคลิป
def truth_for(clip: Clip, n_frames: int) -> Truth:
    if clip.dataset == "nthu":
        return nthu_truth(clip, n_frames)
    if clip.dataset == "dmd":
        return dmd_truth(clip, n_frames)
    return Truth(n_frames)  # clip-level labels only (Clip.kind)


# แปลงเฟรมที่เป็นจริงเป็นช่วงเวลา [เริ่ม, จบ) วินาที; ช่วงที่ห่างกันน้อยรวมเป็นช่วงเดียว
def intervals(track: np.ndarray, fps: float, merge_gap_s: float = 0.0) -> list[tuple[float, float]]:
    """True runs as [start_s, end_s) intervals; runs closer than merge_gap_s are joined."""
    if not track.any():
        return []
    padded = np.concatenate(([False], track, [False])).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    runs = [(s / fps, e / fps) for s, e in zip(edges[::2], edges[1::2], strict=True)]
    merged = [runs[0]]
    for start, end in runs[1:]:
        if start - merged[-1][1] <= merge_gap_s:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged
