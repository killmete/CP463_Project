# เทสต์ eval pipeline ด้วยข้อมูลปลอม (ไม่ต้องมีชุดข้อมูล โมเดล หรือวิดีโอจริง)
"""Evaluation pipeline on synthetic data: no datasets, no model, no video decoding."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from drowsiness.config import load_config
from eval.catalog import Clip, nthu, uta, yawdd
from eval.labels import dmd_truth, intervals, nthu_truth
from eval.metrics import auc, negative_clip, prf, score_clip, summarize
from eval.replay import load_features
from eval.tune import FALSE_ALARM_SLACK_PER_H, GROUPS, nest

FPS = 30.0


def _touch(path: Path, text: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="ascii")
    return path


def _arrays(n: int, *, ear=0.3, mar=0.02, pitch=0.0, yaw=0.0) -> dict[str, np.ndarray]:
    def col(v):
        return np.broadcast_to(np.asarray(v, dtype=float), (n,)).copy()

    return {
        "timestamp": np.arange(n) / FPS,
        "face_found": np.ones(n),
        "ear_left": col(ear),
        "ear_right": col(ear),
        "mar": col(mar),
        "pitch": col(pitch),
        "yaw": col(yaw),
        "roll": np.zeros(n),
    }


# -- catalog -----------------------------------------------------------------------------


# NTHU: รหัสคลิป การแบ่ง tune/test และไฟล์ label ถูกต้อง
def test_nthu_catalog_ids_split_and_labels(tmp_path):
    train = tmp_path / "nthu-ddd/Training_Evaluation_Dataset/Training Dataset/001/night_noglasses"
    _touch(train / "yawning.avi")
    _touch(train / "001_yawning_eye.txt", "0011")
    test = tmp_path / "nthu-ddd/Testing_Dataset"
    _touch(test / "003_noglasses_mix.mp4")
    _touch(test / "test_label_txt/noglasses/003_noglasses_mixing_drowsiness.txt", "01")
    _touch(test / "test_label_txt/wh/003_noglasses_mixing_drowsiness.txt", "11")

    clips = {c.clip_id: c for c in nthu(tmp_path)}
    tr, te = clips["train_001_nightnoglasses_yawning"], clips["test_003_noglasses"]
    assert (tr.split, tr.condition, tr.kind, set(tr.labels)) == (
        "tune",
        "nightnoglasses",
        "yawning",
        {"eye"},
    )
    assert te.split == "test" and te.labels["drowsiness"].parent.name == "noglasses"


# YawDD: แยกชื่อไฟล์ได้ และแบ่งตามคน
def test_yawdd_names_and_subject_split(tmp_path):
    for name in (
        "1-FemaleNoGlasses-Normal.avi",
        "2-MaleGlasses-Talking&Yawning.avi",
        "3-MaleGlasses-Yawning.avi",
        "4-FemaleSunGlasses-Yawning .avi",
    ):
        _touch(tmp_path / "yawdd/Mirror/x" / name)
    _touch(tmp_path / "yawdd/Dash/5-MaleGlasses.avi")
    clips = {c.clip_id: c for c in yawdd(tmp_path)}
    assert set(clips) == {
        "mirror_F01_NoGlasses_Normal",
        "mirror_M02_Glasses_TalkingYawning",
        "mirror_M03_Glasses_Yawning",
        "mirror_F04_SunGlasses_Yawning",
        "dash_M05_Glasses_mixed",
    }
    assert {c.split for c in clips.values()} == {"tune", "test"}


# UTA: fold 1 เป็น test
def test_uta_fold_one_is_test(tmp_path):
    _touch(tmp_path / "uta-rldd/01/10.mov")  # <root>/uta-rldd/<subject>/<0|5|10>.<ext>
    _touch(tmp_path / "uta-rldd/13/0.mp4")
    got = {c.clip_id: (c.split, c.kind) for c in uta(tmp_path)}
    assert got == {"01_10": ("test", "10"), "13_0": ("tune", "0")}


# -- labels ------------------------------------------------------------------------------


# NTHU: อ่าน label ตัวเลข; ไฟล์สั้นกว่าวิดีโอ ส่วนที่เหลือไม่นับ
def test_nthu_digit_labels_and_short_files(tmp_path):
    clip = Clip(
        "nthu",
        "x",
        "001",
        Path("v.avi"),
        "tune",
        labels={
            "mouth": _touch(tmp_path / "m.txt", "0120\n"),
            "eye": _touch(tmp_path / "e.txt", "11"),
        },
    )
    truth = nthu_truth(clip, 6)
    assert truth.tracks["yawn"].tolist() == [False, True, False, False, False, False]
    assert truth.tracks["talking"].tolist() == [False, False, True, False, False, False]
    assert truth.annotated["eyes_closed"].tolist() == [True, True, False, False, False, False]
    assert not truth.has("nod")


# DMD: ช่วงเฟรมรวม frame_end และนับเฉพาะเฟรมที่มี label
def test_dmd_intervals_inclusive_and_family_coverage(tmp_path):
    ann = {
        "openlabel": {
            "frame_intervals": [{"frame_start": 0, "frame_end": 9}],
            "actions": {
                "0": {
                    "type": "eyes_state/close",
                    "frame_intervals": [{"frame_start": 2, "frame_end": 4}],
                },
                "1": {
                    "type": "eyes_state/open",
                    "frame_intervals": [{"frame_start": 0, "frame_end": 1}],
                },
            },
        }
    }
    path = tmp_path / "a.json"
    path.write_text(json.dumps(ann), encoding="utf-8")
    truth = dmd_truth(Clip("dmd", "x", "1", Path("v.mp4"), "tune", labels={"d": path}), 12)
    assert np.flatnonzero(truth.tracks["eyes_closed"]).tolist() == [2, 3, 4]
    assert truth.annotated["eyes_closed"].sum() == 5  # only frames with a definite open/close state
    assert not truth.has("yawn")  # yawning family not annotated in this file


# ช่วงที่ห่างกันน้อยกว่า merge_gap รวมเป็นช่วงเดียว
def test_intervals_merge_gap():
    track = np.array([0, 1, 1, 0, 1, 0, 0, 0, 1], dtype=bool)
    assert intervals(track, 1.0) == [(1, 3), (4, 5), (8, 9)]
    assert intervals(track, 1.0, merge_gap_s=1.0) == [(1, 5), (8, 9)]


# -- metrics -----------------------------------------------------------------------------


# สูตร precision/recall/F1 และ AUC
def test_prf_and_auc():
    assert prf((8, 2, 2)) == pytest.approx(
        {"precision": 0.8, "recall": 0.8, "f1": 0.8, "support": 10}
    )
    assert prf((0, 0, 5))["f1"] == 0.0
    assert np.isnan(prf((0, 0, 0))["f1"])
    assert auc([3, 4], [1, 2]) == 1.0
    assert auc([1], [1]) == 0.5


def _nthu_clip(tmp_path, kind, labels: dict[str, str]) -> Clip:
    files = {k: _touch(tmp_path / f"{k}.txt", v) for k, v in labels.items()}
    return Clip("nthu", "c", "001", Path("v.avi"), "tune", kind=kind, labels=files)


# หลับตานานหลัง calibrate ถูกนับว่าจับได้
def test_score_clip_catches_long_closure_after_calibration(tmp_path):
    cfg = load_config()
    n = int(40 * FPS)
    ear = np.full(n, 0.3)
    closed = slice(int(20 * FPS), int(24 * FPS))  # 4 s eyes closed: microsleep and sleep
    ear[closed] = 0.02
    eye = np.zeros(n, dtype=int)
    eye[closed] = 1
    clip = _nthu_clip(tmp_path, "sleepyCombination", {"eye": "".join(map(str, eye))})
    s = score_clip(clip, _arrays(n, ear=ear), cfg)

    assert 0 < s.scored_s < s.duration_s  # calibration time is not scored
    tp, fp, fn, _tn = s.frame["eyes_closed"]
    assert tp > 0.8 * (closed.stop - closed.start) and fp < 0.1 * FPS
    delay = s.events["closure"][0]
    assert delay is not None and cfg.eyes.microsleep_s <= delay < cfg.eyes.microsleep_s + 0.5
    assert s.drowsy_alerts >= 1 and s.event_fp["closure"] == 0


# จับคู่หาวกับ label และนับหาวที่ผิด
def test_score_clip_yawn_matching_and_false_yawns(tmp_path):
    cfg = load_config(overrides={"yawn": {"mar_threshold": 0.4, "min_duration_s": 1.0}})
    n = int(60 * FPS)
    mar = np.full(n, 0.02)
    mouth = np.zeros(n, dtype=int)
    for start in (20, 30):  # two labelled yawns
        mar[int(start * FPS) : int((start + 3) * FPS)] = 0.7
        mouth[int(start * FPS) : int((start + 3) * FPS)] = 1
    mar[int(45 * FPS) : int(47 * FPS)] = 0.7  # an unlabelled wide mouth: a false yawn
    clip = _nthu_clip(tmp_path, "yawning", {"mouth": "".join(map(str, mouth))})
    s = score_clip(clip, _arrays(n, mar=mar), cfg)
    assert [d is not None for d in s.events["yawn"]] == [True, True]
    assert s.yawn_fp == 1 and s.yawns_detected == 3

    summary = summarize([s])
    assert summary.events["yawn"]["recall"] == 1.0
    assert summary.events["yawn"]["precision"] == pytest.approx(2 / 3)


# false alarm นับเฉพาะคลิปที่ไม่ง่วง
def test_false_alarms_counted_only_on_non_drowsy_clips(tmp_path):
    cfg = load_config()
    n = int(40 * FPS)
    ear = np.full(n, 0.3)
    ear[int(20 * FPS) : int(25 * FPS)] = 0.02
    neg = Clip("uta", "01_0", "01", Path("v"), "test", kind="0")
    pos = Clip("uta", "01_10", "01", Path("v"), "test", kind="10")
    assert negative_clip(neg) and not negative_clip(pos)
    scores = [score_clip(c, _arrays(n, ear=ear), cfg) for c in (neg, pos)]
    summary = summarize(scores)
    assert summary.false_alarms_per_h == pytest.approx(
        scores[0].drowsy_alerts / (scores[0].scored_s / 3600)
    )
    assert summary.negative_hours == pytest.approx(scores[0].scored_s / 3600)
    assert summary.get("events.nod.f1") != summary.get("events.nod.f1")  # NaN: no labels


# cache .npz อ่านกลับได้ค่าเดิม
def test_feature_cache_round_trip(tmp_path):
    csv = tmp_path / "c.csv"
    csv.write_text(
        "timestamp,face_found,ear_left,ear_right,mar,pitch,yaw,roll\n"
        "0.0,1,0.3,0.31,0.02,1.0,2.0,0.0\n0.033,0,,,,,,\n",
        encoding="utf-8",
    )
    first = load_features(csv)
    assert csv.with_suffix(".npz").is_file()
    again = load_features(csv)
    for key in first:
        np.testing.assert_array_equal(first[key], again[key])
    assert np.isnan(first["ear_left"][1]) and first["face_found"].tolist() == [1.0, 0.0]


# -- tuning ------------------------------------------------------------------------------


# key ใน grid ของ tune มีอยู่จริงใน config
def test_tuning_grid_keys_are_real_config_values():
    cfg = load_config()
    for group in GROUPS:
        for key, values in group.grid.items():
            section, name = key.split(".")
            assert isinstance(getattr(getattr(cfg, section), name), float), key
            assert values == tuple(sorted(values)), key
        keys = list(group.grid)
        for combo in itertools.product(*group.grid.values()):
            point = dict(zip(keys, combo, strict=True))
            if group.valid(point):
                load_config(overrides=nest(point))  # every grid point is a valid config
    assert nest({"yawn.mar_threshold": 0.3}) == {"yawn": {"mar_threshold": 0.3}}
    assert FALSE_ALARM_SLACK_PER_H > 0
    assert not any(
        k.split(".")[1] in ("microsleep_s", "sleep_s", "long_s") for g in GROUPS for k in g.grid
    )  # Euro NCAP timings are not tuned
