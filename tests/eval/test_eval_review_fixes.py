"""Regression tests for the milestone-5 code review (metric definitions and tuning rules)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from drowsiness.config import load_config
from eval.catalog import Clip, dmd
from eval.labels import dmd_truth
from eval.metrics import episodes, score_clip, summarize
from eval.replay import _condition_flags
from eval.tune import Group, run_group

FPS = 30.0


def _arrays(n, *, ear=0.3, mar=0.02, pitch=0.0):
    def col(v):
        return np.broadcast_to(np.asarray(v, dtype=float), (n,)).copy()

    return {
        "timestamp": np.arange(n) / FPS,
        "face_found": np.ones(n),
        "ear_left": col(ear),
        "ear_right": col(ear),
        "mar": col(mar),
        "pitch": col(pitch),
        "yaw": np.zeros(n),
        "roll": np.zeros(n),
    }


def _nthu(tmp_path, kind, **labels):
    files = {}
    for name, track in labels.items():
        path = tmp_path / f"{name}.txt"
        path.write_text("".join(map(str, track)), encoding="ascii")
        files[name] = path
    return Clip("nthu", "c", "001", Path("v.avi"), "tune", kind=kind, labels=files)


def _s(seconds: float) -> int:
    return int(round(seconds * FPS))


# -- alert counting ----------------------------------------------------------------------


def test_one_long_closure_is_one_alert_episode(tmp_path):
    n = _s(150)
    ear = np.full(n, 0.3)
    ear[_s(100) : _s(105)] = 0.02  # microsleep -> sleep: warning then alarm, one episode
    # (late in the clip so 5 s closed stays below the PERCLOS warning over the 60 s window)
    s = score_clip(_nthu(tmp_path, "sleepyCombination"), _arrays(n, ear=ear), load_config())
    assert s.drowsy_alerts == 1


def test_episodes_join_short_gaps():
    t = np.arange(10) / 1.0
    flag = np.array([1, 1, 0, 1, 0, 0, 0, 0, 1, 0], dtype=bool)
    assert episodes(flag, t, join_gap_s=3.0) == 2  # gap of 2 s joined, gap of 5 s not
    assert episodes(flag, t, join_gap_s=1.0) == 3
    assert episodes(np.zeros(5, dtype=bool), t[:5], 1.0) == 0


def test_perclos_alert_seen_while_already_drowsy(tmp_path):
    # Repeated 2 s closures every 8 s: microsleep keeps the state at DROWSY_WARNING (or
    # higher), so a PERCLOS warning never shows up as a state-change event of its own.
    n = _s(150)
    ear = np.full(n, 0.3)
    for start in range(20, 150, 8):
        ear[_s(start) : _s(start + 2)] = 0.02
    clip = Clip("uta", "01_10", "01", Path("v"), "tune", kind="10")
    s = score_clip(clip, _arrays(n, ear=ear), load_config())
    assert s.mean_perclos > load_config().perclos.warning
    assert s.perclos_alert


# -- answer keys must not depend on the tuned value -------------------------------------


def test_nod_events_do_not_depend_on_tuned_min_duration(tmp_path):
    n = _s(60)
    pitch = np.zeros(n)
    head = np.zeros(n, dtype=int)
    for start, length in ((20, 1.2), (30, 2.0), (40, 0.6)):
        pitch[_s(start) : _s(start + length)] = -25.0
        head[_s(start) : _s(start + length)] = 1
    clip = _nthu(tmp_path, "slowBlinkWithNodding", head=head)
    counts = {
        d: len(
            score_clip(
                clip,
                _arrays(n, pitch=pitch),
                load_config(overrides={"head_nod": {"min_duration_s": d}}),
            ).events["nod"]
        )
        for d in (0.5, 1.5)
    }
    assert counts == {0.5: 2, 1.5: 2}  # the 0.6 s nod is never an event; the others always are


def test_false_events_outside_labelled_span_are_not_counted(tmp_path):
    cfg = load_config(overrides={"yawn": {"mar_threshold": 0.4, "min_duration_s": 1.0}})
    n = _s(60)
    mar = np.full(n, 0.02)
    mar[_s(45) : _s(48)] = 0.7  # a wide mouth after the label file has ended
    mouth = np.zeros(_s(30), dtype=int)  # labels cover only the first 30 s
    s = score_clip(_nthu(tmp_path, "yawning", mouth=mouth), _arrays(n, mar=mar), cfg)
    assert s.yawns_detected == 1 and s.yawn_fp == 0


def test_closure_events_are_not_joined_across_an_opening(tmp_path):
    n = _s(40)
    ear = np.full(n, 0.3)
    eye = np.zeros(n, dtype=int)
    for a, b in ((20, 21), (21.4, 22.4)):  # two 1 s closures, 0.4 s apart: no microsleep
        ear[_s(a) : _s(b)] = 0.02
        eye[_s(a) : _s(b)] = 1
    s = score_clip(
        _nthu(tmp_path, "sleepyCombination", eye=eye), _arrays(n, ear=ear), load_config()
    )
    assert s.events["closure"] == []  # neither closure is >= microsleep_s on its own


# -- labels -----------------------------------------------------------------------------


def test_dmd_undefined_and_transition_frames_are_not_scored(tmp_path):
    actions = {
        "0": ("eyes_state/open", 0, 3),
        "1": ("eyes_state/closing", 4, 4),
        "2": ("eyes_state/close", 5, 6),
        "3": ("eyes_state/undefined", 7, 9),
    }
    ann = {
        "openlabel": {
            "frame_intervals": [{"frame_start": 0, "frame_end": 9}],
            "actions": {
                k: {"type": t, "frame_intervals": [{"frame_start": a, "frame_end": b}]}
                for k, (t, a, b) in actions.items()
            },
        }
    }
    path = tmp_path / "a.json"
    path.write_text(json.dumps(ann), encoding="utf-8")
    truth = dmd_truth(Clip("dmd", "x", "1", Path("v"), "tune", labels={"d": path}), 10)
    assert np.flatnonzero(truth.annotated["eyes_closed"]).tolist() == [0, 1, 2, 3, 5, 6]
    assert np.flatnonzero(truth.tracks["eyes_closed"]).tolist() == [5, 6]


def test_condition_flags_include_the_first_frame_of_each_run():
    t = np.arange(10) / 10.0
    duration = np.array([0, 0, 0, 0.1, 0.2, 0, 0, 0, 0.1, 0])  # runs start at frames 2 and 7
    flag = np.zeros(10, dtype=bool)
    _condition_flags(flag, duration, t)
    assert np.flatnonzero(flag).tolist() == [2, 3, 4, 7, 8]


def test_dmd_catalog_subjects_never_in_both_splits(tmp_path):
    for subject in ("1", "2", "5", "9", "10", "14"):
        folder = tmp_path / "dmd" / "gA" / subject / "s1"
        folder.mkdir(parents=True)
        (folder / f"gA_{subject}_s1_2019_rgb_face.mp4").write_bytes(b"")
        (folder / f"gA_{subject}_s1_2019_rgb_ann_drowsiness.json").write_text("{}")
    clips = list(dmd(tmp_path))
    tune = {c.subject for c in clips if c.split == "tune"}
    test = {c.subject for c in clips if c.split == "test"}
    assert tune and test and not tune & test
    assert test == {"2", "10"}  # every third subject in numeric order: 1 2 5 9 10 14
    assert all(c.kind == "drowsiness" for c in clips)


# -- tuning rules -----------------------------------------------------------------------


class _InlinePool:
    def imap_unordered(self, fn, jobs):
        return map(fn, jobs)


def test_run_group_caps_alert_rates_and_prefers_the_start_on_ties(tmp_path, monkeypatch):
    # A fake scorer: objective rises with x, alert rate jumps above the cap at x = 3.
    import eval.tune as tune

    def fake_score_points(points, clips, pool):
        return [[p["g.x"]] for p in points], []

    def fake_summarize(xs):
        (x,) = xs

        class S:
            false_alarms_per_h = 5.0 if x >= 3 else 1.0
            distraction_per_h = 0.0
            negative_hours = calm_hours = 1.0

            def get(self, key):
                return {"obj": min(x, 2.0)}[key]

        return S()

    monkeypatch.setattr(tune, "score_points", fake_score_points)
    monkeypatch.setattr(tune, "summarize", fake_summarize)
    monkeypatch.setattr(tune, "_plain", lambda s: {})
    group = Group("g", {"g.x": (1.0, 2.0, 2.5, 3.0, 4.0)}, (), lambda s: s.get("obj"), "obj")
    limits = {"false_alarms_per_h": 2.0, "distraction_per_h": 1.0}
    best, rows = run_group(group, {"g.x": 2.0}, [], _InlinePool(), limits)
    assert best == {"g.x": 2.0}  # 2.0 and 2.5 tie; 3.0 and 4.0 are over the cap
    assert [r["eligible"] for r in rows] == [True, True, True, False, False]


def test_run_group_guard_blocks_a_worse_guarded_metric(monkeypatch):
    import eval.tune as tune

    monkeypatch.setattr(tune, "score_points", lambda pts, c, p: ([[q["g.x"]] for q in pts], []))

    def fake_summarize(xs):
        (x,) = xs

        class S:
            false_alarms_per_h = distraction_per_h = 0.0

            def get(self, key):
                return {"obj": x, "recall": 1.0 - 0.1 * x}[key]

        return S()

    monkeypatch.setattr(tune, "summarize", fake_summarize)
    monkeypatch.setattr(tune, "_plain", lambda s: {})
    group = Group("g", {"g.x": (0.0, 1.0, 2.0)}, (), lambda s: s.get("obj"), "obj",
                  guards=("recall",))  # fmt: skip
    limits = {"false_alarms_per_h": 1.0, "distraction_per_h": 1.0}
    best, _ = run_group(group, {"g.x": 0.0}, [], _InlinePool(), limits)
    assert best == {"g.x": 0.0}  # higher objective, but recall would drop by 0.1 / 0.2


def test_summary_nan_objective_ranks_last():
    assert np.isnan(summarize([]).get("events.nod.f1"))
    assert summarize([]).false_alarms_per_h != pytest.approx(0.0)  # NaN, not zero


def test_run_group_objective_ignores_guard_clips_from_other_datasets(monkeypatch):
    # Guard clips (here NTHU) count for alert rates but not for a DMD-only objective.
    import eval.tune as tune

    class Score:
        def __init__(self, dataset, x):
            self.clip = Clip(dataset, "c", "1", Path("v"), "tune")
            self.x = x

    def fake_score_points(points, clips, pool):
        return [[Score("dmd", p["g.x"]), Score("nthu", -10 * p["g.x"])] for p in points], []

    def fake_summarize(xs):
        xs = list(xs)

        class S:
            false_alarms_per_h = distraction_per_h = 0.0

            def get(self, key):
                return sum(s.x for s in xs)

        return S()

    monkeypatch.setattr(tune, "score_points", fake_score_points)
    monkeypatch.setattr(tune, "summarize", fake_summarize)
    monkeypatch.setattr(tune, "_plain", lambda s: {})
    group = Group("g", {"g.x": (1.0, 2.0)}, ("dmd",), lambda s: s.get("obj"), "obj")
    limits = {"false_alarms_per_h": 1.0, "distraction_per_h": 1.0}
    best, rows = run_group(group, {"g.x": 1.0}, [], _InlinePool(), limits)
    assert best == {"g.x": 2.0}  # with the NTHU score mixed in, 1.0 would have won
    assert [r["objective"] for r in rows] == [1.0, 2.0]


def test_headline_dataset_filter_uses_only_that_dataset(monkeypatch):
    import eval.report as report

    class S:
        def __init__(self, xs):
            self.n = len(xs)

        def get(self, key):
            return float(self.n)

    def fake_score(dataset, split):
        class F:
            clip = Clip(dataset, "c", "1", Path("v"), split)

        return F()

    monkeypatch.setattr(report, "summarize", lambda xs: S(list(xs)))
    monkeypatch.setattr(report, "HEADLINE", (("all", "k", False), ("dmd only", "k", False, "dmd")))
    clips = [fake_score(d, sp) for d in ("dmd", "nthu", "nthu") for sp in ("tune", "test")]
    table = report.headline_table({"default": clips, "proposed": clips})
    assert "| all | 3.00 | 3.00 | 3.00 | 3.00 |" in table
    assert "| dmd only | 1.00 | 1.00 | 1.00 | 1.00 |" in table
