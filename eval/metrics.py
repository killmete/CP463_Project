"""Stage 2b: score one replayed clip against its labels, and summarize many clips.

Frame-level scores (precision / recall / F1) compare logic's per-frame conditions with
per-frame labels. Event-level scores ask whether each labelled event (a yawn, a long
eye closure, a nod, a long look-aside) was caught, and how fast. False alarms are
drowsiness alert *episodes* on clips where the driver is not drowsy: a stretch of
frames with any drowsiness reason active, with gaps shorter than alerts.rearm_s joined
(one long closure that escalates warning -> alarm is one episode). Frames before
calibration ends are never scored (logic is still learning the driver's baseline).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from drowsiness.config import Config
from eval.catalog import Clip
from eval.labels import intervals, truth_for
from eval.replay import load_features, replay, to_frames

FRAME_LABELS = {  # label -> Trace attribute with logic's matching per-frame condition
    "eyes_closed": "closed",
    "look_aside": "off_road",
    "nod": "nod",
    "drowsy": "drowsy_reason",
}
YAWN_MATCH_S = (-1.0, 2.5)  # a yawn counted from 1 s before to 2.5 s after the label matches
EVENT_GRACE_S = 1.0  # an alert up to this long after a labelled event ends still catches it
MIN_YAWN_S = 0.5  # shorter labelled "yawns" are not yawns for this metric
# Labelled nods at least this long are events. Fixed, not cfg.head_nod.min_duration_s:
# that value is tuned, and the answer key must not move with the grid point.
MIN_NOD_S = 1.0


def negative_clip(clip: Clip) -> bool:
    """Clips where nobody is drowsy: any drowsiness alert there is a false alarm."""
    return (
        (clip.dataset == "nthu" and clip.kind == "nonsleepyCombination")
        or (clip.dataset == "yawdd" and clip.kind in ("Normal", "Talking"))
        or (clip.dataset == "uta" and clip.kind == "0")
        or (clip.dataset == "dmd" and clip.kind == "distraction")
    )


def calm_clip(clip: Clip) -> bool:
    """Clips without gaze labels where the driver should not look away for long."""
    return (clip.dataset == "uta" and clip.kind == "0") or (
        clip.dataset == "yawdd" and clip.kind in ("Normal", "Talking")
    )


@dataclass
class ClipScore:
    clip: Clip
    duration_s: float
    scored_s: float  # after calibration
    fallback: bool
    face_ratio: float
    n_frames: int = 0
    frame: dict[str, np.ndarray] = field(default_factory=dict)  # label -> [tp, fp, fn, tn]
    events: dict[str, list[float | None]] = field(default_factory=dict)  # label -> delays/None
    event_fp: dict[str, int] = field(default_factory=dict)  # label -> detections with no event
    yawns_detected: int = 0
    yawn_fp: int = 0
    drowsy_alerts: int = 0  # drowsiness alert episodes after calibration
    distraction_alerts: int = 0  # distraction alert episodes after calibration
    mean_perclos: float = float("nan")
    drowsy_frac: float = float("nan")
    perclos_alert: bool = False  # PERCLOS warning/alarm active at any time after calibration


def _caught(flag: np.ndarray, t: np.ndarray, start: float, end: float) -> float | None:
    """Delay from `start` to the first flagged frame in [start, end + grace], or None."""
    window = (t >= start) & (t <= end + EVENT_GRACE_S) & flag
    hits = np.flatnonzero(window)
    return float(t[hits[0]] - start) if hits.size else None


def _false_onsets(flag, t, scored, labelled_ivs) -> int:
    """Detections that start on a scored frame and near no labelled event.

    Rising edges are taken on the whole flag, so a rule already on when a scored
    stretch begins (end of calibration, end of an unlabelled stretch) is not a new
    detection.
    """
    rising = np.flatnonzero(flag & ~np.concatenate(([False], flag[:-1])))
    return sum(
        bool(scored[i])
        and not any(a - EVENT_GRACE_S <= t[i] <= b + EVENT_GRACE_S for a, b in labelled_ivs)
        for i in rising
    )


def episodes(flag: np.ndarray, t: np.ndarray, join_gap_s: float) -> int:
    """Runs of `flag`; runs separated by less than join_gap_s count once."""
    if not flag.any():
        return 0
    padded = np.concatenate(([False], flag, [False])).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    starts, ends = edges[::2], edges[1::2] - 1
    gaps = t[starts[1:]] - t[ends[:-1]]
    return 1 + int((gaps >= join_gap_s).sum())


def score_clip(clip: Clip, arrays: dict[str, np.ndarray], cfg: Config) -> ClipScore:
    frames = to_frames(arrays)
    tr = replay(frames, cfg)
    t = tr.t
    n = len(t)
    fps = 1.0 / float(np.median(np.diff(t))) if n > 1 else 30.0
    start = tr.calibrated_t if tr.calibrated_t is not None else float("inf")
    valid = t >= start
    face = arrays["face_found"] == 1.0
    score = ClipScore(
        clip, n / fps, float(valid.sum()) / fps, tr.fallback, float(face.mean()) if n else 0.0, n
    )
    truth = truth_for(clip, n)

    for label, attr in FRAME_LABELS.items():
        if not truth.has(label):
            continue
        gt, pred = truth.tracks[label], getattr(tr, attr)
        m = valid & truth.annotated[label]
        score.frame[label] = np.array(
            [
                (pred & gt & m).sum(),
                (pred & ~gt & m).sum(),
                (~pred & gt & m).sum(),
                (~pred & ~gt & m).sum(),
            ]
        )

    def labelled(label: str, min_s: float, merge_gap_s: float) -> list[tuple[float, float]]:
        ivs = intervals(truth.tracks[label] & truth.annotated[label], fps, merge_gap_s)
        return [(a, b) for a, b in ivs if b - a >= min_s and a >= start]

    event_rules = (
        # A closure must be continuous to be a microsleep, so closures are never joined.
        ("closure", "eyes_closed", tr.reason("microsleep") | tr.reason("sleep"),
         cfg.eyes.microsleep_s, 0.0),
        ("nod", "nod", tr.reason("head_nod"), MIN_NOD_S, 0.5),
        ("long_look_aside", "look_aside", tr.reason("long_distraction"),
         cfg.distraction.long_s, 0.5),
    )  # fmt: skip
    for label, source, flag, min_s, gap in event_rules:
        if not truth.has(source):
            continue
        score.events[label] = [_caught(flag, t, a, b) for a, b in labelled(source, min_s, gap)]
        every = intervals(truth.tracks[source], fps, merge_gap_s=gap)
        score.event_fp[label] = _false_onsets(flag, t, valid & truth.annotated[source], every)

    yawn_times = [y for y in tr.yawn_times if y >= start]
    score.yawns_detected = len(yawn_times)
    if truth.has("yawn"):
        known = truth.annotated["yawn"]
        yawn_times = [y for y in yawn_times if known[min(int(np.searchsorted(t, y)), n - 1)]]
        used: set[int] = set()
        delays: list[float | None] = []
        for a, b in labelled("yawn", MIN_YAWN_S, 0.5):
            match = next(
                (
                    k
                    for k, y in enumerate(yawn_times)
                    if k not in used and a + YAWN_MATCH_S[0] <= y <= b + YAWN_MATCH_S[1]
                ),
                None,
            )
            if match is not None:
                used.add(match)
            delays.append(None if match is None else yawn_times[match] - a)
        score.events["yawn"] = delays
        # A leftover yawn is false only if it is near no labelled yawn at all (including
        # labels too short to score, or that started during calibration).
        every = intervals(truth.tracks["yawn"] & known, fps, 0.5)
        score.yawn_fp = score.event_fp["yawn"] = sum(
            not any(a + YAWN_MATCH_S[0] <= y <= b + YAWN_MATCH_S[1] for a, b in every)
            for k, y in enumerate(yawn_times)
            if k not in used
        )

    rearm = cfg.alerts.rearm_s
    score.drowsy_alerts = episodes(tr.drowsy_reason & valid, t, rearm)
    distracted = tr.reason("long_distraction") | tr.reason("short_distraction")
    score.distraction_alerts = episodes(distracted & valid, t, rearm)
    perclos_on = tr.reason("perclos_warning") | tr.reason("perclos_alarm")
    score.perclos_alert = bool((perclos_on & valid).any())
    if valid.any():
        score.mean_perclos = (
            float(np.nanmean(tr.perclos[valid]))
            if np.isfinite(tr.perclos[valid]).any()
            else float("nan")
        )
        score.drowsy_frac = float(tr.drowsy_reason[valid].mean())
    return score


def score(clip: Clip, features_csv, cfg: Config) -> ClipScore:
    return score_clip(clip, load_features(features_csv), cfg)


# -- summaries -----------------------------------------------------------------------


def prf(counts) -> dict[str, float]:
    tp, fp, fn = (float(x) for x in counts[:3])
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else (0.0 if tp + fn else float("nan"))
    return {"precision": p, "recall": r, "f1": f1, "support": tp + fn}


def auc(pos: list[float], neg: list[float]) -> float:
    """P(score of a random positive > a random negative), ties count half."""
    pos_a, neg_a = np.asarray(pos, float), np.asarray(neg, float)
    pos_a, neg_a = pos_a[np.isfinite(pos_a)], neg_a[np.isfinite(neg_a)]
    if not pos_a.size or not neg_a.size:
        return float("nan")
    diff = pos_a[:, None] - neg_a[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


YAWDD_POSITIVE = ("Yawning", "TalkingYawning", "mixed")  # dash clips contain yawns too


@dataclass
class Summary:
    clips: int
    hours: float
    frame: dict[str, dict[str, float]]  # label -> precision/recall/f1/support
    events: dict[str, dict[str, float]]  # label -> recall/precision/f1/events/median_delay_s
    false_alarms_per_h: float  # drowsiness alert episodes per hour on non-drowsy clips
    negative_hours: float
    distraction_per_h: float  # distraction alert episodes per hour on calm clips
    calm_hours: float
    yawdd: dict[str, float]  # clip level: a yawning clip should count >= 1 yawn, others 0
    uta: dict[str, float]  # alert (0) vs drowsy (10) videos

    def get(self, key: str) -> float:
        """Dotted lookup used by tuning objectives, e.g. 'events.yawn.f1'."""
        node = self
        for part in key.split("."):
            if isinstance(node, float):
                return float("nan")  # e.g. no labels of that kind in these clips
            if isinstance(node, dict):
                node = node.get(part, float("nan"))
            else:
                node = getattr(node, part)
        return float(node)


def summarize(scores: Iterable[ClipScore]) -> Summary:
    scores = list(scores)
    frame_counts: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(4))
    caught: dict[str, list[float | None]] = defaultdict(list)
    false_events: dict[str, int] = defaultdict(int)
    alerts = neg_s = distractions = calm_s = 0.0
    yawdd_hit: dict[bool, list[bool]] = {True: [], False: []}
    uta_perclos: dict[str, list[float]] = {"0": [], "10": []}
    uta_alert: dict[str, list[bool]] = {"0": [], "10": []}
    for s in scores:
        for label, c in s.frame.items():
            frame_counts[label] += c
        for label, delays in s.events.items():
            caught[label] += delays
            false_events[label] += s.event_fp.get(label, 0)
        if negative_clip(s.clip):
            alerts += s.drowsy_alerts
            neg_s += s.scored_s
        if calm_clip(s.clip):
            distractions += s.distraction_alerts
            calm_s += s.scored_s
        if s.clip.dataset == "yawdd" and s.scored_s > 0:
            yawdd_hit[s.clip.kind in YAWDD_POSITIVE].append(s.yawns_detected > 0)
        if s.clip.dataset == "uta" and s.clip.kind in uta_perclos and s.scored_s > 0:
            uta_perclos[s.clip.kind].append(s.mean_perclos)
            uta_alert[s.clip.kind].append(s.perclos_alert)
    events = {}
    for label, delays in caught.items():
        hits = [d for d in delays if d is not None]
        counts = prf((len(hits), false_events[label], len(delays) - len(hits)))
        events[label] = {
            **counts,
            "events": float(len(delays)),
            "median_delay_s": float(np.median(hits)) if hits else float("nan"),
        }
    tpr = float(np.mean(yawdd_hit[True])) if yawdd_hit[True] else float("nan")
    fpr = float(np.mean(yawdd_hit[False])) if yawdd_hit[False] else float("nan")
    u_tpr = float(np.mean(uta_alert["10"])) if uta_alert["10"] else float("nan")
    u_fpr = float(np.mean(uta_alert["0"])) if uta_alert["0"] else float("nan")
    return Summary(
        clips=len(scores),
        hours=sum(s.duration_s for s in scores) / 3600,
        frame={k: prf(v) for k, v in frame_counts.items()},
        events=events,
        false_alarms_per_h=alerts / (neg_s / 3600) if neg_s else float("nan"),
        negative_hours=neg_s / 3600,
        distraction_per_h=distractions / (calm_s / 3600) if calm_s else float("nan"),
        calm_hours=calm_s / 3600,
        yawdd={
            "yawn_clip_tpr": tpr,
            "other_clip_fpr": fpr,
            "balanced_acc": (tpr + 1 - fpr) / 2,
            "clips": float(len(yawdd_hit[True]) + len(yawdd_hit[False])),
        },
        uta={
            "perclos_auc": auc(uta_perclos["10"], uta_perclos["0"]),
            "drowsy_alert_rate": u_tpr,
            "alert_false_rate": u_fpr,
            "youden": u_tpr - u_fpr,
            "videos": float(len(uta_alert["0"]) + len(uta_alert["10"])),
        },
    )
