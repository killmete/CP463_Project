"""Stage 3: choose thresholds by grid search on the tune split (never the test split).

    python -m eval.tune --workers 16   # -> eval/reports/tuning.json, proposed_config.yaml

Groups are tuned one after another, each starting from the best values found so far
(eye closure first, because head-nod and PERCLOS depend on it). Every grid point is
replayed through the real DriverMonitor on every relevant tune clip. The result is a
*proposal*: default_config.yaml is never changed here. A person reviews the report and
copies values over by hand.

Not tuned, because they are definitions rather than detector settings:
- Euro NCAP timings (eyes.microsleep_s, eyes.sleep_s, distraction long/short);
- eyes.closed_ratio, the "80 %" of PERCLOS-P80. What is tuned instead is
  eyes.closed_ear_fraction, the estimate of the closed-eye EAR that P80 is measured
  against (calibration only ever sees open eyes, so that estimate is the uncertain part).

False-alarm guard: every group is also replayed on the same set of "guard" clips (all
non-drowsy and calm tune clips). A grid point is eligible only if its drowsiness and
distraction alert rates on that set stay within FALSE_ALARM_SLACK_PER_H of the
*original defaults*, so the slack never adds up across groups.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import multiprocessing as mp
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

from eval.catalog import Clip, catalog
from eval.extract import feature_path
from eval.metrics import ClipScore, Summary, calm_clip, negative_clip, score_clip, summarize

log = logging.getLogger("eval.tune")
REPORTS = Path(__file__).resolve().parent / "reports"

FALSE_ALARM_SLACK_PER_H = 1.0  # allowed rise in alert episodes per hour over the defaults
GUARD_DROP = 0.02  # a guarded metric may not fall more than this below the group's start
_RATES = ("false_alarms_per_h", "distraction_per_h")


@dataclass(frozen=True)
class Group:
    name: str
    grid: dict[str, tuple[float, ...]]  # dotted config key -> values to try
    datasets: tuple[str, ...]
    objective: Callable[[Summary], float]
    objective_text: str
    valid: Callable[[dict[str, float]], bool] = lambda _p: True
    guards: tuple[str, ...] = ()  # Summary keys that must not get worse (see GUARD_DROP)


def _mean(*values: float) -> float:
    finite = [v for v in values if v == v]  # drop NaN
    return sum(finite) / len(finite) if finite else float("nan")


GROUPS = (
    Group(
        "eyes",
        {"eyes.closed_ear_fraction": (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4)},
        ("dmd",),
        lambda s: s.get("frame.eyes_closed.f1"),
        "frame F1 for eyes closed (DMD: labels mark fully closed eyes)",
        guards=("events.closure.recall",),
    ),
    Group(
        "yawn",
        {
            "yawn.mar_threshold": (0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6),
            "yawn.min_duration_s": (0.5, 0.75, 1.0, 1.5),
        },
        ("nthu", "dmd", "yawdd"),
        lambda s: _mean(s.get("events.yawn.f1"), s.get("yawdd.balanced_acc")),
        "mean of yawn event F1 (NTHU, DMD) and YawDD clip balanced accuracy",
    ),
    Group(
        "head_nod",
        {
            "head_nod.pitch_delta_deg": (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0),
            "head_nod.min_duration_s": (0.5, 1.0, 1.5, 2.0),
        },
        ("nthu",),
        lambda s: s.get("events.nod.f1"),
        "nod event F1 (NTHU; labelled nods of at least 1 s)",
    ),
    Group(
        "distraction",
        {
            "distraction.yaw_deg": (15.0, 20.0, 25.0, 30.0, 40.0),
            "distraction.pitch_deg": (15.0, 20.0, 25.0, 35.0),
        },
        ("nthu", "dmd"),
        lambda s: s.get("frame.look_aside.f1"),
        "frame F1 for looking aside (NTHU, DMD)",
    ),
    Group(
        "perclos",
        {
            "perclos.warning": (0.08, 0.1, 0.12, 0.15, 0.2),
            "perclos.alarm": (0.15, 0.2, 0.25, 0.3, 0.4),
        },
        ("uta",),
        lambda s: s.get("uta.youden"),
        "UTA-RLDD: share of drowsy (10) videos with a PERCLOS alert, minus that of alert (0)",
        valid=lambda p: p["perclos.warning"] < p["perclos.alarm"],
    ),
)


def nest(flat: dict[str, float]) -> dict:
    """{'yawn.mar_threshold': 0.3} -> {'yawn': {'mar_threshold': 0.3}} (load_config overrides)."""
    out: dict = {}
    for key, value in flat.items():
        section, name = key.split(".")
        out.setdefault(section, {})[name] = value
    return out


def score_grid(job: tuple[Clip, Path, list[dict[str, float]]]) -> list[ClipScore] | str:
    """Score one clip under every grid point (features loaded once). Errors are returned
    as a message, so one bad file cannot stop a long run."""
    from drowsiness.config import load_config
    from eval.replay import load_features

    clip, path, points = job
    try:
        arrays = load_features(path)
        return [score_clip(clip, arrays, load_config(overrides=nest(p))) for p in points]
    except Exception as exc:
        return f"{clip.dataset}/{clip.clip_id}: {type(exc).__name__}: {exc}"


def score_points(points, clips, pool) -> tuple[list[list[ClipScore]], list[str]]:
    per_point: list[list[ClipScore]] = [[] for _ in points]
    errors: list[str] = []
    for result in pool.imap_unordered(score_grid, [(c, p, points) for c, p in clips]):
        if isinstance(result, str):
            errors.append(result)
            log.warning("skipped %s", result)
            continue
        for k, s in enumerate(result):
            per_point[k].append(s)
    return per_point, errors


def guard_clips(tune: list[tuple[Clip, Path]]) -> list[tuple[Clip, Path]]:
    return [(c, p) for c, p in tune if negative_clip(c) or calm_clip(c)]


def run_group(
    group: Group,
    base: dict[str, float],
    clips: list[tuple[Clip, Path]],
    pool,
    limits: dict[str, float],
) -> tuple[dict[str, float], list[dict]]:
    """Score every grid point; return the best eligible params and all rows.

    `limits` caps each alert rate (see _RATES), measured on every clip given (the guard
    clips included). The objective and guards only see the group's own datasets, so for
    example NTHU's drooping-eye labels on guard clips never enter the DMD eye objective.
    `base` holds the current values, and the point equal to it is the start: guards
    compare against it, and ties prefer it.
    """
    keys = list(group.grid)
    points = []
    for combo in itertools.product(*group.grid.values()):
        point = base | dict(zip(keys, combo, strict=True))
        if group.valid(point):
            points.append(point)
    per_point, errors = score_points(points, clips, pool)
    rows = []
    for point, scores in zip(points, per_point, strict=True):
        everything = summarize(scores)
        own = (
            summarize(s for s in scores if s.clip.dataset in group.datasets)
            if group.datasets
            else everything
        )
        rows.append(
            {
                "params": {k: point[k] for k in keys},
                "objective": group.objective(own),
                **{rate: getattr(everything, rate) for rate in _RATES},
                "guards": {g: own.get(g) for g in group.guards},
                "summary": _plain(own),
            }
        )
    start = next((r for r in rows if all(r["params"][k] == base[k] for k in keys)), None)

    def ok(row) -> bool:
        within = all(not row[rate] > limits[rate] for rate in _RATES)  # NaN rate: no data
        kept = start is None or all(
            not row["guards"][g] < start["guards"][g] - GUARD_DROP for g in group.guards
        )
        return within and kept

    def rank(row):  # best objective; on a tie keep the value closest to the current one
        obj = row["objective"]
        drift = sum(abs(row["params"][k] - base[k]) / max(abs(base[k]), 1e-9) for k in keys)
        return (round(obj, 3) if obj == obj else float("-inf"), -drift)

    eligible = [r for r in rows if ok(r)] or ([start] if start else [])
    if not eligible:
        raise ValueError(f"{group.name}: no grid point is within the caps and none is the start")
    for r in rows:
        r["eligible"] = r in eligible
    if errors:
        rows[0]["errors"] = errors
    return max(eligible, key=rank)["params"], rows


def _plain(summary: Summary) -> dict:
    return {
        "clips": summary.clips,
        "hours": summary.hours,
        "frame": summary.frame,
        "events": summary.events,
        "false_alarms_per_h": summary.false_alarms_per_h,
        "negative_hours": summary.negative_hours,
        "distraction_per_h": summary.distraction_per_h,
        "calm_hours": summary.calm_hours,
        "yawdd": summary.yawdd,
        "uta": summary.uta,
    }


def defaults() -> dict[str, float]:
    from drowsiness.config import load_config

    cfg = load_config()
    return {
        key: float(getattr(getattr(cfg, key.split(".")[0]), key.split(".")[1]))
        for g in GROUPS
        for key in g.grid
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--features", type=Path, default=Path("data/features"))
    parser.add_argument("--out", type=Path, default=REPORTS)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--groups", nargs="+", choices=[g.name for g in GROUPS])
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    tune = [(c, feature_path(args.features, c)) for c in catalog(args.data) if c.split == "tune"]
    missing = sum(not p.is_file() for _, p in tune)
    tune = [(c, p) for c, p in tune if p.is_file()]
    if missing:
        log.warning("%d tune clips have no features yet and are skipped", missing)
    guard = guard_clips(tune)

    start_values = defaults()
    current = dict(start_values)
    with mp.get_context("spawn").Pool(args.workers) as pool:
        (baseline_scores,), _ = score_points([start_values], guard, pool)
        baseline = summarize(baseline_scores)
        base_rates = {rate: getattr(baseline, rate) for rate in _RATES}
        if base_rates["false_alarms_per_h"] != base_rates["false_alarms_per_h"]:
            raise SystemExit("no non-drowsy tune clips: false alarms cannot be measured")
        if base_rates["distraction_per_h"] != base_rates["distraction_per_h"]:
            log.warning("no calm tune clips (UTA 0, YawDD Normal/Talking): distraction uncapped")
        limits = {rate: v + FALSE_ALARM_SLACK_PER_H for rate, v in base_rates.items()}
        log.info(
            "defaults on %d guard clips (%.1f h non-drowsy, %.1f h calm): %s",
            len(guard),
            baseline.negative_hours,
            baseline.calm_hours,
            {k: round(v, 2) for k, v in base_rates.items()},
        )
        report = {
            "start": start_values,
            "baseline": base_rates,
            "limits": limits,
            "slack_per_h": FALSE_ALARM_SLACK_PER_H,
            "guard_clips": len(guard),
            "skipped_clips": missing,
            "groups": {},
        }
        for group in GROUPS:
            if args.groups and group.name not in args.groups:
                continue
            ids = {(c.dataset, c.clip_id) for c, _ in guard}
            clips = guard + [
                (c, p)
                for c, p in tune
                if c.dataset in group.datasets and (c.dataset, c.clip_id) not in ids
            ]
            t0 = time.perf_counter()
            best, rows = run_group(group, current, clips, pool, limits)
            before = next(
                (r for r in rows if all(r["params"][k] == current[k] for k in best)), None
            )
            after = next(r for r in rows if r["params"] == best)
            log.info(
                "%s: %s  objective %s -> %.3f  (%d clips, %d points, %.0fs)",
                group.name,
                best,
                "n/a" if before is None else f"{before['objective']:.3f}",
                after["objective"],
                len(clips),
                len(rows),
                time.perf_counter() - t0,
            )
            current.update(best)
            report["groups"][group.name] = {
                "objective": group.objective_text,
                "datasets": group.datasets,
                "guards": group.guards,
                "clips": len(clips),
                "before": before,
                "best": after,
                "grid": rows,
            }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tuning.json").write_text(
        json.dumps(report, indent=1, default=float), encoding="utf-8"
    )
    changed = {k: v for k, v in current.items() if v != start_values[k]}
    header = (
        "# Proposed by eval/tune.py on the tune split. Review eval/reports/ before\n"
        "# copying any value into src/drowsiness/default_config.yaml.\n"
    )
    (args.out / "proposed_config.yaml").write_text(
        header + yaml.safe_dump(nest(changed), sort_keys=True), encoding="utf-8"
    )
    log.info("proposed changes: %s", changed or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
