"""Stage 4: compare the shipped thresholds with the tuned proposal on both splits.

    python -m eval.report --workers 16      # writes eval/reports/m5-evaluation.md + figures/

Reads eval/reports/tuning.json and proposed_config.yaml (from eval.tune). Every clip with
features is replayed under both configs. The test split was never used for tuning, so
its numbers are the honest ones; tune-split numbers are shown only to expose overfitting.
Only aggregate numbers and charts are written: no frames, faces or per-person data.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from eval.catalog import Clip, catalog
from eval.extract import feature_path, video_frame_count
from eval.metrics import ClipScore, Summary, summarize
from eval.tune import FALSE_ALARM_SLACK_PER_H, GUARD_DROP, REPORTS, defaults, score_grid

log = logging.getLogger("eval.report")
# config ที่เทียบกัน, สีของกราฟ, ชื่อชุดข้อมูลที่แสดง
CONFIGS = ("default", "proposed")
# sequential blue ramp, light -> dark (validated reference palette)
_BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
_INK, _INK2, _GRID = "#0b0b0b", "#52514e", "#e4e3df"
_NAMES = {"nthu": "NTHU-DDD", "dmd": "DMD", "yawdd": "YawDD", "uta": "UTA-RLDD"}


# dict ซ้อน -> key แบบมีจุด (กลับด้านของ nest ใน tune)
def flatten(nested: dict) -> dict[str, float]:
    return {f"{sec}.{k}": float(v) for sec, vals in (nested or {}).items() for k, v in vals.items()}


# จัดรูปตัวเลขในตาราง (NaN -> –)
def _f(x: float, pct: bool = False) -> str:
    if x is None or x != x:
        return "–"
    return f"{x:.0%}" if pct else f"{x:.2f}"


# -- tables ----------------------------------------------------------------------------

# แถวของตารางผลหลัก: ชื่อแถว, key ใน Summary, แสดงเป็น % ไหม
HEADLINE = (  # (row label, Summary key, as percent[, only this dataset])
    # NTHU's eye label marks drooping eyes, not closed ones (see eval/README.md)
    ("Eyes closed, frame F1 (DMD)", "frame.eyes_closed.f1", False, "dmd"),
    ("Long closure caught (≥ microsleep_s, DMD)", "events.closure.recall", True, "dmd"),
    ("Yawn events: recall", "events.yawn.recall", True),
    ("Yawn events: precision", "events.yawn.precision", True),
    ("YawDD clips: yawning clips with a yawn", "yawdd.yawn_clip_tpr", True),
    ("YawDD clips: other clips with a yawn", "yawdd.other_clip_fpr", True),
    ("Head nod events: F1 (NTHU)", "events.nod.f1", False),
    ("Looking aside, frame F1", "frame.look_aside.f1", False),
    ("Long look-aside caught (≥ long_s)", "events.long_look_aside.recall", True),
    ("NTHU drowsy frames, F1", "frame.drowsy.f1", False),
    ("UTA PERCLOS AUC, drowsy vs alert", "uta.perclos_auc", False),
    ("UTA drowsy videos with a PERCLOS alert", "uta.drowsy_alert_rate", True),
    ("UTA alert videos with a PERCLOS alert", "uta.alert_false_rate", True),
    ("Drowsiness alert episodes/h, non-drowsy clips", "false_alarms_per_h", False),
    ("Distraction alert episodes/h, calm clips", "distraction_per_h", False),
)


# ตารางผลหลัก: default vs proposed บน tune และ test (ตัวเลข test คือตัวที่เชื่อได้)
def headline_table(scores: dict[str, list[ClipScore]]) -> str:
    cache: dict[tuple[str, str, str | None], Summary] = {}

    def summary(cfg: str, split: str, dataset: str | None) -> Summary:
        if (cfg, split, dataset) not in cache:
            cache[(cfg, split, dataset)] = summarize(
                s
                for s in scores[cfg]
                if s.clip.split == split and dataset in (None, s.clip.dataset)
            )
        return cache[(cfg, split, dataset)]

    lines = [
        "| Metric | Test: default | Test: proposed | Tune: default | Tune: proposed |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, key, pct, *only in HEADLINE:
        dataset = only[0] if only else None
        cells = [
            _f(summary(cfg, split, dataset).get(key), pct)
            for split in ("test", "tune")
            for cfg in CONFIGS
        ]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# คลิปที่ถอดรหัสได้ไม่ครบ (เฟรมน้อยกว่า header)
def _short(s: ClipScore) -> bool:
    """Fewer feature rows than the video header promises (a decode that stopped early)."""
    expected = video_frame_count(s.clip.video)
    return expected > 0 and s.n_frames < 0.98 * expected


# ตารางความครอบคลุม: จำนวนคลิป ชั่วโมง เฟรมที่เจอหน้า ต่อชุดข้อมูล
def coverage_table(scores: list[ClipScore], manifest: dict[str, dict]) -> str:
    lines = [
        "| Dataset | Split | Clips | Hours | Face found | Calibration fallback | "
        "Never calibrated | Cut short |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    groups: dict[tuple[str, str], list[ClipScore]] = defaultdict(list)
    for s in scores:
        groups[(s.clip.dataset, s.clip.split)].append(s)
    for (ds, split), items in sorted(groups.items()):
        hours = sum(s.duration_s for s in items) / 3600
        total = sum(s.duration_s for s in items)
        face = sum(s.face_ratio * s.duration_s for s in items) / total if total else 0.0
        lines.append(
            f"| {_NAMES[ds]} | {split} | {len(items)} | {hours:.1f} | {face:.0%} | "
            f"{sum(s.fallback for s in items)} | "
            f"{sum(s.scored_s == 0 for s in items)} | {sum(_short(s) for s in items)} |"
        )
    done = {f"{s.clip.dataset}/{s.clip.clip_id}" for s in scores}
    failed = sorted(k for k, e in manifest.items() if not e.get("ok") and k not in done)
    if failed:
        shown = ", ".join(failed[:10]) + (" …" if len(failed) > 10 else "")
        lines.append(f"\n{len(failed)} clip(s) have no features (extraction failed): {shown}")
    return "\n".join(lines)


# ตารางผลบน test แยกตามเงื่อนไขการถ่าย (เช่น แว่น/กลางคืน) ของ 1 ชุดข้อมูล
def condition_table(scores: list[ClipScore], dataset: str, keys) -> str:
    by: dict[str, list[ClipScore]] = defaultdict(list)
    for s in scores:
        if s.clip.dataset == dataset and s.clip.split == "test":
            by[s.clip.condition].append(s)
    if not by:
        return "_no test clips_"
    head = "| Condition | Clips | " + " | ".join(label for label, _, _ in keys) + " |"
    lines = [head, "|---|---:|" + "---:|" * len(keys)]
    for cond, items in sorted(by.items()):
        sm = summarize(items)
        lines.append(
            f"| {cond} | {len(items)} | "
            + " | ".join(_f(sm.get(k), pct) for _, k, pct in keys)
            + " |"
        )
    return "\n".join(lines)


# อัตราเตือนผิดบน test และเตือนถ้าค่าที่เสนอเกินเพดานที่ใช้ตอน tune
def test_rates(scores: dict[str, list[ClipScore]], names: list[str]) -> tuple[str, list[str]]:
    """Alert rates on held-out non-drowsy/calm clips for every config, and warnings
    where the proposal breaks the tuning cap (defaults + slack) on the test split."""
    summ = {n: summarize(s for s in scores[n] if s.clip.split == "test") for n in names}
    base = summ["default"]
    lines = [
        f"Test split: {base.negative_hours:.1f} h non-drowsy, {base.calm_hours:.1f} h calm.",
        "",
        "| Config | Drowsiness alerts/h | Distraction alerts/h |",
        "|---|---:|---:|",
    ]
    for n in names:
        label = n if n in CONFIGS else f"default + `{n}` only"
        lines.append(
            f"| {label} | {_f(summ[n].false_alarms_per_h)} | {_f(summ[n].distraction_per_h)} |"
        )
    warnings = []
    for rate, text in (
        ("false_alarms_per_h", "drowsiness"),
        ("distraction_per_h", "distraction"),
    ):
        before, after = getattr(base, rate), getattr(summ["proposed"], rate)
        if after > before + FALSE_ALARM_SLACK_PER_H:
            warnings.append(
                f"on the test split the proposal raises {text} alert episodes from "
                f"{before:.2f}/h to {after:.2f}/h, more than the {FALSE_ALARM_SLACK_PER_H:g}/h "
                "tuning allows. See the per-change table below before accepting it."
            )
    return "\n".join(lines), warnings


# ค่าที่เลือกอยู่ขอบ grid: ค่าที่ดีกว่าอาจอยู่นอกช่วงที่ลอง
def grid_edges(tuning: dict, base: dict[str, float]) -> list[str]:
    """Proposed values at the edge of their grid: the best value may lie beyond it."""
    out = []
    for name, g in tuning["groups"].items():
        for key, value in g["best"]["params"].items():
            tried = sorted({r["params"][key] for r in g["grid"]})
            if value != base[key] and value in (tried[0], tried[-1]):
                out.append(
                    f"`{key}` = {value:g} is at the edge of the values tried "
                    f"({tried[0]:g}–{tried[-1]:g}) in group {name}."
                )
    return out


# ตารางสรุปการ tune แต่ละกลุ่ม: ค่าเริ่ม, ค่าที่เสนอ, คะแนน, อัตราเตือน
def tuning_table(tuning: dict) -> str:
    lines = [
        "| Group | Objective (tune split) | Clips | Start | Proposed | Score "
        "| Drowsiness alerts/h | Distraction alerts/h |",
        "|---|---|---:|---|---|---|---|---|",
    ]
    for name, g in tuning["groups"].items():
        before, best = g["before"] or {}, g["best"]

        def fmt(params):
            return ", ".join(f"`{k}` {v:g}" for k, v in params.items())

        lines.append(
            f"| {name} | {g['objective']} | {g['clips']} | {fmt(before.get('params', {}))} | "
            f"{fmt(best['params'])} | {_f(before.get('objective'))} → {_f(best['objective'])} | "
            f"{_f(before.get('false_alarms_per_h'))} → {_f(best['false_alarms_per_h'])} | "
            f"{_f(before.get('distraction_per_h'))} → {_f(best['distraction_per_h'])} |"
        )
    return "\n".join(lines)


# -- figures ---------------------------------------------------------------------------


# รูปแบบกราฟให้เหมือนกันทุกรูป
def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.tick_params(colors=_INK2, labelsize=9)


# กราฟผล grid search: 1 ค่า = กราฟเส้น, 2 ค่า = heatmap ของ objective
def grid_figure(name: str, group: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    rows = group["grid"]
    keys = list(rows[0]["params"])
    if len(keys) != 2:
        xs = [r["params"][keys[0]] for r in rows]
        fig, ax = plt.subplots(figsize=(5, 3), dpi=150)
        ax.plot(xs, [r["objective"] for r in rows], color=_BLUES[4], lw=2, marker="o", ms=6)
        ax.set_xlabel(keys[0], color=_INK2)
        ax.set_ylabel("objective", color=_INK2)
    else:
        xv = sorted({r["params"][keys[1]] for r in rows})
        yv = sorted({r["params"][keys[0]] for r in rows})
        z = np.full((len(yv), len(xv)), np.nan)
        ok = np.zeros_like(z, dtype=bool)
        for r in rows:
            i, j = yv.index(r["params"][keys[0]]), xv.index(r["params"][keys[1]])
            z[i, j], ok[i, j] = r["objective"], r.get("eligible", True)
        cmap = LinearSegmentedColormap.from_list("blues", _BLUES)
        cmap.set_bad("#f0efec")
        fig, ax = plt.subplots(figsize=(1.1 * len(xv) + 2, 0.5 * len(yv) + 1.6), dpi=150)
        im = ax.imshow(np.ma.masked_invalid(z), cmap=cmap, aspect="auto", origin="lower")
        best = group["best"]["params"]
        for i, y in enumerate(yv):
            for j, x in enumerate(xv):
                if np.isnan(z[i, j]):
                    continue
                dark = im.norm(z[i, j]) > 0.55
                mark = "" if ok[i, j] else " ×"
                pick = best[keys[0]] == y and best[keys[1]] == x
                ax.text(
                    j,
                    i,
                    f"{z[i, j]:.2f}{mark}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if dark else _INK,
                    weight="bold" if pick else "normal",
                )
                if pick:
                    ax.add_patch(
                        plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=_INK, lw=1.5)
                    )
        ax.set_xticks(range(len(xv)), [f"{v:g}" for v in xv])
        ax.set_yticks(range(len(yv)), [f"{v:g}" for v in yv])
        ax.set_xlabel(keys[1], color=_INK2)
        ax.set_ylabel(keys[0], color=_INK2)
        cb = fig.colorbar(im, ax=ax, shrink=0.8)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=8, colors=_INK2)
    _style(ax)
    ax.set_title(f"{name}: {group['objective']}", fontsize=9, color=_INK, loc="left", wrap=True)
    fig.tight_layout()
    fig.savefig(path, facecolor="#fcfcfb")
    plt.close(fig)


# กราฟ PERCLOS ของวิดีโอ UTA ตื่น vs ง่วง
def uta_figure(scores: list[ClipScore], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kinds = ("0", "5", "10")
    names = ("alert (0)", "low vigilance (5)", "drowsy (10)")
    data = [
        [
            s.mean_perclos
            for s in scores
            if s.clip.dataset == "uta" and s.clip.kind == k and s.mean_perclos == s.mean_perclos
        ]
        for k in kinds
    ]
    fig, ax = plt.subplots(figsize=(5, 3.2), dpi=150)
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data):
        ax.scatter(
            i + rng.uniform(-0.15, 0.15, len(vals)),
            vals,
            s=14,
            color=_BLUES[4],
            alpha=0.6,
            linewidths=0,
        )
        if vals:
            ax.hlines(np.median(vals), i - 0.3, i + 0.3, color=_INK, lw=2)
    ax.set_xticks(range(3), [f"{n}\nn={len(d)}" for n, d in zip(names, data, strict=True)])
    ax.set_ylabel("mean PERCLOS-P80 per video", color=_INK2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", color=_GRID, lw=0.8)
    ax.set_axisbelow(True)
    _style(ax)
    ax.set_title(
        "UTA-RLDD, tune and test videos: PERCLOS by self-reported state\n"
        "(proposed config; bar = median)",
        fontsize=9,
        color=_INK,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(path, facecolor="#fcfcfb")
    plt.close(fig)


# -- main ------------------------------------------------------------------------------


# python -m eval.report: replay ทุกคลิปด้วย default, proposed และทีละค่าที่เปลี่ยน
# แล้วเขียน m5-evaluation.md + รูป (มีแค่ตัวเลขรวม ไม่มีภาพหน้าหรือข้อมูลรายคน)
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--features", type=Path, default=Path("data/features"))
    parser.add_argument("--reports", type=Path, default=REPORTS)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    tuning = json.loads((args.reports / "tuning.json").read_text(encoding="utf-8"))
    proposed_text = (args.reports / "proposed_config.yaml").read_text(encoding="utf-8")
    base = {k: float(v) for k, v in tuning["start"].items()}  # what tuning started from
    if base != defaults():
        log.warning(
            "default_config.yaml no longer matches the values tuning started from; "
            "'default' below means the tuning start values %s",
            base,
        )
    proposed = flatten(yaml.safe_load(proposed_text) or {})
    names = [*CONFIGS, *proposed]  # default, proposed, then each change on its own
    points = [base, base | proposed] + [base | {k: v} for k, v in proposed.items()]

    clips: list[tuple[Clip, Path]] = [
        (c, feature_path(args.features, c)) for c in catalog(args.data)
    ]
    clips = [(c, p) for c, p in clips if p.is_file()]
    log.info("replaying %d clips under %d configs", len(clips), len(points))
    scores: dict[str, list[ClipScore]] = {name: [] for name in names}
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for result in pool.imap_unordered(score_grid, [(c, p, points) for c, p in clips]):
            if isinstance(result, str):
                log.warning("skipped %s", result)
                continue
            for name, s in zip(names, result, strict=True):
                scores[name].append(s)

    manifest: dict[str, dict] = {}
    mpath = args.features / "manifest.jsonl"
    if mpath.is_file():
        for line in mpath.read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            manifest[f"{e['dataset']}/{e['clip_id']}"] = e  # latest entry wins

    figs = args.reports / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    for name, group in tuning["groups"].items():
        grid_figure(name, group, figs / f"tune-{name}.png")
    uta_figure(scores["proposed"], figs / "uta-perclos.png")

    nthu_keys = (
        ("Drowsy frames F1", "frame.drowsy.f1", False),
        ("False alarms/h", "false_alarms_per_h", False),
    )
    dmd_keys = (
        ("Eyes closed F1", "frame.eyes_closed.f1", False),
        ("Yawn recall", "events.yawn.recall", True),
        ("Look-aside F1", "frame.look_aside.f1", False),
    )
    yawdd_keys = (
        ("Yawning clips with a yawn", "yawdd.yawn_clip_tpr", True),
        ("Other clips with a yawn", "yawdd.other_clip_fpr", True),
    )
    prop = scores["proposed"]
    rate_md, warnings = test_rates(scores, names)
    warnings += grid_edges(tuning, base)
    warning_md = "\n>\n".join(f"> **Warning:** {w}" for w in warnings)
    changes = (
        "\n".join(
            f"- `{k}`: {base[k]:g} → **{v:g}**"
            for k, v in flatten(yaml.safe_load(proposed_text)).items()
        )
        or "- none"
    )
    figure_md = "\n\n".join(f"![{n}](figures/tune-{n}.png)" for n in tuning["groups"])
    text = f"""# Milestone 5 evaluation

Generated by `python -m eval.report` from `eval/reports/tuning.json`. Every number comes
from replaying extracted features through the real `DriverMonitor`. Frames before
calibration ends are not scored. Splits are by subject: see `eval/catalog.py`.

## Proposed threshold changes (not applied)

{changes}

These values came from a grid search on the **tune** split (`python -m eval.tune`). A
point was eligible only if two alert rates stayed within {FALSE_ALARM_SLACK_PER_H:g} per hour of the
original defaults, both measured on the same {tuning["guard_clips"]} non-drowsy and calm tune
clips:
- drowsiness alert episodes, defaults {tuning["baseline"]["false_alarms_per_h"]:.2f}/h;
- distraction alert episodes, defaults {tuning["baseline"]["distraction_per_h"]:.2f}/h.

For eye closure, long-closure recall also had to stay within {GUARD_DROP:g} of its starting
value. `src/drowsiness/default_config.yaml` is unchanged until a person accepts them.
Head-nod values have no held-out check: NTHU-DDD's testing set has no nod labels.

{warning_md}

## Alert rates on the test split, one change at a time

Same rule as in tuning, checked on the held-out non-drowsy and calm clips. Each
"only" row applies one proposed change to the defaults. These are few hours of video,
so a handful of episodes moves the rate a lot.

{rate_md}

## Headline results

{headline_table(scores)}

"–" means the split has no labels for that metric. For example, NTHU-DDD's testing set only
has drowsiness labels, and nod labels exist only in its training set.

## Tuning

{tuning_table(tuning)}

{figure_md}

Cells marked × were not eligible: an alert rate went over its cap, or a guarded
metric got worse. The outlined cell is the proposal.

## By condition (test split, proposed config)

**NTHU-DDD** (active-IR video; `night*` clips are 15 fps)

{condition_table(prop, "nthu", nthu_keys)}

**DMD** (session group)

{condition_table(prop, "dmd", dmd_keys)}

**YawDD** (camera position and eyewear)

{condition_table(prop, "yawdd", yawdd_keys)}

![UTA PERCLOS](figures/uta-perclos.png)

## Data coverage

{coverage_table(prop, manifest)}
"""
    out = args.reports / "m5-evaluation.md"
    out.write_text(text, encoding="utf-8")
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
