"""Catalog of evaluation clips: which video, which subject, which split, which labels.

Splits are by subject, never by frame:
- "tune": used to choose thresholds;
- "test": held out, only for the final numbers.
NTHU-DDD keeps its own split (training + evaluation sets = tune, testing set = test).
UTA-RLDD uses its own fold 1 as test. DMD and YawDD hold out every third subject.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

DATASETS = ("nthu", "dmd", "yawdd", "uta")  # extraction order: labelled per-frame data first
VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}


@dataclass(frozen=True)
class Clip:
    dataset: str
    clip_id: str  # unique and filesystem-safe; the feature CSV is <dataset>/<clip_id>.csv
    subject: str
    video: Path
    split: str  # "tune" or "test"
    condition: str = ""  # e.g. "night_glasses", "dash", "mirror"
    kind: str = ""  # behaviour / class, e.g. "yawning", "mix", "10", "Talking"
    labels: dict[str, Path] = field(default_factory=dict)  # label name -> file


def _norm(name: str) -> str:
    return name.lower().replace("_", "")


def _every_third(subjects: list[str]) -> set[str]:
    """Deterministic held-out subjects: every third one, in the order given."""
    return set(list(subjects)[1::3])


# -- NTHU-DDD ------------------------------------------------------------------------


def nthu(root: Path) -> Iterator[Clip]:
    base = root / "nthu-ddd"
    train = base / "Training_Evaluation_Dataset" / "Training Dataset"
    for video in sorted(train.glob("*/*/*.avi")):
        subject, scenario, behaviour = video.parent.parent.name, video.parent.name, video.stem
        labels = {
            kind: video.parent / f"{subject}_{behaviour}_{kind}.txt"
            for kind in ("drowsiness", "eye", "head", "mouth")
        }
        yield Clip(
            "nthu",
            f"train_{subject}_{_norm(scenario)}_{behaviour}",
            subject,
            video,
            "tune",
            _norm(scenario),
            behaviour,
            {k: p for k, p in labels.items() if p.is_file()},
        )

    evaluation = base / "Training_Evaluation_Dataset" / "Evaluation Dataset"
    for video in sorted(evaluation.glob("*/*_mix.mp4")):
        subject, scenario = video.stem.split("_", 1)[0], video.stem.split("_", 1)[1][:-4]
        labels = {}
        for path in video.parent.glob(f"{subject}_*_mixing_*.txt"):
            scen, kind = path.stem[len(subject) + 1 :].rsplit("_mixing_", 1)
            if _norm(scen) == _norm(scenario):
                labels[kind] = path
        yield Clip(
            "nthu",
            f"eval_{subject}_{_norm(scenario)}",
            subject,
            video,
            "tune",
            _norm(scenario),
            "mix",
            labels,
        )

    testing = base / "Testing_Dataset"
    label_files = sorted((testing / "test_label_txt").rglob("*_mixing_drowsiness.txt"))
    for video in sorted(testing.glob("*_mix.mp4")):
        subject, scenario = video.stem.split("_", 1)[0], video.stem.split("_", 1)[1][:-4]
        labels = {}
        for path in label_files:
            if path.parent.name == "wh":
                continue  # duplicate copies of the per-scenario folders
            stem = path.stem[: -len("_mixing_drowsiness")]
            subj, scen = stem.split("_", 1)
            if subj == subject and _norm(scen) == _norm(scenario):
                labels["drowsiness"] = path
        yield Clip(
            "nthu",
            f"test_{subject}_{_norm(scenario)}",
            subject,
            video,
            "test",
            _norm(scenario),
            "mix",
            labels,
        )


# -- DMD -----------------------------------------------------------------------------


def dmd(root: Path) -> Iterator[Clip]:
    base = root / "dmd"
    videos = sorted(base.glob("g*/*/s*/*_rgb_face.mp4"))
    test = _every_third(sorted({v.parent.parent.name for v in videos}, key=int))
    for video in videos:
        subject = video.parent.parent.name
        prefix = video.name[: -len("_rgb_face.mp4")]
        labels = {}
        for kind in ("drowsiness", "distraction"):
            path = video.parent / f"{prefix}_rgb_ann_{kind}.json"
            if path.is_file():
                labels[kind] = path
        kind = next(iter(labels), "")
        group = video.parent.parent.parent.name
        clip_id = re.sub(r"[^0-9A-Za-z_-]", "-", prefix)  # several recordings per session
        yield Clip(
            "dmd",
            clip_id,
            subject,
            video,
            "test" if subject in test else "tune",
            group,
            kind,
            labels,
        )


# -- YawDD ---------------------------------------------------------------------------

_YAWDD = re.compile(
    r"^(\d+)-(Female|Male)([A-Za-z]*?)(?:-(Normal|Talking|Yawning|Talking&Yawning))?(?:\.avi)?$",
    re.IGNORECASE,
)


def yawdd(root: Path) -> Iterator[Clip]:
    base = root / "yawdd"
    parsed = []
    for video in sorted(base.rglob("*")):
        if video.suffix.lower() != ".avi":
            continue
        m = _YAWDD.match(video.stem.strip())
        if m is None:
            continue
        number, gender, eyewear, behaviour = m.groups()
        if behaviour:
            behaviour = "TalkingYawning" if "&" in behaviour else behaviour.capitalize()
        camera = "dash" if "Dash" in video.parts else "mirror"
        parsed.append(
            (
                video,
                f"{gender[0]}{int(number):02d}",
                eyewear or "NoGlasses",
                camera,
                behaviour or "mixed",
            )
        )
    test = _every_third(sorted({p[1] for p in parsed}))
    for video, subject, eyewear, camera, behaviour in parsed:
        clip_id = f"{camera}_{subject}_{eyewear}_{behaviour}"
        yield Clip(
            "yawdd",
            clip_id,
            subject,
            video,
            "test" if subject in test else "tune",
            f"{camera}_{eyewear.lower()}",
            behaviour,
        )


# -- UTA-RLDD ------------------------------------------------------------------------


def uta(root: Path) -> Iterator[Clip]:
    base = root / "uta-rldd"
    for video in sorted(base.glob("*/*")):
        if video.suffix.lower() not in VIDEO_EXTS or not video.parent.name.isdigit():
            continue
        subject = video.parent.name
        fold = (int(subject) - 1) // 12 + 1  # the dataset's 5 folds of 12 subjects
        yield Clip(
            "uta",
            f"{int(subject):02d}_{video.stem}",
            subject,
            video,
            "test" if fold == 1 else "tune",
            f"fold{fold}",
            video.stem.split("_")[0],  # "10_1", "10_2": one drowsy session in two parts
        )


SOURCES = {"nthu": nthu, "dmd": dmd, "yawdd": yawdd, "uta": uta}


def catalog(root: Path, datasets: tuple[str, ...] = DATASETS) -> list[Clip]:
    return [clip for name in datasets for clip in SOURCES[name](root)]
