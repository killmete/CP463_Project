"""Stage 1: run every catalogued clip through perception once and save per-frame features.

    python -m eval.extract --workers 10                 # all datasets, resumable
    python -m eval.extract --datasets nthu dmd --limit 5

Output: data/features/<dataset>/<clip_id>.csv (feature CSV v1, see docs/architecture.md),
plus data/features/manifest.jsonl with one line of stats per clip. A clip whose CSV
exists is skipped, so an interrupted run continues where it stopped. Each CSV is written
to a temporary name and renamed when complete, so a partial file is never mistaken for
a finished one. Each worker process builds a fresh detector per clip (MediaPipe's video
mode needs increasing timestamps). Logic is not run here, so thresholds can be tuned
later without repeating this step.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import time
from dataclasses import asdict
from pathlib import Path

from eval.catalog import DATASETS, Clip, catalog

log = logging.getLogger("eval.extract")


# ตั้งค่า log ของแต่ละ process ลูก
def _init_worker() -> None:
    logging.basicConfig(level=logging.WARNING)


# path ของ CSV ผลลัพธ์: <ชุดข้อมูล>/<clip_id>.csv
def feature_path(out_dir: Path, clip: Clip) -> Path:
    return out_dir / clip.dataset / f"{clip.clip_id}.csv"


# จำนวนเฟรมจาก header ของไฟล์ ใช้ตรวจว่าถอดรหัสวิดีโอได้ครบไหม
def video_frame_count(video: Path) -> int:
    """Frame count from the container header (0 if unknown); used to spot cut-short decodes."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()


# รัน perception กับ 1 คลิปแล้วเขียน CSV; ไม่ raise error คืนผลใน manifest แทน
def extract_clip(job: tuple[Clip, Path]) -> dict:
    """Run one clip; returns its manifest entry (never raises, errors are reported)."""
    from drowsiness.capture import VideoFileSource
    from drowsiness.config import load_config
    from drowsiness.feature_log import FeatureCsvWriter
    from drowsiness.perception import build_perception

    clip, out_dir = job
    target = feature_path(out_dir, clip)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".partial")
    entry = {"dataset": clip.dataset, "clip_id": clip.clip_id, "video": str(clip.video)}
    start = time.perf_counter()
    frames = faces = 0
    try:
        # A fresh detector per clip: MediaPipe's VIDEO mode needs increasing timestamps
        # and tracks the face from frame to frame, so clips must not share one.
        # เปิด perception, วิดีโอ, และ CSV ชั่วคราว (.partial) พร้อมกัน
        with (
            build_perception(load_config()) as perception,
            VideoFileSource(clip.video) as source,
            FeatureCsvWriter(partial) as writer,
        ):
            entry["fps"] = source.fps
            entry["expected_frames"] = video_frame_count(clip.video)
            while (packet := source.read()) is not None:
                out = perception.process(packet)
                writer.write(out, packet.index)
                frames += 1
                faces += out.features.face_found
        # เขียนเสร็จแล้วค่อยเปลี่ยนชื่อ: ไฟล์ที่ไม่ครบจะไม่ถูกนับว่าเสร็จ
        os.replace(partial, target)
        entry.update(ok=True, frames=frames, face_ratio=faces / frames if frames else 0.0)
    # วิดีโอเสียไม่ทำให้ทั้งการรันหยุด: ลบไฟล์ชั่วคราวแล้วบันทึก error
    except Exception as exc:  # a bad video must not stop the whole run
        partial.unlink(missing_ok=True)
        entry.update(ok=False, error=f"{type(exc).__name__}: {exc}", frames=frames)
    entry["seconds"] = round(time.perf_counter() - start, 1)
    return entry


# python -m eval.extract: หาคลิป -> ข้ามคลิปที่ทำแล้ว -> แบ่งงานหลาย process
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("data/features"))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--workers", type=int, default=max(1, min(10, (os.cpu_count() or 2) - 2)))
    parser.add_argument("--limit", type=int, help="at most N clips per dataset (smoke runs)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    clips = catalog(args.data, tuple(args.datasets))
    # --limit: เอาแค่ N คลิปต่อชุดข้อมูล (ทดลองรันเร็วๆ)
    if args.limit:
        per: dict[str, int] = {}
        clips = [
            c
            for c in clips
            if (per := per | {c.dataset: per.get(c.dataset, 0) + 1})[c.dataset] <= args.limit
        ]
    # ข้ามคลิปที่มี CSV แล้ว จึงรันต่อจากที่หยุดได้ (resumable)
    todo = [c for c in clips if not feature_path(args.out, c).is_file()]
    log.info(
        "%d clips, %d already done, %d to extract with %d workers",
        len(clips),
        len(clips) - len(todo),
        len(todo),
        args.workers,
    )
    if not todo:
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "catalog.jsonl").write_text(
        "".join(json.dumps(asdict(c), default=str) + "\n" for c in clips), encoding="utf-8"
    )
    failed = 0
    start = time.perf_counter()
    with (
        # maxtasksperchild: a fresh process now and then keeps MediaPipe's memory in check
        mp.get_context("spawn").Pool(
            args.workers, initializer=_init_worker, maxtasksperchild=20
        ) as pool,
        open(args.out / "manifest.jsonl", "a", encoding="utf-8") as manifest,
    ):
        jobs = [(c, args.out) for c in todo]
        # รับผลทีละคลิปเมื่อเสร็จ เขียน manifest และแสดงเวลาที่เหลือ (eta)
        for i, entry in enumerate(pool.imap_unordered(extract_clip, jobs), 1):
            manifest.write(json.dumps(entry) + "\n")
            manifest.flush()
            failed += not entry["ok"]
            elapsed = time.perf_counter() - start
            eta = elapsed / i * (len(todo) - i)
            status = f"faces {entry['face_ratio']:.0%}" if entry["ok"] else entry["error"]
            log.info(
                "[%d/%d] %s/%s %d frames in %.0fs (%s) - eta %.0f min",
                i,
                len(todo),
                entry["dataset"],
                entry["clip_id"],
                entry["frames"],
                entry["seconds"],
                status,
                eta / 60,
            )
    log.info("done in %.1f min, %d failed", (time.perf_counter() - start) / 60, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
