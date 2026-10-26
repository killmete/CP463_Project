"""Main processing loop and command-line entry point.

Milestone 1: capture -> display with FPS and latency overlay. Perception, logic and
alerts plug into `run_loop` in later milestones; no detection rules belong here.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from drowsiness.capture import CameraError, FrameSource, open_source
from drowsiness.config import Config, ConfigError, load_config

log = logging.getLogger("drowsiness")

WINDOW_NAME = "Drowsiness Detection"
_QUIT_KEYS = {ord("q"), 27}  # q, Esc


@dataclass(frozen=True, slots=True)
class RunStats:
    frames: int
    duration_s: float
    mean_fps: float
    latency_p50_ms: float
    latency_p95_ms: float
    dropped_frames: int
    interrupted: bool = False


class FpsMeter:
    """Frames per second over a rolling time window (wall clock)."""

    def __init__(self, window_s: float) -> None:
        self.window_s = window_s
        self._times: deque[float] = deque()

    def tick(self, now: float) -> float:
        self._times.append(now)
        while now - self._times[0] > self.window_s:
            self._times.popleft()
        span = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / span if span > 0 else 0.0


def draw_overlay(image: np.ndarray, fps: float, latency_ms: float, budget_ms: float) -> None:
    over_budget = latency_ms > budget_ms
    color = (0, 0, 255) if over_budget else (0, 200, 0)  # BGR
    text = f"FPS {fps:5.1f} | latency {latency_ms:5.1f} ms"
    cv2.rectangle(image, (0, 0), (300, 28), (0, 0, 0), thickness=-1)
    cv2.putText(image, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def run_loop(
    source: FrameSource, cfg: Config, display: bool, max_frames: int | None = None
) -> RunStats:
    """Process frames from an opened source until it ends, max_frames, or the user quits.

    A live source that fails raises (e.g. CameraError) rather than ending quietly.
    """
    fps_meter = FpsMeter(cfg.runtime.fps_window_s)
    latencies_ms: list[float] = []
    fps = 0.0
    last_latency_ms = 0.0
    start = time.perf_counter()

    interrupted = False

    try:
        while max_frames is None or len(latencies_ms) < max_frames:
            packet = source.read()
            if packet is None:
                break

            # M2+: features = perception.process(packet); state = logic.update(features); ...

            fps = fps_meter.tick(time.perf_counter())
            if display:
                draw_overlay(packet.image, fps, last_latency_ms, cfg.runtime.latency_budget_ms)
                cv2.imshow(WINDOW_NAME, packet.image)
                key = cv2.waitKey(1) & 0xFF
                visible = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE)
                if key in _QUIT_KEYS or visible < 1:
                    break

            last_latency_ms = (time.perf_counter() - packet.grabbed_at) * 1000.0
            latencies_ms.append(last_latency_ms)
    except KeyboardInterrupt:
        interrupted = True  # Ctrl+C: stop cleanly and still report stats
    finally:
        if display:
            cv2.destroyAllWindows()

    duration = time.perf_counter() - start
    frames = len(latencies_ms)
    return RunStats(
        frames=frames,
        duration_s=duration,
        mean_fps=frames / duration if duration > 0 else 0.0,
        latency_p50_ms=float(np.percentile(latencies_ms, 50)) if frames else 0.0,
        latency_p95_ms=float(np.percentile(latencies_ms, 95)) if frames else 0.0,
        dropped_frames=source.dropped_frames,
        interrupted=interrupted,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drowsiness", description="Real-time driver drowsiness and inattention detection."
    )
    parser.add_argument("--source", help="camera index (e.g. 0) or path to a video file")
    parser.add_argument("--config", type=Path, help="YAML config merged over the defaults")
    parser.add_argument("--no-display", action="store_true", help="run without a window")
    parser.add_argument("--max-frames", type=int, help="stop after N frames")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    overrides: dict = {}
    if args.source is not None:
        overrides["capture"] = {"source": args.source}
    if args.max_frames is not None:
        overrides["runtime"] = {"max_frames": args.max_frames}
    try:
        cfg = load_config(args.config, overrides)
    except ConfigError as e:
        logging.basicConfig(level=logging.ERROR)
        log.error("%s", e)
        return 2

    logging.basicConfig(
        level=cfg.runtime.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    display = cfg.runtime.display and not args.no_display

    try:
        with open_source(cfg.capture) as source:
            stats = run_loop(source, cfg, display, cfg.runtime.max_frames)
    except (CameraError, OSError) as e:
        # Includes a live camera lost mid-run: monitoring stopped, so this is a failure.
        log.error("%s", e)
        return 1
    except cv2.error as e:
        log.error("OpenCV error: %s (headless OpenCV? try --no-display)", e)
        return 1
    except KeyboardInterrupt:  # while opening the source, before the loop started
        log.info("interrupted")
        return 0

    if stats.interrupted:
        log.info("interrupted")
    log.info(
        "processed %d frames in %.1f s: %.1f fps, latency p50 %.1f ms / p95 %.1f ms, %d dropped",
        stats.frames,
        stats.duration_s,
        stats.mean_fps,
        stats.latency_p50_ms,
        stats.latency_p95_ms,
        stats.dropped_frames,
    )
    return 0
