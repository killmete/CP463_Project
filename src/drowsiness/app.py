"""Main processing loop and command-line entry point.

capture -> perception -> logic -> alerts (banner, sound, event log), plus the debug
overlay and feature CSV. No detection rules belong here: they live in drowsiness.logic,
and alerts only present what logic decided.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from drowsiness.alerts import AlertManager, build_alerts
from drowsiness.capture import CameraError, FrameSource, open_source
from drowsiness.config import Config, ConfigError, load_config
from drowsiness.feature_log import FeatureCsvWriter
from drowsiness.logic import DriverMonitor
from drowsiness.perception import Perception, PerceptionSetupError, build_perception
from drowsiness.ui import draw_banner, draw_logic, draw_perception, draw_status

log = logging.getLogger("drowsiness")

WINDOW_NAME = "Drowsiness Detection"
_QUIT_KEYS = {ord("q"), 27}  # q, Esc
_RECALIBRATE_KEY = ord("c")


@dataclass(frozen=True, slots=True)
class RunStats:
    frames: int
    duration_s: float
    mean_fps: float
    latency_p50_ms: float
    latency_p95_ms: float
    dropped_frames: int
    interrupted: bool = False
    perception_p50_ms: float | None = None
    perception_p95_ms: float | None = None
    alert_events: int = 0
    alerts_sounded: int = 0


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


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def run_loop(
    source: FrameSource,
    cfg: Config,
    display: bool,
    max_frames: int | None = None,
    perception: Perception | None = None,
    feature_log: FeatureCsvWriter | None = None,
    monitor: DriverMonitor | None = None,
    alerts: AlertManager | None = None,
) -> RunStats:
    """Process frames from an opened source until it ends, max_frames, or the user quits.

    A live source that fails raises (e.g. CameraError) rather than ending quietly.
    Without `perception` (which `feature_log` and `monitor` need) this is the
    capture-only loop. `alerts` presents the monitor's output (banner, sound, log).
    """
    fps_meter = FpsMeter(cfg.runtime.fps_window_s)
    latencies_ms: list[float] = []
    perception_ms: list[float] = []
    alert_events = alerts_sounded = 0
    decision = None  # the last one is kept on screen if logic skips a frame
    fps = 0.0
    last_latency_ms = 0.0
    start = time.perf_counter()

    interrupted = False

    try:
        while max_frames is None or len(latencies_ms) < max_frames:
            packet = source.read()
            if packet is None:
                break

            out = perception.process(packet) if perception is not None else None
            if out is not None:
                perception_ms.append(out.perception_ms)
                if feature_log is not None:
                    feature_log.write(out, packet.index)
            logic_out = None
            if out is not None and monitor is not None:
                try:
                    logic_out = monitor.update(out.features)
                except ValueError as exc:  # e.g. a camera clock step backwards: skip the frame
                    log.warning("logic skipped frame %d: %s", packet.index, exc)
                else:
                    alert_events += len(logic_out.events)
                    if alerts is not None:  # logs the events; sound runs on its own thread
                        decision = alerts.handle(logic_out, out.features.timestamp)
                        alerts_sounded += decision.sounded

            fps = fps_meter.tick(time.perf_counter())
            if display:
                if out is not None:
                    draw_perception(packet.image, out, cfg.overlay)
                if logic_out is not None and cfg.overlay.logic:
                    draw_logic(packet.image, logic_out)
                draw_status(packet.image, fps, last_latency_ms, cfg.runtime.latency_budget_ms)
                if decision is not None and alerts is not None:  # last: nothing covers it
                    draw_banner(packet.image, decision, decision.timestamp, cfg.alerts.flash_hz,
                                sound_off=not alerts.sound_available)  # fmt: skip
                cv2.imshow(WINDOW_NAME, packet.image)
                key = cv2.waitKey(1) & 0xFF
                visible = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE)
                if key in _QUIT_KEYS or visible < 1:
                    break
                if key == _RECALIBRATE_KEY and monitor is not None:
                    monitor.recalibrate()

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
        latency_p50_ms=_percentile(latencies_ms, 50),
        latency_p95_ms=_percentile(latencies_ms, 95),
        dropped_frames=source.dropped_frames,
        interrupted=interrupted,
        perception_p50_ms=_percentile(perception_ms, 50) if perception is not None else None,
        perception_p95_ms=_percentile(perception_ms, 95) if perception is not None else None,
        alert_events=alert_events,
        alerts_sounded=alerts_sounded,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drowsiness", description="Real-time driver drowsiness and inattention detection."
    )
    parser.add_argument("--source", help="camera index (e.g. 0) or path to a video file")
    parser.add_argument("--config", type=Path, help="YAML config merged over the defaults")
    parser.add_argument("--no-display", action="store_true", help="run without a window")
    parser.add_argument("--max-frames", type=int, help="stop after N frames")
    parser.add_argument(
        "--log-features", type=Path, metavar="CSV", help="write per-frame features to a CSV"
    )
    parser.add_argument(
        "--no-perception", action="store_true", help="capture only (no landmark model)"
    )
    parser.add_argument("--mute", action="store_true", help="no alert sound (visual only)")
    parser.add_argument(
        "--no-debug", action="store_true", help="hide the debug overlay (the banner stays)"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    overrides: dict = {"capture": {}, "runtime": {}, "perception": {}, "alerts": {}}
    if args.source is not None:
        overrides["capture"]["source"] = args.source
    if args.max_frames is not None:
        overrides["runtime"]["max_frames"] = args.max_frames
    if args.log_features is not None:
        overrides["runtime"]["log_features"] = str(args.log_features)
    if args.no_perception:
        overrides["perception"]["enabled"] = False
    if args.mute:
        overrides["alerts"]["sound"] = False
    if args.no_display and args.source is not None and not args.source.isdigit():
        # A headless file replay runs faster than real time: alarms would stack up.
        overrides["alerts"]["sound"] = False
    if args.no_debug:
        overrides["overlay"] = {"landmarks": False, "values": False, "logic": False}
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
    if cfg.runtime.log_features is not None and not cfg.perception.enabled:
        log.error("--log-features needs perception (remove --no-perception)")
        return 2

    feature_log = None
    try:
        # Closed in reverse order on every path: alerts (stops sound), feature log,
        # source, perception.
        with contextlib.ExitStack() as stack:
            perception = (
                stack.enter_context(build_perception(cfg)) if cfg.perception.enabled else None
            )
            source = stack.enter_context(open_source(cfg.capture))
            if cfg.runtime.log_features is not None:
                # Opened only once the source works, so a bad --source leaves no stub CSV.
                feature_log = stack.enter_context(
                    FeatureCsvWriter(
                        cfg.runtime.log_features, blendshapes=cfg.perception.output_blendshapes
                    )
                )
            monitor = alerts = None
            if perception is not None:
                monitor = DriverMonitor(cfg)
                alerts = stack.enter_context(build_alerts(cfg))
            stats = run_loop(
                source, cfg, display, cfg.runtime.max_frames, perception, feature_log, monitor,
                alerts,
            )  # fmt: skip
    except PerceptionSetupError as e:  # before CameraError/OSError: a missing model is setup
        log.error("%s", e)
        return 2
    except (CameraError, OSError) as e:
        # Includes a live camera lost mid-run: monitoring stopped, so this is a failure.
        log.error("%s", e)
        return 1
    except cv2.error as e:
        log.error("OpenCV error: %s (headless OpenCV? try --no-display)", e)
        return 1
    except KeyboardInterrupt:  # while loading the model or opening the source
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
    if stats.perception_p50_ms is not None:
        log.info(
            "perception p50 %.1f ms / p95 %.1f ms", stats.perception_p50_ms, stats.perception_p95_ms
        )
    if stats.perception_p50_ms is not None:
        log.info("%d alert events, %d sounded", stats.alert_events, stats.alerts_sounded)
    if feature_log is not None:
        log.info("features written to %s", feature_log.path)
    return 0
