"""The individual detectors (lit. review §2A, §3, §6; thresholds in default_config.yaml).

| reason            | condition                                                     | state
|-------------------|---------------------------------------------------------------|----------------
| sleep             | eyes closed continuously > eyes.sleep_s (Euro NCAP)           | DROWSY_ALARM
| perclos_alarm     | PERCLOS-P80 over perclos.window_s >= perclos.alarm            | DROWSY_ALARM
| no_face           | face not found continuously >= no_face.timeout_s              | NO_FACE
| long_distraction  | head off-road continuously >= distraction.long_s (Euro NCAP)  | DISTRACTED
| short_distraction | off-road >= short_total_s within short_window_s (Euro NCAP)   | DISTRACTED
| microsleep        | eyes closed continuously >= eyes.microsleep_s                 | DROWSY_WARNING
| perclos_warning   | PERCLOS-P80 >= perclos.warning                                | DROWSY_WARNING
| head_nod          | head >= pitch_delta_deg below neutral, eyes not open, >= 1 s  | DROWSY_WARNING
| yawning           | >= warning_count yawns (MAR > threshold >= 1 s) in window     | DROWSY_WARNING

REASONS lists them in priority order. Pure: no I/O, no clocks.
"""

from __future__ import annotations

from drowsiness.config import Config
from drowsiness.logic.calibration import Calibration
from drowsiness.logic.signals import Condition, EventTimes, MedianSmoother, TimeWindow
from drowsiness.types import DriverState, FrameFeatures

REASON_STATE: dict[str, DriverState] = {
    "sleep": DriverState.DROWSY_ALARM,
    "perclos_alarm": DriverState.DROWSY_ALARM,
    "no_face": DriverState.NO_FACE,
    "long_distraction": DriverState.DISTRACTED,
    "short_distraction": DriverState.DISTRACTED,
    "microsleep": DriverState.DROWSY_WARNING,
    "perclos_warning": DriverState.DROWSY_WARNING,
    "head_nod": DriverState.DROWSY_WARNING,
    "yawning": DriverState.DROWSY_WARNING,
}
REASONS = tuple(REASON_STATE)  # priority order, highest first


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


class RuleEngine:
    """Turns FrameFeatures into active reasons and live metrics, given a calibration."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        sig = cfg.signals
        gap = sig.max_gap_s
        self._eye_smoother = MedianSmoother(sig.smoothing_s)
        self._mouth_smoother = MedianSmoother(sig.smoothing_s)
        self._eyes_closed = Condition(gap, True, sig.carry_min_s, sig.max_carry_s)
        self._mouth_open = Condition(gap, adverse=False)
        self._head_dropped = Condition(gap, True, sig.carry_min_s, sig.max_carry_s)
        self._off_road = Condition(gap, True, sig.carry_min_s, sig.max_carry_s)
        self._perclos = TimeWindow(cfg.perclos.window_s)
        self._off_road_window = TimeWindow(cfg.distraction.short_window_s)
        self._yawns = EventTimes(cfg.yawn.count_window_s)
        self._yawn_counted = False
        self._face_since: float | None = None
        self._missing_since: float | None = None
        self._no_face_start: float | None = None
        self._short_fired = False
        self._on_road_since: float | None = None
        self._prev_t: float | None = None
        self._prev_closed: bool | None = None
        self._prev_off_road: bool | None = None

    def reset(self) -> None:
        for part in (
            self._eye_smoother, self._mouth_smoother, self._eyes_closed, self._mouth_open,
            self._head_dropped, self._off_road, self._perclos,
            self._off_road_window, self._yawns,
        ):  # fmt: skip
            part.reset()
        self._yawn_counted = self._short_fired = False
        self._face_since = self._on_road_since = None
        self._missing_since = self._no_face_start = None
        self._prev_t = self._prev_closed = self._prev_off_road = None

    def update_no_face_only(self, f: FrameFeatures) -> tuple[list[str], dict[str, float]]:
        """During calibration only the no-face rule runs (there is no baseline yet)."""
        no_face_s = self._update_no_face(f)
        reasons = ["no_face"] if no_face_s >= self.cfg.no_face.timeout_s else []
        return reasons, {"no_face_s": no_face_s}

    def update(self, f: FrameFeatures, cal: Calibration) -> tuple[list[str], dict[str, float]]:
        t, cfg = f.timestamp, self.cfg
        metrics: dict[str, float] = {}

        # Time since the previous frame belongs to the previous frame's values.
        if self._prev_t is not None:
            self._attribute(self._perclos, self._prev_t, t, self._prev_closed)
            self._attribute(self._off_road_window, self._prev_t, t, self._prev_off_road)

        no_face_s = self._update_no_face(f)
        metrics["no_face_s"] = no_face_s

        # Eyes: openness relative to the calibrated open EAR, mean of available eyes.
        openness = self._eye_smoother.update(t, self._openness(f, cal) if f.face_found else None)
        if openness is None:
            closed_now = None
        elif cal.fallback:
            closed_now = openness < cfg.eyes.ear_fallback_closed
        else:
            frac = cfg.eyes.closed_ear_fraction
            closed_now = openness < frac + cfg.eyes.closed_ratio * (1.0 - frac)  # P80
        closed = self._eyes_closed.update(t, closed_now)
        closed_s = self._eyes_closed.duration(t)
        metrics["closed_s"] = closed_s
        if openness is not None:
            metrics["eye_openness"] = openness

        # PERCLOS-P80 over the window, once enough of it is known.
        closed_total, known_total = self._perclos.totals(t)
        perclos = None
        if known_total >= cfg.perclos.min_valid_ratio * cfg.perclos.window_s:
            perclos = closed_total / known_total
            metrics["perclos"] = perclos

        # Mouth: a yawn is MAR above threshold for min_duration_s, counted once per opening.
        mar = self._mouth_smoother.update(t, f.mar if f.face_found else None)
        mouth_open = self._mouth_open.update(
            t, None if mar is None else mar > cfg.yawn.mar_threshold
        )
        mouth_open_s = self._mouth_open.duration(t)
        if mouth_open_s >= cfg.yawn.min_duration_s and not self._yawn_counted:
            self._yawns.add(t)
            self._yawn_counted = True
        elif mouth_open is False:
            self._yawn_counted = False  # only a seen closed mouth ends a yawn, not a dropout
        yawn_count = self._yawns.count(t)
        metrics["yawn_count"] = float(yawn_count)

        # Head pose relative to neutral.
        pitch_d = yaw_d = None
        if f.face_found and f.pitch is not None and f.yaw is not None:
            pitch_d, yaw_d = f.pitch - cal.neutral_pitch, f.yaw - cal.neutral_yaw
            metrics["pitch_delta_deg"], metrics["yaw_delta_deg"] = pitch_d, yaw_d
        # Nod: head dropped while the eyes are not confidently open. Head down with
        # open eyes (e.g. looking at a phone) is left to the distraction rules.
        dropped_now = None
        if pitch_d is not None:
            dropped_now = pitch_d < -cfg.head_nod.pitch_delta_deg and closed is not False
        self._head_dropped.update(t, dropped_now)
        nod_s = self._head_dropped.duration(t)
        metrics["nod_s"] = nod_s
        # Off-road: head turned or tilted beyond the limits, with eyes not closed
        # (closed eyes are drowsiness, handled above).
        off_now = None
        if pitch_d is not None and yaw_d is not None:
            turned = (
                abs(yaw_d) > cfg.distraction.yaw_deg or abs(pitch_d) > cfg.distraction.pitch_deg
            )
            # A blink while looking away is not evidence of looking back: unknown, so
            # the off-road timer holds instead of resetting.
            off_now = None if closed is True else turned
        off_road = self._off_road.update(t, off_now)
        off_road_s = self._off_road.duration(t)
        off_total, _ = self._off_road_window.totals(t)
        # Short distraction is a one-shot trigger (Euro NCAP): once it has fired, the
        # window restarts after the driver has been back on the road for clear_s.
        if off_total >= cfg.distraction.short_total_s:
            self._short_fired = True
        if off_road is False:
            self._on_road_since = self._on_road_since if self._on_road_since is not None else t
            if self._short_fired and t - self._on_road_since >= cfg.state_machine.clear_s:
                self._off_road_window.reset()
                self._short_fired, off_total = False, 0.0
        else:
            self._on_road_since = None
        metrics["off_road_s"], metrics["off_road_total_s"] = off_road_s, off_total

        # Carried values are inferred, not observed: keep them out of the windows.
        self._prev_t = t
        self._prev_closed = None if self._eyes_closed.carrying else closed
        self._prev_off_road = None if self._off_road.carrying else off_road

        active = {
            "sleep": closed_s > cfg.eyes.sleep_s,
            "perclos_alarm": perclos is not None and perclos >= cfg.perclos.alarm,
            "no_face": no_face_s >= cfg.no_face.timeout_s,
            "long_distraction": off_road_s >= cfg.distraction.long_s,
            "short_distraction": off_total >= cfg.distraction.short_total_s,
            "microsleep": closed_s >= cfg.eyes.microsleep_s,
            "perclos_warning": perclos is not None and perclos >= cfg.perclos.warning,
            "head_nod": nod_s >= cfg.head_nod.min_duration_s,
            "yawning": yawn_count >= cfg.yawn.warning_count,
        }
        return [r for r in REASONS if active[r]], metrics

    @staticmethod
    def _openness(f: FrameFeatures, cal: Calibration) -> float | None:
        if cal.fallback:
            return _mean([e for e in (f.ear_left, f.ear_right) if e is not None])
        pairs = ((f.ear_left, cal.ear_open_left), (f.ear_right, cal.ear_open_right))
        return _mean([ear / open_ for ear, open_ in pairs if ear is not None and open_])

    def _update_no_face(self, f: FrameFeatures) -> float:
        """No-face run length, with symmetric hysteresis of max_gap_s.

        A run starts only after the face has been missing continuously for max_gap_s
        (so an occasional missed detection never starts one), and ends only after the
        face has been seen continuously for max_gap_s (so one spurious detection does
        not end it). While a returning face is being confirmed the run stops growing.
        """
        t, gap = f.timestamp, self.cfg.signals.max_gap_s
        if not f.face_found:
            self._missing_since = self._missing_since if self._missing_since is not None else t
            if t - self._missing_since >= gap:
                self._face_since = None  # a real dropout cancels a pending confirmation
                if self._no_face_start is None:
                    self._no_face_start = self._missing_since
        else:
            self._missing_since = None
            self._face_since = self._face_since if self._face_since is not None else t
            if self._no_face_start is not None and t - self._face_since >= gap:
                self._no_face_start = None
        if self._no_face_start is None:
            return 0.0
        if self._face_since is not None:
            return self._face_since - self._no_face_start  # confirming a returning face
        return t - self._no_face_start

    def _attribute(self, window: TimeWindow, t0: float, t1: float, value: bool | None) -> None:
        """Credit [t0, t1) to `value`, at most max_gap_s of it: a longer frame gap is unknown."""
        window.add(t0, min(t1, t0 + self.cfg.signals.max_gap_s), value)
