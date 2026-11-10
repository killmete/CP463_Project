# Architecture

Owner: system-architect. `CLAUDE.md` has the overview, pipeline diagram and core
rules; this file records decisions that are too detailed for it.
Method: `docs/literature-review.md` §2A and §6.

## Stage outputs
| Stage | Output | Consumers |
|---|---|---|
| capture | `FramePacket` | perception, ui (draws on `image`) |
| perception | `PerceptionOutput` = `FrameFeatures` + `landmarks` + `perception_ms` | logic (features only), ui, feature log |
| logic | `DriverState`, `AlertEvent` | alerts, ui |

Only `FrameFeatures` crosses into logic. Landmarks stay with perception and ui,
so logic can be replayed from a feature CSV without images.

## Perception (milestone 2)
```
perception/
  base.py                 LandmarkDetector protocol, LandmarkResult
  mediapipe_detector.py   the only module that imports mediapipe (lazy)
  landmarks.py            MediaPipe 478-point index sets (eyes, mouth, pose points)
  features.py             pure functions: ear(), mar(), head_pose()
  preprocess.py           CLAHE (applied to a copy given to the detector)
  pipeline.py             Perception.process(FramePacket) -> PerceptionOutput
  fetch_model.py          python -m drowsiness.perception.fetch_model (sha256-checked)
ui/overlay.py             debug overlay (landmarks, values, FPS/latency)
feature_log.py            FeatureCsvWriter / read_feature_csv (replay format v1)
```

## Conventions
- **Left/right** are the driver's anatomical sides. Frames are never mirrored
  before perception. `ear_left` uses MediaPipe indices 362, 385, 387, 263, 373, 380;
  `ear_right` uses 33, 160, 158, 133, 153, 144. In an unmirrored front camera the
  driver's left eye has the larger image x.
- **Angles** (degrees, camera frame, uncalibrated): pitch + = head up,
  yaw + = turned to the driver's left, roll + = tilted toward the driver's left
  shoulder. Neutral offsets are removed by logic calibration, not perception.
- **Geometry** is computed in pixel coordinates (normalised x·W, y·H), so EAR/MAR
  do not depend on the frame aspect ratio.
- **MediaPipe timestamps:** `ts_ms = round(timestamp * 1000)`, bumped to
  `last + 1` if not strictly increasing. `FrameFeatures.timestamp` keeps the
  original `FramePacket.timestamp`. One detector instance per source.
- **Several faces:** keep the largest (the driver is closest to the camera).

## Feature CSV v1 (`--log-features`)
UTF-8, comma-separated, one header row, one row per processed frame.
Columns in order: `timestamp, frame_index, face_found, ear_left, ear_right, mar,
pitch, yaw, roll, gaze_h, gaze_v, perception_ms`, then one `bs_<name>` column per
MediaPipe blendshape (fixed 52-name list, MediaPipe order) when blendshapes are
enabled. `face_found` is 0/1; `None` is an empty cell; floats are written with
`repr()` so they round-trip exactly. The reader rebuilds `FrameFeatures`
(ignores `frame_index` and `perception_ms`); no `bs_*` columns or all empty
-> `blendshapes=None`. Blendshape names outside the fixed list are not written.

## Logic (milestone 3)
Method: `docs/literature-review.md` §2A (EAR, PERCLOS-P80, MAR, head pose), §3
(Euro NCAP timings), §6 (rule list, P80 formula).
```
logic/
  calibration.py    Calibrator -> Calibration (per-eye open EAR, neutral pitch/yaw/roll)
  signals.py        time-weighted samples, gap handling, median smoothing, sliding windows
  rules.py          one detector per condition -> active reasons + metrics
  state_machine.py  priority + hysteresis -> DriverState, AlertEvents on transitions
  __init__.py       DriverMonitor(cfg).update(FrameFeatures) -> LogicOutput
```
- **Pure:** no I/O, OpenCV or clocks; only `FrameFeatures.timestamp` is used.
  Timestamps must strictly increase (otherwise `ValueError`). Same input gives the same output.
- **Time weighting:** each sample holds from its timestamp to the next one,
  capped at `signals.max_gap_s` (0.3 s). PERCLOS and the off-road totals add up seconds, not frames.
- **Unknown data** (face not found, or the value is `None`): for up to
  `max_gap_s`, the last value is kept. After that, an *adverse* condition (eyes
  closed, head off-road, head dropped) **carries** if it had been active for at
  least `signals.carry_min_s` (0.5 s, longer than a blink): it stays active
  until a valid frame contradicts it, for at most `signals.max_carry_s` (30 s).
  Anything else becomes unknown. Unknown and carried time is left out of the
  PERCLOS and off-road windows and never resets them. A face that drops out of
  view during a closure can therefore still reach the sleep alarm, while a
  blink just before the face is lost gives `NO_FACE`, not `sleep`.
- **No-face hysteresis:** a no-face run starts only after the face has been
  missing continuously for `max_gap_s` (timed from the first missed frame), and
  ends only after the face has been seen continuously for `max_gap_s`. So
  occasional missed detections of a visible face never raise `NO_FACE`, and one
  spurious detection doesn't reset the timer.
- **Blinks while looking away** make the off-road value unknown (held), not
  "on-road", so they don't reset the distraction timer.
- **Short distraction** is one-shot (Euro NCAP): after it fires, the 30 s window
  restarts once the driver has been on-road for `state_machine.clear_s`.
- **Yawns** are counted once per mouth opening; only a seen closed mouth ends
  it, not a face dropout.
- **Eyes:** each eye is normalised by its own calibrated open EAR, and openness is the
  mean of the eyes that are available (not the min: noise in one eye, yaw
  foreshortening and one-eye occlusion would cause false closures). The closed
  EAR is estimated as `eyes.closed_ear_fraction` x open, because calibration only
  sees open eyes. P80 closed: `ear < closed + closed_ratio * (open - closed)`.
- **Calibration:** median of the valid frames over `duration_s`. If it fails,
  it retries up to `max_attempts` times, then falls back to the absolute EAR
  threshold and emits `calibration_fallback`. The fallback keeps the median
  neutral pose if enough frames had a valid pose, else zero offsets.
- **Recalibration** (`c` key, `.recalibrate()`): the rules keep running on the
  old calibration until the new one succeeds, so it never pauses detection or
  silences an alarm (`metrics["calibrating"]` is 1 meanwhile). Success replaces
  the baseline and restarts the timers; failure keeps the old baseline. Events:
  `recalibrating`, then `calibrated` or `recalibration_failed`, all `INFO` and
  without a state change. Only the first calibration uses `CALIBRATING`.
- **Nod vs pitch-down distraction:** a nod needs the eyes to be *not confidently open*
  (closed or unknown). Head down with open eyes counts only as distraction.
- **Priority:** `DROWSY_ALARM` > `NO_FACE` > `DISTRACTED` > `DROWSY_WARNING` > `NORMAL`.
  `CALIBRATING` replaces `NORMAL` until calibration ends, and `NO_FACE` is still
  shown while calibrating. Escalation is immediate. A state is left only after
  its condition has been clear for `state_machine.clear_s`.
- **Events:** one `AlertEvent` per state transition, never per frame.
  `reason` is the top active condition: `sleep`, `microsleep`, `perclos_alarm`,
  `perclos_warning`, `yawning`, `head_nod`, `long_distraction`,
  `short_distraction`, `no_face`, `calibrated`, `calibration_fallback`, `cleared`,
  plus the INFO events `recalibrating` and `recalibration_failed` (which don't
  change the state; a calibration that ends during e.g. `NO_FACE` is also INFO).
  `details` always includes `previous_state` and `reasons` (all active
  reasons, comma-joined), plus metrics for each reason (see the M3 plan).
