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
| alerts (M4) | `AlertDecision` (in `alerts/`, not `types.py`) | ui banner, sinks (log, sound, webhook) |

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

## Alerts (milestone 4)
Implemented in milestone 4. Alerts only *present* what logic decided. They never
change `DriverState` or create `AlertEvent`s. Timing uses the frame timestamp
(`FrameFeatures.timestamp`), never the wall clock, so a replayed clip gives the
same alert sequence. **No change to `types.py`.**

`docs/literature-review.md` covers detection, not how warnings are presented. So the
cooldown, repeat and escalation timings below are UX defaults, to be checked in M5.
(research-analyst: add a short "warning presentation" note that cites Euro NCAP
[R1][R2] if it specifies how warnings should escalate; we don't assume it does.)

```
alerts/
  __init__.py     build_alerts(cfg) -> AlertManager (context manager), AlertDecision, SoundCue
  manager.py      AlertManager: policy (cooldown, repeat, escalation) + dispatch to sinks
  messages.py     reason -> driver-facing text (ASCII only: cv2 Hershey fonts)
  sinks.py        AlertSink protocol, LogSink, WebhookSink (stub, off by default)
  sound.py        tone patterns (numpy), SoundSink with a worker thread, silent fallback
ui/
  banner.py       draw_banner(image, decision, now, flash_hz): driver-facing banner + border
  overlay.py      debug panels (unchanged, except the values panel moves up above the banner)
```

### Interface
- `AlertManager(cfg.alerts, sinks).handle(logic_out, now) -> AlertDecision`.
  It is called once for each frame that logic processed (`now = out.features.timestamp`).
  The policy is pure. Afterwards each sink gets `sink.emit(decision)`, which must
  return immediately (anything slow runs on the sink's own thread).
  `.close()` stops the sound and joins the threads (it is entered in app's `ExitStack`).
- `AlertDecision` (frozen, defined in `alerts/`, used by `ui/` and the sinks):
  `timestamp`, `state`, `level` (the presented `AlertLevel`), `reason` (or None), `message`
  (banner text or None), `flash` (bool), `sound` (`SoundCue`: NONE / WARNING /
  ALARM / STOP), `escalated` (bool), `toast` (INFO text or None), `events`
  (this frame's logic events, for LogSink).
- **Banner reason:** the first entry of `logic_out.reasons` whose
  `REASON_STATE` (public in `drowsiness.logic`) is the current state. If there is none
  (the state is held during `clear_s`), it is the reason of the event that entered the state.

### Policy (all times come from frame timestamps)
| Situation | Sound | Banner |
|---|---|---|
| ALARM-level event (`DROWSY_ALARM`) | ALARM pattern right away; the cooldown never suppresses it | red, flashing |
| ALARM state persists | ALARM pattern again every `alarm_repeat_s` | red, flashing |
| WARNING-level event (`DROWSY_WARNING`, `DISTRACTED`, `NO_FACE`) | WARNING chime, unless the same reason was sounded less than `cooldown_s` ago | amber |
| WARNING state persists | chime again every `warning_repeat_s` | amber |
| A more serious reason within the same state (e.g. yawning -> microsleep; no logic event) | announced like a new warning (cooldown, escalation count); it is re-armed only after a less serious reason has held for `rearm_s` (3 s), so a reason flickering at its threshold is announced once | the new reason's text |
| De-escalation ALARM -> `DROWSY_WARNING` (same drowsy episode, recovering) | STOP, no new chime | amber |
| De-escalation ALARM -> `NO_FACE` / `DISTRACTED` (e.g. asleep and slumped out of view) | WARNING chime at once (replaces the alarm, no silent gap) | amber |
| `cleared` / back to `NORMAL` or `CALIBRATING` | STOP | none |
| INFO event (calibrated, calibration_fallback, recalibrating, recalibration_failed) | never | grey toast for `info_display_s` |
| Escalation: at least `escalation_count` `DROWSY_WARNING` announcements (sounded or held back by the cooldown) within `escalation_window_s`; bypasses the cooldown | that warning plays the ALARM pattern once (no loop) | red, steady, "Drowsy again - stop and rest"; repeats use the WARNING chime |

- Every sounded announcement (repeats included) updates the "last sounded" time for its
  reason. So if the same warning comes back just after it cleared, it stays quiet
  (logged at DEBUG as "not sounded (cooldown)") unless it escalates.
- Escalation only changes the presentation (`escalated=True`, the state is unchanged,
  and it is logged). Fatigue builds up over time, so repeated microsleep, PERCLOS, yawn
  or nod warnings should get louder rather than being muted by the cooldown.
  Distraction and no-face warnings are not counted. If evaluation shows that
  escalation matters for detection, it moves into logic as a rule in M5, after a
  research review.
- Latency: the decision and the sound command come from the same frame as the
  event (well within 0.5 s). `emit` only puts the command in a queue.

### Driver view
- The banner is always drawn when there is a window. The debug panels can be hidden
  (`overlay.logic: false`, or `--no-debug`, which turns off landmarks, values and logic).
- Layout: a full-width bar at the bottom (about 12% of the frame height). For WARNING
  and ALARM there is also an 8 px coloured border around the frame, which the driver
  can see out of the corner of the eye. The perception values panel moves up by the
  banner height.
- `NORMAL`: no bar, just a small green "Monitoring" tag (shows the system is running).
  `CALIBRATING` (and recalibration, `metrics["calibrating"]`): grey bar "Calibrating -
  look ahead". `NO_FACE`: amber bar "Face not visible - check camera".
- Flashing: ALARM alternates between red and dark at `flash_hz` (the phase comes from
  `now`), at most 3 Hz (photosensitivity limit).
- Messages (`messages.py`, at most 32 characters so they fit in 640 px): sleep "WAKE UP -
  pull over safely"; perclos_alarm "Very drowsy - pull over and rest"; microsleep
  "Eyes closing - take a break"; perclos_warning "Drowsy - take a break soon";
  head_nod "Head nodding - take a break"; yawning "Frequent yawns - take a
  break"; long_distraction "Look at the road"; short_distraction "Keep your eyes on
  the road"; no_face "Face not visible - check camera". An unknown reason falls back
  to a generic text for its level, so the banner never crashes or goes blank.

### Sound
- Patterns are generated once with numpy at start-up (float32, with a 5 ms fade in and
  out to avoid clicks). There are no audio files in the repo. WARNING: two 880 Hz beeps
  (150 ms on, 100 ms off). ALARM: about 1 s of alternating 1000 Hz / 1500 Hz beeps.
  Tone shapes are code constants (presentation, not thresholds); the volume is in config.
- `SoundSink` owns a daemon worker thread and a small queue. `emit` calls
  `put_nowait`; when the queue is full, the oldest command is dropped, and STOP always
  gets in. The worker calls `sounddevice.play` (non-blocking, replaces the sound that
  is playing) or `stop`.
- `sounddevice` is imported lazily, **in the worker thread**. If import, device query
  or play raises any exception (`ImportError`, including when the mediapipe guard has set
  `sys.modules["sounddevice"] = None`, `OSError`, or `PortAudioError`), the sink logs one
  warning ("audio unavailable, alerts are visual only") and does nothing from then on.
  The backend can be injected, so tests never touch real audio.
- With `alerts.sound: false` or `--mute`, no SoundSink is created at all.

### Config (`alerts:` section) and CLI
```yaml
alerts:
  sound: true              # false or --mute: visual only
  volume: 0.5              # (0, 1]
  sample_rate: 22050
  cooldown_s: 30.0         # the same warning reason is not sounded again within this time
  warning_repeat_s: 10.0   # chime again while a WARNING state persists
  alarm_repeat_s: 2.0      # sound again while DROWSY_ALARM persists
  escalation_count: 3      # sounded drowsiness warnings ...
  escalation_window_s: 300.0  # ... within this window -> alarm-style presentation
  rearm_s: 3.0             # a reason that eased must stay gone this long to be re-announced
  info_display_s: 3.0      # how long an INFO toast stays up
  flash_hz: 2.0            # ALARM banner flash, at most 3
  webhook_url: null        # http(s) URL; null = off
  webhook_timeout_s: 2.0
overlay:
  logic: true              # new: debug logic panel (top right)
```
Validation (in `config.py`, on top of the existing "> 0" check): `volume <= 1`,
`flash_hz <= 3`, `alarm_repeat_s < warning_repeat_s`, `escalation_count >= 2`,
and `webhook_url` must start with `http://` or `https://`. CLI: `--mute` sets
`alerts.sound=false`; `--no-debug` sets `overlay.landmarks/values/logic=false`.

### app.py wiring
When perception is on, `alerts = stack.enter_context(build_alerts(cfg))` (capture-only
mode has no alerts). On each frame, after `monitor.update`:
`decision = alerts.handle(logic_out, out.features.timestamp)`. With a display,
`draw_banner(image, decision, now, cfg.alerts.flash_hz)` is drawn last. A frame that logic skipped
(`ValueError`) is not passed to alerts. `log_event` moves from `app.py` to
`LogSink` (same format, plus "sounded/suppressed/escalated" at DEBUG level).
`RunStats` gets a new `alerts_sounded` field.

### Webhook (stub)
Off by default. When a URL is set, the webhook's own thread POSTs JSON `{timestamp, state,
reason, level}` for each sounded WARNING/ALARM announcement, using `urllib` with
`webhook_timeout_s`. Failures are logged and dropped. It never sends images or landmarks.

### Review follow-ups (implemented)
- **SOUND OFF:** if audio is muted or unavailable, the banner shows a "SOUND OFF" tag
  so the driver knows alerts are visual only. After a playback failure (e.g. a headset
  unplugged) the sound worker retries the device every 10 s when there is something to
  play, and logs when audio is back.
- **Headless file replays are muted** (`--source clip.mp4 --no-display`): they run
  faster than real time, so repeats would stack up.
- If logic skips a frame, the last banner stays on screen (no flicker).
- The INFO `recalibrating` event shows no toast (the "Calibrating" bar already does), and
  the `cleared` event never replaces a toast.
- The webhook payload also carries `escalated`.
