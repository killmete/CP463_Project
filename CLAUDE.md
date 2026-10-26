# CP463 Project – Driver Drowsiness & Inattention Detection

Real-time system that watches the driver through a dashcam/webcam (20–30 FPS)
and raises an alert message when it detects drowsiness (eyes closing,
microsleep, yawning, head nodding) or inattention (looking away, face not
visible).

## Stack
Python 3.11 · OpenCV · MediaPipe Face Landmarker (Tasks API) · NumPy ·
PyYAML · pytest · ruff

## Pipeline
```
FrameSource ──(frame, t)──▶ Perception ──FrameFeatures──▶ Logic ──DriverState/AlertEvent──▶ Alerts
 capture/                    perception/                   logic/                           alerts/, ui/
 (camera, video file)        (landmarks, EAR, MAR,         (calibration, PERCLOS,           (overlay, sound,
                              head pose, gaze)              rules, state machine)            console, webhook)
```
`app.py` wires the stages; it contains no detection rules.

## Layout (planned)
```
src/drowsiness/
  types.py          # shared contracts – owned by system-architect
  config.py         # loads config/default.yaml
  app.py            # main loop + CLI  (python -m drowsiness)
  capture/          # FrameSource: CameraSource, VideoFileSource
  perception/       # LandmarkDetector, features.py (EAR/MAR/pose/gaze)
  logic/            # calibration, signals, rules, state_machine
  alerts/           # AlertManager, sinks, messages
  ui/               # overlay drawing
config/default.yaml # ALL thresholds
eval/               # evaluation scripts, labels, reports
tests/              # unit, replay, perf
data/, models/      # gitignored
```

## Shared contracts (`src/drowsiness/types.py`)
- `FrameFeatures`: `timestamp` (s, monotonic), `face_found`, `ear_left`,
  `ear_right`, `mar`, `pitch`, `yaw`, `roll` (degrees), optional `gaze`,
  `blendshapes`. Missing values are `None`.
- `DriverState` enum: `CALIBRATING`, `NORMAL`, `DROWSY_WARNING`,
  `DROWSY_ALARM`, `DISTRACTED`, `NO_FACE`.
- `AlertEvent`: `timestamp`, `state`, `reason`, `level`, `details`.

Changing these requires `system-architect` sign-off and updating every caller.

## Core rules
- Logic is time-based (seconds), never frame counts.
- Perception is stateless per frame; logic is pure (no I/O); alerts never decide.
- Thresholds live in `config/default.yaml`.
- Nothing blocking on the frame loop. Budget: < 50 ms per frame, alert within
  ~0.5 s of a rule firing.
- Never commit datasets, model weights, secrets, or face recordings of people.

## Agent team (`.claude/agents/`)
| Agent | Owns | Use for |
|---|---|---|
| `system-architect` | `types.py`, `CLAUDE.md`, `docs/` | plans, contracts, task breakdown – start here |
| `vision-engineer` | `perception/` | landmarks, EAR/MAR, head pose, gaze |
| `detection-logic-engineer` | `logic/` | timing rules, PERCLOS, calibration, state machine |
| `realtime-pipeline-engineer` | `capture/`, `app.py`, `config.py` | capture, threading, FPS/latency, CLI |
| `alert-ux-engineer` | `alerts/`, `ui/` | messages, overlay, sound, cooldown |
| `data-eval-engineer` | `eval/`, `data/`, `models/` | datasets, labels, metrics, tuning, ML |
| `qa-test-engineer` | `tests/`, `.github/` | unit/replay tests, CI |
| `code-reviewer` | (read-only) | review every diff before commit |

### Workflow
1. `system-architect` plans the feature and defines contract changes.
2. Specialists implement in parallel when they don't share files
   (perception / capture / alerts are independent; logic depends on the
   `FrameFeatures` contract only).
3. `qa-test-engineer` adds/updates tests and runs the suite.
4. `data-eval-engineer` measures impact on labeled clips when detection
   behavior changes.
5. `code-reviewer` reviews; fix blockers, then commit.

### Suggested milestones
1. Scaffold + contracts + webcam loop showing FPS.
2. Landmarks + EAR/MAR/head pose overlay; feature CSV logging.
3. Rules: eye closure, PERCLOS, yawn, nod, looking away, no face + calibration.
4. Alerts: overlay banner, sound, cooldown/escalation.
5. Evaluation on public datasets + own clips; threshold tuning; report.
6. (Optional) CNN eye-state model, night/IR robustness, notifications.
