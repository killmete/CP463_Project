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
