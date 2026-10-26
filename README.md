# Driver Drowsiness & Inattention Detection (CP463)

Real-time system that watches the driver through a dashcam or webcam
(20–30 FPS) and shows an alert when it detects drowsiness (long eye closure,
microsleep, yawning, head nodding) or inattention (looking away, face not
visible).

Method: MediaPipe facial landmarks → eye/mouth openness and head pose →
time-based rules based on PERCLOS and Euro NCAP driver-monitoring
definitions. See [docs/literature-review.md](docs/literature-review.md).

> Status: milestone 1 – camera/video input and a live loop showing FPS and
> latency. Detection comes in the next milestones.

## Install

Python 3.11–3.13.

```
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e . --no-deps
```

OpenCV is an extra so you can pick the build: `pip install -e ".[gui]"`
(window display) or `pip install -e ".[headless]"` (servers/CI, no window).
Don't install both.

## Run

```
python -m drowsiness --source 0                    # webcam; q or Esc quits
python -m drowsiness --source clip.mp4 --no-display --max-frames 300
python -m drowsiness --config my.yaml              # merged over the defaults
```

Default settings and thresholds are in
[src/drowsiness/default_config.yaml](src/drowsiness/default_config.yaml).

## Test

```
pytest -q                      # DROWSINESS_CAMERA_TESTS=1 also runs the real-webcam test
ruff check . && ruff format --check .
```
