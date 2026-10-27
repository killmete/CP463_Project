# Driver Drowsiness & Inattention Detection (CP463)

Real-time system that watches the driver through a dashcam or webcam
(20–30 FPS) and shows an alert when it detects drowsiness (long eye closure,
microsleep, yawning, head nodding) or inattention (looking away, face not
visible).

Method: MediaPipe facial landmarks → eye/mouth openness and head pose →
time-based rules based on PERCLOS and Euro NCAP driver-monitoring
definitions. See [docs/literature-review.md](docs/literature-review.md).

> Status: milestone 2 – per-frame perception (EAR per eye, MAR, head
> pitch/yaw/roll, blendshapes) with a debug overlay and feature CSV logging.
> Detection rules and alerts come in the next milestones.

## Install

Python 3.11–3.13.

```
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
pip install --no-deps -r requirements-mediapipe.txt
pip install -e . --no-deps
python -m drowsiness.perception.fetch_model   # -> models/face_landmarker.task (gitignored)
```

Install mediapipe only with `--no-deps`: a plain `pip install mediapipe`
pulls in `opencv-contrib-python`, which overwrites the OpenCV build below.
Linux also needs `sudo apt-get install libportaudio2 libegl1 libgles2`, because
`import mediapipe` loads `sounddevice` and EGL/GLES. If `python` is 3.14 or newer, create
the venv with 3.13 (e.g. `py -V:3.13 -m venv .venv`): PyYAML has no wheels
there yet.

OpenCV is an extra so you can pick the build: `pip install -e ".[gui]"`
(window display) or `pip install -e ".[headless]"` (servers/CI, no window).
Don't install both.

## Run

```
python -m drowsiness --source 0                    # webcam + debug overlay; q or Esc quits
python -m drowsiness --source clip.mp4 --no-display --max-frames 300
python -m drowsiness --source 0 --log-features out.csv   # per-frame feature CSV
python -m drowsiness --source 0 --no-perception    # capture only (no model needed)
python -m drowsiness --config my.yaml              # merged over the defaults
```

The overlay shows eye and mouth contours, a head-direction line, EAR (left /
right = the driver's own eyes), MAR, pitch/yaw/roll and perception time. The
CSV format is described in [docs/architecture.md](docs/architecture.md).
For low light, set `preprocess.clahe: true` in your config.

Default settings and thresholds are in
[src/drowsiness/default_config.yaml](src/drowsiness/default_config.yaml).

## Test

```
pytest -q                      # DROWSINESS_CAMERA_TESTS=1 also runs the real-webcam tests
DROWSINESS_PERF_SOURCE=face.jpg pytest -q -s -m perf   # face speed check on your own image/clip
ruff check . && ruff format --check .
```
