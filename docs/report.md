# Real-Time Driver Drowsiness and Inattention Detection from a Single Camera

CP463 course project: final report

## Abstract

We built a system that watches a driver through an ordinary webcam or dashcam.
It warns the driver when they show signs of drowsiness (long eye closure,
microsleep, a high PERCLOS, frequent yawning, head nodding) or inattention
(looking away from the road, face not visible).

**How it works**
- Each frame goes through MediaPipe Face Landmarker. Its landmarks give eye and
  mouth openness (EAR, MAR) and head pose.
- Time-based rules then decide. They follow the PERCLOS-P80 measure and the
  Euro NCAP driver-monitoring definitions, and are calibrated to each driver
  over the first five seconds.
- A state machine turns the rules into driver states. An alert layer presents
  them as an on-screen banner and warning or alarm sounds, with cooldown and
  escalation.

**Speed:** about 5 ms per frame on a desktop CPU with no GPU (p95 5.5 ms at
640×480), a tenth of our 50 ms budget. Nothing is trained, and the model file is
3.6 MB.

**Evaluation**
- Data: 1040 clips (54.8 h) from four public datasets: NTHU-DDD, DMD, YawDD and
  UTA-RLDD.
- Splits are by subject. Thresholds were tuned by grid search on the tune split
  only, under a false-alarm cap.

**Results on the held-out test split, after tuning**

| Measure | Result |
|---|---|
| Eyes-closed frame F1 (DMD) | 0.80 |
| YawDD yawning clips with a yawn detected | 87% |
| False yawns on other YawDD clips | 4% |
| UTA-RLDD drowsy vs alert, separated by mean PERCLOS | AUC 0.80 |
| Drowsiness alert episodes on non-drowsy clips | 3.9 per hour |

**Against a simpler baseline:** one fixed eye threshold for every driver catches
more closures, but raises 2–6× as many false alarms. Per-driver calibration is
the better trade-off.

**Weakest parts**
- Head-nod detection (event F1 0.13–0.19).
- Look-aside detection from head pose alone (frame F1 0.41).
- One driver whose eyelid landmarks never close, which causes most missed
  closures on test.

## 1. Introduction

Drowsy and distracted driving cause a large share of road crashes. EU rules have
required a driver drowsiness and attention warning system in all new cars and
commercial vehicles since 7 July 2024 [E1]. The Euro NCAP safety rating also
scores driver-monitoring systems [R1][R3].

The project brief asked for a system that:
- takes video from one RGB camera at 20–30 FPS;
- runs in real time on a normal computer;
- outputs an alert message when the driver is drowsy or not paying attention.

We added four requirements of our own:
- **Explainable:** every alert must be traceable to a measured cause.
- **Standard definitions:** detection follows published definitions (PERCLOS,
  Euro NCAP) rather than thresholds we invented.
- **CPU only:** it must run without a GPU.
- **Measured:** it must be evaluated on public data with a protocol that
  doesn't leak subjects between tuning and testing.

**Users**
- **Primary:** drivers of older cars without a built-in driver-monitoring
  system, using a laptop or a cheap dashcam-style webcam.
- **Secondary:** small fleet operators (delivery, ride-hailing) who want a
  low-cost aftermarket monitor. The optional webhook can log alerts centrally.

**Success criteria**

| # | Criterion | Target | Result | Met? |
|---|---|---|---|---|
| 1 | Real-time on CPU | ≥ 20 FPS and < 50 ms per frame | 171 FPS, p95 5.5 ms (Section 4) | yes |
| 2 | Alert delay | sound queued ≤ 0.5 s after a rule fires | same frame (Section 3.5) | yes |
| 3 | Standard timings | microsleep, sleep and distraction inside Euro NCAP ranges | 1.5 s, 3 s, 3 s, 10 s in 30 s (Section 2) | yes |
| 4 | Fewer misses than the starting thresholds | held-out detection metrics improve after tuning | eye F1 0.76 → 0.80, yawning clips caught 59% → 87% (Section 5.4) | yes |
| 5 | False alarms stay low | ≤ 1 more alert episode per hour than the starting thresholds | +0.8/h on tune, but +1.8/h on test (Section 5.4) | **no, on test** |
| 6 | Generalise across eyewear and lighting | no condition far below the rest | sunglasses and one DMD subject fail (Section 6) | partly |

Criteria 1–3 were fixed at the start of the project. Criteria 4–6 were set when we
designed the evaluation (milestone 5), before any thresholds were tuned.

## 2. Background and choice of method

The full literature review is in `docs/literature-review.md`. In short:
- Surveys group drowsiness detection into four families: physiological
  signals (EEG, ECG), vehicle signals (steering, lane position), video, and
  hybrids [S1][S3]. One camera allows only the video family.
- We compared five video approaches (table below) on CPU speed, which
  behaviours they cover, robustness, training-data needs, explainability and
  match with standards.

| Approach | Verdict |
|---|---|
| Facial landmarks + geometric features + time rules | **Chosen.** Real time on CPU, covers every behaviour we need, needs no training, every alert can be explained |
| Landmarks + learned temporal model (e.g. blink-sequence LSTM [D2]) | Planned upgrade for *early* drowsiness. Needs labelled sequences and reacts over minutes |
| Eye-state CNN on eye crops | Planned upgrade if EAR fails on glasses, pose or night video |
| End-to-end deep video models (CNN-LSTM, two-stream, transformers) | Rejected. Far below real time on CPU: one multi-granularity CNN-LSTM runs at 37 FPS on a GPU but about 3 FPS on a CPU [V2]. Many published NTHU-DDD scores don't state a subject-disjoint protocol, so they can't be compared fairly |
| Object detectors (YOLO family) | Rejected. A per-frame detector still needs all the timing logic built on top |

What the chosen method rests on:
- **Eye Aspect Ratio (EAR):** Soukupová and Čech reduce six eye landmarks to one
  openness value. It is enough for real-time blink detection [P1].
- **PERCLOS-P80:** the share of a minute that the eyes are at least 80% closed
  [P2]. In an NHTSA validation study it was the only measure tested that tracked
  attention lapses consistently (mean r = 0.875 with lapses on a reaction-time
  test) [P3]. A 2023 review notes it can fail to respond to moderate drowsiness
  and is less reliable in real driving than in lab tests [P4], which is why we
  added further cues.
- **Euro NCAP timings** (Driver Engagement protocol and SD-202 test procedure,
  v1.2) [R1][R3]:

  | Event | Euro NCAP definition | Our setting |
  |---|---|---|
  | Microsleep | Eye closure of 1–2 s | 1.5 s |
  | Sleep | Continuous eye closure of 3 s or more | 3 s |
  | Long distraction | One glance away from the road of 3–4 s | 3 s |
  | Short distraction (visual attention time sharing) | 10 s away from the road in total within 30 s | 10 s in 30 s |

## 3. System design

```
FrameSource ──(frame, t)──▶ Perception ──FrameFeatures──▶ Logic ──DriverState/AlertEvent──▶ Alerts
 capture/                    perception/                   logic/                           alerts/, ui/
 (camera, video file)        (landmarks, EAR, MAR,         (calibration, PERCLOS,           (banner, sound,
                              head pose)                    rules, state machine)            log, webhook)
```

Each stage has one job and communicates through typed contracts
(`src/drowsiness/types.py`):
- **Perception** is stateless per frame.
- **Logic** is pure: no I/O and no clock, only frame timestamps. So the same
  feature stream always gives the same decisions.
- **Alerts** only present what logic decided.

This split let us replay hours of dataset video through the real decision code
without images (Section 5).

### 3.1 Perception

**MediaPipe Face Landmarker [I3][K1]** finds 478 face landmarks per frame. If
several faces are found, the largest is kept. From the landmarks we compute:
- **EAR** for each eye, from the six-point eye contour [P1], in pixel
  coordinates so the frame's aspect ratio doesn't matter;
- **MAR** (mouth aspect ratio), the mean inner-lip gap divided by the inner mouth
  width;
- **head pose** (pitch, yaw, roll), by fitting landmarks to a generic 3D face
  with `cv2.solvePnP`.

Two options for low light:
- CLAHE contrast enhancement can be turned on in the config.
- Each frame's features can be written to a CSV, which the evaluation uses.

### 3.2 Calibration

During the first 5 s the driver looks ahead with eyes open. The system takes the
median of these values:
- the open EAR of each eye;
- the neutral head pitch, yaw and roll.

Every later measurement is relative to this baseline. A fixed EAR threshold
generalises poorly across people [M1], which is why we calibrate.

If calibration fails, it is retried. After three failures the system falls back
to an absolute EAR threshold and says so. Pressing `c` recalibrates for a new
driver, and detection keeps running on the old baseline until the new one
succeeds.

### 3.3 Detection rules

All rules are timed in seconds from frame timestamps, never in frame counts, so
they behave the same at 15 and 30 FPS.

| Rule | Condition | Result |
|---|---|---|
| Microsleep | eyes closed ≥ 1.5 s | drowsy warning |
| Sleep | eyes closed ≥ 3 s | drowsy alarm |
| PERCLOS | share of the last 60 s with eyes P80-closed ≥ 0.15 / 0.25 | warning / alarm |
| Yawning | ≥ 3 yawns in 5 min (mouth open above MAR 0.35 for ≥ 0.75 s) | drowsy warning |
| Head nod | head ≥ 15° below neutral for ≥ 1 s, eyes not clearly open | drowsy warning |
| Long distraction | head turned > 30° or pitched > 25° off neutral for ≥ 3 s | distracted |
| Short distraction | ≥ 10 s off-road within 30 s | distracted |
| No face | face missing for 2 s | no-face warning |

**Eye closure (P80):** each eye's openness is its EAR divided by its own
calibrated open EAR, and the two eyes are averaged. An eye is "closed" when:

`EAR < closed + 0.2 × (open − closed)`

Calibration only ever sees open eyes, so the closed-eye EAR is estimated as a
fraction of the open one. This fraction was tuned to 0.2 in Section 5.

**Signal handling**
- Each sample counts for the time until the next one, so a dropped frame
  doesn't change PERCLOS.
- A missing value (face lost, eye not found) is held for at most 0.3 s.
- After that, a condition that was already established (eyes closed, head
  down, head turned away) *carries* until a valid frame contradicts it, for up
  to 30 s. So a driver who slumps out of view with closed eyes still reaches the
  sleep alarm.
- Blinks while looking away do not reset the distraction timer.

### 3.4 State machine

The rules feed a state machine with six states: `CALIBRATING`, `NORMAL`,
`DROWSY_WARNING`, `DROWSY_ALARM`, `DISTRACTED` and `NO_FACE`.

The order of priority is:

`DROWSY_ALARM` > `NO_FACE` > `DISTRACTED` > `DROWSY_WARNING` > `NORMAL`

**Moving between states**
- Escalation is immediate.
- A state is left only after its cause has been clear for 1 s. This hysteresis
  stops flicker.
- Logic emits one alert event per state change, never one per frame.

### 3.5 Alerts

**Banner**
- A full-width bar at the bottom of the video plus a coloured border: amber for
  warnings, flashing red for the alarm. Flashing is capped at 3 Hz
  (photosensitivity limit).
- Short action messages, for example "Eyes closing - take a break" or "WAKE UP -
  pull over safely".

**Sound**
- Generated in code: a two-beep chime for warnings, and an alternating
  two-tone alarm.
- Played on a worker thread so the frame loop never blocks.
- With no audio device, the alerts stay visual and a "SOUND OFF" tag is shown.

**Timing policy**

| Rule | Timing |
|---|---|
| Alarm repeat | every 2 s while the alarm state lasts |
| Warning repeat | every 10 s while the warning state lasts |
| Cooldown | the same warning reason stays silent for 30 s after it sounded |
| Escalation | the third drowsiness warning within 5 minutes plays the alarm sound |

Fatigue builds up over time, so repeated warnings get louder rather than being
muted. All timings come from frame timestamps, so a replay gives the same alert
sequence.

## 4. Implementation

- **Code:** Python 3.11–3.13, with OpenCV, MediaPipe (Tasks API), NumPy and
  PyYAML.
- **Thresholds:** all of them live in one YAML file,
  `src/drowsiness/default_config.yaml`. A user YAML can override it.
- **Command line:** `python -m drowsiness --source 0` uses a webcam, and a video
  file path is also accepted. Other flags give a feature log, a muted mode and
  a mode without the debug overlay.
- **Tests:** 325 automated tests (pytest) cover:
  - feature geometry;
  - every rule's timing boundaries;
  - signal dropouts, calibration and recalibration;
  - the alert policy;
  - the evaluation metrics.
- **CI:** the tests and the ruff linter run on Ubuntu and Windows with Python
  3.11 and 3.13 (GitHub Actions).
- **Manual check:** live webcam runs confirmed the banner and both sounds fire as
  intended.
- **Privacy:** no datasets, face recordings or model weights are in the
  repository.

**Speed**

We ran the full pipeline (capture, perception, logic, alerts) on 600 frames of a
640×480 clip, on a 16-core AMD desktop CPU without a GPU.

| Measure | Result |
|---|---|
| Throughput, unthrottled file input | 171 FPS |
| Latency p50 / p95 | 4.9 / 5.5 ms |
| Frames dropped | 0 |

The budget was 50 ms per frame and an alert within about 0.5 s. The alert sound
is queued on the same frame the rule fires.

**Cost**

| Item | Value |
|---|---|
| Hardware | one desktop PC: 16-core / 32-thread AMD CPU, 15 GB RAM, no GPU used |
| Model | MediaPipe Face Landmarker, 3.6 MB file, used as is (no training or fine-tuning) |
| Training time | none: the rules have no learned weights. "Training" is the threshold grid search below |
| Live inference | 4.9 ms per frame (p50), one process |
| Feature extraction (all datasets) | 5.46 M frames in 25.3 CPU-process hours: about 3.2 h of wall time on 8 parallel processes (17 ms per frame per process under contention), about 0.9 GB RAM each |
| Threshold tuning | about 4 min for 113 grid points × up to 700 clips (16 processes), because it replays saved features instead of video |
| Report generation | about 1 min |
| Money | nothing beyond our own PC: no cloud, no API calls |

**Reproducibility**
- The whole evaluation is deterministic: replays use frame timestamps, logic has
  no randomness, and the grid search is exhaustive. Rerunning gives the same
  numbers.
- The one random draw (dot jitter in one chart) uses a fixed seed (0).
- Every threshold is in `default_config.yaml`. The tuning result is saved in
  `eval/reports/tuning.json`.
- The commands to rebuild every figure and table are in `eval/README.md`.

## 5. Evaluation

### 5.1 Datasets

| Dataset | What it gives us | Clips (tune / test) | Hours (tune / test) |
|---|---|---:|---:|
| NTHU-DDD [N1] (official licensed copy) | Acted drowsiness, active-IR video, day and night, glasses and sunglasses. Per-frame labels for drowsiness, eyes, mouth and head | 376 / 70 | 9.9 / 6.8 |
| DMD [D4] | Real cars. Per-frame eye-state, yawn and gaze-on-road labels | 45 / 20 | 3.0 / 1.3 |
| YawDD [D3] | Short in-car clips: normal, talking, yawning, with varied eyewear | 236 / 113 | 1.8 / 0.9 |
| UTA-RLDD [D2] | 60 people, *real* drowsiness, one video per state (alert, low vigilance, drowsy) | 144 / 36 | 25.2 / 6.0 |
| **Total** | | **801 / 239** | **39.9 / 15.0** |

**Splits are by subject**, so no person appears in both tune and test:
- NTHU-DDD: the official training and evaluation sets are tune, and its testing
  set is test.
- UTA-RLDD: fold 1 is test.
- DMD and YawDD: every third subject is test.

Perception ran once per clip, over several hours on 8 processes. Its features were
then replayed through the real `DriverMonitor`, so tuning and evaluation use the
exact code that runs live. A face was found in 91–100% of frames per dataset.

### 5.2 Metrics

Frame accuracy alone hides what matters for an alerting system, so we measure
three levels:
- **Frame level:** precision, recall and F1 of a rule's condition against
  labels. Examples are eyes closed, looking aside, and head down against nods.
  Calibration time and unlabelled frames are not scored.
- **Event level:** whether each labelled event is caught.
  - A closure or look-aside is caught if its rule fires between the start of the
    event and 1 s after it ends. Only events long enough to count are scored:
    closures ≥ 1.5 s, look-asides ≥ 3 s, nods ≥ 1 s.
  - A yawn counts if a detection falls between 1 s before the label and 2.5 s
    after it.
  - Detections near no label are false events.
- **Alert level:** alert *episodes* per hour on clips where nobody is drowsy, and
  distraction episodes per hour on calm clips. One closure that escalates from
  warning to alarm counts once. Non-drowsy clips are NTHU non-sleepy, YawDD
  normal and talking, UTA alert and DMD distraction sessions.
- **UTA-RLDD per video:** the AUC of mean PERCLOS for drowsy vs alert videos, and
  the share of each class that gets a PERCLOS alert.

### 5.3 Threshold tuning

The default thresholds came from the literature or were rough starting points.
We tuned them by grid search on the tune split, one group at a time, in this
order:

| Group | Objective | Tuned on |
|---|---|---|
| Eye closure | eyes-closed frame F1 | DMD |
| Yawn | yawn event F1 and YawDD clip accuracy | NTHU-DDD, DMD, YawDD |
| Head nod | nod event F1 | NTHU-DDD |
| Distraction | look-aside frame F1 | NTHU-DDD, DMD |
| PERCLOS | UTA drowsy vs alert separation | UTA-RLDD |

**Tuning rules**
- **False-alarm cap:** a grid point was eligible only if both alert rates stayed
  within 1 per hour of the original defaults. They were measured on 308
  non-drowsy and calm tune clips (13.7 h).
- **Eye-closure guard:** long-closure recall could not fall by more than 0.02.
- **Not tuned:** the Euro NCAP timings and the 80% in P80, because they are
  definitions.
- **Proposals only:** the search writes proposed values. A person decides what
  goes into the config.

A test-split check showed the full proposal doubled drowsiness false alarms
(2.08 → 4.76 per hour). Testing one change at a time explained why, and we
applied two of the five groups:

| Threshold | Before | Applied | Why |
|---|---:|---:|---|
| `eyes.closed_ear_fraction` | 0.15 | **0.2** | better eye F1 and closure catch rate on held-out DMD |
| `yawn.mar_threshold` | 0.6 | **0.35** | labelled yawns have a median MAR of only about 0.46 |
| `yawn.min_duration_s` | 1.0 | **0.75** | (tuned together with the line above) |
| `head_nod.pitch_delta_deg` | 15 | 15 (proposed 2) | the search kept lowering it to the edge of the grid; 2° is head-pose noise, and the test split has no nod labels |
| `distraction.pitch_deg` | 25 | 25 (proposed 20) | gain of only +0.01 F1 |
| `perclos.warning` | 0.15 | 0.15 (proposed 0.12) | no gain on the test split, and more false alarms |

### 5.4 Results

Original defaults vs the applied thresholds:

| Metric | Test: before | Test: after | Tune: before | Tune: after |
|---|---:|---:|---:|---:|
| Eyes closed, frame F1 (DMD) | 0.76 | **0.80** | 0.93 | 0.95 |
| Long closures (≥ 1.5 s) caught, DMD | 26% | **39%** | 71% | 80% |
| Yawn events: recall | 38% | **48%** | 47% | 87% |
| Yawn events: precision | 100% | 100% | 98% | 94% |
| YawDD yawning clips with a yawn detected | 59% | **87%** | 52% | 87% |
| YawDD other clips with a yawn detected | 0% | 4% | 0% | 4% |
| Head nod events: F1 (NTHU-DDD) | – | – | 0.13 | 0.19 |
| Looking aside, frame F1 | 0.41 | 0.41 | 0.30 | 0.30 |
| Long look-asides (≥ 3 s) caught | 37% | 37% | 13% | 13% |
| NTHU-DDD drowsy frames, F1 | 0.64 | 0.64 | 0.49 | 0.60 |
| UTA-RLDD PERCLOS AUC, drowsy vs alert | 0.79 | **0.80** | 0.79 | 0.80 |
| UTA drowsy videos with a PERCLOS alert | 17% | 25% | 22% | 24% |
| UTA alert videos with a PERCLOS alert | 0% | 0% | 4% | 4% |
| Drowsiness alert episodes/h, non-drowsy clips | 2.08 | 3.87 | 2.05 | 2.86 |
| Distraction alert episodes/h, calm clips | 12.6 | 12.6 | 4.6 | 4.5 |

"–" means the split has no labels for that metric. NTHU-DDD's testing set has
drowsiness labels only.

**Timing:** a caught long closure was flagged a median of 1.5 s after the eyes
closed. That is the microsleep definition itself, so the rule adds no delay of
its own.

**By condition (test split, applied thresholds)**

NTHU-DDD, drowsy frames F1:

| glasses | night, glasses | night, no glasses | no glasses | sunglasses |
|---:|---:|---:|---:|---:|
| 0.65 | 0.62 | 0.71 | 0.68 | 0.50 |

Night active-IR video does as well as daytime video. Sunglasses are clearly
worst, as expected when the eyes can't be seen.

YawDD, yawning clips with a yawn detected:

| Mirror, no glasses | Mirror, glasses | Mirror, sunglasses | Mirror, glasses + moustache | Dashboard |
|---:|---:|---:|---:|---:|
| 95% | 80% | 80% | 50% | 90% |

DMD, eyes-closed frame F1, by session group:

| gA | gB | gC | gE | gZ |
|---:|---:|---:|---:|---:|
| 0.97 | 0.99 | 0.93 | 0.27 | 0.87 |

Group gE is an outlier. Section 6.3 explains it.

### 5.5 Baseline: one fixed threshold for everyone

Our design calibrates each driver's open-eye EAR. The simpler alternative uses one
absolute EAR threshold for every driver (0.21, a common value). We ran it by forcing
calibration to fall back, keeping everything else the same. The neutral head pose is
still measured.

| Metric | Test: fixed threshold | Test: ours | Tune: fixed threshold | Tune: ours |
|---|---:|---:|---:|---:|
| Eyes closed, frame F1 (DMD) | **0.92** | 0.80 | 0.88 | **0.95** |
| Long closures caught (DMD) | **96%** | 39% | **100%** | 80% |
| NTHU drowsy frames, F1 | 0.68 | 0.64 | 0.70 | 0.60 |
| UTA PERCLOS AUC, drowsy vs alert | **0.90** | 0.80 | 0.86 | 0.80 |
| UTA alert videos with a PERCLOS alert | 0% | 0% | 23% | **4%** |
| Drowsiness alert episodes/h, non-drowsy clips | 8.33 | **3.87** | 17.99 | **2.86** |

The simple baseline catches more closures, but it raises 2.2× (test) to 6.3× (tune)
as many false alarms. On tune it also alerts on almost a quarter of the videos where
the driver is alert.

An alert that fires every few minutes on an alert driver gets switched off, so we
keep per-driver calibration. The missed closures it causes come almost entirely from
one subject (Section 6.3). A better eye-state signal, rather than a fixed threshold,
is the fix.

### 5.6 Ablation: one change at a time

Each tuned value was applied alone to the starting thresholds and replayed on the
held-out non-drowsy and calm clips (3.4 h). This is how we chose which values to
keep.

| Configuration | Drowsiness alerts/h | Effect on detection (test) |
|---|---:|---|
| Starting thresholds | 2.08 | – |
| + `eyes.closed_ear_fraction` 0.2 | 2.68 | eye F1 0.76 → 0.80, closures caught 26% → 39% |
| + `yawn.mar_threshold` 0.35 | 2.68 | yawning clips caught 59% → 87% (with the duration change) |
| + `yawn.min_duration_s` 0.75 | 2.08 | |
| + `perclos.warning` 0.12 | 2.98 | no test gain: **not applied** |
| + `head_nod.pitch_delta_deg` 2 | 2.98 | no test labels to check, and within head-pose noise: **not applied** |
| + `distraction.pitch_deg` 20 | 2.08 | look-aside F1 +0.01: **not applied** |
| **Applied set (eye + yawn)** | **3.87** | |
| All proposed changes | 4.76 | |

## 6. Discussion

### 6.1 What works
- **Eye closure:** frame F1 is 0.93–0.95 on the tune split and 0.80 on the test
  split. The P80 rule, calibrated per driver, tracks real eye closure well.
- **PERCLOS on real drowsiness:** it separates drowsy from alert UTA-RLDD videos
  (AUC 0.80) with no PERCLOS alerts on held-out alert videos.
- **Yawns:** after tuning, 87% of yawning clips are detected with few false
  yawns. Most of the gain came from fixing a threshold that was too high.
- **Speed:** far beyond real time on a CPU.

### 6.2 What doesn't work well yet
- **Head nods:**
  - Event F1 is only 0.13–0.19.
  - Labelled nods drop the head a median of only 8.5°, against our 15°
    threshold.
  - Lowering the threshold enough to catch them reaches the noise level of
    landmark head-pose estimation.
  - A better signal (pitch velocity, or a learned temporal model) is needed.
- **Looking aside:**
  - Frame F1 is 0.30–0.41.
  - The rules use head pose only. Iris gaze isn't computed yet, so eyes-only
    glances are missed.
  - Many labelled "not looking at the road" frames in DMD involve only small
    head movements.
- **Long closures on held-out DMD:** only 39% are caught, compared with 80% on
  the tune split. One driver causes almost all of the gap (Section 6.3).
- **False alarms:**
  - With the applied thresholds there are 2.9 (tune) to 3.9 (test) drowsiness
    alert episodes per hour of non-drowsy driving. We accepted this in return
    for better yawn and closure detection.
  - The test figure rests on 3.4 h of video, so a handful of episodes moves it
    a lot.
  - The cooldown and escalation policy means the driver hears no more than
    this.
- **Distraction alerts on calm clips:** 12.6 per hour on the test split, mostly
  from 20–30 s YawDD clips where drivers glance around the car. UTA-RLDD alone
  gives about 4–6 per hour.

### 6.3 Error analysis

We sorted every miss and every false alarm by what caused it. The checks were run on
the replayed features with the applied thresholds; the counts below are aggregate
only.

**Missed long closures (DMD, test split: 14 of 23 missed)**

| Cause | Count |
|---|---:|
| Eyes judged open: EAR stayed above the closed threshold | 12 |
| Closure broken into pieces shorter than 1.5 s | 2 |
| Face lost, or head turned away | 0 |

- **One driver:** 11 of the 14 misses come from one driver, DMD subject 29
  (session group gE).
- **Why:** for that driver, MediaPipe's eyelid landmarks never come fully
  together. Eyes labelled closed still measure 0.45 × the driver's open EAR.
  Our P80 cut-off is 0.36 × open. For the other DMD groups, labelled-closed eyes
  measure 0.10–0.20 × open, well below the cut-off.
- **Without that driver,** 9 of 11 test closures are caught (82%), the same as on
  tune (80%).
- **What would fix it:** a landmark geometry rule cannot see this closure. The
  fixed-threshold baseline happens to catch it (0.45 × 0.27 ≈ 0.12 < 0.21), but
  over-fires on everyone else (Section 5.5). An appearance-based eye-state CNN
  is the planned fix (Section 7).

**Missed yawns (NTHU-DDD and DMD labels)**

| Cause | Test (11 of 21 missed) | Tune (99 of 775 missed) |
|---|---:|---:|
| Mouth open above the MAR threshold, but for less than 0.75 s | 7 | 58 |
| Mouth never opened past the MAR threshold | 4 | 41 |
| Face lost (e.g. a hand over the mouth) | 0 | 0 |

Most misses are short or half-open yawns, not detection failures. Lowering the
thresholds further would catch them, but tuning showed the false yawns from talking
rise faster than the hits.

**False drowsiness alerts on non-drowsy clips: what set them off**

| Trigger | Test (13 episodes in 3.4 h) | Tune (39 episodes in 13.7 h) |
|---|---:|---:|
| Microsleep (eyes closed ≥ 1.5 s) | 5 | 13 |
| PERCLOS warning | 4 | 9 |
| Yawning (third yawn in 5 min) | 4 | 4 |
| Head nod | 0 | 12 |
| PERCLOS alarm | 0 | 1 |

- **Where they happen:** most are on DMD distraction sessions, where drivers look
  down at the radio or a phone. Looking down most likely narrows the eyes in the
  image, so it reads as closed eyes, or as a nod.
- **What they suggest:** in the next version, a closure or nod rule should not
  fire while the head is pitched down with gaze off-road. This needs the gaze
  feature (Section 7).

### 6.4 Bias across groups

We checked yawn detection on YawDD yawning clips by sex and eyewear, across all
subjects, with 95% Wilson intervals. YawDD records only binary sex, and the
groups are small.

| Group | Yawning clips with a yawn detected |
|---|---|
| Male | 68 / 73 = 93% (85–97%) |
| Female | 56 / 71 = 79% (68–87%) |
| No glasses | 69 / 80 = 86% (77–92%) |
| Glasses | 40 / 46 = 87% (74–94%) |
| Sunglasses | 12 / 13 = 92% (67–99%) |

- **Sex:** the difference between men and women is 14 points. The intervals only
  just overlap, so the difference is plausibly real.
- **A possible cause:** MAR is a ratio of mouth height to width, so drivers who
  yawn with a narrower opening reach the threshold less often.
- **Eyewear:** it doesn't affect yawn detection.
- **Eye-closure rules:** these are hurt by sunglasses. NTHU-DDD sunglasses clips
  score 0.50 drowsy-frame F1, against 0.62–0.71 for the other conditions
  (Section 5.4).
- **Not tested:** we could not test skin tone or age, because none of the
  datasets labels them.

### 6.5 Overfitting and data leakage

**Subject leakage**
- All splits are by subject. A unit test checks the split rule, and we checked
  the full catalogue: no subject is in both splits.
- The remaining risk is YawDD's two camera folders, which number subjects
  separately (Section 6.6).

**Test-set use in model selection**
- The grid search used the tune split only.
- But deciding which tuned values to *keep* looked at test-split false alarms
  (Section 5.6). So the final test numbers are slightly optimistic.
- A fully clean protocol would need a third, untouched split. We judged the data
  too small for that: the test split has only 21 labelled yawns and 23 labelled
  closures.

**Why tune and test differ**

| Measure | Tune | Test (95% CI) |
|---|---|---|
| Yawn recall | 87% (84–89%) | 48% (28–68%) |
| Closure recall | 80% (68–89%) | 39% (22–59%) |

Both gaps are mostly *distribution shift*, not overfitting of thresholds:
- **Yawns:** tune yawns are almost all from NTHU-DDD. Test yawns are only from
  DMD, because NTHU-DDD's testing set has no yawn labels.
- **Closures:** the closure gap is one driver (Section 6.3).
- **The yawn change still helped on test:** YawDD yawning clips caught went from
  59% to 87%, close to the tune value of 87%.

### 6.6 Dataset caveats
- **NTHU-DDD's "sleepy eyes" label marks *drooping* eyes.** On those frames the
  eyes are still about 77% open, so eye thresholds were tuned on DMD, whose
  labels mark fully closed eyes.
- **NTHU-DDD drowsiness is acted, and UTA-RLDD labels are self-reported.** Each
  is only a stand-in for real drowsiness.
- **Possible identity leak in YawDD:** its two camera folders number subjects
  separately, so the same person may appear in both splits.
- **Calibration period not scored:** yawns in a clip's first 5 s fall inside
  calibration and are never scored.

### 6.7 Limitations
- **No recordings of our own yet.** We planned scripted webcam clips covering
  our exact setup, low light and glasses. All results come from public data
  with different cameras.
- **No visible-light night recordings were tested.** NTHU-DDD covers night only
  with active-IR video.
- **Unchecked literature figures:** every figure used in this report was checked
  against its original source. A few numbers in the literature review that the
  report doesn't use are still unchecked (see its Section 7).

## 7. Conclusion and future work

**What we delivered:** a complete, real-time, CPU-only driver monitoring system:
- landmark perception;
- per-driver calibration;
- time-based rules based on PERCLOS and Euro NCAP;
- a state machine with hysteresis;
- driver alerts with escalation;
- a reproducible pipeline that evaluates it on 55 hours of public video.

**What the evaluation showed**
- Eye closure, PERCLOS and yawning work well. Tuning on the tune split improved
  them on the held-out test split.
- Per-driver calibration costs a few missed closures compared with a fixed
  threshold, but cuts false alarms by 2–6×.
- Head-nod and look-away detection are the clear weak points.
- Most misses have a concrete, identified cause: one driver whose eyelid landmarks
  never close, yawns that are short or half-open, and false alarms when the
  driver looks down.

**Next steps, in order of expected value:**
1. Record our own scripted clips (webcam, low light, glasses), to measure false
   alarms in our real setup.
2. Add iris-based gaze to the distraction rule. Also use it to stop closure
   and nod rules firing while the driver looks down at the radio or a phone,
   the main source of false alarms (Section 6.3).
3. Train a small eye-state CNN on MRL Eye for drivers whose eyelid landmarks
   never fully close (Section 6.3), sunglasses and odd head angles, where EAR
   is weakest.
4. Replace the fixed-angle nod rule with one based on pitch velocity, or with
   the planned blink and head-motion LSTM trained on UTA-RLDD.

## References

- [S1] A Survey on Drowsiness Detection – Modern Applications and Methods, 2024.
- [S3] A Review of Recent Developments in Driver Drowsiness Detection Systems, *Sensors*, 2022.
- [E1] Regulation (EU) 2019/2144 (General Safety Regulation): driver drowsiness and attention warning required for new vehicle types from 6 July 2022 and for all new vehicles from 7 July 2024.
- [P1] T. Soukupová and J. Čech, "Real-Time Eye Blink Detection using Facial Landmarks," CVWW 2016.
- [P2] Drowsiness Metrics and Thresholds: The Search for Valid and Reliable Standards, Virginia Tech.
- [P3] D. F. Dinges, M. M. Mallis, G. Maislin and J. W. Powell, "Evaluation of Techniques for Ocular Measurement as an Index of Fatigue and as the Basis for Alertness Management," NHTSA final report, April 1998.
- [P4] T. Abe, "PERCLOS-based technologies for detecting drowsiness: current evidence and future directions," *Sleep Advances*, 2023.
- [R1] Euro NCAP, Technical Bulletin SD 202, Driver Monitoring Test Procedure, version 1.2, July 2026.
- [R3] Euro NCAP, Safe Driving Driver Engagement Protocol, version 1.2, July 2026.
- [I3] Google, MediaPipe Face Landmarker documentation.
- [K1] Y. Kartynnik, A. Ablavatski, I. Grishchenko and M. Grundmann, "Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs," CVPR Workshops 2019 (arXiv 1907.06724).
- [M1] G. Ersoy, M. A. Tatar, E. Tonbul and S. Kırbız, "Improving Driver Drowsiness Detection via Personalized EAR/MAR Thresholds and CNN-Based Classification," arXiv 2604.22479, 2026.
- [V2] "Long-term Multi-granularity Deep Framework for Driver Drowsiness Detection," arXiv 1801.02325, 2018.
- [N1] C.-H. Weng, Y.-H. Lai and S.-H. Lai, "Driver Drowsiness Detection via a Hierarchical Temporal Deep Belief Network," ACCV 2016 Workshops, LNCS 10118, Springer, 2017 (NTHU-DDD).
- [D2] R. Ghoddoosian, M. Galib and V. Athitsos, "A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection," CVPR Workshops 2019 (UTA-RLDD).
- [D3] S. Abtahi, M. Omidyeganeh, S. Shirmohammadi and B. Hariri, "YawDD: A Yawning Detection Dataset," ACM MMSys 2014, pp. 24–28.
- [D4] J. D. Ortega et al., "DMD: A Large-Scale Multi-Modal Driver Monitoring Dataset for Attention and Alertness Analysis," ECCV Workshops 2020.

Links for most bracketed sources are in `docs/literature-review.md`. Euro NCAP
figures were checked against the v1.2 documents (July 2026); the other figures used
here were checked against the original papers or their official records in
November 2026. The full
evaluation output is in `eval/reports/m5-evaluation.md`, and the evaluation
method in `eval/README.md`.

## Appendix: where each grading criterion is covered

This project is in the "other AI" track: rules on top of a pretrained landmark
model, with no agent.

| Criterion | Evidence in this report |
|---|---|
| A1 Problem and goal | Section 1: problem, users, success criteria with targets and results |
| A2 Design and rationale | Section 2: approaches compared and why the simpler or heavier ones were rejected. Section 5.5: measured against a simpler fixed-threshold baseline |
| A3 Implementation and demo | Sections 3–4; `README.md` (install and run); live webcam demo (`python -m drowsiness --source 0`). Reproducibility: Section 4 |
| A4 Evaluation | Section 5: subject-level tune/test split, metrics chosen for alerting, baseline (5.5), ablation (5.6), per-condition results |
| A5 Risks and failure modes | Section 6.3 error analysis with real cases; 6.4 bias; 6.5 overfitting and leakage; 6.6–6.7 caveats and limitations |
| A6 Cost and engineering | Section 4: speed, model size, hardware, extraction and tuning time, money |
