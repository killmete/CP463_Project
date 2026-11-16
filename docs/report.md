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

**Speed:** about 5 ms per frame on a laptop-class CPU (p95 5.5 ms at 640×480),
a tenth of our 50 ms budget.

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

**Weakest parts**
- Head-nod detection (event F1 0.13–0.19).
- Look-aside detection from head pose alone (frame F1 0.41).

## 1. Introduction

Drowsy and distracted driving cause a large share of road crashes. Since 2024, EU
rules have required drowsiness warning systems in all new vehicles [S6]. The
Euro NCAP safety rating also scores driver-monitoring systems [R1][R2].

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
| End-to-end deep video models (CNN-LSTM, two-stream, transformers) | Rejected. Far below real time on CPU [V2]. Their high NTHU-DDD scores are often frame-level without subject-disjoint splits, so they can't be compared fairly [V5] |
| Object detectors (YOLO family) | Rejected. A per-frame detector still needs all the timing logic built on top |

What the chosen method rests on:
- **Eye Aspect Ratio (EAR):** Soukupová and Čech reduce six eye landmarks to one
  openness value. It is enough for real-time blink detection [P1].
- **PERCLOS-P80:** the share of a minute that the eyes are at least 80% closed.
  It is the most validated eye-based drowsiness measure [P2][P3]. It can
  under-detect moderate drowsiness on real roads [P4], which is why we added
  further cues.
- **Euro NCAP timings [R1]:**

  | Event | Definition |
  |---|---|
  | Microsleep | Eyes closed for a short time; we use 1.5 s |
  | Sleep | Eyes closed > 3 s |
  | Long distraction | One glance away from the road ≥ 3 s |
  | Short distraction | ≥ 10 s away from the road in total within 30 s |

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
| Sleep | eyes closed > 3 s | drowsy alarm |
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

Group gE is an outlier and needs a closer look.

## 6. Discussion

### What works
- **Eye closure:** frame F1 is 0.93–0.95 on the tune split and 0.80 on the test
  split. The P80 rule, calibrated per driver, tracks real eye closure well.
- **PERCLOS on real drowsiness:** it separates drowsy from alert UTA-RLDD videos
  (AUC 0.80) with no PERCLOS alerts on held-out alert videos.
- **Yawns:** after tuning, 87% of yawning clips are detected with few false
  yawns. Most of the gain came from fixing a threshold that was too high.
- **Speed:** far beyond real time on a CPU.

### What doesn't work well yet
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
  the tune split. The test set is small: 23 events from 5 subjects.
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

### Dataset caveats
- **NTHU-DDD's "sleepy eyes" label marks *drooping* eyes.** On those frames the
  eyes are still about 77% open, so eye thresholds were tuned on DMD, whose
  labels mark fully closed eyes.
- **NTHU-DDD drowsiness is acted, and UTA-RLDD labels are self-reported.** Each
  is only a stand-in for real drowsiness.
- **Possible identity leak in YawDD:** its two camera folders number subjects
  separately, so the same person may appear in both splits.
- **Calibration period not scored:** yawns in a clip's first 5 s fall inside
  calibration and are never scored.

### Limitations
- **No recordings of our own yet.** We planned scripted webcam clips covering
  our exact setup, low light and glasses. All results come from public data
  with different cameras.
- **No visible-light night recordings were tested.** NTHU-DDD covers night only
  with active-IR video.
- **Unchecked literature figures:** some numbers in the literature review come
  from secondary sources and still need checking against the primary papers.

## 7. Conclusion and future work

**What we delivered:** a complete, real-time, CPU-only driver monitoring system:
- landmark perception;
- per-driver calibration;
- time-based rules based on PERCLOS and Euro NCAP;
- a state machine with hysteresis;
- driver alerts with escalation;
- a reproducible pipeline that evaluates it on 55 hours of public video.

**What the evaluation showed:** eye closure, PERCLOS and yawning work well, and
tuning on held-out data improved them. Head-nod and gaze-based distraction
detection are the clear weak points.

**Next steps, in order of expected value:**
1. Record our own scripted clips (webcam, low light, glasses), to measure false
   alarms in our real setup.
2. Add iris-based gaze to the distraction rule.
3. Replace the fixed-angle nod rule with one based on pitch velocity, or with
   the planned blink and head-motion LSTM trained on UTA-RLDD.
4. Train a small eye-state CNN on MRL Eye for sunglasses and odd head angles,
   where EAR is weakest.

## References

- [S1] A Survey on Drowsiness Detection – Modern Applications and Methods, 2024.
- [S3] A Review of Recent Developments in Driver Drowsiness Detection Systems, *Sensors*, 2022.
- [S6] Driver drowsiness detection (overview of the EU requirement), Wikipedia.
- [P1] T. Soukupová and J. Čech, "Real-Time Eye Blink Detection using Facial Landmarks," CVWW 2016.
- [P2] Drowsiness Metrics and Thresholds: The Search for Valid and Reliable Standards, Virginia Tech.
- [P3] Evaluation of techniques for drowsiness detection, Drexel University dissertation.
- [P4] PERCLOS review, PMC, 2023.
- [R1] Euro NCAP, SD-202 Driver Monitoring Test Procedure v1.2.
- [R2] European NCAP Program Developments to Address Driver Distraction, Drowsiness and Sudden Sickness, PMC.
- [I3] Google, MediaPipe Face Landmarker documentation.
- [K1] Y. Kartynnik et al., "Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs," CVPR Workshops 2019.
- [M1] Improving Driver Drowsiness Detection via Personalized EAR/MAR Thresholds and CNN-Based Classification, arXiv 2604.22479.
- [V2] Long-term Multi-granularity Deep Framework for Driver Drowsiness Detection, arXiv 1801.02325.
- [V5] NTHU-DDD dataset overview.
- [N1] C.-H. Weng, Y.-H. Lai and S.-H. Lai, "Driver Drowsiness Detection via a Hierarchical Temporal Deep Belief Network," ACCV 2016 Workshops (NTHU-DDD).
- [D2] R. Ghoddoosian, M. Galib and V. Athitsos, "A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection," CVPR Workshops 2019 (UTA-RLDD).
- [D3] S. Abtahi et al., "YawDD: A Yawning Detection Dataset," ACM MMSys 2014.
- [D4] J. D. Ortega et al., "DMD: A Large-Scale Multi-Modal Driver Monitoring Dataset for Attention and Alertness Analysis," ECCV Workshops 2020.

Links for the bracketed sources are in `docs/literature-review.md`. The full
evaluation output is in `eval/reports/m5-evaluation.md`, and the evaluation
method in `eval/README.md`.
