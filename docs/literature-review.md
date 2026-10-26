# Literature Review – Camera-Based Driver Drowsiness & Inattention Detection

**Purpose:** choose the detection method for CP463 before designing the system.
**Constraints from the project brief:** one RGB dashcam/webcam, real-time at
20–30 FPS, output = an alert message. Assume a normal laptop CPU (no
guaranteed GPU) and a course-project timeline.

---

## 1. Families of methods

Surveys group drowsiness detection into four families: physiological signals,
vehicle-based signals, image/video analysis, and hybrids of these
[S1][S2][S3].

| Family | Signals | Fit for this project |
|---|---|---|
| Physiological | EEG, ECG, EOG | ✗ Needs body-worn sensors. Cross-subject EEG accuracy is only about 73–77% [S4]. |
| Vehicle-based | Steering angle, lane position, speed | ✗ Needs CAN-bus or a simulator. Best lightweight method reached about 88% [S5]. |
| **Vision-based** | Eyes, mouth, head, gaze from a camera | ✓ The only family our input allows, and the one most active in research [S1]. |
| Hybrid | Vision + other signals | Out of scope (we have one camera only). |

The rest of this review covers vision-based methods only.

## 2. Vision-based approaches

### A. Facial landmarks + geometric features + temporal rules
- **Eye Aspect Ratio (EAR)** [P1]: Soukupová & Čech (2016) reduce 6 eye
  landmarks to one number that shows how open the eye is. It works in real
  time with an ordinary camera and matched state-of-the-art blink detection on
  three datasets. The paper decides blinks with an SVM on a short window of
  EAR values, or with an HMM followed by a state machine.
- **PERCLOS** (percentage of eyelid closure): the share of time in a minute
  that the eyes are at least 80% closed (the "P80" definition, from a 1994
  simulator study) [P2]. In Dinges' validation work it was the only measure
  tested that correlated highly with attention lapses on a reaction-time test
  (PVT), with mean r ≈ 0.88 [P3]. That correlation was higher than drivers'
  own sleepiness ratings. A 2023 review cautions that PERCLOS can miss
  moderate drowsiness and is less reliable on real roads than in lab
  vigilance tests [P4].
- **Mouth Aspect Ratio (MAR)** detects yawns (mouth open above a threshold
  for a sustained time).
- **Head pose** is estimated by matching 2D landmarks to a generic 3D face
  model with `cv2.solvePnP` [I1]. Pitch catches head nods; yaw and pitch
  catch looking away.
- **Implementations:** MediaPipe Face Mesh / Face Landmarker gives 468–478
  landmarks. EAR, MAR and head pose computed from it run at about 30 FPS even
  on a Raspberry Pi 5 CPU [I1][I2]. Face Landmarker also outputs 52
  blendshape scores (including `eyeBlink*` and `jawOpen`, each 0–1) [I3].
- **Weaknesses:** fixed EAR/MAR thresholds do not generalise across people
  [M1]. EAR is sensitive to head pose and lighting [I1]. Accuracy drops with
  glasses and occlusion [I1].
  Mitigation: per-driver thresholds improved accuracy by 2–3% over fixed
  ones [M1], and CLAHE preprocessing helps in low light [M1].

### B. Landmark features + a learned temporal model
- Ghoddoosian et al. (CVPR Workshops 2019) released **UTA-RLDD**: 60 people,
  about 30 hours of *real* (not acted) drowsiness video [D2]. Their baseline
  pulls blink features from EAR and feeds windows of 30 blinks into a
  hierarchical multiscale LSTM (HM-LSTM). It detected *early* drowsiness
  more accurately than human judges.
- Others feed 60-frame windows of EAR/MAR from MediaPipe into a CNN-LSTM on
  NTHU-DDD [I1].
- **Strengths:** catches subtle signs before eyes close for long (blink
  duration and frequency). Inputs are tiny feature vectors, so it is cheap to
  run.
- **Weaknesses:** needs labelled training data; slower to react, since a
  30-blink window covers minutes of video.

### C. Appearance CNN for eye and mouth state (on face crops)
- A small CNN on eye crops cut out using landmarks classifies open vs closed.
  It is trained on MRL Eye (about 85k images) or CEW. Reported accuracy:
  - about 99% within the same dataset [M1][M2];
  - 98.8% for yawn detection [M1].
- **Strengths:** more robust than EAR to head pose, glasses and lighting [M1].
  Its output can replace EAR in the same rule pipeline.
- **Weaknesses:** needs training. A MobileNet student project showed a
  training/validation gap (99.7% vs 89.5%), a sign of overfitting [M3].
  Adds a little time per frame.

### D. End-to-end deep video models
- Methods: CNN-LSTM, two-stream (RGB + optical flow), attention CNNs and
  transformers, mostly evaluated on **NTHU-DDD**.
- Reported results on NTHU-DDD [V1][V2][V3][V4]:

  | Model | Accuracy | Notes |
  |---|---|---|
  | Multi-patch CNN + LSTM | 90.05% | 37 FPS on GPU, about 3 FPS on CPU |
  | Two-stream network | 94.46% | official split, 4 held-out subjects |
  | CNN-LSTM | 97.3% | |
  | Attention CNN / preprints | 99.6% and higher | evaluation protocol unclear |

- **Caveats:**
  - NTHU-DDD drowsiness is *acted* [V5].
  - Many 99%+ results are frame-level scores, often without subject-disjoint
    splits, so they cannot be compared fairly [V5].
  - CPU speed is far below our 20–30 FPS target [V2].
  - These models are black boxes and need GPUs and large labelled datasets.

### E. Object detectors (YOLO family)
- YOLOv5–v11 models trained to detect eye/yawn classes reach mAP50 of about
  0.96–0.99 [V6].
- These are per-frame detectors: closure duration, PERCLOS and distraction
  timing would still have to be built on top. They give no advantage over
  landmarks for our use case.

## 3. Defining inattention (distraction)

Euro NCAP's driver-monitoring protocol gives concrete thresholds we can adopt
[R1][R2]. Euro NCAP is a car-safety rating programme, not a law; separately,
EU rules have required drowsiness detection in all new vehicles since 2024
[S6].

| Event | Euro NCAP definition |
|---|---|
| **Long distraction** | One glance away from the road lasting ≥ 3 s |
| **Short / visual time-sharing distraction** | ≥ 10 s looking away in total within a 30 s window |
| **Microsleep** | Brief eye closure (< 3 s; about 1–2 s is the usual example) |
| **Sleep** | Eye closure > 3 s |

These definitions use head pose and gaze plus eye closure, which approach A
already produces.

## 4. Datasets

| Dataset | Content | Use in this project |
|---|---|---|
| **UTA-RLDD** [D2] | 60 subjects, about 30 h, real drowsiness in 3 classes (alert / low-vigilance / drowsy) | Main drowsiness evaluation |
| **YawDD** [D3] | 322 + 29 in-car videos: talking, singing, yawning; varied eyewear | Yawn / MAR evaluation; talking is a false-positive test |
| **DMD** [D4] | Originally 37 drivers, 41 h of RGB, depth and IR. Since a 2026 revision only RGB real-car video from 14 subjects is unrestricted | Distraction / gaze evaluation |
| **MRL Eye / CEW** [M1][M2] | Eye crops labelled open or closed; MRL includes glasses and varied lighting | Training the optional eye-state CNN (approach C) |
| **Own recordings** | Team webcam clips following a fixed script, under bright, normal, low-light and night-like lighting, with and without glasses | **Night and glasses robustness** (replaces NTHU-DDD); our exact camera setup |
| ~~NTHU-DDD~~ [V5] | Acted drowsiness, day/night, glasses | **Not used.** Access needs a signed agreement emailed to the lab, with replies taking months. Cited only for other papers' results (§2D) |

Access details, usage rules and the recording plan are in
[dataset-access.md](dataset-access.md).

Evaluation protocol, given the leakage concerns above:
- Split by **subject**, never by frame.
- Report **event-level** precision and recall, detection delay, and false
  alarms per hour, not just frame accuracy.

## 5. Comparison

| Criterion (weight) | A. Landmarks + rules | B. Landmarks + temporal ML | C. Eye-state CNN + rules | D. End-to-end deep video | E. YOLO |
|---|---|---|---|---|---|
| Real-time on CPU at 20–30 FPS (high) | ✓✓ about 30 FPS even on a Raspberry Pi | ✓✓ | ✓ small extra cost | ✗ about 3 FPS on CPU | ~ needs GPU for real time |
| Detects microsleep, nodding, yawning, distraction (high) | ✓✓ all of them, directly | ~ drowsiness only | ✓ eyes and mouth | ~ drowsy/not only | ~ per-frame only |
| Robust to pose, glasses, lighting (med) | ~ needs calibration and CLAHE | ~ | ✓ | ✓ when training data matches | ✓ |
| Data and training needs (med) | ✓✓ none | ~ labelled sequences | ~ public eye datasets | ✗ large labelled video | ~ labelled boxes |
| Explainable, easy to tune (med) | ✓✓ | ~ | ✓ | ✗ | ~ |
| Matches standards: PERCLOS, Euro NCAP (med) | ✓✓ | ~ | ✓ | ✗ | ✗ |
| Effort / risk for a course project (med) | ✓✓ low | ~ medium | ~ medium | ✗ high | ~ medium |

## 6. Recommendation

**Use a hybrid built on approach A, with C and B as planned upgrades.**

1. **Core (MVP), approach A:**
   - **Landmarks:** MediaPipe Face Landmarker (478 landmarks + blendshapes)
     produces EAR, MAR, head pose and iris-based gaze.
   - **Decision layer:** time-based rules, using seconds rather than frame
     counts. These follow PERCLOS and Euro NCAP:
     - microsleep: eyes closed ≥ 1.5 s, raises a warning;
     - sleep: eyes closed > 3 s, raises the alarm;
     - PERCLOS over 60 s;
     - yawn frequency;
     - head nods;
     - long distraction: ≥ 3 s away from the road;
     - short distraction: ≥ 10 s away within 30 s;
     - no face.
   - **Calibration:** per-driver baseline from the first seconds, with CLAHE
     for low light.
   - Meets the 20–30 FPS target on a CPU, needs no training, and every alert
     can be explained.
2. **Upgrade 1, approach C:** a lightweight eye-state CNN (MobileNet-sized)
   on landmark-cropped eyes, trained on MRL Eye. Its output replaces or fuses
   with EAR. Add it only if evaluation shows EAR failing on glasses, head
   pose or night footage.
3. **Upgrade 2, approach B (stretch):** blink-sequence features (duration,
   frequency) feeding a small LSTM, trained and tested on UTA-RLDD, for
   *early* drowsiness warnings.
4. **Not recommended as the primary method:**
   - **D (end-to-end video):** too slow on CPU, acted training data, and
     benchmark numbers that cannot be compared fairly.
   - **E (YOLO):** no timing logic, which we would have to build anyway.

### Changes to the current design
- Raise the "looking away" rule from 2 s to **3 s** (Euro NCAP long
  distraction).
- Add a **short-distraction** rule: ≥ 10 s total within 30 s.
- Add a **sleep** rule (eyes closed > 3 s) as the top alarm level.
- Approximate PERCLOS-P80 with EAR below 20% of the way from closed to
  calibrated open, i.e. `EAR < closed + 0.2 × (open − closed)`.

## 7. Limitations of this review
- Several figures come from secondary sources, reviews or preprints. Before
  citing them in the final report, check them against the primary papers,
  especially the 99%+ NTHU-DDD claims and the original 1994 P80 study.
- Default thresholds (EAR ratio, MAR, angles) are starting points. The
  `data-eval-engineer` agent must tune them on our own data.
- None of our evaluation data includes true infrared night video, which real
  in-car night cameras use. Night robustness is tested only on our own dim,
  visible-light "night-like" recordings.

---

## Sources
**Surveys**
- [S1] [A Survey on Drowsiness Detection – Modern Applications and Methods (2024)](https://www.researchgate.net/publication/380279984_A_Survey_on_Drowsiness_Detection_-_Modern_Applications_and_Methods)
- [S2] [Intelligent driver monitoring systems: a survey of drowsiness detection technologies (Springer AIR)](https://link.springer.com/article/10.1007/s10462-026-11505-w)
- [S3] [A Review of Recent Developments in Driver Drowsiness Detection Systems (Sensors, PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8914892/)
- [S4] [Calibration-Free EEG-based Driver Drowsiness Detection with Online Test-Time Adaptation](https://arxiv.org/pdf/2511.22030)
- [S5] [Comparison of Lightweight Methods for Vehicle Dynamics-Based Driver Drowsiness Detection](https://arxiv.org/pdf/2506.07014)
- [S6] [Driver drowsiness detection (Wikipedia)](https://en.wikipedia.org/wiki/Driver_drowsiness_detection)

**Eye closure and PERCLOS**
- [P1] [Soukupová & Čech, Real-Time Eye Blink Detection using Facial Landmarks (CVWW 2016)](https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf)
- [P2] [Drowsiness Metrics and Thresholds: The Search for Valid and Reliable Standards (VT)](https://vtechworks.lib.vt.edu/items/ad3c3cb5-061d-43e6-b329-b338bbe305cc)
- [P3] [Evaluation of techniques for drowsiness detection (Drexel dissertation)](https://researchdiscovery.drexel.edu/esploro/outputs/doctoral/Evaluation-of-techniques-for-drowsiness-detection/991021889097704721)
- [P4] [PERCLOS review (PMC, 2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10108649/)

**Implementations (MediaPipe)**
- [I1] [DrowSAFE: real-time drowsiness detection with MediaPipe (Hackster)](https://www.hackster.io/MohamedAliBedair/drowsafe-real-time-driver-drowsiness-detection-system-aeee70)
- [I2] [Low-cost AIoT drowsiness detection using MediaPipe and ESP32](https://granthaalayahpublication.org/journals/granthaalayah/article/view/7036)
- [I3] [MediaPipe face mesh / blendshapes package docs](https://pub.dev/documentation/mediapipe_face_mesh/2.9.0/)

**Eye-state CNNs**
- [M1] [Improving Driver Drowsiness Detection via Personalized EAR/MAR Thresholds and CNN-Based Classification (arXiv 2604.22479)](https://arxiv.org/pdf/2604.22479)
- [M2] [EyeNet (Univ. of Milan)](https://air.unimi.it/handle/2434/818554)
- [M3] [MobileNetV1 eye-state classifier (Khwopa Engineering College)](https://elibrary.khwopa.edu.np/handle/123456789/875)

**End-to-end video models and detectors**
- [V1] [Real-time driver drowsiness detection using transformer architectures (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12092738)
- [V2] [Long-term Multi-granularity Deep Framework for Driver Drowsiness Detection](https://arxiv.org/pdf/1801.02325)
- [V3] [Dynamic driver drowsiness detection with attention-enhanced CNNs (Sci. Reports)](https://www.nature.com/articles/s41598-025-33727-8)
- [V4] [Real-Time Drivers' Drowsiness Detection and Analysis through Deep Learning](https://arxiv.org/pdf/2511.12438)
- [V5] [NTHU-DDD dataset overview](https://www.emergentmind.com/topics/nthu-ddd-dataset)
- [V6] [Vision-Based Driver Drowsiness Monitoring: Comparative Analysis of YOLOv5–v11](https://arxiv.org/html/2509.17498v1)

**Datasets**
- [D2] [Ghoddoosian et al., A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection (UTA-RLDD)](https://arxiv.org/pdf/1904.07312)
- [D3] [YawDD: Yawning Detection Dataset (IEEE DataPort)](https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset)
- [D4] [DMD: A Large-Scale Multi-Modal Driver Monitoring Dataset (ECCVW 2020)](https://arxiv.org/abs/2008.12085)

**Regulation and standards**
- [R1] [Euro NCAP SD-202 Driver Monitoring Test Procedure v1.2](https://cdn.euroncap.com/cars/assets/SD_202_Driver_Monitoring_Test_Procedure_v1_2_d3420cb629.pdf)
- [R2] [European NCAP Program Developments to Address Driver Distraction, Drowsiness and Sudden Sickness (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10790826/)
