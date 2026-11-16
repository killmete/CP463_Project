# Evaluation (milestone 5)

Measures how well the rules in `src/drowsiness/logic/` match labelled public datasets,
and proposes threshold changes. It never edits `default_config.yaml`: a person reviews
the report and copies values over by hand.

```
python -m eval.extract --workers 8     # 1. videos -> data/features/<dataset>/<clip>.csv  (~2-3 h, resumable)
python -m eval.tune --workers 16       # 2. grid search on the tune split -> eval/reports/tuning.json,
                                       #    eval/reports/proposed_config.yaml
python -m eval.report --workers 16     # 3. default vs proposed, tune and test splits -> eval/reports/m5-evaluation.md
```

Datasets go in `data/` (gitignored). See `docs/dataset-access.md` for where to get them
and their license terms. Only use the official licensed NTHU-DDD copy. Never commit
videos, frames or feature files: they are recordings of people's faces.

## Stages

| Module | Does |
|---|---|
| `catalog.py` | Finds every clip and assigns subject, split, condition and kind (NTHU-DDD, DMD, YawDD, UTA-RLDD). |
| `extract.py` | Runs perception once per clip. One worker process per clip, with a fresh detector for each clip. Writes feature CSV v1 atomically and logs stats to `data/features/manifest.jsonl`. |
| `labels.py` | Maps each dataset's annotations onto common per-frame tracks: `eyes_closed`, `yawn`, `talking`, `nod`, `look_aside`, `drowsy`. |
| `replay.py` | Feeds features through the real `DriverMonitor`, frame by frame, and records states, active reasons, yawns and alerts. The CSV is cached as `.npz`. |
| `metrics.py` | Frame and event scores for one clip, then summaries: false alarms per hour, YawDD clip-level results, UTA PERCLOS AUC. |
| `tune.py` | Grid search, one parameter group at a time. See the false-alarm guard below. |
| `report.py` | The markdown report plus charts in `eval/reports/figures/`. |

Memory: each extraction worker uses about 0.9 GB. On 16 GB of RAM, use 8 workers at most.

## Splits (by subject, never by frame)

- **NTHU-DDD:** the training and evaluation sets are *tune*; the testing set (14
  subjects, drowsiness labels only) is *test*.
- **UTA-RLDD:** fold 1 (subjects 1–12) is *test*.
- **DMD and YawDD:** every third subject, in sorted order, is *test*.

## What is scored

- **Frame level:**
  - logic's eyes-closed condition vs `eyes_closed`;
  - off-road vs `look_aside`;
  - head-dropped vs `nod`;
  - any drowsiness reason vs NTHU `drowsy`.

  Only frames after calibration and inside each label file's annotated span count.
- **Events:**
  - a labelled closure ≥ `eyes.microsleep_s`, a nod ≥ 1 s, or a
    look-aside ≥ `distraction.long_s` is caught if its rule fires between the event's
    start and 1 s after its end;
  - a yawn is matched to a counted yawn from 1 s before the label to 2.5 s after it.
  - Detections that start inside labelled frames but near no labelled event are false
    events.
  - Event precision = caught labelled events / (caught + false events). It mixes a
    count of labels with a count of detections, which is the usual convention for
    event matching.
- **False alarms:** drowsiness alert *episodes* per hour on clips where nobody is drowsy.
  These are NTHU `nonsleepyCombination`, YawDD Normal/Talking, UTA class 0 and DMD
  distraction sessions. An episode is a stretch with any drowsiness reason active, with
  gaps shorter than `alerts.rearm_s` joined. One closure that escalates from warning to
  alarm counts once. The alert layer's cooldown means the driver hears no more than
  this.
- **Distraction alerts:** episodes per hour on "calm" clips without gaze labels (UTA
  class 0, YawDD Normal/Talking). YawDD clips last 20–30 s and the drivers glance
  around the car, so a few alerts there make a large hourly rate: the test split has
  only 0.37 h of them. UTA alone is the steadier number (about 4/h on tune, 6/h on test
  with the defaults).
- **UTA-RLDD:**
  - the AUC of mean PERCLOS per video, drowsy (10) vs alert (0);
  - the share of each class that gets a PERCLOS alert.

## Tuning rules

- **Not tuned:** Euro NCAP timings, and `eyes.closed_ratio` (the 80 % in PERCLOS-P80).
  These are definitions.
- **Tuned instead:** `eyes.closed_ear_fraction`, the estimate of closed-eye EAR that P80
  is measured against. Calibration only sees open eyes, so this estimate is the
  uncertain part.
- **False-alarm guard:** every group is also replayed on the same set of non-drowsy and
  calm tune clips. A point is eligible only if both alert rates stay within 1 per hour
  of the *original defaults*, so the allowance does not add up across groups.
- **Eye-closure guard:** long-closure recall may not drop by more than 0.02.
- **Ties:** go to the value closest to the current one.
- **Answer keys:** these never depend on a tuned value. A labelled nod is an event if it
  lasts at least 1 s, whatever `head_nod.min_duration_s` is being tried.

## Known label mismatches

- NTHU-DDD's eye label (`sleepy eyes`) marks *drooping* eyes. On labelled frames the
  median openness is about 0.77 of the driver's open EAR. The PERCLOS-P80 rule counts
  an eye as closed only below about 0.32. Frame-level recall against this label is low
  by design, so eye-closure thresholds are tuned on DMD, whose label marks fully
  closed eyes.
- NTHU-DDD's testing set only has drowsiness labels, and nod labels exist only in its
  training set. Metrics without labels on a split show as "–".
- DMD eye frames labelled `undefined`, `opening` or `closing`, and frames with no gaze
  label, are left out. They are neither positives nor negatives.
- YawDD's Dash and Mirror folders number their subjects separately. The same person may
  appear under different numbers, so a small identity leak between the YawDD splits
  can't be ruled out.
- Yawns during the first 5 s of a clip (calibration) are never scored. This lowers the
  hit rate on short YawDD clips.
