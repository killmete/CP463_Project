# Dataset Access – Downloads, Usage Rules and Own Recordings

Checked on 2026-10-26. Access processes change, so re-check each page before
downloading. Fill in everything in `[brackets]`.

## Summary

We use datasets we can download straight away, NTHU-DDD (obtained under the
lab's signed license, 2026-11), and our own recordings.

| Dataset | Used for | How to get it | Size | Action |
|---|---|---|---|---|
| **UTA-RLDD** | Main drowsiness eval (real, not acted) | Direct Google Drive link | 111.3 GB | Download (§1) |
| **NTHU-DDD** | Drowsiness with per-frame labels; **IR video, night and glasses** | Signed license from the lab | – | **Have it** (official copy) |
| **DMD** (drowsiness + distraction parts) | Distraction / gaze eval | Direct download, 18+, academic only | not stated | Download (§1) |
| **YawDD** | Yawn / MAR eval; varied eyewear | Free IEEE account login | 4.94 GB | Download (§1) |
| **MRL Eye** | Train eye-state CNN (optional); includes glasses, varied lighting | Public download | ~85k images | Download (§1) |
| **CEW** | Train/test eye-state CNN (optional) | Direct Google Drive links | 2.6–20 MB | Download (§1) |
| **Own recordings** | Our exact camera setup; low light and glasses on our webcam; false-alarm checks | Record ourselves | – | Plan (§3) + consent form (§4) before recording |

### NTHU-DDD: official copy only
NTHU-DDD needs a signed license agreement emailed to the lab. We dropped it while
waiting for a reply, then received the official copy (2026-11), so it is back in
use. It is the only dataset here with **infrared night driving and glasses** for
drowsiness, and it has per-frame labels. Use only the copy obtained under the
license; never the Kaggle "NTHU-DDD2" re-upload, and never share ours.

---

## 1. Downloads – checklist

### UTA-RLDD
- Page: <https://sites.google.com/view/utarldd/home>. The data is in a Google
  Drive folder linked from there.
- 180 videos (60 people × 3 states), **111.3 GB**: check disk space first, and
  download to `data/uta-rldd/` (git-ignored).
- Use the dataset's own 5 folds: one fold for testing, the other four for
  training. Splits are by person.
- Questions: Reza Ghoddoosian, `reza_ghoddoosian@honda-ri.com`.

### NTHU-DDD (Driver Drowsiness Detection, NTHU CV Lab)
- Already obtained under the lab's license. Keep the license email/agreement with
  the team lead, and follow its terms where they are stricter than §2.
- Store it in `data/nthu-ddd/` (git-ignored), keeping the original folder structure.
- Content (checked against our copy, 7 GB): all video is **active-infrared** (IR),
  640×480; night scenarios are 15 FPS, the others 30 FPS. Scenarios: noglasses,
  glasses, sunglasses, night_noglasses, night_glasses.
  - **Training** (18 subjects, 356 AVI): 4 clips per scenario: `yawning`,
    `slowBlinkWithNodding`, `sleepyCombination`, `nonsleepyCombination`
    (talking, laughing, looking aside). Subject 005 has no night_glasses clips.
  - **Evaluation** (4 subjects, 20 MP4): one `mix` clip per scenario.
  - **Testing** (14 subjects, 70 MP4): one `mix` clip per scenario.
- Labels: one digit per frame. Training and evaluation have four files per clip:
  `drowsiness` (0 still / 1 drowsy), `eye` (1 sleepy eyes), `head` (1 nodding,
  2 looking aside), `mouth` (1 yawning, 2 talking/laughing). Testing has
  `drowsiness` only. The lab's evaluation code (C/Matlab) is included.
- Use training + evaluation for tuning and testing for the final numbers. Split
  by subject, never by frame.
- Drowsiness is **acted**, so it complements UTA-RLDD (real) rather than replacing it.

### DMD (Driver Monitoring Dataset)
- Page: <https://dmd.vicomtech.org/> → "Download Complete Dataset". It has
  separate services for **Drowsiness**, **Distraction** and **Gaze & Hands**;
  we need Drowsiness and Distraction.
- You must be **18 or older**, and use is **academic only**.
- After a 2026 revision, only **RGB real-car** video remains, and only subjects
  **1, 5, 6, 7, 9, 10, 13, 14, 23, 28, 29, 33, 36, 37** are unrestricted. Use
  only those.
- Contact: `info-dmd@vicomtech.org`.

### YawDD
- Page: <https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset>.
  Listed as Open Access: log in with a free IEEE account (membership not
  needed). The page also mentions a subscription; if it asks you to pay, check
  whether the university library has IEEE DataPort access before paying anything.
- Files: `YawDD.rar.gz` (4.94 GB) **and** the Participants Information file;
  download both, because the second one controls which videos can be shown (§2).

### MRL Eye
- Page: <https://mrl.cs.vsb.cz/eyedataset.html>. It's described as publicly
  available, but the page refused our automated check, so open it in a browser.
  Note the license shown there before using it. Contact listed: Radovan Fusek.

### CEW (Closed Eyes in the Wild)
- Page: <https://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/ClosedEyeDatabases.html>.
  Three Google Drive links; take "eye patches 24x24" (2.6 MB) and "raw face
  images" (20 MB).
- Contact: `x.tan@nuaa.edu.cn`, `f.song@nuaa.edu.cn`.

---

## 2. Usage rules the whole team must follow

| Dataset | Commercial use | Redistribute / share link | Show faces in report or slides | Citation required |
|---|---|---|---|---|
| UTA-RLDD | Not stated – treat as no | Not stated – treat as no | **Only 36 of 60 participants** allowed it; identities must never be revealed | Yes |
| NTHU-DDD | No (license agreement) | **No** | **No** unless the license allows it | Yes |
| DMD | No (CC BY-NC-ND 4.0) | No modified copies | Allowed subjects only | Yes |
| YawDD | Research use | Not stated – treat as no | **Only videos not marked "No"** in Participants Information | Yes |
| MRL Eye | Check page | Check page | – | Yes |
| CEW | No | **No – don't share the link** | – | Yes |
| Own recordings | No | No | Only people who ticked "allow" on the consent form | – |

Practical rules:
- Keep all data in `data/` or `recordings/` (both git-ignored). Never commit it,
  upload it to Drive folders others can see, or post it in chat groups.
- Each team member downloads their own copy. Don't pass files around.
- Before putting any face image in the report or slides, check it against the
  rules above. When in doubt, use frames from our own consenting recordings.
- Delete the data when the course ends unless the license says otherwise.

### Citations to put in the report
- **UTA-RLDD:** R. Ghoddoosian, M. Galib, V. Athitsos, "A Realistic Dataset and
  Baseline Temporal Model for Early Drowsiness Detection," CVPR Workshops, 2019.
- **DMD:** J. D. Ortega, N. Kose, P. Cañas, M.-A. Chao, A. Unnervik, M. Nieto,
  O. Otaegui, L. Salgado, "DMD: A Large-Scale Multi-modal Driver Monitoring
  Dataset for Attention and Alertness Analysis," ECCV 2020 Workshops,
  pp. 387–405.
- **YawDD:** S. Abtahi, M. Omidyeganeh, S. Shirmohammadi, B. Hariri, "YawDD: A
  Yawning Detection Dataset," Proc. ACM Multimedia Systems, 2014, pp. 24–28,
  doi:10.1145/2557642.2563678.
- **NTHU-DDD:** C.-H. Weng, Y.-H. Lai, S.-H. Lai, "Driver Drowsiness Detection via
  a Hierarchical Temporal Deep Belief Network," ACCV 2016 Workshops (use the
  citation the lab's license asks for, if different).
- **CEW:** F. Song, X. Tan, X. Liu, S. Chen, "Eyes Closeness Detection from Still
  Images with Multi-scale Histograms of Principal Oriented Gradients," Pattern
  Recognition, 2014.
- **MRL Eye:** use the citation given on its page.

---

## 3. Own recordings – plan

These are the only data from our exact camera and room setup. NTHU-DDD covers
infrared night driving and glasses; these clips add low light and glasses on our
own webcam, plus false-alarm checks (talking, mirror glances) no dataset labels.

### Setup
- **Camera:** the webcam we'll demo with, placed like a dashcam: in front of
  the "driver", slightly below eye level, 50–80 cm away, face fully in frame.
- **Format:** 640×480, 30 FPS, constant frame rate (avoid phone recordings,
  which are usually variable frame rate). Any recorder works if set to a
  constant 30 FPS, e.g. OBS Studio. Check a clip opens with
  `python -m drowsiness --source <clip> --no-display`; it rejects files with an
  implausible frame rate.
- **Participants:** every team member, plus classmates who consent. Aim for at
  least 6 people, with at least 2 who wear glasses.

### Conditions (record the full script under each)

| Lighting | Eyewear |
|---|---|
| Bright (daylight or well-lit room) | No glasses |
| Normal indoor | Glasses (own prescription) |
| **Low light** – room lights off, one dim lamp | Sunglasses (optional, 1–2 people) |
| **Night-like** – lights off, only a screen or small lamp below the face, like a dashboard | |

Each person records every lighting level, with and without glasses if they
wear them.

### Script (about 2 minutes per clip; follow it in order)
1. **Calibration, 10 s:** look straight ahead, eyes open, blink normally.
2. **Normal driving, 30 s:** natural blinks and small head movements; this is
   the "should not alert" baseline.
3. **Microsleep:** close eyes for about 2 s, twice.
4. **Sleep:** close eyes for about 4 s, once.
5. **Drowsy blinking, 20 s:** slow, heavy blinks (raises PERCLOS).
6. **Yawning:** 3 yawns, a few seconds apart.
7. **Talking or singing, 10 s:** mouth moves but no yawn (false-alarm check).
8. **Head nod:** let the head drop forward for about 1–2 s, twice.
9. **Short glance, ~1 s:** check a side mirror (should **not** alert).
10. **Long distraction, ~4 s:** look down at a phone in your lap.
11. **Repeated glances:** about 6 glances of ~2 s each within 30 s (short
    distraction rule).
12. **No face, ~3 s:** turn away fully or cover the camera.

### Labels and files
- Name files by participant ID and condition, e.g.
  `recordings/P03_nightlike_glasses.mp4`. Never use real names.
- Label each script step as `[start_s, end_s, label]` intervals in
  `eval/labels/<clip>.json`. Write down the start time of each step while
  recording to make this quick.

### Known limitation
Only NTHU-DDD has true **infrared** night video (what real in-car night cameras
use). Our own "night-like" clips are dim visible-light video. Report the two
separately.

---

## 4. Own recordings – consent form

Record nobody without signed consent. Check whether your university requires
ethics approval for recording people; for a course project with only team
members it usually doesn't, but ask your instructor.

> **Consent to be recorded – CP463 Driver Drowsiness Detection Project**
>
> I, [full name], agree to be video-recorded by the [team name] project team
> for the course [course code, University].
>
> **What will be recorded:** short webcam videos of my face while I simulate
> driving behaviours (normal driving, blinking, closing my eyes, yawning,
> nodding, looking away), with and without glasses and in different lighting,
> including dim, night-like lighting.
>
> **How it will be used:** only to develop, test and tune the project's
> detection system, and, if I tick the box below, as images in the project
> report and presentation.
>
> **How it will be stored:** on team members' private computers only, never
> uploaded publicly or committed to the project's GitHub repository. Deleted by
> [date, e.g. end of semester + 1 month].
>
> I can withdraw at any time by telling [contact name/email]; my recordings will
> then be deleted.
>
> ☐ I allow still images of my face to appear in the project report and
> presentation.
> ☐ I do **not** allow my face to appear in any report or presentation.
>
> Signature: ____________  Date: ____________

Keep signed forms with the team lead (not in the repo).
