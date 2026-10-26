# Dataset Access – Requests, Downloads and Usage Rules

Checked on 2026-10-26. Access processes change, so re-check each page before you
apply. Fill in everything in `[brackets]`.

## Summary

| Dataset | Used for | How to get it | Size | Action |
|---|---|---|---|---|
| **NTHU-DDD** | Drowsiness eval (acted; night, glasses) | **Signed license agreement by email** | – | **Send request (§1) – slow, do it first** |
| **UTA-RLDD** | Main drowsiness eval (real) | Direct Google Drive link | 111.3 GB | Download (§2) |
| **DMD** (drowsiness + distraction parts) | Distraction / gaze eval | Direct download, 18+, academic only | not stated | Download (§2) |
| **YawDD** | Yawn / MAR eval | Free IEEE account login | 4.94 GB | Download (§2) |
| **MRL Eye** | Train eye-state CNN (optional, M3+) | Public download | ~85k images | Download (§2) |
| **CEW** | Train/test eye-state CNN (optional) | Direct Google Drive links | 2.6–20 MB | Download (§2) |
| **Own recordings** | Our camera setup, glasses, low light | Record ourselves | – | **Consent form (§4) before recording** |

Only NTHU-DDD needs a formal request. Everything else is a download, but each one
has usage terms we must follow (§3).

---

## 1. NTHU-DDD – request email

**Where:** NTHU CV Lab dataset page:
<http://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/>. Download the
**Dataset License Agreement** there. The page refused connections from our
automated check, so open it in a normal browser; if it's down, try again later
or from the university network.

**What to send:**
- [ ] The Dataset License Agreement, filled in and **signed**. If it asks for a
      supervisor/advisor, have your course instructor sign as well.
- [ ] Registration form, if the page provides one (the lab's ACCV 2016 workshop
      page asked for both together).
- [ ] Send from your **university email address**, not a personal one.

**To:** `cvlablai636@my.nthu.edu.tw`
**Subject (exactly):** `Dataset on Driver Drowsiness Detection from Video`

> Dear NTHU Computer Vision Lab,
>
> I am a [year/program, e.g. 3rd-year undergraduate in Computer Science] student
> at [University], working on a course project for [course code and name,
> e.g. CP463 ...] under the supervision of [instructor name, title].
>
> Our project develops a real-time, camera-based driver drowsiness and
> inattention detection system using facial landmarks (eye closure, PERCLOS,
> yawning and head pose). We would like to use the NTHU Driver Drowsiness
> Detection dataset to evaluate our method, in particular under night-time and
> eyeglasses conditions.
>
> Please find attached the signed Dataset License Agreement [and registration
> form]. We will use the dataset for non-commercial academic research only,
> will not redistribute it, will store it only on our team's private machines,
> and will cite your work in any report or publication that uses it.
>
> Team members who will access the data: [names, student IDs].
>
> Thank you for considering our request.
>
> Best regards,
> [Your name]
> [Program, University]
> [University email] · [Supervisor name and email]

**Expect a slow reply.** One user reported waiting more than 4 months with no
answer. If there's no response after about 3 weeks, send one polite follow-up
replying to your original email. Plan the project so NTHU-DDD is optional:
UTA-RLDD is the main drowsiness benchmark.

**Don't use the Kaggle "NTHU-DDD2" mirror as a substitute** unless its license
clearly allows it. Reports describe the official data as covered by a
non-disclosure agreement, so a re-upload may not be legitimate.

---

## 2. Direct downloads – checklist

### UTA-RLDD
- Page: <https://sites.google.com/view/utarldd/home>. The data is in a Google
  Drive folder linked from there.
- 180 videos (60 people × 3 states), **111.3 GB**: check disk space first, and
  download to `data/uta-rldd/` (git-ignored).
- Use the dataset's own 5 folds: one fold for testing, the other four for
  training. Splits are by person.
- Questions: Reza Ghoddoosian, `reza_ghoddoosian@honda-ri.com`.

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
  download both, because the second one controls which videos can be shown (§3).

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

## 3. Usage rules the whole team must follow

| Dataset | Commercial use | Redistribute / share link | Show faces in report or slides | Citation required |
|---|---|---|---|---|
| NTHU-DDD | No | No | Check the license agreement | Yes |
| UTA-RLDD | Not stated – treat as no | Not stated – treat as no | **Only 36 of 60 participants** allowed it; identities must never be revealed | Yes |
| DMD | No (CC BY-NC-ND 4.0) | No modified copies | Allowed subjects only | Yes |
| YawDD | Research use | Not stated – treat as no | **Only videos not marked "No"** in Participants Information | Yes |
| MRL Eye | Check page | Check page | – | Yes |
| CEW | No | **No – don't share the link** | – | Yes |

Practical rules:
- Keep all data in `data/` (git-ignored). Never commit it, upload it to Drive
  folders others can see, or post it in chat groups.
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
- **CEW:** F. Song, X. Tan, X. Liu, S. Chen, "Eyes Closeness Detection from Still
  Images with Multi-scale Histograms of Principal Oriented Gradients," Pattern
  Recognition, 2014.
- **NTHU-DDD, MRL Eye:** use the citation given on their pages or in the
  license agreement.

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
> nodding, looking away), with and without glasses and in different lighting.
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

Keep signed forms with the team lead (not in the repo). Name recordings by
participant ID (e.g. `P01_eyes_closed_lowlight.mp4`), not by name.
