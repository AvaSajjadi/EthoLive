# EthoLive 0.4, mouse (model card)

> This is the development record, kept as written (newest sections last). Paths under `outputs/` and `backups/` refer to
> the development workspace; the recorded outputs behind the paper's numbers are in [`results/`](../results/), and the
> model files are listed in [`models/README.md`](../models/README.md). Commands use the `etholive` package of this repository.

EthoLive (from ethology, the study of animal behavior, and live). The name is kept in one constant (`MODEL_NAME` in `etholive/model.py`).

EthoLive is one model built from the pieces of the models this project used. One file: `models/etholive/mouse.pt` (10.8 MB).

```
camera picture
  Section 1, Eyes         network taken from YOLO26n-pose, trained on the MARS hand-labeled pictures;
                          our own picture preparation, mouse choice (one black = a, one white = b) and short-miss memory
  -> 7 keypoints per mouse
  ScopeScore steps        9 methods, fixed (not learned): quality flags, swap fix, per-keypoint One-Euro, motion,
                          body axis, spine center and bend, distance to the picture edge
  -> 120 features
  Section 2, Behavior     logits = frame branch (this frame) + memory branch (causal, last 2.1 s)
  -> attack / investigation / mount
```

| Piece | Taken from | Where in EthoLive |
|---|---|---|
| backbone, neck, keypoint head (2.9 M weights) | YOLO26n-pose (Ultralytics, AGPL-3.0) | Section 1 |
| teaching pictures and keypoint labels; 7 keypoints, black / white identity | MARS (Segalin et al., 2021, CC BY-NC 4.0) | Section 1 |
| 9 tracking and geometry methods | ScopeScore (Sajjadi, 2026) | between the sections (`scopescore_mouse.py`, `quality.py`, `filters.py`, `features.py`) |
| 102 measurements, class balancing, fitted coefficients | the hackathon classifier, Model B | Section 2, frame branch (starts from it) |
| causal dilated convolution | new | Section 2, temporal branch |

The two sections are trained separately: the MARS set has pictures with keypoints but no behavior labels; CalMS21 has
behavior labels on keypoints but no pictures. MARS's own 560 MB networks are not in EthoLive; the file names them
(with md5s) as the offline route for recorded videos. CalMS21's keypoints are MARS keypoints.

## Section 1: Eyes

- Same output as the Ultralytics predictor with the same weights: on 300 pictures of a real CalMS21 video, the largest
  keypoint difference is 0.0002 px (CPU) / 0.0 px (Apple GPU), and both find the same mice in every picture
  (`python -m etholive.model check-eyes`). No accuracy was gained or lost: the MARS accuracy of the network is
  in `outputs/mars_accuracy.json` (median 6.5 px).
- Speed (MacBook, Apple GPU): 6.7 ms per picture after fusing each convolution with its batch norm (11.6 ms unfused).
  Half precision gave 6.2 ms but moved keypoints by up to 6 px, so it is not used.

## Section 2: Behavior (validation, 14 sessions, 3 seeds; never test)

`outputs/etholive/section2_validation.csv`. Features: all 9 ScopeScore methods. Rule v1 for alerts.

| Model | Macro F1 | Attack F1 | Investigation F1 | Mount F1 | Alert precision | Alert recall | False alerts / h |
|---|---|---|---|---|---|---|---|
| Frame branch alone (= Model B, Section 2 before training) | 0.636 | | | | | | |
| Temporal head alone, methods 1–3 (MouseScope 0.1 design) | 0.739 ± 0.013 | 0.708 | 0.829 | 0.679 | 0.80 | 0.70 | 13.7 |
| Temporal head alone, all 9 methods | 0.739 ± 0.007 | 0.708 | 0.830 | 0.681 | 0.77 | 0.77 | 16.8 |
| **EthoLive Section 2 (frame + memory branch), all 9 methods** | **0.743 ± 0.010** | 0.719 | 0.825 | 0.685 | **0.86** | 0.70 | **9.5** |

EthoLive 0.3 shipped seed 0 (declared before training): macro F1 0.745. EthoLive 0.4 ships the team (below).

## Improvement: EthoLive 0.4 (validation, never test)

`scripts/improve_etholive.py`, `outputs/etholive/improve.csv`. Each design trained with 3 seeds; "team" = the 3 nets vote
(mean of their probabilities). The team needs no new training and adds about 20 ms per 15-frame chunk.

| Design | Single net (mean of 3) | Team of 3 | Team: attack F1 | Team: mount F1 | Team: false alerts / h |
|---|---|---|---|---|---|
| **width 64, memory 2.1 s (0.3 design)** | 0.743 | **0.771** | 0.741 | 0.733 | 11.5 |
| width 64, memory 4.2 s | 0.735 | 0.766 | 0.734 | 0.732 | 12.6 |
| width 128, memory 2.1 s | 0.730 | 0.763 | 0.751 | 0.707 | 8.0 |

Longer memory and a wider net did not help; voting helped every design by about +0.03 macro F1 (three times the seed
spread), most of all mount (0.685 -> 0.733). EthoLive 0.4 = the 0.3 design, seeds 0, 1, 2 voting. Choosing the best of the
three teams on validation makes 0.771 slightly optimistic.

## End to end

Real CalMS21 video, 600 pictures, Section 1 -> ScopeScore steps -> Section 2 through the streaming scorer
(`live_scoring.LiveScorer`): 75 pictures per second on the MacBook (the camera gives 30). Live scores equal whole-session
scores (test). The chunk size of that measurement was not recorded. Re-measured on 2026-11-26 with
`scripts/bench_end_to_end.py` (Apple M5, MPS, 600 frames, video decoding excluded; `results/benchmarks/`): 81-99 frames per
second with chunks of 15 frames (LiveTracker's live-camera setting) and 61-63 with chunks of 5. A score row is emitted
once its chunk is full, so throughput is not latency: with chunks of 15 a frame's score arrives up to 0.5 s after the frame.

## Protocol and limits

- All choices used train and validation only. The CalMS21 test split was used once, for MouseScope 0.1
  (`docs/mousescope.md`); a test run of EthoLive would be the second look and must be reported as such.
- Differences between the Section 2 variants are about one seed spread; the clearest gain is fewer false attack alerts.
- Section 1 is not retrained here; its accuracy is the MARS-trained YOLO network's.
- Earlier versions are kept: `backups/mousescope_0.1_2026-10-31/`, `backups/livescope_0.2_2026-11-01/`.

## Fly pack (EthoLive fly, `models/etholive/fly.pt`, 11.2 MB with eyes; now fly 0.7, see below)

Data: Caltech Fly vs. Fly, Aggression set (Eyjolfsdottir et al., 2014; CC0): 10 movies of 2 male flies, 30 min each at
30 fps, 144 x 144 px chambers. Official split: movies 1-5 train, 6-10 test; EthoLive keeps movie 3 for validation
(`splits/flyvsfly_split.json`). Code: `flyvsfly.py` (loader, labels, scores), `scopescore_fly.py` (ScopeScore steps),
`scripts/build_fly_features.py`, `scripts/train_fly.py`, `scripts/pack_fly.py`.

- Keypoints (5 per fly) from the Fly vs. Fly tracks: head, thorax, abdomen, left and right wing tip.
- Behaviors: lunge, wing threat, tussle, hold (+ other). Charge (172 frames in all) and "unknown" are left out.
- ScopeScore steps for flies (94 features): quality flags; head-first fix (the head is the end away from the wings;
  0.05% of frames); per-point One-Euro (head = tip, abdomen = start); motion with fly-sized settings; body axis;
  wing angles, spread and asymmetry (the tool's jaw); facing; distance to the chamber edge.
- Section 2: the same design as the mouse; the frame branch starts from a fly logistic regression (no hackathon model).
- Section 1 (eyes): added later, see "Fly eyes" below.

Validation (movie 3, never test; `outputs/etholive/fly/validation.csv`):

| Model | Macro F1 | Lunge F1 | Wing threat F1 | Tussle F1 | Hold F1 | Lunge bouts found | Lunge bouts right |
|---|---|---|---|---|---|---|---|
| Frame branch alone | 0.351 | 0.297 | 0.303 | 0.465 | 0.341 | 96% | 48% |
| Frame + memory branch (mean of 3 seeds) | 0.482 | 0.401 | 0.626 | 0.517 | 0.381 | 79% | 45% |
| **Team of 3 (shipped)** | **0.526** | 0.434 | 0.693 | 0.560 | 0.416 | 82% | 52% |

Limits: one validation movie (tussle 261 and hold 111 frames), so the fly numbers are noisy; lunges last about 4 frames,
so frame F1 undersells them (82% of lunges are found).

Tried and not kept: adding Boy meets boy movies 1-3 (official train half, 741k frames) to training lowered validation
macro F1 from 0.526 to 0.458 (lunge 0.43 -> 0.27, wing threat 0.69 -> 0.51; `outputs/etholive/fly_bmb/validation.csv`).
Likely causes: Boy meets boy does not label tussle or hold, and its labels come from another context and annotation
pass, so many of its "other" frames teach the wrong thing. The shipped fly file uses the Aggression set only.

## Held-out test (run once, 2026-11-02; `scripts/test_run_etholive.py`, `outputs/test_run_etholive/`)

Both model files and the feature code were checksummed before any test data was read (`frozen_checksums.txt`).
95% intervals resample whole sessions (mouse) or whole movies (fly).

**Mouse, CalMS21 Task 1 test, 19 sessions.** Second look at this split (the first: MouseScope 0.1, 2026-10-31).
Attack rule v1 for every model.

| Model | Macro F1 (95% CI) | Attack F1 | Investigation F1 | Mount F1 | Alert precision | Alert recall | False alerts / h |
|---|---|---|---|---|---|---|---|
| Model B (hackathon) | 0.755 (0.666–0.779) | 0.592 | 0.782 | 0.890 | 0.83 | 0.52 | 13.2 |
| MouseScope 0.1 | 0.818 (0.772–0.837) | 0.730 | 0.823 | 0.901 | 0.94 | 0.67 | 5.4 |
| **EthoLive 0.4** | **0.847 (0.811–0.868)** | **0.778** | **0.848** | **0.915** | **0.99** | 0.67 | **1.2** |

Paired differences (same resampled sessions): EthoLive - MouseScope 0.1: macro F1 +0.019 to +0.063, attack F1
+0.022 to +0.151, alert precision +0.012 to +0.187, alert recall -0.05 to +0.06 (no change). EthoLive - Model B:
macro F1 +0.064 to +0.169, alert recall +0.07 to +0.17.

**Fly, Fly vs. Fly Aggression movies 6-10.** First look at these movies.

| Model | Macro F1 (95% CI) | Lunge F1 | Wing threat F1 | Tussle F1 | Hold F1 | Lunges found | Lunge calls right |
|---|---|---|---|---|---|---|---|
| Frame branch alone (logistic regression) | 0.373 (0.255–0.473) | 0.317 | 0.261 | 0.533 | 0.381 | 93% | 44% |
| **EthoLive 0.4 fly** | **0.524 (0.359–0.608)** | 0.451 | 0.604 | 0.619 | 0.422 | 83% | 50% |

Paired difference in macro F1: +0.092 to +0.191. Only 5 test movies, so the fly intervals are wide.

## External test: new videos (run once, 2026-11-02; `scripts/external_test.py`, `outputs/external_test/`)

**Mouse: CalMS21 Task 2, 77 videos (1.5 M frames), labelled by 5 annotators.** None of the videos is in Task 1
(keypoint fingerprints, `calms21_task2_overlap.csv`). Same camera setup as training; other people's labelling styles.

| Model | Macro F1 (95% CI) | Attack F1 | Investigation F1 | Mount F1 | Alert precision | Alert recall | False alerts / h |
|---|---|---|---|---|---|---|---|
| Model B | 0.697 (0.665–0.720) | 0.575 | 0.713 | 0.804 | 0.80 | 0.55 | 15.8 |
| MouseScope 0.1 | 0.760 (0.729–0.784) | 0.686 | 0.755 | 0.838 | 0.88 | 0.68 | 10.8 |
| **EthoLive 0.4** | **0.773 (0.741–0.799)** | **0.704** | **0.769** | **0.847** | **0.90** | 0.63 | **8.3** |

EthoLive - MouseScope 0.1: macro F1 +0.010 to +0.018, attack F1 +0.008 to +0.031. EthoLive is best for every one of
the 5 annotators (macro F1 0.646 to 0.833); the annotator matters more than the model (annotator 3: 0.65, annotator 1: 0.83).

**Fly: Boy meets boy movies 4-6 (another lab setup: bigger arena, flies 29 px long), poses rescaled (flyvsfly.calibrate).**

| | Lunge F1 | Wing threat F1 | Lunges found | Lunge calls right |
|---|---|---|---|---|
| EthoLive 0.4 fly | 0.03 | 0.20 | 40% | 4% |
| Second human annotator (agreement with the first) | 0.82 | 0.82 | 100% | 80% |

The fly model does not transfer to this setup: it was trained on 4 Aggression movies from one setup with many fights,
and in calm Boy meets boy recordings it calls far too many lunges. A fly model for other setups needs training data from
them. This is a limit of the fly pack, not of the method: the mouse pack, trained on 56 sessions, holds on 77 new videos.

## Fly 0.5: fixed for a second camera setup (2026-11-02)

`scripts/train_fly_v2.py --sqrt`, `models/etholive/fly.pt` (0.4 kept in `backups/etholive_fly_0.4_2026-11-02/`).
Three changes, all chosen on validation (Aggression 3 + Boy meets boy 3):
1. both setups in training (Aggression 1, 2, 4, 5 + Boy meets boy 1, 2), Boy meets boy rescaled with `flyvsfly.calibrate`;
2. partial labels: Boy meets boy does not label tussle or hold, so its other frames count as right for any of other /
   tussle / hold (`etholive._PartialLoss`);
3. gentler class weights (square root of "balanced"): rare actions such as lunge were over-called in calm recordings.

Validation (combined = mean of Aggression macro F1 and Boy meets boy mean F1 of lunge and wing threat):

| Model | Combined | Aggression macro F1 | BMB lunge F1 | BMB wing threat F1 |
|---|---|---|---|---|
| fly 0.4 (Aggression only) | 0.314 | 0.526 | 0.02 | 0.18 |
| both setups, balanced weights (team) | 0.464 | 0.485 | 0.14 | 0.75 |
| **both setups, gentler weights (team) = fly 0.5** | **0.541** | **0.540** | **0.29** | **0.80** |

Test, run once (`scripts/test_run_fly_v2.py`, `outputs/test_run_fly_v2/`). **Second look at both fly test sets**
(the first: `outputs/test_run_etholive/`, `outputs/external_test/`); 0.5 was chosen before this run.

| Test set | Model | Macro F1 | Lunge F1 | Wing threat F1 | Tussle F1 | Hold F1 | Lunges found | Lunge calls right |
|---|---|---|---|---|---|---|---|---|
| Aggression 6-10 | fly 0.4 | 0.524 | 0.451 | 0.604 | 0.619 | 0.422 | 83% | 50% |
| Aggression 6-10 | **fly 0.5** | **0.554** | **0.492** | **0.655** | 0.617 | **0.454** | **94%** | 49% |
| Boy meets boy 4-6 | fly 0.4 | | 0.031 | 0.201 | | | 40% | 4% |
| Boy meets boy 4-6 | **fly 0.5** | | **0.191** | **0.660** | | | **85%** | **19%** |
| Boy meets boy 4-6 | second human annotator | | 0.82 | 0.82 | | | 100% | 80% |

Still open: lunges in calm recordings (19% of lunge calls are right; humans agree at 0.82). Fly eyes and live calibration came
later (sections below).

## Fly eyes (Section 1 for flies, 2026-11-02; now in `models/etholive/fly.pt`)

`scripts/fly_pose_dataset.py`, `scripts/train_yolo_pose.py --degrees 180`, `scripts/check_fly_eyes.py`, `scripts/pack_fly.py`.
- Network: YOLO26n-pose, 1 class ("fly"), 5 points, 160 x 160 input, trained on Fly vs. Fly Aggression frames with the
  data set's FlyTracker points as labels (automatic, not hand-marked): every 60th frame plus every 5th frame during a
  labelled action (train movies 1, 2, 4, 5: 8,744 pictures); movie 3 for validation (900 pictures); test movies not read.
- `etholive.FlyEyes`: the picture keeps its own size; the second fly is the best detection not overlapping the first;
  fly a / b identity is kept by matching to the last positions (the flies look alike).
- The first eyes (calm frames only) never showed spread wings: wing threat F1 0.01. Adding action frames fixed it.

Validation movie 3, every frame (54,000):

| | Value |
|---|---|
| both flies found | 99.998% of pictures |
| median point error vs FlyTracker | head 1.2 px, thorax 0.7, abdomen 0.8, wings 1.1 (a fly is about 19 px long) |
| identity swaps vs FlyTracker | 0.37 per minute |
| speed (MacBook, Apple GPU) | 190 pictures per second (112 including the brain, live) |

| Fly 0.5 brain on | Macro F1 | Lunge | Wing threat | Tussle | Hold |
|---|---|---|---|---|---|
| FlyTracker points | 0.540 | 0.458 | 0.664 | 0.535 | 0.502 |
| **EthoLive fly eyes** | **0.582** | 0.432 | **0.748** | **0.555** | **0.591** |

Not yet tested on the test movies (they have been used twice for the brain); the eyes were chosen on movie 3 only.

## Fly: tried after 0.5, not kept (validation only, 2026-11-02)

| Try | Validation combined | Kept? |
|---|---|---|
| fly 0.5 (shipped) | 0.541 | |
| + head ("tip") motion features: head speed, acceleration, approach to the other fly, wing opening rate (`scripts/train_fly_v2.py --sqrt --tip`, `outputs/etholive/fly_v3/`) | 0.527 | no |
| + per-behavior alarm levels (`scripts/fly_alarm_levels.py`): fitted on one half of each validation movie, scored on the other half | held-out halves 0.458 -> 0.457 and 0.513 -> 0.450 | no |

The alarm levels raise the score on the frames they are fitted on (0.541 -> 0.554) but not on frames they did not see:
with 2 validation movies there are too few lunges in calm recordings to set them reliably. The code stays
(`scope_model.apply_levels`, `log_levels` in the model file) for when more labelled data exists.

## Fly 0.6: brain trained on the eyes' own points (2026-11-04)

`scripts/train_fly_on_eyes.py --mix|--eyes`, `scripts/pack_fly.py`, results in
`outputs/etholive/fly_on_eyes_{mix,eyes}/validation.csv`; 0.5 with eyes kept in `backups/etholive_fly_0.5_with_eyes_2026-11-04/`.
Idea: live, the brain only sees the eyes' points, but fly 0.5 learned from the FlyTracker points. Same recipe as 0.5; only
Aggression has video, so Boy meets boy keeps tracker points. Fly 0.6 = `--eyes`.

Validation (Aggression 3 from the eyes + Boy meets boy 3; every row scored on those same points; team of 3 unless marked):

| Brain | Combined | Aggression macro F1 | Boy meets boy lunge + wing | Boy meets boy lunge | Lunge calls right (Boy meets boy) |
|---|---|---|---|---|---|
| fly 0.5 | 0.553 | 0.565 | 0.542 | 0.285 | 15% |
| `--mix` (tracker + eye points) | 0.562 | | | 0.230 | |
| **`--eyes` = fly 0.6** | **0.576** | **0.595** | **0.557** | **0.315** | **26%** |
| `--eyes`, single seeds | 0.525-0.545 | 0.555-0.575 | 0.490-0.516 | 0.206-0.304 | |

The packed file reproduces 0.576 exactly. Limits: the gain is small and comes from the team (each single seed is below fly
0.5's team), measured on one movie per setup. The eyes were trained on pictures from Aggression 1, 2, 4, 5, so their points
there are cleaner than on a new video; movie 3 was never shown to them.

Fly 0.6 did not hold up on the test movies: it scored below fly 0.5 there (see "Fly 0.7: test" below).

## Fly: running live (2026-11-04)

Two faults found by running a whole movie through `LiveScorer`; neither showed in the earlier tests:
- **A live fly run crashed** when no fly was found (`KeyError`): the camera's "nothing found" rows have mouse keypoints.
  Each frame is now filled to the file's animals and keypoints, so missing flies count as lost tracking.
- **Live runs had no calibration**: `flyvsfly.calibrate` was only used offline, and on another camera the brain then
  scores nothing (Boy meets boy 3: lunge and wing threat F1 0.00). A fly camera now uses `setup="auto"`
  (LiveTracker's `monitor_runner.build_scorer` -> `LiveScorer` -> `flyvsfly.calibrate_live`): the scale comes from 60 s of frames showing
  both flies, the picture size from the camera. Recorded pose files are not measured: use `flyvsfly.calibrate` first.

A first try with the first 5 seconds failed (lunge 0.02, wing threat 0.09): these recordings open with one fly, at first
often measured at half its length, and the second fly appears only after 20 to 100 seconds (Boy meets boy 1-3; the "scoring
starts" time below is on the wrong clock too, see fly 0.8).

Whole validation movies through the live path, fly 0.6 (`scripts/check_fly_live.py`; fly 0.7 below):

| Movie | Scoring starts | Scale | | Macro F1 | Lunge | Wing threat |
|---|---|---|---|---|---|---|
| Aggression 3, from the eyes | 54 s | 0.992 | live | 0.595 | 0.447 | 0.728 |
| | | | offline, same frames | 0.597 | 0.447 | 0.725 |
| Boy meets boy 3 | 202 s | 0.682 (5-minute scale 0.673) | live | | 0.280 | 0.788 |
| | | | offline, same frames | | 0.315 | 0.798 |

A 6 s buffer matches whole-session scoring on flies (6,000 frames of Aggression 3: largest difference 0.008, same decisions;
32 ms per 15-frame chunk). Lunge F1 is sensitive to the scale: x1.02 gives 0.267, x1.06 gives 0.205 (wing threat 0.784, 0.759).

Limits: nothing is scored until both flies have been in view for 60 s, and frames older than the 6 s buffer at that moment
get no score. Boy meets boy has no video, so that row used tracker points; **the eyes have only ever seen one camera**, and
no fly camera has been run for real. Eyes on another camera, and a full-arena picture at 160 x 160 input, are untested.

## Fly 0.7: lookahead (2026-11-04)

**Read "Fly 0.8" below: every Boy meets boy number in this section and the next was measured on the wrong clock.**

`scripts/train_fly_lookahead.py --look K --fold F`, `scripts/pack_fly.py`, results in `outputs/etholive/fly_look/`; 0.6 kept in
`backups/etholive_fly_0.6_2026-11-04/`. The brain sees K frames past the frame it labels: the labels move K frames later,
the network is unchanged, and the file's `info["lookahead_frames"]` tells `BehaviorScorer.score_buffer` to give frame t the
output of frame t + K. Why: lunges last about 4 frames and 66% of false lunge frames on Aggression 3 lay within 15 frames
of a real lunge, so the brain found lunges but had to call them before seeing how they end. **Fly 0.7 = K 10 (0.33 s).**

Team of 3, three held-out pairs (Aggression movie, Boy meets boy movie), same recipe as fly 0.6:

| K | (3, 3) | (2, 1) | (5, 2) | Mean combined | Mean Aggression macro F1 | Mean Boy meets boy lunge |
|---|---|---|---|---|---|---|
| 0 | 0.576 | 0.513 | 0.523 | 0.537 | 0.525 | 0.338 |
| 6 | 0.677 | 0.595 | 0.613 | 0.628 | 0.636 | 0.426 |
| **10** | **0.710** | 0.606 | 0.643 | 0.653 | 0.619 | 0.560 |
| 15 | 0.724 | 0.609 | 0.636 | 0.656 | 0.603 | 0.570 |

K 3 ran on pair (3, 3) only (0.654). K 15 was added after the declared grid (3, 6, 10) because K 10 was at its edge. K 10 over
K 15: the mean is a tie, and K 15 costs 5 more frames, hold and Aggression macro F1. K 6 is best for Aggression alone. The
shipped team is the pair (3, 3) team, so Aggression 3 and Boy meets boy 3 stay out of its training.

Validation, fly 0.6 vs fly 0.7 (pair (3, 3)):

| | Combined | Agg lunge | Agg wing threat | Agg tussle | Agg hold | BMB lunge | BMB wing threat |
|---|---|---|---|---|---|---|---|
| fly 0.6 | 0.576 | 0.438 | 0.725 | 0.559 | 0.656 | 0.315 | 0.798 |
| **fly 0.7** | **0.710** | **0.703** | **0.762** | 0.528 | 0.620 | **0.696** | **0.836** |

Whole validation movies through the live path (camera measured by `setup="auto"`):

| Movie | | Macro F1 | Lunge | Wing threat | Lunge found (bouts) | Lunge calls right (bouts) |
|---|---|---|---|---|---|---|
| Aggression 3 | live | 0.652 | 0.704 | 0.762 | 92% | 71% |
| | offline, same frames | 0.654 | 0.704 | 0.762 | 92% | 71% |
| Boy meets boy 3 | live | | 0.684 | 0.834 | 100% | 42% |
| | offline, same frames | | 0.696 | 0.836 | 100% | 32% |

Checks: the packed file reproduces 0.710 against the original labels. No future leak: scrambling every point after a frame
changes nothing at or before it (largest change 1e-11, poses through to the net). Scale (Boy meets boy 3 lunge): x0.94 0.617,
x0.98 0.687, x1.02 0.693, x1.06 0.632, so much less sensitive than fly 0.6.

Limits:
- Validation numbers above are optimistic: K and the epochs were chosen on these same pairs. Use the test results below.
- Aggression 2 and 5 scored cleaner eye points in pairs (2, 1) and (5, 2) (the eyes trained on pictures from them); both arms share it.
- Tussle and hold fall (0.559 -> 0.528, 0.656 -> 0.620), and they rest on few frames.
- Boy meets boy lunge bouts: still more than half of calls are wrong, though frame F1 is high. 147 lunge frames in that movie.
- Live delay: 10 frames on top of the chunk, so a score can arrive about 0.8 s after its frame; the last 10 frames of a run get none.
- Scripts that read `.head` on the fly file directly get scores 10 frames late (`train_fly_v2.py`, `train_fly_on_eyes.py`, and the
  finished test scripts `external_test.py`, `test_run_fly_v2.py`): score through `score_buffer` instead.

## Fly 0.7: test, run once (2026-11-04)

`scripts/test_run_fly_v3.py`, `outputs/test_run_fly_v3/`, frozen at commit 3bf9e0f before reading test data. **Third look at both
fly test sets**, first for 0.6, 0.7 and the fly eyes. Nothing was changed after this run. Intervals resample whole movies
(5 and 3 movies, so they are wide), paired across models. Fly 0.5 on tracker points reproduces its earlier test exactly.

| Test set | Model | Macro F1 (95%) | Lunge | Wing threat | Tussle | Hold | Lunge bouts found | Lunge bouts right |
|---|---|---|---|---|---|---|---|---|
| Aggression 6-10, from video | fly 0.5 | 0.568 (0.399-0.613) | 0.467 | 0.705 | 0.597 | 0.505 | 94% | 42% |
| | fly 0.6 | 0.547 (0.375-0.614) | 0.470 | 0.631 | 0.580 | 0.506 | 90% | 48% |
| | **fly 0.7** | **0.598 (0.448-0.636)** | **0.665** | **0.719** | 0.566 | 0.441 | 90% | **63%** |
| Aggression 6-10, tracker points | fly 0.5 | 0.554 (0.390-0.625) | 0.492 | 0.655 | 0.617 | 0.454 | 94% | 49% |
| | fly 0.6 | 0.510 (0.343-0.589) | 0.477 | 0.636 | 0.559 | 0.369 | 87% | 50% |
| | **fly 0.7** | **0.579 (0.401-0.646)** | **0.653** | **0.678** | 0.559 | 0.425 | 89% | **63%** |

| Test set | Model | Lunge (95%) | Wing threat | Lunge bouts found | Lunge bouts right |
|---|---|---|---|---|---|
| Boy meets boy 4-6, tracker points | fly 0.5 | 0.191 (0.051-0.277) | 0.660 | 85% | 19% |
| | fly 0.6 | 0.174 (0.045-0.251) | 0.607 | 85% | 15% |
| | **fly 0.7** | **0.474 (0.388-0.496)** | 0.653 | 81% | **46%** |

Paired differences, 0.7 minus 0.5: Aggression macro F1 from video -0.000 to +0.083, tracker points -0.002 to +0.060;
Boy meets boy lunge F1 +0.219 to +0.337.

What this says:
- **The lunge gain is real**: Aggression lunge 0.47 -> 0.67, Boy meets boy lunge 0.19 -> 0.47 (interval excludes zero).
- **Overall macro F1 on Aggression is not a proven gain over fly 0.5**: the interval just touches zero. Tussle and hold fall.
- **Validation overstated it**: 0.710 combined and 0.70 Boy meets boy lunge became a much smaller test gain.
- **Fly 0.6's validation gain did not hold**: on test it is below fly 0.5. Eye-point training alone is not supported.
- Both fly test sets have now been used three times: treat them as spent, and report that count in the paper. New claims need new data.

## Mouse: lookahead tried, not kept (validation only, 2026-11-04)

`scripts/mouse_lookahead.py`, `outputs/etholive/mouse_look/validation.csv`; same design, 3 seeds, team.

| K | Macro F1 | Attack F1 | Investigation | Mount | Attack false alerts/h |
|---|---|---|---|---|---|
| 0 (shipped) | 0.771 | 0.741 | 0.838 | 0.733 | 11.5 |
| 3 | 0.769 | 0.756 | 0.833 | 0.718 | 12.6 |
| 6 | 0.773 | 0.757 | 0.843 | 0.718 | 5.7 |
| 10 | 0.778 | 0.765 | 0.842 | 0.728 | 12.6 |

Inside seed noise (about 0.01 macro F1), and false alerts do not move consistently, so `mouse.pt` stays. The gain looks
specific to very short behaviors such as 4-frame lunges; mouse behaviors last much longer.

## Fly 0.8: the true clock (2026-11-04)

**Fault found:** Boy meets boy is 200 fps, not 30. The raw movie's header says so, and lunges last 18 to 20 frames there, 4 in
Aggression (0.10 s and 0.13 s). Every earlier fly version read it as 30 fps, so its speeds and time windows were 6.7 times off,
and all earlier Boy meets boy results (including the three test runs) describe fast footage read at the wrong clock.

`flyvsfly.to_fps` resamples it, `scripts/train_fly_lookahead.py --bmb30` trains on it, `outputs/etholive/fly_look30/`; 0.7 kept in
`backups/etholive_fly_0.7_2026-11-04/`. **Fly 0.8 = K 6 (0.2 s), Boy meets boy at 30 fps**, the team of pair (3, 3). A fly run at another
frame rate is now refused (LiveTracker, `monitor_runner.build_scorer`).

Mean of 3 held-out pairs (team of 3):

| | Aggression macro F1 | Aggression lunge | BMB lunge, 30 fps footage | BMB lunge bouts found | BMB lunge bouts right | BMB wing threat, 30 fps footage |
|---|---|---|---|---|---|---|
| old clock, K 10 (fly 0.7 recipe) | 0.619 | 0.688 | 0.204 | 88% | 16% | 0.501 |
| true clock, K 0 | 0.534 | 0.419 | 0.188 | 48% | 18% | 0.621 |
| **true clock, K 6 = fly 0.8** | 0.620 | **0.739** | **0.527** | 83% | **54%** | **0.647** |
| true clock, K 10 | 0.621 | 0.745 | 0.516 | 72% | 47% | 0.641 |

The rule declared before running looked at Aggression macro F1 only, which is a tie. The comparison on the same 30 fps footage was
added after seeing that, and it decided both the fix and K. Old-clock models score much higher on their own 200 fps footage, but a
real camera on that rig runs at 30.

**Not run on any test set.** Both labelled fly test sets are spent (3 looks), and those looks used the wrong clock for Boy meets boy. Any
new run there would be a 4th look. No fly version has a clean test number on the true clock.

## Fly: second camera from video (2026-11-04)

Raw Boy meets boy videos are on the Caltech record (`movie1-6.avi`, about 7 GB each; `movie3.avi` is in `data/raw/flyvsfly/new_bmb_video/`,
checked against the tracks). `scripts/check_fly_video30.py`, Boy meets boy 3, picture -> eyes -> live scorer, 30 fps:

| Points | Camera scale measured (true 0.676) | Lunge | Wing threat | Lunge bouts right |
|---|---|---|---|---|
| tracker points | 0.676 | 0.491 | 0.683 | 47% |
| eyes, picture at full size | 0.844 | 0.122 | 0.176 | 6% |
| eyes, flies at the trained size (scale given by hand) | 0.746 | 0.198 | 0.255 | 9% |

**Fly 0.8 does not work from video on a camera it has not seen.** On 500 sampled pictures the eyes look good (both flies in 100%, median
thorax error 2 px of a 29 px fly; at the old 160 px input, 51%, so `FlyEyes` now keeps a big picture big), but they measure flies too
short and the stream is too noisy for the brain. Fixed as far as it goes in fly 0.9, below.

## Fly: Courtship (2026-11-04)

Courtship set: 31 movies never seen by any model, same chamber and clock as Aggression, male and female, no aggression labels.
`scripts/check_fly_courtship.py`, train half only; **the test half (ids 17-31) is untouched, kept for one run on a final model.**
Calls per hour, train half, tracker points (fly 0.8): lunge 520-720, wing threat 420-570, tussle 1,400-1,800, hold about 2,000; fly 0.5
and 0.7 are as high. On 5 movies, by true behavior (fly 0.8):

| True behavior | Frames | Called |
|---|---|---|
| none | 102,807 | other 97.8%, lunge 0.2%, wing threat 2.0% |
| copulation | 160,605 | hold 47%, tussle 18%, other 30% |
| wing extension | 4,081 | wing threat 56%, other 43% |
| circling | 962 | wing threat 60%, other 39% |

The model is a male-male fight model: it has no courtship classes and calls them the nearest fight behavior. Do not use it on mixed-sex
pairs. Improved in fly 0.9, below.

Other data checked, neither used: FlyMABe2022 (Janelia, CC BY 4.0; `data/raw/flymabe/test1_v3.npz`): about 10 flies per arena, 150 fps; its
development split has 1,279 labelled aggression fly-frames (about 8.5 s) over 6 videos, and every fighting fly is female, so it cannot test a
male-male model. Zenodo 20698002 has per-video counts only.

## Fly 0.9: two-camera eyes and hard negatives (2026-11-05)

`models/etholive/fly.pt`; 0.8 kept in `backups/etholive_fly_0.8_2026-11-05/`. Three changes:
- **Eyes on two cameras** (`scripts/fly_pose_dataset2.py`, `models/pose/fly_pose2_best.pt`): fine-tuned on the Aggression pictures plus 2,375 crops of Boy
  meets boy 1, 2, shrunk so flies have the training size. Boy meets boy 3 held out: both flies in every sampled picture, median error 0.9-1.8 px, 95% within about 3 px.
- **Eyes that learn the camera** (`FlyEyes.learn_size`): they start at full size, measure the flies and rescale, and never stop. A first version with 4 rounds spent them
  on the opening, before the second fly arrived, and stayed at full size. Now they settle at 0.70-0.73 (true 0.67-0.68): flies read 4-8% short.
- **Brain on eye points with Courtship hard negatives** (`scripts/extract_fly_eyes2.py`, `scripts/train_fly_eyes2.py --arm neg`): the team of pair (3, 3).

Mean of 3 held-out pairs, everything on eye points, Boy meets boy from its video (`outputs/etholive/fly_eyes2/report.csv`):

| | fly 0.8 brain | tracker negatives | eye brain, no negatives | **eye brain + negatives = fly 0.9** |
|---|---|---|---|---|
| Aggression macro F1 | 0.599 | 0.573 | 0.588 | 0.587 |
| Boy meets boy lunge | 0.134 | 0.211 | 0.391 | 0.342 |
| Boy meets boy wing threat | 0.462 | 0.432 | 0.526 | 0.535 |
| lunge calls per hour on dev movies, outside copulation | 191 | 142 | 110 | 90 |
| wing threat calls per hour | 169 | 125 | 194 | 58 |
| copulation called a fight | 21% | 10% | 10% | 0% |

These report numbers score whole movies including the one-fly opening, so they run lower than the live check below.
Hard negatives on tracker points looked far better than they were (29 lunge calls per hour on tracker points, 142 on eye points): mostly a tracker-versus-eyes shortcut.
Rule: fewer false calls wins unless a score falls by more than the seed spread (Aggression 0.021, Boy meets boy lunge 0.057). Negatives fell 0.001 and 0.049: kept.
**The rule for the eyes was broken and overruled:** "without hurting Aggression" fails, as Aggression macro F1 is 0.587 against 0.620 for fly 0.8 on the old eyes, more than the seed
spread. Shipped anyway, because the old eyes fail on every other camera.

Live path, whole movies, from video (`scripts/check_fly_video30.py`, `scripts/check_fly_live.py`):

| Movie | Model | Scale | Macro F1 | Lunge | Wing threat | Lunge bouts found | Lunge bouts right |
|---|---|---|---|---|---|---|---|
| Boy meets boy 3 | fly 0.8, old eyes | 0.844 | | 0.122 | 0.176 | 88% | 6% |
| | fly 0.8 brain, new eyes | 0.719 | | 0.352 | 0.380 | 88% | 21% |
| | **fly 0.9** | 0.719 | | **0.384** | **0.443** | 75% | **26%** |
| | fly 0.8 on tracker points (best case) | 0.676 | | 0.491 | 0.683 | 88% | 47% |
| Aggression 3 | **fly 0.9** live | 0.989 | 0.613 | 0.719 | 0.631 | 88% | 75% |
| | fly 0.9 offline, same frames | | 0.612 | 0.718 | 0.626 | 88% | 75% |

Fly 0.9 no longer reads tracker points well (Boy meets boy 3 lunge 0.243): use it with its eyes.

**Fresh test, run once** (`scripts/test_run_courtship.py`, `outputs/test_run_courtship/`, frozen by checksum): Courtship test half, 15 movies no model or choice had read, from video.
False calls only: male and female pairs with no aggression labels, so this says nothing about finding fights. Both models on the same eye points.

| | fly 0.8 | **fly 0.9** |
|---|---|---|
| plain frames called a fight (310,465) | 0.6% | **0.3%** |
| copulation (480,291) | 15.4% | **0.1%** |
| wing extension (12,093) | 20.5% | **4.5%** |
| circling (3,227) | 33.7% | **7.6%** |
| attempted copulation (436) | 21.1% | **1.6%** |
| lunge calls per hour, outside copulation | 95 | **56** |
| wing threat calls per hour | 160 | **71** |
| tussle calls per hour | 14 | 18 |
| hold calls per hour | 16 | **5** |

Limits of fly 0.9:
- **No clean test of finding fights.** Aggression 6-10 and Boy meets boy 4-6 are spent, and 0.9 was not run on them. All detection numbers are validation; choices were made on these pairs.
- **Second camera is better, not solved**: 3 in 4 lunge calls wrong on Boy meets boy 3. Only 2 movies taught the eyes that camera; its brain saw 2 movies of it too.
- **Two cameras are not every camera.** A third camera may need its own frames for the eyes. No real fly camera has been run.
- 56 lunge calls per hour on courting pairs is still high; a male may really attack, so some may be true.

## Scout: a first look at any video (2026-11-05)

`etholive/scout.py`. Every model is built for exactly two animals, and a video with another number used to reach it silently. `scout.look(video)` says which animal and
how many; `scout.analyze(video, out_dir)` then takes the path. Command line: `python -m etholive.scout video.mp4 --out folder`.

| Found | Path | Output |
|---|---|---|
| nothing, or not a mouse or fly | stop | `summary.json` with the reason |
| 1 animal | tracking only (position, speed, distance) | `track.csv` |
| 2 animals | the species' full model, camera measured for flies | `scores.csv`, bouts in `summary.json` |
| 3 or more | count only, no behavior scores | `summary.json` |

How it decides, on 60 pictures spread over the video: species from which eyes are confident (right eyes 0.93-0.98, wrong eyes 0.00-0.20); count is the largest number seen in 30% of
pictures, after dropping detections far shorter than the sure ones and fixed spots. Flies are counted at the training size. Videos go to 30 per second by the file's header.

`scripts/check_scout.py`: **19 of 19**: 7 full videos, 3 real one-fly openings, and 9 built scenes (real animals pasted on real empty backgrounds: 0, 1, 2, 3, 4, 6 flies, 1 and 3 mice, an empty mouse frame).
The thresholds were set while looking at these same cases, and SHARE was raised from 0.25 to 0.3 after 6 flies read as 7, so this is a development check, not a test.
A screen recording was correctly rejected. Whole runs: Aggression 3, 485 lunge bouts called (410 labelled); mouse064, attack 75.7 s called, 74.7 s labelled, F1 0.84 (a training video);
investigation 66.7 s called, 30.0 s labelled; one-fly clip, 6,000 frames at 200 fps read as 30.0 s, tracking path.

Limits: only one mouse video, and no real scene with 0, 1 or 3+ mice or 3+ flies: those are built scenes. **A lone white mouse is untested**: it cannot be cut out from pale bedding, and its built
scenes were invalid, so they were dropped from the check. Counts above 3 are approximate. Species is mouse or fly only. A resting fly in 70% of pictures, or at the edge in 30%, would be taken for
a fixed spot. Live cameras do not use the scout yet: it needs a video to look through. The frame rate is taken from the file's header.

Found on the way: a dark mark at the top of the Boy meets boy arena scores 0.5-0.7 as a fly, and the live eyes picked it in 0.7-4.7% of thorax points. Telling the eyes to skip it did not help
(Boy meets boy 3: lunge 0.313 against 0.384, wing threat 0.486 against 0.443): not kept, `FlyEyes.ignore` stays empty. A size rule in the eyes was rejected first: real flies measure under half length in up to 10% of frames.

## Filmed screens: the corner stretch per animal (2026-11-05)

The corner stretch (`screen_map`) always went to the mouse frame, 1024 x 570. A square fly video came out 1.8 times too wide. `screen_map.FRAME` now gives the frame per animal, and `spec["animal"]` selects it (default mouse).
`scripts/check_fly_screen.py`, a simulation (tilt, blur, glare, noise), Aggression 3, first 10 minutes: the file itself lunge 0.712; filmed and stretched to 144 x 144, 0.704, 100% of flies found; filmed and stretched to 1024 x 570, 0.000, 44% found.
Wing threat and hold are not comparable here: the first 10 minutes hold few of them. Real screens add moire and flicker, so a camera will do worse than this.
`scout.identify(pictures, corners)` is the function for an "Identify animal" button: it stretches the pictures each animal's way, keeps the confident eyes, and returns the animal, its frame and the count.

## Fly courtship: a separate model (2026-11-06, now 0.2, see below)

`models/etholive/fly_courtship.pt`, `scripts/train_fly_courtship.py`, `scripts/pack_fly_courtship.py`, `outputs/etholive/fly_courtship/report.csv`. Same eyes, ScopeScore steps and network as fly 0.9, its own brain: wing extension,
circling, attempted copulation, copulation. The fight model is unchanged. Each frame takes one label (attempt, then circling, wing extension, copulation), because a male circles while extending a wing.

Validation, team of 3 on eye points, mean of 2 folds of the Courtship train half (test half not read):

| K | Macro F1 | Wing extension | Circling | Attempted copulation | Copulation |
|---|---|---|---|---|---|
| 0 | 0.589 | 0.454 | 0.590 | 0.317 | 0.998 |
| **6 (shipped)** | **0.641** | 0.460 | **0.700** | **0.407** | 0.999 |

Bouts at K 6: wing extension 60% found, 62% right; circling 88%, 66%; attempted copulation 83%, 42% (37 true bouts over both folds); copulation all 10 found, but split into pieces, so 47% of called bouts are right.
The packed file is the fold A team and reproduces its 0.599. Copulation is easy and long, and 57% of frames, so the macro F1 flatters the model. Wing extension is weak, and a published tool reports F1 0.95 at 30 fps on its own data: not comparable,
but a sign there is room. Not run on the test half, not run live, and the scout does not route to it yet.

## Climbing: the first experiment block (2026-11-06)

`etholive/assay.py`, `scripts/check_climbing.py`, `scripts/make_climb_count_pack.py`, tests in `tests/test_assay.py`. Plain code, no learning: `ClimbPlan` holds the tube area, vials, scale, line, window and real frame rate, and no setting has a default that stands for a name.
It finds what is darker than the clip's empty scene (the brightest value of each pixel), corrects a change of light, splits blobs into up to 5 flies, and reports per vial the flies above the line at the end, the most ever above it, and the first time above it.

On 6 public clips (FreeClimber examples, MIT, `data/raw/assay_clips/`), typical flies per frame against FreeClimber's filtered detections: 115 vs 91, 98 vs 66, 118 vs 97, 160 vs 72, 24 vs 20, 31 vs 28. This is agreement between tools, not accuracy: FreeClimber drops detections, and neither is checked by hand yet.
Limits: a fly that never moves joins the empty scene and is missed; flies are not told apart, so "most ever above" is a lower bound of "ever crossed"; `climbing_4` is clearly too high; clips are 6 s long. Hand counts: `hand_counts/climbing/tuning/`.

2026-11-19, hand counts (Ava, 20 pictures, 100 vials; only "above" counted). First score: our code overcounted (mean error 3.77 above the line, FreeClimber 1.46): flicker in empty tube, tube edges, specks, and a fly size taken from noise. Fix: subtract the local median of the difference (`local_px`), keep a spot only if its darkest pixel reaches `peak_share` of the clip's 90th percentile (so faint-fly cameras still work), drop spots under `min_fly_part` of a fly. After: 1.08, 65/100 exact. These pictures are now the tuning set; `hand_counts/climbing/test/` holds 10 fresh pictures for one final score. Scoring: `scripts/score_climb_hand_counts.py`. Later the same day Ava recounted all 20 pictures properly (above and below), and frames are now lined up with the middle frame first (`steady`), because tapping shakes the camera: 1.04 above the line, FreeClimber 1.51. Tuning stopped there. One-time test on the 10 fresh pictures (50 vials): 0.44 flies off above the line, 38/50 exact (FreeClimber 0.72, 34/50). Same videos as tuning, one counter, so it says the code works on new moments of these clips, not on new cameras.

## Exchanging flies a and b: fly 0.10 and courtship 0.2 (2026-11-07, shipped)

Wing spread separated wing extension for fly b only, because which fly is "a" is arbitrary. So each training movie is also shown with every a and b column exchanged (`--swap`, `--arm swap`), a form of data augmentation. A hand rule (widest spread over a threshold) lost first: held-out F1 0.42-0.48 against the network's 0.59-0.64.

Courtship, mean of 2 folds, rule set before running (beat 0.641): macro F1 0.689, wing extension 0.594 (0.460), found 79% (60%), circling 0.723 (0.700), attempted copulation 0.440 (0.407). Packed as `fly_courtship.pt` 0.2 (fold A team).

Fights, mean of 3 pairs, eye points, against fly 0.9 (`outputs/etholive/fly_eyes2/report.csv`): Aggression macro F1 0.628 (0.587), hold 0.548 (0.434), Boy meets boy lunge 0.454 (0.342), lunge bouts right 35% (22%), found 84% (76%); Courtship dev false calls per hour: lunge 78 (90), wing threat 23 (58), tussle 6 (21), hold 13 (24). Better on all 3 pairs. Packed as `fly.pt` 0.10 (pair (3, 3) team); 0.9 in `backups/etholive_fly_0.9_2026-11-07/`.

Live path from video: Boy meets boy 3 lunge 0.500 (0.384), wing threat 0.442 (0.443), lunges found 88% (75%), right 44% (26%). Aggression 3: macro F1 0.664 (0.613), lunge 0.763, hold 0.670, 77% right; offline, same frames 0.669.
Validation only: no fly version has a clean test of finding fights. The Courtship test half was used once, on 0.9, and was not rerun. Mice were not tried: their a and b are black and white, not arbitrary.

## One animal in an arena (2026-11-07)

`assay.ArenaPlan`, `track_one`, `measure_arena`: an animal of any coat against the empty arena, inside the arena, animal-sized, nearest to where it was, then time in zones, entries, distance. No learning. `scripts/check_arena.py` compares it with the published DeepLabCut body centre on Sturman 2020 videos (videos CC BY 4.0, tracks GPL, research check), with the same zones.

| Test | Videos | Position error, median | Percent time, mean gap | Distance gap |
|---|---|---|---|---|
| Open field, centre zone | 20 of 20 | 6.2 px | 0.41 points, largest 1.12, r 0.997 | 2.7% |
| Plus maze, 4 arms | 7 of 8 | 5.8 px | 0.96 points, largest 2.76 | 7.3% |

Entries differ by about 2 per video in the open field. A comparison of two trackers, not hand-scored truth. EPM_15, a 38 s clip with a person moving in view, failed: the mouse is lost. First runs failed in useful ways: the tracker followed the experimenter, hence the area limit; and pairing open field files by frame count matched wrong tracks, so that run was thrown out and files were paired by path instead. Zones came from the reference's maze points, so a user must still mark them. Mice here are all one strain, one lab.

More blocks in `assay.py`, plain code with tests on made-up data: `zone_sequence` and `alternation` (Y-maze and other entry orders), `still_periods` (the time rule behind freezing, immobility and video sleep, with thresholds from the scientist), `choice_index` (with non-choosers in or out of the denominator), `load_table` (device and tally files, refusing a missing column). None is checked on real labelled data yet, and no public Y-maze video with hand scores was found.

One fly in an arena (`scripts/check_fly_walk.py`): 80 s of Boy meets boy 1 before the second fly arrives, from video, against the FlyTracker thorax: seen in 96% of frames, median error 4.5 px on a 29 px fly, distance 85.1 cm against 89.6 cm, at 12 px per mm. One clip.

## Mouse: CalMS21 Task 3, tried, not kept (2026-11-07)

`scripts/train_mouse_task3.py`, `outputs/etholive/mouse_task3/`. Seven new behaviors, one small model each, 1 to 4 training videos per behavior, each video split in time for validation. Test run once, the first look at this test set here.

| Behavior | Validation F1 | Test F1 | Test videos |
|---|---|---|---|
| approach | 0.323 | 0.276 | 25 |
| disengaged | 0.117 | 0.000 | 1 |
| groom | 0.626 | 0.345 | 13 |
| intromission | 0.754 | 0.383 | 10 |
| mount attempt | 0.251 | 0.035 | 12 |
| sniff face | 0.251 | 0.097 | 47 |
| white rearing | 0.747 | 0.459 | 1 |
| **mean** | **0.438** | **0.228** | |

Published class-averaged F1 is about 0.34. So this falls short, and validation inside the same videos overstated it by about half: a model of a few videos learns those videos. Not packed. Any further try needs a new idea, such as starting from the mouse 0.4 network, and the test set has now been used once.

## Mouse Task 3 from mouse 0.4: not kept, test not read (2026-11-07)

`scripts/train_mouse_task3_probe.py`: mouse 0.4's three memory branches as a fixed description of each frame (192 numbers), one logistic regression per behavior on top. Validation mean F1 0.397, below the scratch models' 0.438, so by the rule set beforehand the Task 3 test set was not read again (still one look).

## The mouse eyes on another lab's camera (2026-11-07)

`scripts/check_mouse_eyes_crim13.py`: CRIM13 out-of-distribution frames (1,274, new intruder mice), body points marked by five annotators (median), CC BY 4.0, never used in training.
Points found: 98%. The two datasets define left and right ears oppositely (`--swap-ears` matches them); with that, median error per point is 11-17 px on a mouse 117 px from nose to tail: nose 11, neck 11, ears 13-14, tail base 14-17.
45% of points lie within 10% of body length, 80% within 20%. Hips cannot be compared: CRIM13 marks knees. This checks the eyes only, not behavior on that camera, which needs CRIM13's behavior labels in the full 250 GB set.

## Chat agent for experiments (2026-11-07)

`etholive/assay_agent.py`, tests in `tests/test_assay_agent.py`. The agent talks; tools measure. Tools: list the experiments, look at a video (scout), choose an experiment, record answers into the question form (`plan_form.py`), run the measurement (`assay.py`).
Held by code, not by the prompt: no measurement before the form is complete; a reply quoting a number that no tool returned and the scientist did not give is sent back once, then marked; the video is checked against what the scientist says.
The language model is swappable (`ClaudeBrain` uses the Anthropic API, Claude Opus 5.5, medium effort, server-side fallback on refusal; `ScriptedBrain` plays fixed turns for tests; a local model can implement the same `step`).
Tested only with the scripted brain: no API credentials were on this computer, so no conversation with a real model has been run. Experiments covered: climbing and one animal with zones.

## Local language model for the agent (2026-11-07)

Ollama v0.40.10 (official release, checksum checked, signed) in `~/.local/ollama/`, serving on 127.0.0.1 only; nothing leaves the computer. `OllamaBrain` in `assay_agent.py`; `LIVE_TRACKER_LOCAL_MODEL=qwen3:14b python -m etholive.assay_agent [video]`.
Start the server: `OLLAMA_HOST=127.0.0.1:11434 ~/.local/ollama/ollama serve`. Models: `qwen3:14b` (9.3 GB, used), `qwen3:8b` (5.2 GB, too weak here). Apple M5, 26 GB memory.

`scripts/eval_assay_agent.py`: six scripted conversations on real clips (climbing, open field), scored by code on what the agent did. First runs: 14B 20 of 24, 8B 14 of 24. Fixes came from reading the traces:
the animal check (trained on two setups) made the model refuse; it is now advice only. The model sent answers flat while the tool wanted them wrapped, and the error said only "'answers'", so it looped: both forms are now accepted, with a clear error.
The model once ran a measurement with a line height the scientist never gave; telling it not to guess was not enough, so every number saved to the form must now appear in what the scientist typed (0 counts as "none"). Warnings from the checks are added to the reply by code.
After the fixes: 23, 21, 21, then 24 of 24 as each was made; the last run had all six right. Replies take 40 s to 2 minutes. These six scenarios were used to make the fixes, so they flatter the agent: new conversations, and real scientists, are the real test.

