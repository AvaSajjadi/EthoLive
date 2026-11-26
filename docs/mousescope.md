# MouseScope 0.1 (model card)

MouseScope is Live Tracker's one model: a camera picture goes in, and the keypoints of both mice, a `tracking_ok` flag
and per-frame probabilities of attack, investigation and mount come out. It is one file, `models/mousescope.pt`
(27 MB), made by `python -m etholive.mousescope pack`. The code is `etholive/mousescope.py`.

## What is inside

| Part | What it is | Where it comes from |
|---|---|---|
| Vision | YOLO26n-pose, 2.9 million parameters, 2 classes (black mouse, white mouse), 7 keypoints each; input 384 x 640, confidence 0.15, a missed mouse is held for 3 pictures | Ultralytics `yolo26n-pose.pt`, fine-tuned on the MARS hand-labeled set (`scripts/train_yolo_pose.py`) |
| Methods | Tracking-quality flags, causal One-Euro smoothing, 25 motion features plus 77 body-shape features | ScopeScore methods adapted to mice (`quality.py`, `filters.py`, `features.py`, `basic_features.py`); not learned, the same code as model B |
| Behavior | TemporalHead: a causal dilated convolution network (dilations 1, 2, 4, 8, 16; width 64; 90,052 parameters). It sees the last 63 frames (2.1 s) | Trained here on CalMS21 Task 1 |

The monitor, the rules and the researcher's Confirm are not part of the model and did not change. MouseScope
scores; the approved rule decides what is an alert.

## Training

- Data: CalMS21 Task 1, split by session (`splits/split_manifest.json`): 56 train, 14 validation. The 19 test sessions
  were not read.
- Loss: cross-entropy with balanced class weights (as model B), on random 256-frame windows, AdamW, learning rate 0.001.
- The epoch was chosen on validation macro F1 (best epoch 9 of 15, early stopping after 6 epochs without a gain).
- Reproduce: `python -m etholive.mousescope train`, then `... pack`.

## Held-out test result (19 sessions, 262,107 frames, one run)

`scripts/test_run_mousescope.py`, run once on 2026-10-31. Both models were frozen first: their SHA-256 checksums were
recorded in `outputs/test_run/frozen_checksums.txt` before any test data was read, and the script refuses to run if a
model changed. Same 102 features, same code, attack rule v1 for both. Intervals resample whole sessions (2000 times).

| Measure | Model B | MouseScope |
|---|---|---|
| Macro F1 (attack, investigation, mount) | 0.755 (95%: 0.666 to 0.779) | **0.818** (0.772 to 0.837) |
| Attack F1 | 0.592 | **0.730** |
| Investigation F1 | 0.782 | **0.823** |
| Mount F1 | 0.890 | **0.901** |
| Attack alerts that were right | 82.5% of 183 (47% to 91%) | **94.4% of 234** (78% to 98%) |
| Attack bouts found | 135 of 258, 52% (47% to 61%) | **173 of 258, 67%** (59% to 72%) |
| Short attack bouts (1.1 s or less) found | 47 of 126 | **61 of 126** |
| False alerts per hour | 13.2 | **5.4** |
| Alert fires after the attack starts, median | 0.90 s | **0.57 s** |

Paired difference (MouseScope minus model B, the same resampled sessions for both; `paired_difference.json`):
macro F1 +0.040 to +0.108, alert precision +0.054 to +0.295, alert recall +0.043 to +0.194 (95% intervals). None
includes zero.

## Results (validation, 14 sessions, 94,305 frames)

`scripts/compare_mousescope.py`. Both models get the same 102 features, and both use the demo attack rule v1
(ask 0.6, at least 0.5 s), which was chosen for model B.

| Measure | Model B | MouseScope |
|---|---|---|
| Macro F1 (attack, investigation, mount) | 0.650 | **0.753** |
| Attack F1 | 0.570 | **0.729** |
| Investigation F1 | 0.780 | **0.832** |
| Mount F1 | 0.601 | **0.698** |
| Attack alerts that were right | 83.5% of 79 | **89.5% of 86** |
| Attack bouts found | 36 of 56 | **38 of 56** |
| Short attack bouts (1.1 s or less) found | 5 of 18 | 5 of 18 |
| False alerts per hour | 14.9 | **10.3** |
| Alert fires after the attack starts, median | **0.35 s** | 0.65 s |

Seed check: two more training runs (seeds 1 and 2) gave macro F1 0.728 and 0.735, so the gain over model B is not one
lucky run. The last epoch of each run, with no choice made on validation, gave 0.71 to 0.73.

Live check: streamed frame by frame through `live_scoring.LiveScorer` (6 s buffer, chunks of 5), the scores equal the
whole-session scores (largest difference 6e-07), at about 25 ms per chunk against a budget of 167 ms.

## Limits

- The test set has now been used once. Any further change to MouseScope must be judged on validation (or new data),
  not by running the test again.
- On validation, alerts came about 0.3 s later than with model B and short attacks were not found more often; on test
  both improved. With only 4 validation sessions that contain attacks, the validation numbers are noisy.
- Rule v1 was tuned for model B. A rule tuned for MouseScope would have to be chosen on validation.
- Like model B, it expects a top-down view like CalMS21. Another cage or camera angle gives scores that mean little.
- On Windows, the Intel settings (`models/pose/pose_model.json`) still use the OpenVINO pose copy; MouseScope's
  behavior part works there, its pose part runs from the `.pt` weights.

## Fallback

When `models/mousescope.pt` is missing, runs use model B (`run_control.scoring_model`). The previous Mac pose
settings are kept in `models/pose/pose_model_mac_yolo.json`.

## Credits

Ultralytics YOLO26 (AGPL-3.0); MARS hand-labeled pose data (Segalin et al., 2021, Caltech); CalMS21 (Sun et al.,
2021); ScopeScore methods (Sajjadi, 2026, manuscript submitted).
