# LiveScope 0.2, mouse (model card)

"LiveScope" is a working name, kept in one constant (`MODEL_NAME` in `etholive/scope_model.py`).

One file, `models/scope/mouse.pt` (27 MB), holds the whole mouse model:

| Part | What it is | Source |
|---|---|---|
| Animal pack | keypoints, 2 mice, torso, skeleton, behaviors | `species.py` (CalMS21 setup) |
| Live pose | YOLO26n-pose, fine-tuned on the MARS hand-labeled set | weights inside the file |
| Offline pose | MARS top-view (named with md5 checksums, not carried: 560 MB, CC BY-NC 4.0) | `models/mars/`, `scripts/mars_pose_offline.py` |
| ScopeScore methods | 9 methods (below), settings inside the file | `quality.py`, `filters.py`, `features.py`, `scopescore_mouse.py` |
| Temporal head | causal dilated 1-D CNN, 63-frame (2.1 s) receptive field, 120 features | `scope_model.py` |
| Model B | logistic regression on 102 features, as plain numbers (no scikit-learn needed) | `models/model_b.joblib` |
| Fusion | 0.6 × temporal head + 0.4 × model B (weight chosen on validation) | `scripts/build_scope_model.py` |

Build: `python scripts/build_scope_features.py`, `python scripts/ablation_scopescore.py`, `python scripts/build_scope_model.py`.

## ScopeScore methods in the model

| # | ScopeScore (surgical tool) | Mouse version | Code |
|---|---|---|---|
| 1 | tracking-quality flags, missing stays missing | flags per keypoint | `quality.py` |
| 2 | causal One-Euro smoothing | body center | `filters.py`, `features.py` |
| 3 | motion of a tracked point | speed, turning, pauses, pair motion | `features.py` |
| 4 | tip first, from this frame's geometry | nose / tail-base swap fix (0.46% of mouse-frames) | `scopescore_mouse.fix_nose_tail_swaps` |
| 5 | asymmetric One-Euro per endpoint (published settings) | nose = tip 0.6 / 0.04, tail base = start 0.1 / 0.001, rest = center 0.6 / 0.01 | `smooth_keypoints` |
| 6 | PCA axis for S | body axis, length, width; the other mouse in front / beside | `geometry_features` |
| 7 | geodesic center C | spine center at half the spine length | `geometry_features` |
| 8 | polyline bend | spine straightness | `geometry_features` |
| 9 | FOV distance and tip-edge state | distance to the picture edge, at-edge flag (8 px) | `geometry_features` |

Not measurable on CalMS21 (keypoints only, no video): KLT optical flow with the detector blend
(`point_stabilizer.py`, off by default) and the drawing methods (Douglas-Peucker, trails).

## Ablation (validation, 14 sessions, 3 seeds each; mean)

`outputs/ablation_scopescore.csv`. Rule v1 for the alert columns.

| Features | Macro F1 | Attack F1 | Investigation F1 | Mount F1 | Alert precision | Alert recall | False alerts / h |
|---|---|---|---|---|---|---|---|
| methods 1–3 (MouseScope 0.1 features) | 0.739 ± 0.013 | 0.708 | 0.829 | 0.679 | 0.80 | 0.70 | 13.7 |
| + swap fix (4) | 0.737 ± 0.007 | 0.700 | 0.825 | 0.687 | 0.79 | 0.70 | 13.7 |
| + geometry (6–9) | 0.731 ± 0.002 | 0.728 | 0.829 | 0.635 | 0.86 | 0.76 | 10.7 |
| + swap fix + geometry | 0.735 ± 0.004 | 0.726 | 0.830 | 0.649 | 0.88 | 0.74 | 9.2 |
| + swap fix + geometry + per-keypoint One-Euro (all 9) | **0.739 ± 0.007** | 0.708 | 0.830 | 0.681 | 0.77 | 0.77 | 16.8 |

Reading: on macro F1 no variant differs from the 0.1 features by more than the seed spread. The geometry
methods (6–9) raise attack F1 and make the alerts cleaner (precision 0.80 → 0.86–0.88, fewer false alerts) but
lower mount F1. The selection rule, written before the results, picked the best mean macro F1: all 9 methods.

## Final model (validation, seed 0, rule v1)

`outputs/scope_model_validation.json`

| Model | Macro F1 | Attack F1 | Mount F1 | Alert precision | Alert recall | False alerts / h |
|---|---|---|---|---|---|---|
| Model B | 0.650 | 0.570 | 0.601 | 0.84 | 0.64 | 14.9 |
| MouseScope 0.1 (its seed 0 was the best of 3; mean 0.739) | 0.753 | 0.729 | 0.698 | 0.90 | 0.68 | 10.3 |
| LiveScope 0.2 (all 9 methods + model B at 0.4) | 0.739 | 0.715 | 0.671 | 0.89 | 0.73 | 11.5 |

Live: 34 ms per chunk of 5 frames (budget 167 ms); live scores equal whole-session scores (test).

## Limits and protocol

- Every choice above used train and validation only. The CalMS21 test split was used once, for MouseScope 0.1
  (`docs/mousescope.md`); a test run of 0.2 would be the second look and must be reported as such.
- The fusion weight was chosen on the same validation sessions as the epoch, so the validation number is slightly
  optimistic.
- The swap fix cannot tell a real swap from a tightly curled mouse; it changed 0.46% of mouse-frames.
- The ScopeScore One-Euro settings were published for instruments at 24 fps; mice at 30 fps move faster.
