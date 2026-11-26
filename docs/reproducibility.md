# Reproducing the results

Every command runs from the repository root after `pip install -e ".[video,figures,dev]"`. Scripts write to
`outputs/` (git-ignored); the recorded outputs behind the manuscript are in `results/` with the same sub-paths.

## Environment

The recorded results were produced on one computer:

| | |
|---|---|
| Hardware | Apple MacBook, M5 chip, 24 GB memory; PyTorch MPS backend |
| Python | 3.12 |
| Packages (as re-run on 2026-11-26) | torch 2.14.1, ultralytics 8.4.190, scikit-learn 1.9.1, pandas 2.3.3, numpy 2.5.4, scipy 1.18.1 |
| Seeds | 0, 1, 2 for every trained network (three-seed team); 0 for the simulation and the permutation fallback |

CUDA was not used. MPS kernels are not bit-for-bit deterministic, so retraining reproduces numbers to about the seed
spread (0.01 class-averaged F1 for mice), not exactly. Scoring a saved model file is deterministic.

## 1. Data

Nothing is redistributed here. Put the files under `data/raw/` (git-ignored):

| Dataset | Source | Place |
|---|---|---|
| CalMS21 Task 1 (`task1_classic_classification.zip`: `calms21_task1_train.json`, `calms21_task1_test.json`) | Caltech DATA record s0vdx-0k302, CC BY 4.0 | `data/raw/calms21/` |
| One CalMS21 Task 1 video (for the eyes and speed checks), `task1_videos_mp4/train/mouse064_task1_annotator1.mp4` | same record | `data/raw/calms21_video/` |
| Fly vs. Fly (Aggression, Boy-meets-boy, Courtship tracks and labels) | Caltech DATA record zrznw-w7386, CC0 | `data/raw/flyvsfly/` |
| MARS hand-labelled pictures (pose training only) | Caltech DATA record 2011 | see `docs/pose_model.md` |

The CalMS21 JSON layout is documented in [`calms21_schema.md`](calms21_schema.md). The session split is committed:
[`splits/split_manifest.json`](../splits/split_manifest.json) (56 train, 14 validation, 19 official test sessions;
checked disjoint, test equal to the official test set, no duplicated session across splits).

## 2. Features

```bash
python -m etholive.classify features --train data/raw/calms21/calms21_task1_train.json \
                                     --test  data/raw/calms21/calms21_task1_test.json   # data/processed/
python scripts/build_scope_features.py      # mouse feature steps -> data/processed_scope/
python scripts/build_fly_features.py        # fly features -> data/processed_fly/
python scripts/extract_fly_eyes2.py         # fly points from our pose network (needs the videos)
```

## 3. Training

| What | Command | Output |
|---|---|---|
| Mouse frame-branch start (Model B) | `python -m etholive.classify train --name model_b --features basic,motion` | `models/model_b.*` |
| Mouse Section 2, three seeds | `python -m etholive.model train` | `outputs/etholive/behavior_seed{0,1,2}.pt` |
| Pack the mouse file | `python -m etholive.model pack --behavior outputs/etholive/behavior_seed0.pt outputs/etholive/behavior_seed1.pt outputs/etholive/behavior_seed2.pt` | `models/etholive/mouse.pt` |
| Design comparison (width, memory) | `python scripts/improve_etholive.py` | `outputs/etholive/improve.csv` |
| Fly fights, per fold | `python scripts/train_fly_eyes2.py --arm swap --fold 3` (folds 3, 2, 5), then `--report`; pack with `scripts/pack_fly.py` | `outputs/etholive/fly_eyes2/` |
| Fly courtship | `python scripts/train_fly_courtship.py --look 6 --fold A --swap` (folds A, B), then `--report`; pack with `scripts/pack_fly_courtship.py` | `outputs/etholive/fly_courtship/` |
| Pose networks | `scripts/mars_pose_to_yolo.py`, `scripts/train_yolo_pose.py` or `colab/train_yolo_pose_colab.ipynb`; fly: `scripts/fly_pose_dataset2.py` | see `docs/pose_model.md` |

## 4. Manuscript results

| Result | Command | Recorded output | Re-run on 2026-11-26 |
|---|---|---|---|
| Lookahead table and figure (Table 1, Figure 1) | `python scripts/lookahead_sweep.py --data mouse`; `--data fights --fold 2`, `3`, `5`; `--data courtship --fold A`, `B`; then `--report` | `results/paper/lookahead/{results,summary}.csv` | no (about a day of GPU time); the summary and statistics were recomputed from `results.csv` |
| Duration test (Spearman, exact permutation p) | part of `--report` | `results/paper/lookahead/stats.json` (Monte Carlo, 200,000 permutations), `stats_exact.json` (exact) | yes: rho -0.727, exact p 0.0144 (K=6); rho -0.545, p 0.0876 (K=31) |
| Robustness to dropping behaviors (Table 2) | `python scripts/robustness_duration.py` (written 2026-11-26; the original analysis was an unsaved inline script) | `results/paper/lookahead/robustness.json` | yes: every value of Table 2 reproduced |
| Bayes-optimal reference (Section 5, Figure 2) | `python scripts/synthetic_lag.py` | `results/paper/synthetic_lag/results.csv` | yes, byte-identical |
| Figures | `python scripts/paper_figures.py` | `paper/figures/*` | yes, byte-identical PNGs |
| Shipped mouse file on validation | `python scripts/verify_mouse_validation.py` | | yes: macro F1 0.7707 (attack 0.741, investigation 0.838, mount 0.733), equal to the recorded 0.771 |
| CalMS21 Task 1 test (Table 3) | `python scripts/test_run_etholive.py` (refuses to run if a result exists) | `results/test_run_etholive/` | no: re-running would be another look at the test split |
| CalMS21 Task 2 and Boy-meets-boy (external test) | `python scripts/external_test.py mouse` / `fly` | `results/external_test/` | no (test data) |
| Throughput | `python scripts/bench_end_to_end.py --chunk 15` (and `--chunk 5`) | `results/benchmarks/end_to_end_2026-11-26_apple_m5.jsonl` | yes: 81 to 99 fps (chunk 15), 61 to 63 fps (chunk 5) |
| Pose network speed alone | `python scripts/bench_yolo_pose_speed.py`, `bench_yolo_pose_sustained.py` | | no |
| Pose accuracy against MARS hand labels | `python scripts/accuracy_mars_handlabels.py` | `results/mars_accuracy.json` | no |

## 5. Checks of causality and leakage

- `tests/test_etholive.py::test_section_2_never_looks_at_future_frames` scrambles all inputs after frame `t` and
  requires an unchanged output at `t`.
- `tests/test_alerts.py::test_decision_uses_only_past_frames` does the same for the alert rule.
- The streaming scorer (`etholive.live_scoring.LiveScorer`) recomputes features over a trailing buffer; its equality
  with whole-session scoring is tested in `tests/test_live_scoring.py` (the real-data check runs with
  `LIVE_TRACKER_SLOW=1`).
- `quality.interpolate_short_gaps` fills gaps of up to 3 frames from both sides, so it is not causal. On CalMS21
  keypoints it fills nothing (0 of 1,320,270 validation keypoint rows; low-confidence points keep their coordinates and
  are dropped later). On fly eye points it fills 0 to 0.003% of rows (5 movies
  measured on 2026-11-26); the frozen flag is disabled for flies.
- `scripts/check_leakage.py` predates the current model files and stops on files without `fit_params`; the split checks
  above were re-done directly.

## 6. Alert-level metrics

`etholive.alerts.run_rule` implements the bout rule of the LiveTracker streaming monitor without its journal. On the
14 validation sessions, with Model B and MouseScope 0.1 scores and four rule settings each, it gave results identical
to the LiveTracker monitor in all 8 comparisons (checked on 2026-11-26 before the move).

## 7. Runs that are records, not recipes

`scripts/test_run_*.py` and `scripts/external_test.py` were each run once. Their recorded checksums refer to the
original file paths (`src/live_tracker/<name>.py`) in the LiveTracker development tree; in this repository the same
modules are in `etholive/` with only their import lines changed, so those `.py` checksums no longer match. The model file
checksums still do.
