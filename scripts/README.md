# Scripts

Run from the repository root. Each script's header states what it does, what it reads and writes, and, for
experiments, the decision rule declared before it ran.

**Manuscript**

| Script | Purpose |
|---|---|
| `lookahead_sweep.py` | the cost of being live: one network, window moved from all-past to centred (Table 1, Figure 1) |
| `robustness_duration.py` | the duration trend after dropping behaviors and datasets (Table 2) |
| `synthetic_lag.py` | exact fixed-lag decoder on simulated behaviors (Section 5, Figure 2) |
| `paper_figures.py` | figures from `results/` |
| `verify_mouse_validation.py` | the shipped mouse file on the 14 validation sessions |
| `bench_end_to_end.py` | throughput, picture to behavior probabilities |

**Data preparation**

| Script | Purpose |
|---|---|
| `fetch_public_datasets.py` | download public fly and mouse datasets |
| `build_scope_features.py`, `build_fly_features.py` | feature tables for training |
| `extract_fly_eyes2.py` | fly points from the fly pose network |
| `mars_pose_to_yolo.py`, `fly_pose_dataset.py`, `fly_pose_dataset2.py` | pose training sets |

**Training**

| Script | Purpose |
|---|---|
| `train_yolo_pose.py` | fine-tune the pose network |
| `improve_etholive.py`, `mouse_lookahead.py` | mouse design and lookahead comparisons (validation) |
| `build_scope_model.py`, `ablation_scopescore.py` | feature-step variants and their ablation |
| `train_fly*.py`, `fly_alarm_levels.py` | fly model versions 0.1 to 0.10 (the model card explains each) |
| `train_fly_courtship.py` | fly courtship model |
| `train_mouse_task3.py`, `train_mouse_task3_probe.py` | CalMS21 Task 3 attempts (not kept) |
| `pack_fly.py`, `pack_fly_courtship.py` | write the one-file fly models |

**Evaluation and checks**

| Script | Purpose |
|---|---|
| `test_run_etholive.py`, `test_run_mousescope.py`, `test_run_fly_v2.py`, `test_run_fly_v3.py`, `test_run_courtship.py`, `external_test.py` | one-time test runs (records; they refuse to run twice) |
| `compare_mousescope.py` | MouseScope 0.1 against Model B on validation |
| `check_fly_*.py`, `check_mouse_eyes_crim13.py` | pose and live-path checks on held-out movies and another lab's camera |
| `eval_yolo_pose.py`, `accuracy_mars_handlabels.py`, `bench_yolo_pose_speed.py`, `bench_yolo_pose_sustained.py`, `realtime_pose_test.py` | pose accuracy and speed |
| `check_leakage.py` | split and artifact leakage check (see `docs/reproducibility.md` for its current limitation) |
| `mars_pose_offline.py`, `download_mars_models.py` | the original MARS networks (offline pose route, accuracy reference) |
