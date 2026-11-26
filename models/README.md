# Model files

Weights are not stored in git. They are attached to a GitHub release of this repository and checked against
[`CHECKSUMS.sha256`](CHECKSUMS.sha256) after download:

```bash
cd models && shasum -a 256 -c CHECKSUMS.sha256
```

| File | What it is | Size | Notes |
|---|---|---|---|
| `etholive/mouse.pt` | EthoLive 0.4, mouse: pose network (Section 1), feature settings, three-seed behavior team (Section 2) | 11 MB | the file scored on the CalMS21 Task 1 test set (checksum recorded in `results/test_run_etholive/frozen_checksums.txt` before the run) |
| `etholive/fly.pt` | EthoLive fly 0.10: two-camera fly pose network and fight behavior team (lunge, wing threat, tussle, hold), K = 6 | 11 MB | validation numbers only; see the model card |
| `etholive/fly_courtship.pt` | EthoLive fly courtship 0.2 (wing extension, circling, attempted copulation, copulation), K = 6 | 11 MB | validation numbers only |
| `pose/mouse_pose_best.pt` | YOLO26n-pose fine-tuned on the MARS hand-labelled pictures (the network inside `mouse.pt`) | 18 MB | Ultralytics format, for `scripts/eval_yolo_pose.py` and the benchmarks |
| `pose/fly_pose2_best.pt`, `pose/fly_pose_best.pt` | fly pose networks (two cameras; first camera only) | 6 MB each | trained on FlyTracker points, not hand labels |
| `model_b.joblib`, `model_b.json` | logistic regression on 102 features (LiveTracker hackathon Model B) | 6 KB | starting point of the mouse frame branch |
| `model_a.joblib`, `model_a.json` | logistic regression on basic features only | 5 KB | baseline |
| `mousescope.pt`, `mousescope_head.pt` | MouseScope 0.1, the earlier version in the CalMS21 test table | 26 MB | kept for the comparison |
| `pose/pose_model_mac_yolo.json`, `pose/pose_model_mac.json` | pose settings (input size, confidence, hold) | | in git |

The model files are plain dictionaries of tensors and settings saved with `torch.save`; they contain no pickled code
from this package. Load them with `etholive.model.load_eyes` and `etholive.scope_model.load_behavior`.

## Licenses of the weights

- The pose networks are fine-tuned from Ultralytics YOLO26n-pose (AGPL-3.0).
- The mouse pose network was trained on the MARS hand-labelled pictures (CC BY-NC 4.0): research use only.
- The behavior networks were trained on CalMS21 (CC BY 4.0) and Fly vs. Fly (CC0).

## MARS (optional offline pose route)

[`mars/README.md`](mars/README.md) describes the original MARS networks (Segalin et al., 2021), used only by
`scripts/mars_pose_offline.py` and `scripts/accuracy_mars_handlabels.py`. `python scripts/download_mars_models.py`
downloads them and checks their md5 sums.
