"""Build the LiveScope 0.2 mouse file (models/livescope/mouse.pt; superseded by EthoLive, etholive.py) from the ablation, on VALIDATION only (never test).

Declared before looking at the results:
  - the feature variant is the one with the best mean validation macro F1 over the ablation seeds;
  - its head is the seed-0 run (no picking the luckiest seed);
  - model B is mixed in with the weight in WEIGHTS that gives the best validation macro F1 (0 = no model B);
  - rule v1 stays the attack rule, so alerts compare fairly with MouseScope 0.1 and model B.

Run from the repo root:  python scripts/build_scope_model.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ablation_scopescore import features                               # noqa: E402
from build_scope_features import VARIANTS                              # noqa: E402
from compare_mousescope import report                                  # noqa: E402
from etholive import scope_model as sm                             # noqa: E402
from etholive import scopescore_mouse as sc                        # noqa: E402
from etholive.calms21 import check_manifest                        # noqa: E402
from etholive.evaluate import summary_metrics                      # noqa: E402

WEIGHTS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
OFF = sc.ScopeParams(swap_fix=False, geometry=False, keypoint_smoothing=False)


def main():
    manifest = check_manifest(json.loads((ROOT / "splits/split_manifest.json").read_text()))
    val = manifest["validation"]
    labels = pd.read_parquet(ROOT / "data/processed/labels.parquet")
    labels = labels[labels["session"].isin(val)]
    ablation = pd.read_csv(ROOT / "outputs/ablation_scopescore.csv")
    means = ablation.groupby("variant")["macro_f1"].mean().sort_values(ascending=False)
    variant = means.index[0]
    print("mean validation macro F1 by variant:\n" + means.round(4).to_string())
    print(f"chosen: {variant}")
    methods = (VARIANTS[variant] if variant in VARIANTS else OFF).to_dict()

    head_path = ROOT / f"outputs/ablation_heads/{variant}_seed0.pt"
    head, norm, info = sm.load_head(head_path)
    feats = features(variant)
    va = sm.session_arrays(feats, None, val, info["feature_columns"], norm)
    model_b = sm.model_b_part(ROOT / "models/model_b.joblib", ROOT / "models/model_b.json")
    tables = {sid: table for sid, (_, _, table) in va.items()}
    d = feats[feats["session"].isin(val)].sort_values(["session", "frame"])
    p_head = np.concatenate([sm.predict_proba(head, x) for x, _, _ in va.values()])
    p_b = sm.model_b_proba(model_b, d)
    base = pd.concat(tables.values(), ignore_index=True)
    grid = []
    for w in WEIGHTS:
        s = sm._behavior_columns(base.copy(), sm.fuse(p_head, p_b, w), sm.MOUSE)
        grid.append({"model_b_weight": w, **summary_metrics(s, labels)})
    grid = pd.DataFrame(grid)
    weight = float(grid.loc[grid["macro_f1"].idxmax(), "model_b_weight"])
    print(grid[["model_b_weight", "macro_f1", "attack_f1", "investigation_f1", "mount_f1"]].round(4).to_string(index=False))
    print(f"model B weight: {weight}")

    out = ROOT / "models/livescope"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(head_path, out / "mouse_head.pt")
    info.update(name=sm.MODEL_NAME, version=sm.MODEL_VERSION, methods=methods, model_b_weight=weight,
                chosen_from="outputs/ablation_scopescore.csv (variant with the best mean validation macro F1, seed 0)")
    (out / "mouse_head.json").write_text(json.dumps(info, indent=2, default=float) + "\n")
    settings = json.loads((ROOT / sm.POSE_SETTINGS["mouse"]).read_text())
    path = sm.pack(out / "mouse.pt", out / "mouse_head.pt", ROOT / settings["weights"], settings,
                   methods=methods, model_b=model_b, model_b_weight=weight)

    final = sm._behavior_columns(base.copy(), sm.fuse(p_head, p_b, weight), sm.MOUSE)
    rows = [report(f"{sm.MODEL_NAME} {sm.MODEL_VERSION}", final, labels)]
    old = pd.read_csv(ROOT / "outputs/mousescope_vs_model_b.csv", index_col=0)      # one column per model
    num = lambda v: float(v) if str(v).replace(".", "", 1).replace("-", "", 1).isdigit() else v     # noqa: E731
    old = [{"model": "MouseScope 0.1" if "MouseScope" in c else c, **{k: num(v) for k, v in old[c].items()}}
           for c in old.columns]
    result = {"variant": variant, "methods": methods, "model_b_weight": weight,
              "ablation_means": means.to_dict(), "weight_grid": grid.to_dict("records"),
              "validation": rows + old}
    (ROOT / "outputs/scope_model_validation.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(pd.DataFrame(result["validation"])[["model", "macro_f1", "attack_f1", "investigation_f1", "mount_f1",
                                               "alert_precision", "alert_recall", "false_alerts_per_hour"]].round(3).to_string(index=False))
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
