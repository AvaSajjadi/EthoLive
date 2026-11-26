"""Can EthoLive's Section 2 be improved? VALIDATION only (never test).

Each design is trained with 3 seeds; the table gives each seed and the team of the 3 (mean of their probabilities).
  v0.3   width 64,  memory 63 frames (2.1 s)      the shipped design (seeds from `python -m etholive.model train`)
  long   width 64,  memory 127 frames (4.2 s)     one more dilated block; still inside live_scoring's 6 s buffer
  wide   width 128, memory 63 frames

Run from the repo root:  python scripts/improve_etholive.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from etholive import model as etholive                                         # noqa: E402
from etholive import scope_model as sm                             # noqa: E402
from etholive.calms21 import check_manifest                        # noqa: E402
from etholive.evaluate import summary_metrics                      # noqa: E402

DESIGNS = {"v0.3": {"width": 64, "dilations": (1, 2, 4, 8, 16)},
           "long": {"width": 64, "dilations": (1, 2, 4, 8, 16, 32)},
           "wide": {"width": 128, "dilations": (1, 2, 4, 8, 16)}}
SEEDS = (0, 1, 2)
OUT = ROOT / "outputs/etholive"
KEEP = ["macro_f1", "attack_f1", "investigation_f1", "mount_f1", "attack_event_precision", "attack_event_recall",
        "attack_false_alerts_per_hour"]


def path_of(design, seed):
    return OUT / (f"behavior_seed{seed}.pt" if design == "v0.3" else f"behavior_{design}_seed{seed}.pt")


def main():
    manifest = check_manifest(json.loads((ROOT / "splits/split_manifest.json").read_text()))
    feats = pd.read_parquet(ROOT / "data/processed_scope/geometry_swap_smooth.parquet")
    labels = pd.read_parquet(ROOT / "data/processed/labels.parquet")
    labels = labels[labels["session"].isin(set(manifest["train"]) | set(manifest["validation"]))]
    part = sm.model_b_part(ROOT / "models/model_b.joblib", ROOT / "models/model_b.json")
    out = OUT / "improve.csv"
    rows = pd.read_csv(out).to_dict("records") if out.is_file() else []
    done = {(r["design"], str(r["seed"])) for r in rows}
    va = None
    for design, kw in DESIGNS.items():
        nets = []
        for seed in SEEDS:
            path = path_of(design, seed)
            if not path.is_file():
                print(f"== {design}, seed {seed}", flush=True)
                net, norm, info = etholive.train(feats, labels, manifest, part, seed=seed, log=lambda *_: None, **kw)
                etholive.save_behavior(path, net, norm, info)
            net, norm, info = etholive._behavior_from(torch.load(path, map_location="cpu", weights_only=False))
            nets.append(net)
            if va is None:
                va = sm.session_arrays(feats, labels, manifest["validation"], info["feature_columns"], norm)
            if (design, str(seed)) not in done:
                m = summary_metrics(sm.score_sessions(net, va), labels)
                rows.append({"design": design, "seed": seed, **{k: m[k] for k in KEEP}})
        if (design, "team") not in done:
            m = summary_metrics(sm.score_sessions(etholive.Team(nets).eval(), va), labels)
            rows.append({"design": design, "seed": "team", **{k: m[k] for k in KEEP}})
        pd.DataFrame(rows).to_csv(out, index=False)
    t = pd.DataFrame(rows)
    t["kind"] = t["seed"].astype(str).where(t["seed"].astype(str) == "team", "single (mean of 3)")
    print(t.groupby(["design", "kind"])[KEEP].mean().round(3).to_string())


if __name__ == "__main__":
    main()
