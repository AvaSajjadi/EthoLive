"""EthoLive mouse: does the fly's lookahead (scripts/train_fly_lookahead.py) help the mouse? VALIDATION only (never test).

Same design and recipe as mouse 0.4 (3 seeds, team). Labels move K frames later, so the output at frame t is about frame
t - K; a session's first K frames are left out. Declared before running: K = 3, 6, 10 against K = 0, judged on team macro
F1 and on attack false alerts per hour. An alert comes K frames later, which matters more here than for flies, so this
script only measures: changing models/etholive/mouse.pt is a separate decision.

Run from the repo root:  python scripts/mouse_lookahead.py
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

LOOKS = (0, 6, 3, 10)
SEEDS = (0, 1, 2)
OUT = ROOT / "outputs/etholive/mouse_look"
KEEP = ["macro_f1", "attack_f1", "investigation_f1", "mount_f1", "attack_event_precision", "attack_event_recall",
        "attack_false_alerts_per_hour"]


def moved(feats, labels, look):
    first = feats.groupby("session")["frame"].transform("min")
    feats = feats[feats["frame"] >= first + look]
    labels = labels.assign(frame=labels["frame"] + look)
    return feats, labels.merge(feats[["session", "frame"]], on=["session", "frame"], how="inner")     # same frames as the features


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = check_manifest(json.loads((ROOT / "splits/split_manifest.json").read_text()))
    base = pd.read_parquet(ROOT / "data/processed_scope/geometry_swap_smooth.parquet")
    labels0 = pd.read_parquet(ROOT / "data/processed/labels.parquet")
    labels0 = labels0[labels0["session"].isin(set(manifest["train"]) | set(manifest["validation"]))]
    base = base[base["session"].isin(set(manifest["train"]) | set(manifest["validation"]))]
    part = sm.model_b_part(ROOT / "models/model_b.joblib", ROOT / "models/model_b.json")
    rows = []
    for look in LOOKS:
        feats, labels = moved(base, labels0, look)
        nets, va = [], None
        for seed in SEEDS:
            path = (ROOT / f"outputs/etholive/behavior_seed{seed}.pt") if look == 0 else OUT / f"k{look}_seed{seed}.pt"
            if not path.is_file():
                print(f"== K {look} seed {seed}", flush=True)
                net, norm, info = etholive.train(feats, labels, manifest, part, seed=seed, log=lambda *_: None,
                                                 width=64, dilations=(1, 2, 4, 8, 16))
                info["lookahead_frames"] = look
                etholive.save_behavior(path, net, norm, info)
            net, norm, info = etholive._behavior_from(torch.load(path, map_location="cpu", weights_only=False))
            nets.append(net)
            va = va or sm.session_arrays(feats, labels, manifest["validation"], info["feature_columns"], norm)
            rows.append({"look": look, "seed": seed, **{k: summary_metrics(sm.score_sessions(net, va), labels)[k] for k in KEEP}})
        m = summary_metrics(sm.score_sessions(etholive.Team(nets).eval(), va), labels)
        rows.append({"look": look, "seed": "team", **{k: m[k] for k in KEEP}})
        pd.DataFrame(rows).to_csv(OUT / "validation.csv", index=False)
        print(pd.DataFrame(rows).query("look == @look").round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
