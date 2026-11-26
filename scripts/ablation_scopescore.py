"""ScopeScore ablation: the temporal head trained on each feature variant, VALIDATION only (never test).

Variants: the 0.1 features (3 ScopeScore methods) and the data/processed_scope tables from
build_scope_features.py (more ScopeScore methods). Each is trained with several seeds, so the
paper can show the spread, not one lucky run.

Run from the repo root:  python scripts/ablation_scopescore.py [--seeds 0 1 2]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from etholive import scope_model as sm                            # noqa: E402
from etholive.calms21 import check_manifest                       # noqa: E402
from etholive.classify import load_feature_set                    # noqa: E402
from etholive.evaluate import summary_metrics                     # noqa: E402

VARIANTS = ("base", "geometry", "swap", "geometry_swap", "geometry_swap_smooth")


def features(name):
    if name == "base":
        return load_feature_set(ROOT / "data/processed", ["basic", "motion"])
    return pd.read_parquet(ROOT / f"data/processed_scope/{name}.parquet")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    a = ap.parse_args(argv)
    manifest = check_manifest(json.loads((ROOT / "splits/split_manifest.json").read_text()))
    labels = pd.read_parquet(ROOT / "data/processed/labels.parquet")
    labels = labels[labels["session"].isin(set(manifest["train"]) | set(manifest["validation"]))]
    out = ROOT / "outputs/ablation_scopescore.csv"
    rows = pd.read_csv(out).to_dict("records") if out.is_file() else []
    done = {(r["variant"], r["seed"]) for r in rows}
    heads = ROOT / "outputs/ablation_heads"
    heads.mkdir(parents=True, exist_ok=True)
    for name in a.variants:
        feats = features(name)
        for seed in a.seeds:
            if (name, seed) in done:
                continue
            print(f"== {name}, seed {seed}", flush=True)
            head, norm, info = sm.train(feats, labels, manifest, seed=seed, log=lambda *_: None)
            va = sm.session_arrays(feats, labels, manifest["validation"], info["feature_columns"], norm)
            m = summary_metrics(sm.score_sessions(head, va), labels)
            sm.save_head(heads / f"{name}_seed{seed}.pt", head, norm, info)
            rows.append({"variant": name, "seed": seed, "n_features": info["n_features"],
                         "best_epoch": info["best_epoch"], **m})
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"   macro F1 {m['macro_f1']:.3f}  attack {m['attack_f1']:.3f}  "
                  f"investigation {m['investigation_f1']:.3f}  mount {m['mount_f1']:.3f}", flush=True)
    t = pd.DataFrame(rows)
    print(t.groupby("variant")[["macro_f1", "attack_f1", "investigation_f1", "mount_f1"]].agg(["mean", "std"]).round(3))


if __name__ == "__main__":
    main()
