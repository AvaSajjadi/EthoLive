"""EthoLive fly pack, Section 2: the same recipe as the mouse, on Fly vs. Fly Aggression. VALIDATION only (never test).

  1. the frame branch's start: median fill, scaling, class-balanced logistic regression; C chosen on validation
  2. Section 2 = frame branch (starts as that classifier) + memory branch (causal, 2.1 s); 3 seeds
  3. the team of the 3 seeds
Writes outputs/etholive/fly/ (frame_start.joblib, behavior_seed*.pt, validation.csv).

Run from the repo root:  python scripts/train_fly.py [--with-boy-meets-boy]
  --with-boy-meets-boy  also train on Boy meets boy movies 1-3 (official train half); same validation movie;
                        the frame branch's start keeps the C chosen without it (0.01); writes outputs/etholive/fly_bmb/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from etholive import model as etholive                                       # noqa: E402
from etholive import flyvsfly as fv                                 # noqa: E402
from etholive import scope_model as sm                              # noqa: E402
from etholive import species as sp                                  # noqa: E402
from etholive.classify import feature_columns                      # noqa: E402

SEEDS = (0, 1, 2)
OUT = ROOT / "outputs/etholive/fly"
KEEP = ["macro_f1"] + [f"{b}_f1" for b in fv.BEHAVIORS] + ["lunge_bout_recall", "lunge_bout_precision"]


def frame_scores(proba, table):
    s = table.reset_index(drop=True).copy()
    for b in fv.BEHAVIORS:
        s[b] = proba[:, sp.FLY.classes.index(b)]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-boy-meets-boy", action="store_true")
    a = ap.parse_args()
    global OUT
    out_dir = OUT.parent / "fly_bmb" if a.with_boy_meets_boy else OUT
    OUT = out_dir
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "splits/flyvsfly_split.json").read_text())
    feats = pd.read_parquet(ROOT / "data/processed_fly/features.parquet")
    labels = pd.read_parquet(ROOT / "data/processed_fly/labels.parquet")
    grid = (0.01, 0.1, 1.0)
    if a.with_boy_meets_boy:
        bf = pd.read_parquet(ROOT / "data/processed_fly/bmb_features.parquet")
        feats = pd.concat([feats, bf], ignore_index=True)
        labels = pd.concat([labels, pd.read_parquet(ROOT / "data/processed_fly/bmb_labels.parquet")], ignore_index=True)
        manifest = dict(manifest, train=manifest["train"] + sorted(bf["session"].unique()))
        grid = (0.01,)
    cols = feature_columns(feats)
    d = feats.merge(labels[["session", "frame", "label"]], on=["session", "frame"]).sort_values(["session", "frame"])
    tr = d[d["session"].isin(manifest["train"]) & (d["label"] != fv.IGNORE)]
    va = d[d["session"].isin(manifest["validation"])]
    rows = []

    best = None
    for C in grid:
        pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=0))
        pipe.fit(tr[cols].astype("float32").to_numpy(), tr["label"].to_numpy())
        proba = pipe.predict_proba(va[cols].astype("float32").to_numpy())
        order = [list(pipe.classes_).index(c) for c in sp.FLY.classes]
        m = fv.metrics(frame_scores(proba[:, order], va[["session", "frame", "time_s", "tracking_ok"]]), labels)
        print(f"frame-branch start, C={C}: validation macro F1 {m['macro_f1']:.3f}", flush=True)
        if best is None or m["macro_f1"] > best[1]["macro_f1"]:
            best = (pipe, m, C)
    pipe, m, C = best
    joblib.dump(pipe, OUT / "frame_start.joblib")
    part = sm.linear_start_from(pipe, cols, m["macro_f1"], classes=sp.FLY.classes)
    if part["classes"] != list(sp.FLY.classes):
        raise SystemExit(f"frame-branch start classes {part['classes']} are not in the pack order {list(sp.FLY.classes)}")
    rows.append({"model": f"frame branch alone (C={C})", "seed": "", **{k: m[k] for k in KEEP}})

    arrays, nets = None, []
    for seed in SEEDS:
        path = OUT / f"behavior_seed{seed}.pt"
        if not path.is_file():
            print(f"== Section 2, seed {seed}", flush=True)
            net, norm, info = etholive.train(feats, labels, manifest, part, species=sp.FLY, seed=seed,
                                             metrics=fv.metrics)
            info["features"] = "data/processed_fly/features.parquet"
            etholive.save_behavior(path, net, norm, info)
        net, norm, info = etholive._behavior_from(torch.load(path, map_location="cpu", weights_only=False))
        nets.append(net)
        arrays = arrays or sm.session_arrays(feats, None, manifest["validation"], info["feature_columns"], norm)
        m = fv.metrics(sm.score_sessions(net, arrays, species=sp.FLY), labels)
        rows.append({"model": "frame + memory branch", "seed": seed, **{k: m[k] for k in KEEP}})
    m = fv.metrics(sm.score_sessions(etholive.Team(nets).eval(), arrays, species=sp.FLY), labels)
    rows.append({"model": "team of 3", "seed": "", **{k: m[k] for k in KEEP}})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "validation.csv", index=False)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
