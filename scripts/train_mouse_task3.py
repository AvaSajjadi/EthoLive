"""EthoLive mouse: the 7 new behaviors of CalMS21 Task 3, each a small model of its own. Few-shot: 1 to 4 training videos per behavior.

Declared before running. Same ScopeScore features and network as mouse 0.4 (3 seeds, team, balanced weights, no lookahead), the frame branch started from a logistic regression.
Each training video is split in time: first 75% to train, last 25% to choose the epoch. The test file is read only by --test, once, after every design choice. Frame F1 for each
behavior, and their mean, are the numbers; published class-averaged F1 is about 0.34.

Run from the repo root:  python scripts/train_mouse_task3.py --behavior groom    then --report    then --test
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from etholive import calms21                                         # noqa: E402
from etholive import model as etholive                                        # noqa: E402
from etholive import scope_model as sm                               # noqa: E402
from etholive import scopescore_mouse as ssm                         # noqa: E402
from etholive import species as sp                                   # noqa: E402
from etholive.classify import feature_columns                        # noqa: E402

DATA = ROOT / "data/raw/datasets/calms21_task3/task3_new_behaviors"
CACHE = ROOT / "data/processed_task3"
OUT = ROOT / "outputs/etholive/mouse_task3"
BEHAVIORS = ("approach", "disengaged", "groom", "intromission", "mount_attempt", "sniff_face", "whiterearing")
SEEDS = (0, 1, 2)
TRAIN_SHARE = 0.75


def pack(b):
    return replace(sp.MOUSE, name=f"mouse_{b}", behaviors=(b,))


def tables(split, behavior):
    """Features and labels of one behavior and one file."""
    CACHE.mkdir(parents=True, exist_ok=True)
    fp, lp = CACHE / f"{split}_{behavior}.parquet", CACHE / f"{split}_{behavior}_labels.parquet"
    if fp.is_file():
        return pd.read_parquet(fp), pd.read_parquet(lp)
    with open(DATA / f"calms21_task3_{split}.json") as f:
        records = json.load(f)[behavior]
    params = ssm.ScopeParams(**etholive.load_behavior(ROOT / "models/etholive/mouse.pt").methods)
    feats, labels = [], []
    for sid, rec in records.items():
        f = ssm.session_features(calms21._session_to_long(sid, rec), params)
        feats.append(f.astype({c: "float32" for c in f.columns if c not in ("session", "frame", "time_s", "tracking_ok")}))
        a = np.asarray(rec["annotations"])
        labels.append(pd.DataFrame({"session": sid, "frame": np.arange(len(a)), "time_s": np.arange(len(a)) / 30.0, "label": np.where(a > 0, behavior, "other")}))
    feats, labels = pd.concat(feats, ignore_index=True), pd.concat(labels, ignore_index=True)
    feats.to_parquet(fp, index=False)
    labels.to_parquet(lp, index=False)
    return feats, labels


def split_in_time(feats, labels):
    """Each video becomes <id>/early and <id>/late."""
    last = feats.groupby("session")["frame"].transform("max")
    tag = lambda d, cut: d["session"] + np.where(d["frame"] < cut, "/early", "/late")          # noqa: E731
    cut = (last * TRAIN_SHARE).astype(int)
    f = feats.assign(session=tag(feats, cut))
    limits = feats.groupby("session")["frame"].max().mul(TRAIN_SHARE).astype(int)
    l = labels.assign(session=tag(labels, labels["session"].map(limits)))
    return f, l


def metrics_for(b):
    def metrics(scores, labels):
        d = labels[["session", "frame", "label"]].merge(scores, on=["session", "frame"], how="inner")
        t, q = (d["label"] == b).to_numpy(), d[b].to_numpy() >= 0.5
        tp = int((t & q).sum())
        return {"macro_f1": 2 * tp / max(1, t.sum() + q.sum()), "precision": tp / max(1, q.sum()), "recall": tp / max(1, t.sum()), "frames": int(t.sum())}
    return metrics


def train(b):
    out = OUT / b
    out.mkdir(parents=True, exist_ok=True)
    feats, labels = split_in_time(*tables("train", b))
    sessions = sorted(feats["session"].unique())
    man = {"train": [s for s in sessions if s.endswith("/early")], "validation": [s for s in sessions if s.endswith("/late")], "test": []}
    cols = feature_columns(feats)
    d = feats[feats["session"].isin(man["train"])].merge(labels[["session", "frame", "label"]], on=["session", "frame"])
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.01, class_weight="balanced", max_iter=2000, random_state=0))
    pipe.fit(d[cols].astype("float32").to_numpy(), d["label"].to_numpy())
    p = pack(b)
    coef = np.vstack([-pipe[-1].coef_[0], pipe[-1].coef_[0]]) if len(pipe[-1].classes_) == 2 else pipe[-1].coef_
    pipe[-1].coef_, pipe[-1].intercept_ = coef, np.r_[-pipe[-1].intercept_[0], pipe[-1].intercept_[0]]
    start = sm.linear_start_from(pipe, cols, classes=p.classes)
    for seed in SEEDS:
        path = out / f"behavior_seed{seed}.pt"
        if path.is_file():
            continue
        print(f"== {b} seed {seed}", flush=True)
        net, norm, info = etholive.train(feats, labels, man, start, species=p, seed=seed, metrics=metrics_for(b), log=lambda *_: None)
        etholive.save_behavior(path, net, norm, info)


def team(b):
    nets = []
    for s in SEEDS:
        net, norm, info = etholive._behavior_from(torch.load(OUT / b / f"behavior_seed{s}.pt", map_location="cpu", weights_only=False))
        nets.append(net)
    return etholive.Team(nets).eval(), norm, info


def evaluate(b, feats, labels, sessions):
    t, norm, info = team(b)
    s = sm.score_sessions(t, sm.session_arrays(feats, None, sessions, info["feature_columns"], norm), species=pack(b))
    return metrics_for(b)(s, labels)


def report():
    rows = []
    for b in BEHAVIORS:
        if (OUT / b / "behavior_seed2.pt").is_file():
            f, l = split_in_time(*tables("train", b))
            rows.append({"behavior": b, **evaluate(b, f, l, [s for s in f["session"].unique() if s.endswith("/late")])})
    t = pd.DataFrame(rows).rename(columns={"macro_f1": "f1"})
    t.to_csv(OUT / "validation.csv", index=False)
    print(t.round(3).to_string(index=False))
    print("mean F1", round(t["f1"].mean(), 3))


def test():
    path = OUT / "test.csv"
    if path.exists():
        raise SystemExit("already run; not run again")
    rows = []
    for b in BEHAVIORS:
        f, l = tables("test", b)
        rows.append({"behavior": b, "videos": f["session"].nunique(), **evaluate(b, f, l, sorted(f["session"].unique()))})
    t = pd.DataFrame(rows).rename(columns={"macro_f1": "f1"})
    t.to_csv(path, index=False)
    print(t.round(3).to_string(index=False))
    print("mean F1", round(t["f1"].mean(), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavior", choices=BEHAVIORS)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    report() if a.report else test() if a.test else train(a.behavior)


if __name__ == "__main__":
    main()
