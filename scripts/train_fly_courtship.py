"""EthoLive fly courtship: a separate model for wing extension, circling, attempted copulation and copulation. Never reads the Courtship test
half (ids 17-31) or any fight test movie.

Declared before running. Same network and recipe as fly 0.9 (sqrt weights, 3 seeds, team), on eye points (scripts/extract_fly_eyes2.py).
  folds    A: validate on wild 9-11, hyper 4-5, train on the rest of the train half.   B: validate on wild 1-3, hyper 1-2.
  K        0 and 6 frames of lookahead; the mean of the folds decides.
Labels overlap (a male circles while extending a wing), so each frame takes one: attempted copulation, then circling, wing extension, copulation.
"unknown" bouts are ignored. The labels are of the pair, as with fights: they do not say which fly.

--swap: each training movie is also shown with flies a and b exchanged, since which fly is "a" is arbitrary and wing spread separated wing extension for one of them only.
Declared before running: kept only if the mean macro F1 of the folds beats 0.641 (K 6 without it).

Run from the repo root:  python scripts/train_fly_courtship.py --look 6 --fold A [--swap]     then --report
"""

from __future__ import annotations

import argparse
import sys
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
sys.path.insert(0, str(ROOT / "scripts"))

from etholive import model as etholive                                       # noqa: E402
from etholive import flyvsfly as fv                                 # noqa: E402
from etholive import scope_model as sm                              # noqa: E402
from etholive import species as sp                                  # noqa: E402
from etholive.classify import feature_columns                      # noqa: E402
import train_fly_eyes2 as te                                            # noqa: E402
import train_fly_lookahead as tl                                        # noqa: E402
import train_fly_negatives as tn                                        # noqa: E402

PACK = sp.FLY_COURTSHIP
ORDER = ("copul_attempt", "circle", "wing_extension", "copulation")
MOVIES = tn.TRAIN + tn.DEV
FOLDS = {"A": tn.DEV, "B": [("wild", 1), ("wild", 2), ("wild", 3), ("hyper", 1), ("hyper", 2)]}
OUT = ROOT / "outputs/etholive/fly_courtship"
SEEDS = (0, 1, 2)


def labels_of(group, movie, frames):
    a = fv._mat(tn.DATA / group / f"movie{movie}" / f"movie{movie}_actions.mat")
    n = int(frames.max()) + 1
    label = np.full(n, "other", dtype=object)
    names = [str(x[0]) for x in a["behs"][0]]
    for b in ("unknown",) + tuple(reversed(ORDER)):         # the first in ORDER is written last, so it wins
        if b not in names:
            continue
        j = names.index(b)
        for fly in range(a["bouts"].shape[0]):
            for start, end, *_ in np.atleast_2d(a["bouts"][fly, j]):
                if end > start:
                    label[int(start) - 1:int(end)] = fv.IGNORE if b == "unknown" else b
    return pd.DataFrame({"session": tn.sid(group, movie), "frame": frames, "time_s": frames / fv.FPS, "label": label[frames]})


def tables():
    feats = [te.features(f"{g}{m}") for g, m in MOVIES]
    labels = [labels_of(g, m, f["frame"].to_numpy()) for (g, m), f in zip(MOVIES, feats)]
    return pd.concat(feats, ignore_index=True), pd.concat(labels, ignore_index=True)


def bouts(v):
    d = np.diff(np.r_[0, v.astype(int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def metrics(scores, labels):
    d = labels[["session", "frame", "label"]].merge(scores, on=["session", "frame"], how="inner", validate="one_to_one")
    d = d[d["label"] != fv.IGNORE].sort_values(["session", "frame"])
    names = list(PACK.behaviors)
    p = d[names].to_numpy(float)
    pred = np.asarray(names + ["other"])[np.c_[p, 1 - p.sum(axis=1)].argmax(axis=1)]
    out, f1s = {}, []
    for b in names:
        t, q = d["label"].to_numpy() == b, pred == b
        tp = int((t & q).sum())
        f1 = 2 * tp / max(1, t.sum() + q.sum())
        found = true_n = right = pred_n = 0
        for _, g in d.assign(pred=pred).groupby("session"):
            tt, qq = (g["label"] == b).to_numpy(), (g["pred"] == b).to_numpy()
            tb, qb = bouts(tt), bouts(qq)
            true_n += len(tb)
            pred_n += len(qb)
            found += sum(qq[s:e].any() for s, e in tb)
            right += sum(tt[s:e].any() for s, e in qb)
        out.update({f"{b}_f1": f1, f"{b}_found": found / max(1, true_n), f"{b}_right": right / max(1, pred_n), f"{b}_bouts": true_n})
        f1s.append(f1)
    out["macro_f1"] = float(np.mean(f1s))
    return out


def start_from(feats, labels, sessions, cols):
    d = feats[feats["session"].isin(sessions)].merge(labels[["session", "frame", "label"]], on=["session", "frame"])
    d = d[d["label"].isin(PACK.classes)]
    keep = (d["label"] != "other") & (d["label"] != "copulation") | (d["frame"] % 10 == 0)
    d = d[keep]
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.01, class_weight="balanced", max_iter=2000, random_state=0))
    pipe.fit(d[cols].astype("float32").to_numpy(), d["label"].to_numpy())
    return pipe


def manifest(fold):
    held = [tn.sid(*m) for m in FOLDS[fold]]
    return {"train": [tn.sid(*m) for m in MOVIES if m not in FOLDS[fold]], "validation": held,
            "test": [fv.session_id(m) for m in fv.SPLIT["test"]]}


def swapped(feats, labels, sessions):
    """The training movies again, with every per-fly column of a and b exchanged."""
    f = feats[feats["session"].isin(sessions)]
    names = {c: c[:-1] + ("b" if c.endswith("_a") else "a") for c in f.columns if c.endswith(("_a", "_b"))}
    f = f.rename(columns=names)[list(feats.columns)].assign(session=lambda d: d["session"] + "/swap")
    lab = labels[labels["session"].isin(sessions)].assign(session=lambda d: d["session"] + "/swap")
    return pd.concat([feats, f], ignore_index=True), pd.concat([labels, lab], ignore_index=True)


def train(look, fold, swap=False):
    out = OUT / f"k{look}_fold{fold}{'_swap' if swap else ''}"
    out.mkdir(parents=True, exist_ok=True)
    feats, labels = tables()
    man = manifest(fold)
    cols = feature_columns(feats)
    if swap:
        feats, labels = swapped(feats, labels, man["train"])
        man = {**man, "train": man["train"] + [s + "/swap" for s in man["train"]]}
    start = sm.linear_start_from(start_from(feats, labels, man["train"], cols), cols, classes=PACK.classes)
    moved = tl.moved_labels(labels, look)
    for seed in SEEDS:
        path = out / f"behavior_seed{seed}.pt"
        if path.is_file():
            continue
        print(f"== K {look} fold {fold} seed {seed}", flush=True)
        net, norm, info = etholive.train(feats, moved, man, start, species=PACK, seed=seed, metrics=metrics, balance="sqrt", log=lambda *_: None)
        info["features"] = f"train_fly_courtship.py --look {look} --fold {fold}{' --swap' if swap else ''}"
        info["lookahead_frames"] = look
        etholive.save_behavior(path, net, norm, info)


def report():
    feats, labels = tables()
    rows = []
    for look, tag in ((0, ""), (6, ""), (6, "_swap")):
        for fold in FOLDS:
            folder = OUT / f"k{look}_fold{fold}{tag}"
            if not (folder / "behavior_seed2.pt").is_file():
                continue
            team, norm, info = tn.team_from(folder)
            man = manifest(fold)
            s = sm.score_sessions(team, sm.session_arrays(feats, None, man["validation"], info["feature_columns"], norm), species=PACK)
            rows.append({"look": f"{look}{tag}", "fold": fold, **metrics(s, tl.moved_labels(labels[labels["session"].isin(man["validation"])], look))})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "report.csv", index=False)
    print(t.T.round(3).to_string())
    print(t.groupby("look").mean(numeric_only=True).T.round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--look", type=int, choices=(0, 6))
    ap.add_argument("--fold", choices=sorted(FOLDS))
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--swap", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    report() if a.report else train(a.look, a.fold, a.swap)


if __name__ == "__main__":
    main()
