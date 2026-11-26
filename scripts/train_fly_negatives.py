"""EthoLive fly: Courtship movies as hard negatives. Never reads a test movie, nor the Courtship test half (ids 17-31).

Copulation and wing extension get called hold and wing threat: the brain has no courtship classes. This adds Courtship train-half
movies to training, every frame "other" (bouts labelled "unknown" are ignored), tracker points.

Declared before running. Recipe = fly 0.8 (train_fly_lookahead.py --look 6 --bmb30), folds 3, 2, 5.
  train    wild 1-8, hyper 1-3            dev    wild 9-11, hyper 4-5 (false calls, in no model's training)
  kept only if false calls on the dev movies fall and Aggression macro F1 and Boy meets boy lunge (30 fps) do not fall by more than
  the seed spread. Courtship has no aggression labels; a male may really attack, so this may teach some wrong "other" frames.

Run from the repo root:  python scripts/train_fly_negatives.py --fold 3     then  --report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from etholive import model as etholive                                       # noqa: E402
from etholive import flyvsfly as fv                                 # noqa: E402
from etholive import scope_model as sm                              # noqa: E402
from etholive import scopescore_fly as sf                           # noqa: E402
from etholive import species as sp                                  # noqa: E402
from etholive.classify import feature_columns                      # noqa: E402
import train_fly_lookahead as tl                                        # noqa: E402
import train_fly_v2 as v2                                               # noqa: E402

DATA = ROOT / "data/raw/flyvsfly/new_courtship/Courtship"
TRAIN = [("wild", m) for m in range(1, 9)] + [("hyper", m) for m in range(1, 4)]
DEV = [("wild", m) for m in range(9, 12)] + [("hyper", m) for m in (4, 5)]
LOOK = 6
OUT = ROOT / "outputs/etholive/fly_neg"
CACHE = ROOT / "data/processed_fly/courtship"


def sid(group, movie):
    return fv.session_id(movie, f"courtship_{group}")


def true_marks(group, movie, n):
    """Per frame: the labelled courtship behavior, "none" otherwise."""
    a = fv._mat(DATA / group / f"movie{movie}" / f"movie{movie}_actions.mat")
    mark = np.full(n, "none", dtype=object)
    for j, b in enumerate(str(x[0]) for x in a["behs"][0]):
        for fly in range(a["bouts"].shape[0]):
            for start, end, *_ in np.atleast_2d(a["bouts"][fly, j]):
                if end > start:
                    mark[int(start) - 1:int(end)] = b
    return mark


def tables(movies):
    CACHE.mkdir(parents=True, exist_ok=True)
    feats, labels = [], []
    for group, movie in movies:
        path = CACHE / f"{group}{movie}.parquet"
        if path.is_file():
            f = pd.read_parquet(path)
        else:
            poses, _ = fv.load_movie(DATA / group, movie, part=f"courtship_{group}")
            f = sf.session_features(poses, v2.PARAMS)
            f = f.astype({c: "float32" for c in f.columns if c not in ("session", "frame", "time_s", "tracking_ok")})
            f.to_parquet(path, index=False)
            print(f"  features: {group} movie{movie}", flush=True)
        mark = true_marks(group, movie, int(f["frame"].max()) + 1)
        labels.append(pd.DataFrame({"session": sid(group, movie), "frame": f["frame"].to_numpy(), "time_s": f["time_s"].to_numpy(),
                                    "label": np.where(mark[f["frame"].to_numpy()] == "unknown", fv.IGNORE, "other")}))
        feats.append(f)
    return pd.concat(feats, ignore_index=True), pd.concat(labels, ignore_index=True)


def team_from(folder):
    nets = []
    for s in (0, 1, 2):
        net, norm, info = etholive._behavior_from(torch.load(folder / f"behavior_seed{s}.pt", map_location="cpu", weights_only=False))
        nets.append(net)
    return etholive.Team(nets).eval(), norm, info


def false_calls(team, norm, info):
    """Calls on the dev movies: bouts per hour outside copulation, and the share of frames of each courtship behavior called a fight."""
    rows = []
    for group, movie in DEV:
        f = pd.read_parquet(CACHE / f"{group}{movie}.parquet") if (CACHE / f"{group}{movie}.parquet").is_file() else tables([(group, movie)])[0]
        s = sm.score_sessions(team, sm.session_arrays(f, None, [sid(group, movie)], info["feature_columns"], norm), species=sp.FLY)
        s = s.sort_values("frame")
        mark = true_marks(group, movie, int(s["frame"].max()) + 1 + LOOK)
        frames = np.maximum(s["frame"].to_numpy() - LOOK, 0)
        p = s[list(fv.BEHAVIORS)].to_numpy(float)
        rows.append(pd.DataFrame({"mark": mark[frames], "call": np.asarray(fv.BEHAVIORS + ("other",))[np.c_[p, 1 - p.sum(1)].argmax(1)]}))
    d = pd.concat(rows, ignore_index=True)
    free = d[d["mark"] != "copulation"]
    hours = len(free) / fv.FPS / 3600
    out = {f"{b}_bouts_per_h_no_copulation": int((np.diff(np.r_[0, (free["call"] == b).astype(int), 0]) == 1).sum()) / hours for b in fv.BEHAVIORS}
    for m in ("none", "copulation", "wing_extension", "circle"):
        g = d[d["mark"] == m]
        out[f"{m}_called_fight"] = float((g["call"] != "other").mean()) if len(g) else np.nan
    return out


def bmb30_score(team, norm, info, fold):
    f, l = tl.bmb30_tables()
    bm = fv.session_id(tl.FOLDS[fold][1], "boymeetsboy")
    s = sm.score_sessions(team, sm.session_arrays(f[f["session"] == bm], None, [bm], info["feature_columns"], norm), species=sp.FLY)
    return fv.metrics(s, tl.moved_labels(l[l["session"] == bm].assign(label=lambda d: d["label"].replace({v2.REST: "other"})), LOOK))


def report():
    rows = []
    for name, base in (("fly 0.8", ROOT / "outputs/etholive/fly_look30/k6_fold{}"), ("hard negatives", OUT / "fold{}")):
        for fold in tl.FOLDS:
            folder = Path(str(base).format(fold))
            if not (folder / "behavior_seed2.pt").is_file():
                continue
            team, norm, info = team_from(folder)
            feats, labels, manifest, *_ = tl.fold_tables(fold, True)
            agg = fv.session_id(tl.FOLDS[fold][0])
            s = sm.score_sessions(team, sm.session_arrays(feats, None, [agg], info["feature_columns"], norm), species=sp.FLY)
            a = fv.metrics(s, tl.moved_labels(labels[labels["session"] == agg], LOOK))
            b = bmb30_score(team, norm, info, fold)
            rows.append({"model": name, "fold": fold, "aggression_macro_f1": a["macro_f1"], "aggression_lunge": a["lunge_f1"],
                         "aggression_wing": a["wing_threat_f1"], "aggression_hold": a["hold_f1"], "bmb_lunge": b["lunge_f1"], "bmb_wing": b["wing_threat_f1"],
                         **false_calls(team, norm, info)})
            print(f"  {name} fold {fold}", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "report.csv", index=False)
    print(t.groupby("model", sort=False).mean(numeric_only=True).drop(columns="fold").T.round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, choices=sorted(tl.FOLDS))
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.report:
        return report()
    out = OUT / f"fold{a.fold}"
    out.mkdir(exist_ok=True)
    feats, labels, manifest, agg_f, agg_l, agg_train = tl.fold_tables(a.fold, True)
    neg_f, neg_l = tables(TRAIN)
    labels = tl.moved_labels(pd.concat([labels, neg_l], ignore_index=True), LOOK)
    feats = pd.concat([feats, neg_f], ignore_index=True)
    manifest = {**manifest, "train": manifest["train"] + [sid(*m) for m in TRAIN]}
    cols = feature_columns(feats)
    start = sm.linear_start_from(tl.frame_start(a.fold, agg_f, agg_l, agg_train, cols, out), cols, classes=sp.FLY.classes)
    for seed in tl.SEEDS:
        path = out / f"behavior_seed{seed}.pt"
        if path.is_file():
            continue
        print(f"== fold {a.fold} seed {seed}", flush=True)
        net, norm, info = etholive.train(feats, labels, manifest, start, species=sp.FLY, seed=seed, metrics=v2.metrics,
                                         partial={v2.REST: ("other", "tussle", "hold")}, balance="sqrt", log=lambda *_: None)
        info["features"] = f"train_fly_negatives.py --fold {a.fold}"
        info["lookahead_frames"] = LOOK
        etholive.save_behavior(path, net, norm, info)


if __name__ == "__main__":
    main()
