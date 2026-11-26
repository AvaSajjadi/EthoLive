"""EthoLive fly: let the brain see K frames past the frame it labels. Never reads a test movie.

Lunges last about 4 frames and 66% of false lunge frames on Aggression 3 lie within 15 frames of a real lunge: the brain
finds the lunge but must call it before seeing how it ends. Live scores already wait for a 15-frame chunk, so a short
lookahead adds little delay. Done by moving the labels K frames later, so the network is unchanged: its output at frame
t, from frames up to t, is about frame t - K.

Declared before running. Recipe = fly 0.6 (train_fly_on_eyes.py --eyes: sqrt weights, partial labels, 3 seeds, team).
  step 1  K = 3, 6, 10 on fold 3 (the usual validation pair)
  step 2  the best K against K = 0 on the other folds; kept only if better on the mean of the 3 folds
Folds (held-out Aggression movie + held-out Boy meets boy movie): 3 = (3, 3), 2 = (2, 1), 5 = (5, 2).
Caveat: the eyes were trained on pictures from Aggression 1, 2, 4, 5, so folds 2 and 5 score somewhat cleaner points; both
arms of a comparison share that. The epoch is chosen on the held-out pair, in every arm.

--bmb30 (fly 0.8 candidate): Boy meets boy is really 200 fps, and earlier runs read it as 30, so its speeds and windows were
wrong by 6.7 times. This flag resamples it to 30 fps first (flyvsfly.to_fps). Declared before running: K = 0, 6, 10 on the 3 folds;
kept only if better on Aggression macro F1, which the change does not rescore. Boy meets boy is compared only among --bmb30 runs:
F1 per frame at 30 fps is another measure than at 200.

Run from the repo root:  python scripts/train_fly_lookahead.py --look K --fold 3 [--bmb30]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
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
import train_fly_v2 as v2                                               # noqa: E402

FOLDS = {3: (3, 3), 2: (2, 1), 5: (5, 2)}
AGG = (1, 2, 3, 4, 5)
BMBS = (1, 2, 3)
SEEDS = (0, 1, 2)


def bmb30_tables():
    """Boy meets boy at 30 fps: same steps as train_fly_v2.bmb_tables, on the resampled recordings (train and validation movies)."""
    path, lpath = ROOT / "data/processed_fly/bmb30.parquet", ROOT / "data/processed_fly/bmb30_labels.parquet"
    if path.is_file() and lpath.is_file():
        return pd.read_parquet(path), pd.read_parquet(lpath)
    feats, labels = [], []
    for movie in BMBS:
        poses, lab = fv.to_fps(*fv.load_movie(ROOT / "data/raw/flyvsfly/BoyMeetsBoy", movie, part="boymeetsboy"), fv.BMB_FPS)
        f = sf.session_features(poses, v2.PARAMS, setup=fv.calibrate(poses))
        num = [c for c in f.columns if c not in ("session", "frame", "time_s", "tracking_ok")]
        feats.append(f.astype({c: "float32" for c in num}))
        labels.append(lab.assign(label=lab["label"].replace({"other": v2.REST})))
        print(f"  boy meets boy movie{movie} at 30 fps: {len(lab)} frames", flush=True)
    feats, labels = pd.concat(feats, ignore_index=True), pd.concat(labels, ignore_index=True)
    feats.to_parquet(path, index=False)
    labels.to_parquet(lpath, index=False)
    return feats, labels


def fold_tables(fold, bmb30=False):
    """(features, labels, manifest): Aggression from the eyes' points, held-out movie under its plain session id."""
    agg_held, bmb_held = FOLDS[fold]
    agg_f, agg_l = v2.agg_tables()
    bmb_f, bmb_l = bmb30_tables() if bmb30 else v2.bmb_tables()
    eye = pd.read_parquet(ROOT / "data/processed_fly/eyes/features.parquet")
    eye = eye.assign(session=eye["session"].str.replace("/eyes", "", regex=False))
    train = [fv.session_id(m) for m in AGG if m != agg_held]
    rename = {s: s + "/eyes" for s in train}
    feats = pd.concat([eye.assign(session=eye["session"].replace(rename)), bmb_f], ignore_index=True)
    labels = pd.concat([agg_l.assign(session=agg_l["session"].replace(rename)), bmb_l], ignore_index=True)
    manifest = {"train": [rename[s] for s in train] + [fv.session_id(m, "boymeetsboy") for m in BMBS if m != bmb_held],
                "validation": [fv.session_id(agg_held), fv.session_id(bmb_held, "boymeetsboy")],
                "test": [fv.session_id(m) for m in fv.SPLIT["test"]]
                        + [fv.session_id(m, "boymeetsboy") for m in fv.BMB_SPLIT["test"]]}
    return feats, labels, manifest, agg_f, agg_l, train


def moved_labels(labels, look):
    """Output at frame t is about frame t - K: each label moves K frames later; a session's first K frames have none."""
    moved = labels.assign(frame=labels["frame"] + look)
    first = labels[labels["frame"] < labels.groupby("session")["frame"].transform("min") + look].assign(label=fv.IGNORE)
    return pd.concat([first, moved], ignore_index=True).sort_values(["session", "frame"], ignore_index=True)


def frame_start(fold, agg_f, agg_l, sessions, cols, out):
    if fold == 3:
        return joblib.load(ROOT / "outputs/etholive/fly/frame_start.joblib")
    path = out.parent / f"frame_start_fold{fold}.joblib"
    if not path.is_file():
        v2.OUT = out
        joblib.dump(v2.fit_start(agg_f, agg_l, sessions, cols), path)
    return joblib.load(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--look", type=int, required=True)
    ap.add_argument("--fold", type=int, required=True, choices=sorted(FOLDS))
    ap.add_argument("--bmb30", action="store_true")
    a = ap.parse_args()
    out = ROOT / f"outputs/etholive/fly_look{'30' if a.bmb30 else ''}/k{a.look}_fold{a.fold}"
    out.mkdir(parents=True, exist_ok=True)
    feats, labels, manifest, agg_f, agg_l, agg_train = fold_tables(a.fold, a.bmb30)
    labels = moved_labels(labels, a.look)
    cols = feature_columns(feats)
    start = sm.linear_start_from(frame_start(a.fold, agg_f, agg_l, agg_train, cols, out), cols, classes=sp.FLY.classes)
    rows, nets, arrays = [], [], None
    for seed in SEEDS:
        path = out / f"behavior_seed{seed}.pt"
        if not path.is_file():
            print(f"== K {a.look} fold {a.fold} seed {seed}", flush=True)
            net, norm, info = etholive.train(feats, labels, manifest, start, species=sp.FLY, seed=seed, metrics=v2.metrics,
                                             partial={v2.REST: ("other", "tussle", "hold")}, balance="sqrt",
                                             log=lambda *_: None)
            info["features"] = f"train_fly_lookahead.py --look {a.look} --fold {a.fold}{' --bmb30' if a.bmb30 else ''}"
            info["lookahead_frames"] = a.look
            etholive.save_behavior(path, net, norm, info)
        net, norm, info = etholive._behavior_from(torch.load(path, map_location="cpu", weights_only=False))
        nets.append(net)
        arrays = arrays or sm.session_arrays(feats, None, manifest["validation"], info["feature_columns"], norm)
        rows.append({"look": a.look, "fold": a.fold, "seed": seed,
                     **v2.metrics(sm.score_sessions(net, arrays, species=sp.FLY), labels)})
    rows.append({"look": a.look, "fold": a.fold, "seed": "team",
                 **v2.metrics(sm.score_sessions(etholive.Team(nets).eval(), arrays, species=sp.FLY), labels)})
    t = pd.DataFrame(rows)
    t.to_csv(out / "validation.csv", index=False)
    print(t[["look", "fold", "seed", "macro_f1", "aggression_macro_f1", "bmb_lunge_wing_f1", "aggression_lunge_f1",
             "aggression_wing_threat_f1", "aggression_tussle_f1", "aggression_hold_f1", "bmb_lunge_f1", "bmb_wing_threat_f1",
             "bmb_lunge_bout_precision"]].round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
