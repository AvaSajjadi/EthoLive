"""EthoLive fly 0.5: trained on two camera setups, chosen on validation from both. Never reads a test movie.

Fixes the fly pack failing on Boy meets boy (outputs/external_test/results_fly.json):
  1. every recording is rescaled to the training fly size (flyvsfly.calibrate) and the edge distance is capped at
     72 px (half the Aggression chamber; no change for Aggression, whose values never exceed it);
  2. Boy meets boy labels only lunge and wing threat (plus charge, touch, wing extension). Its other frames get the
     partial label "rest": the net is right there with any of other / tussle / hold (etholive.train, partial=...),
     so it learns what calm flies look like without being told that a tussle is not a tussle.

Declared before training:
  train       Aggression 1, 2, 4, 5 + Boy meets boy 1, 2
  validation  Aggression 3 (macro F1 of 4 behaviors) + Boy meets boy 3 (mean F1 of lunge and wing threat);
              the epoch is chosen on the mean of the two
  test        Aggression 6-10 and Boy meets boy 4-6, once, in scripts/test_run_fly_v2.py (the second look at both)
  frame-branch start: the Aggression logistic regression of 0.4 (outputs/etholive/fly/frame_start.joblib)

Run from the repo root:  python scripts/train_fly_v2.py [--sqrt]
  --sqrt  gentler class weights (square root of "balanced"); writes outputs/etholive/fly_v2_sqrt/
  --tip   also the head ("tip") motion features (scopescore_fly.tip_motion_features); the frame-branch start is refitted
          on the new features (Aggression train, C = 0.01 as before); writes outputs/etholive/fly_v3/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from etholive import model as etholive                                       # noqa: E402
from etholive import flyvsfly as fv                                 # noqa: E402
from etholive import scope_model as sm                              # noqa: E402
from etholive import scopescore_fly as sf                           # noqa: E402
from etholive import species as sp                                  # noqa: E402
from etholive.classify import feature_columns                      # noqa: E402

OUT = ROOT / "outputs/etholive/fly_v2"
SEEDS = (0, 1, 2)
REST = "rest"                                                           # Boy meets boy "other": other, tussle or hold
BMB = {"train": [1, 2], "validation": [3]}


TIP = "--tip" in sys.argv
PARAMS = sf.FlyParams(tip_motion=TIP)


def agg_tables():
    if not TIP:
        return (pd.read_parquet(ROOT / "data/processed_fly/features.parquet"),
                pd.read_parquet(ROOT / "data/processed_fly/labels.parquet"))
    path = ROOT / "data/processed_fly/features_tip.parquet"
    if not path.is_file():
        feats = []
        for movie in fv.SPLIT["train"] + fv.SPLIT["validation"]:
            poses, _ = fv.load_movie(ROOT / "data/raw/flyvsfly/Aggression", movie)
            f = sf.session_features(poses, PARAMS)
            feats.append(f.astype({c: np.float32 for c in f.columns if c not in ("session", "frame", "time_s", "tracking_ok")}))
            print(f"  aggression movie{movie} (tip motion)", flush=True)
        pd.concat(feats, ignore_index=True).to_parquet(path, index=False)
    return pd.read_parquet(path), pd.read_parquet(ROOT / "data/processed_fly/labels.parquet")


def bmb_tables():
    tag = "_tip" if TIP else ""
    path = ROOT / f"data/processed_fly/bmb_calibrated{tag}.parquet"
    lpath = ROOT / "data/processed_fly/bmb_calibrated_labels.parquet"
    if path.is_file() and lpath.is_file():
        return pd.read_parquet(path), pd.read_parquet(lpath)
    feats, labels, setups = [], [], []
    for movie in BMB["train"] + BMB["validation"]:
        poses, lab = fv.load_movie(ROOT / "data/raw/flyvsfly/BoyMeetsBoy", movie, part="boymeetsboy")
        setup = fv.calibrate(poses)
        f = sf.session_features(poses, PARAMS, setup=setup)
        num = [c for c in f.columns if c not in ("session", "frame", "time_s", "tracking_ok")]
        feats.append(f.astype({c: np.float32 for c in num}))
        labels.append(lab.assign(label=lab["label"].replace({"other": REST})))
        setups.append({"movie": movie, **setup})
        print(f"  boy meets boy movie{movie}: scale {setup['scale']:.3f}", flush=True)
    feats, labels = pd.concat(feats, ignore_index=True), pd.concat(labels, ignore_index=True)
    feats.to_parquet(path, index=False)
    labels.to_parquet(lpath, index=False)
    pd.DataFrame(setups).to_csv(ROOT / "data/processed_fly/bmb_calibration.csv", index=False)
    return feats, labels


def fit_start(feats, labels, sessions, cols):
    """The frame branch's start on the given (Aggression train) sessions: same recipe as train_fly.py, C = 0.01."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    d = feats.merge(labels[["session", "frame", "label"]], on=["session", "frame"])
    d = d[d["session"].isin(sessions) & d["label"].isin(sp.FLY.classes)]
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.01, class_weight="balanced", max_iter=2000, random_state=0))
    pipe.fit(d[cols].astype("float32").to_numpy(), d["label"].to_numpy())
    joblib.dump(pipe, OUT / "frame_start.joblib")
    return pipe


def metrics(scores, labels):
    """Validation score: mean of Aggression macro F1 (4 behaviors) and Boy meets boy mean F1 of lunge and wing threat."""
    agg = scores[scores["session"].str.contains("/aggression/")]
    bmb = scores[scores["session"].str.contains("/boymeetsboy/")]
    a = fv.metrics(agg, labels)
    b = fv.metrics(bmb, labels.assign(label=labels["label"].replace({REST: "other"})))
    bmb_f1 = float(np.mean([b["lunge_f1"], b["wing_threat_f1"]]))
    return {"macro_f1": float(np.mean([a["macro_f1"], bmb_f1])), "aggression_macro_f1": a["macro_f1"],
            "bmb_lunge_wing_f1": bmb_f1, **{f"aggression_{k}": a[k] for k in ("lunge_f1", "wing_threat_f1", "tussle_f1", "hold_f1")},
            **{f"bmb_{k}": b[k] for k in ("lunge_f1", "wing_threat_f1", "lunge_bout_recall", "lunge_bout_precision")}}


def main():
    global OUT
    balance = "sqrt" if "--sqrt" in sys.argv else "balanced"
    if balance == "sqrt":
        OUT = OUT.parent / "fly_v2_sqrt"
    if TIP:
        OUT = OUT.parent / "fly_v3"
    OUT.mkdir(parents=True, exist_ok=True)
    agg_f, agg_l = agg_tables()
    bmb_f, bmb_l = bmb_tables()
    feats = pd.concat([agg_f, bmb_f], ignore_index=True)
    labels = pd.concat([agg_l, bmb_l], ignore_index=True)
    base = json.loads((ROOT / "splits/flyvsfly_split.json").read_text())
    manifest = {"train": base["train"] + [fv.session_id(m, "boymeetsboy") for m in BMB["train"]],
                "validation": base["validation"] + [fv.session_id(m, "boymeetsboy") for m in BMB["validation"]],
                "test": base["test"] + [fv.session_id(m, "boymeetsboy") for m in fv.BMB_SPLIT["test"]],
                "note": "EthoLive fly 0.5: Aggression + Boy meets boy (see scripts/train_fly_v2.py)"}
    (ROOT / "splits/flyvsfly_v2_split.json").write_text(json.dumps(manifest, indent=2) + "\n")
    cols = feature_columns(feats)
    if TIP:
        start = sm.linear_start_from(fit_start(feats, labels, base["train"], cols), cols, classes=sp.FLY.classes)
    else:
        start = sm.linear_start_from(joblib.load(ROOT / "outputs/etholive/fly/frame_start.joblib"), cols,
                                     classes=sp.FLY.classes)
    rows, nets = [], []
    arrays = None
    for seed in SEEDS:
        path = OUT / f"behavior_seed{seed}.pt"
        if not path.is_file():
            print(f"== seed {seed}", flush=True)
            net, norm, info = etholive.train(feats, labels, manifest, start, species=sp.FLY, seed=seed, metrics=metrics,
                                             partial={REST: ("other", "tussle", "hold")}, balance=balance)
            info["features"] = "Aggression features.parquet + bmb_calibrated.parquet"
            etholive.save_behavior(path, net, norm, info)
        net, norm, info = etholive._behavior_from(torch.load(path, map_location="cpu", weights_only=False))
        nets.append(net)
        arrays = arrays or sm.session_arrays(feats, None, manifest["validation"], info["feature_columns"], norm)
        rows.append({"model": "frame + memory branch", "seed": seed,
                     **metrics(sm.score_sessions(net, arrays, species=sp.FLY), labels)})
    rows.append({"model": "team of 3", "seed": "",
                 **metrics(sm.score_sessions(etholive.Team(nets).eval(), arrays, species=sp.FLY), labels)})
    old = etholive.load_behavior(ROOT / "models/etholive/fly.pt")             # the shipped fly file, same validation movies
    oa = sm.session_arrays(feats, None, manifest["validation"], old.feature_columns, old.norm)
    rows.insert(0, {"model": f"shipped fly {old.info['version']}", "seed": "",
                    **metrics(sm.score_sessions(old.head, oa, species=sp.FLY), labels)})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "validation.csv", index=False)
    print(t[["model", "seed", "macro_f1", "aggression_macro_f1", "bmb_lunge_wing_f1", "aggression_lunge_f1",
             "aggression_wing_threat_f1", "aggression_tussle_f1", "aggression_hold_f1", "bmb_lunge_f1",
             "bmb_wing_threat_f1", "bmb_lunge_bout_precision"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
