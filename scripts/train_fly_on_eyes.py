"""EthoLive fly: train the brain on points from its own eyes, not only on the FlyTracker points. Never reads a test movie.

Live, the brain only ever sees the eyes' points (a bit noisier, other small habits), but fly 0.5 learned from the
tracker's. On validation movie 3 the eyes' points already scored higher (0.582 vs 0.540, check_fly_eyes.py), so the
brain may do better still if it learns from them.

Only the Aggression movies come with video, so only they get eye points (Boy meets boy keeps its tracker points).
The eyes were trained on pictures from Aggression 1, 2, 4, 5 (every 60th frame, every 5th during an action), so their
points there are somewhat cleaner than on a new movie; movie 3 was never shown to the eyes.

Declared before training (same recipe as fly 0.5: train_fly_v2.py --sqrt, 3 seeds, team):
  --mix   train  Aggression 1, 2, 4, 5 tracker points + the same movies' eye points + Boy meets boy 1, 2
  --eyes  train  Aggression 1, 2, 4, 5 eye points only + Boy meets boy 1, 2
  validation (both)  Aggression 3 from the eyes' points + Boy meets boy 3; fly 0.5 scored on the same, for comparison

Run from the repo root:  python scripts/train_fly_on_eyes.py --mix|--eyes
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
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
import train_fly_v2 as v2                                               # noqa: E402

AGG = ROOT / "data/raw/flyvsfly/Aggression"
CACHE = ROOT / "data/processed_fly/eyes"
SEEDS = (0, 1, 2)
NUM = lambda f: [c for c in f.columns if c not in ("session", "frame", "time_s", "tracking_ok")]   # noqa: E731


def eye_poses(movie, eyes):
    path = CACHE / f"poses_movie{movie}.parquet"
    if path.is_file():
        return pd.read_parquet(path)
    cap = cv2.VideoCapture(str(AGG / f"movie{movie}" / f"movie{movie}.mp4"))
    rows, f = [], 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        p = eyes.find_mice(image)
        if p is not None:
            rows.append(p.assign(frame=f))
        f += 1
    seen = pd.concat(rows, ignore_index=True).set_index(["frame", "mouse", "keypoint"])
    full = seen.reindex(pd.MultiIndex.from_product([range(f), list(fv.FLIES), list(fv.KEYPOINTS)],
                                                   names=["frame", "mouse", "keypoint"])).reset_index()
    full = full.assign(session=fv.session_id(movie), time_s=full["frame"] / fv.FPS)
    full["score"] = full["score"].fillna(0.0)
    full = full[["session", "mouse", "keypoint", "frame", "time_s", "x", "y", "score"]]
    full.to_parquet(path, index=False)
    print(f"  eyes on aggression movie{movie}: {f} frames", flush=True)
    return full


def eye_features(movies, params):
    """Features of the eyes' points, one session per movie; session ids end in "/eyes" except the validation movie's,
    which takes the tracker session's place (so labels and the validation list stay as they are)."""
    path = CACHE / "features.parquet"
    if path.is_file():
        return pd.read_parquet(path)
    eyes = etholive.load_eyes(ROOT / "models/etholive/fly.pt", device="mps")
    feats = []
    for movie in movies:
        f = sf.session_features(eye_poses(movie, eyes), params)
        if movie not in fv.SPLIT["validation"]:
            f = f.assign(session=f["session"] + "/eyes")
        feats.append(f.astype({c: np.float32 for c in NUM(f)}))
    out = pd.concat(feats, ignore_index=True)
    out.to_parquet(path, index=False)
    return out


def main():
    mode = "mix" if "--mix" in sys.argv else "eyes" if "--eyes" in sys.argv else None
    if mode is None:
        raise SystemExit("say --mix or --eyes")
    out = ROOT / f"outputs/etholive/fly_on_eyes_{mode}"
    out.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    shipped = etholive.load_behavior(ROOT / "models/etholive/fly.pt")
    params = sf.FlyParams(**shipped.methods)

    agg_f, agg_l = v2.agg_tables()
    bmb_f, bmb_l = v2.bmb_tables()
    eye_f = eye_features(fv.SPLIT["train"] + fv.SPLIT["validation"], params)
    train_eyes = [fv.session_id(m) + "/eyes" for m in fv.SPLIT["train"]]
    eye_l = pd.concat([agg_l[agg_l["session"] == fv.session_id(m)].assign(session=fv.session_id(m) + "/eyes")
                       for m in fv.SPLIT["train"]], ignore_index=True)
    val = fv.session_id(fv.SPLIT["validation"][0])
    tracker_train = [fv.session_id(m) for m in fv.SPLIT["train"]] if mode == "mix" else []
    feats = pd.concat([agg_f[agg_f["session"].isin(tracker_train)], eye_f, bmb_f], ignore_index=True)
    labels = pd.concat([agg_l, eye_l, bmb_l], ignore_index=True)
    manifest = {"train": tracker_train + train_eyes + [fv.session_id(m, "boymeetsboy") for m in v2.BMB["train"]],
                "validation": [val] + [fv.session_id(m, "boymeetsboy") for m in v2.BMB["validation"]],
                # ids only, so check_manifest can confirm no overlap; no test features are loaded or read
                "test": [fv.session_id(m) for m in fv.SPLIT["test"]]
                        + [fv.session_id(m, "boymeetsboy") for m in fv.BMB_SPLIT["test"]]}
    cols = feature_columns(feats)
    start = sm.linear_start_from(__import__("joblib").load(ROOT / "outputs/etholive/fly/frame_start.joblib"), cols,
                                 classes=sp.FLY.classes)
    rows, nets, arrays = [], [], None
    for seed in SEEDS:
        path = out / f"behavior_seed{seed}.pt"
        if not path.is_file():
            print(f"== seed {seed}", flush=True)
            net, norm, info = etholive.train(feats, labels, manifest, start, species=sp.FLY, seed=seed, metrics=v2.metrics,
                                             partial={v2.REST: ("other", "tussle", "hold")}, balance="sqrt")
            info["features"] = f"train_fly_on_eyes.py --{mode}"
            etholive.save_behavior(path, net, norm, info)
        net, norm, info = etholive._behavior_from(torch.load(path, map_location="cpu", weights_only=False))
        nets.append(net)
        arrays = arrays or sm.session_arrays(feats, None, manifest["validation"], info["feature_columns"], norm)
        rows.append({"model": f"trained on eyes ({mode})", "seed": seed,
                     **v2.metrics(sm.score_sessions(net, arrays, species=sp.FLY), labels)})
    rows.append({"model": "team of 3", "seed": "",
                 **v2.metrics(sm.score_sessions(etholive.Team(nets).eval(), arrays, species=sp.FLY), labels)})
    oa = sm.session_arrays(feats, None, manifest["validation"], shipped.feature_columns, shipped.norm)
    rows.insert(0, {"model": f"shipped fly {shipped.info['version']}", "seed": "",
                    **v2.metrics(sm.score_sessions(shipped.head, oa, species=sp.FLY), labels)})
    t = pd.DataFrame(rows)
    t.to_csv(out / "validation.csv", index=False)
    print("validation: Aggression 3 from the eyes' points + Boy meets boy 3")
    print(t[["model", "seed", "macro_f1", "aggression_macro_f1", "bmb_lunge_wing_f1", "aggression_lunge_f1",
             "aggression_wing_threat_f1", "aggression_tussle_f1", "aggression_hold_f1", "bmb_lunge_f1",
             "bmb_wing_threat_f1"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
