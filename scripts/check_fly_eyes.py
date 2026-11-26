"""Check EthoLive's fly eyes on the validation movie (Aggression 3; never a test movie).

  1. points: the eyes against the FlyTracker points on every frame (fly identity matched per frame), and how often
     the eyes' fly a / fly b labels swap relative to the tracker;
  2. behavior from the camera: the shipped fly brain scored on points from the eyes, against the same brain on the
     tracker's points.

Run from the repo root:  python scripts/check_fly_eyes.py <fly eyes weights .pt>
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from etholive import model as etholive                                       # noqa: E402
from etholive import flyvsfly as fv                                 # noqa: E402
from etholive import scope_model as sm                              # noqa: E402
from etholive import scopescore_fly as sf                           # noqa: E402
from etholive import species as sp                                  # noqa: E402

MOVIE = 3


def main():
    weights = sys.argv[1]
    eyes = etholive.eyes_from_yolo(weights, imgsz=(160, 160), conf=0.25, hold=3, kind="fly")
    eyes.network.to("mps")
    track, labels = fv.load_movie(ROOT / "data/raw/flyvsfly/Aggression", MOVIE)
    track = sf.fix_head_abdomen(track)
    cap = cv2.VideoCapture(str(ROOT / f"data/raw/flyvsfly/Aggression/movie{MOVIE}/movie{MOVIE}.mp4"))
    rows, t0, f = [], time.time(), 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        p = eyes.find_mice(image)
        if p is not None:
            rows.append(p.assign(frame=f))
        f += 1
    secs = time.time() - t0
    seen = pd.concat(rows, ignore_index=True)
    sid = fv.session_id(MOVIE)
    seen = seen.assign(session=sid, time_s=seen["frame"] / fv.FPS)

    # 1. points
    t = track.set_index(["frame", "mouse", "keypoint"])[["x", "y"]]
    e = seen.set_index(["frame", "mouse", "keypoint"])[["x", "y"]]
    errs, swapped, both = {k: [] for k in fv.KEYPOINTS}, [], 0
    th = lambda d, fr, fly: d.loc[(fr, fly, "thorax")].to_numpy(float)        # noqa: E731
    frames_e = set(seen["frame"])
    for fr in range(0, f, 10):
        if fr not in frames_e:
            continue
        try:
            ta, tb, ea, eb = th(t, fr, "a"), th(t, fr, "b"), th(e, fr, "a"), th(e, fr, "b")
        except KeyError:
            continue
        if not np.isfinite(np.r_[ta, tb]).all():
            continue
        both += 1
        swap = np.hypot(*(ea - tb)) + np.hypot(*(eb - ta)) < np.hypot(*(ea - ta)) + np.hypot(*(eb - tb))
        swapped.append(swap)
        pair = {"a": "b", "b": "a"} if swap else {"a": "a", "b": "b"}
        for fly in ("a", "b"):
            for k in fv.KEYPOINTS:
                tp, ep = t.loc[(fr, pair[fly], k)].to_numpy(float), e.loc[(fr, fly, k)].to_numpy(float)
                if np.isfinite(tp).all():
                    errs[k].append(float(np.hypot(*(tp - ep))))
    sw = np.array(swapped)
    points = {"pictures": f, "pictures_per_s": f / secs, "both_flies_found": float(seen.groupby("frame")["mouse"].nunique().eq(2).sum() / f),
              "median_error_px": {k: float(np.median(v)) for k, v in errs.items()},
              "identity_switches_per_min": float(np.sum(sw[1:] != sw[:-1]) / (len(sw) * 10 / fv.FPS / 60))}
    # 2. behavior
    b = etholive.load_behavior(ROOT / "models/etholive/fly.pt")
    p = sf.FlyParams(**b.methods)
    out = {"points": points}
    for name, poses in (("tracker points", track), ("EthoLive eyes", seen)):
        full = poses.set_index(["frame", "mouse", "keypoint"]).reindex(
            pd.MultiIndex.from_product([range(f), list(fv.FLIES), list(fv.KEYPOINTS)], names=["frame", "mouse", "keypoint"]))
        full = full.reset_index().assign(session=sid, time_s=lambda d: d["frame"] / fv.FPS)
        full["score"] = full["score"].fillna(0.0)
        feats = sf.session_features(full, p)
        s = b.score_buffer(feats, set(feats["frame"]))
        out[name] = fv.metrics(s, labels)
    (ROOT / "outputs/etholive/fly_eyes_check.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(json.dumps(points, indent=1, default=lambda v: round(v, 3)))
    keep = ["macro_f1", "lunge_f1", "wing_threat_f1", "tussle_f1", "hold_f1", "lunge_bout_recall", "lunge_bout_precision"]
    print(pd.DataFrame({n: {k: out[n][k] for k in keep} for n in ("tracker points", "EthoLive eyes")}).T.round(3).to_string())


if __name__ == "__main__":
    main()
