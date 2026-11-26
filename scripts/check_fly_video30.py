"""The whole fly pipeline on a second camera: Boy meets boy 3 (never trained on) from its raw video, picture -> eyes -> live scorer
with the camera measured, at the true 30 fps clock. Needs data/raw/flyvsfly/new_bmb_video/movie3.avi. Validation movie, not test.
Compares with the same model on the tracker's points, and with an earlier fly file given as an argument.

Run from the repo root:  python scripts/check_fly_video30.py [other fly file]
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from etholive import model as etholive                                       # noqa: E402
from etholive import flyvsfly as fv                                 # noqa: E402
from etholive.live_scoring import LiveScorer                        # noqa: E402

KEEP = ["lunge_f1", "wing_threat_f1", "lunge_bout_recall", "lunge_bout_precision"]
SID = fv.session_id(3, "boymeetsboy")
SCALE = next((float(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--eye-scale=")), None)   # flies at the trained size
IGNORE = "--ignore-spots" in sys.argv                                                                 # the scout's fixed spots
CACHE = ROOT / f"data/processed_fly/eyes/poses_bmb3_30fps{'' if SCALE is None else f'_s{SCALE}'}{'_ignore' if IGNORE else ''}.parquet"


def eye_points(n_raw):
    if CACHE.is_file():
        return pd.read_parquet(CACHE)
    keep = np.unique(np.round(np.arange(0, (n_raw - 1) / fv.BMB_FPS, 1 / fv.FPS) * fv.BMB_FPS).astype(int))
    eyes = etholive.load_eyes(ROOT / "models/etholive/fly.pt", device="mps")
    if SCALE is not None:
        eyes.scale, eyes.learning = SCALE, False
    if IGNORE:
        from live_tracker import scout   # optional: the fixed-spot finder lives in LiveTracker (experiment not kept)
        video = ROOT / "data/raw/flyvsfly/new_bmb_video/movie3.avi"
        eyes.ignore = [tuple(s) for s in scout.look(video).fixed_spots]
        eyes.scale, eyes.learning = None, True
        print("  ignored spots:", eyes.ignore, flush=True)
    cap = cv2.VideoCapture(str(ROOT / "data/raw/flyvsfly/new_bmb_video/movie3.avi"))
    rows, i = [], 0
    for new, raw in enumerate(keep):
        while i < raw:
            cap.grab()
            i += 1
        ok, image = cap.read()
        i += 1
        if not ok:
            break
        p = eyes.find_mice(image)
        if p is not None:
            rows.append(p.assign(frame=new))
        if new % 6000 == 0:
            print(f"  eyes: frame {new} of {len(keep)}", flush=True)
    grid = pd.MultiIndex.from_product([range(len(keep)), list(fv.FLIES), list(fv.KEYPOINTS)], names=["frame", "mouse", "keypoint"])
    full = pd.concat(rows, ignore_index=True).set_index(["frame", "mouse", "keypoint"]).reindex(grid).reset_index()
    full = full.assign(session=SID, time_s=full["frame"] / fv.FPS)
    full["score"] = full["score"].fillna(0.0)
    full = full[["session", "mouse", "keypoint", "frame", "time_s", "x", "y", "score"]]
    full.to_parquet(CACHE, index=False)
    return full


def live(b, poses, labels):
    scorer = LiveScorer(b, b.feature_columns, session=SID, chunk=15, fps=fv.FPS, setup="auto", picture_size=lambda: (640, 480))
    rows = []
    for f, g in poses.groupby("frame", sort=True):
        rows += scorer.push(int(f), g)
    rows += scorer.flush()
    d = pd.DataFrame(rows)
    return {"first scored s": d["frame"].min() / fv.FPS, "scale": scorer.setup["scale"], **{k: fv.metrics(d, labels)[k] for k in KEEP}}


def main():
    track, labels = fv.load_movie(ROOT / "data/raw/flyvsfly/BoyMeetsBoy", 3, part="boymeetsboy")
    n_raw = len(labels)
    track, labels = fv.to_fps(track, labels, fv.BMB_FPS)
    eyes = eye_points(n_raw)
    files = {"fly file": ROOT / "models/etholive/fly.pt"}
    others = [a for a in sys.argv[1:] if not a.startswith("--")]
    if others:
        files["other file"] = Path(others[0])
    out = {}
    for name, path in files.items():
        b = etholive.load_behavior(path)
        print(name, etholive.read(path)["version"], flush=True)
        for src, poses in (("eyes from video", eyes), ("tracker points", track)):
            out[f"{name}, {src}"] = live(b, poses, labels)
            print(pd.DataFrame(out).T.tail(1).round(3).to_string(), flush=True)
    print(pd.DataFrame(out).T.round(3).to_string())


if __name__ == "__main__":
    main()
