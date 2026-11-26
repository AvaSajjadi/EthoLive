"""Eye points for every fly recording used in training and validation, from the two-camera eyes (models/pose/fly_pose2_best.pt).
Aggression 1-5, Boy meets boy 1-3 from the raw video at 30 fps (the eyes learn the camera themselves), Courtship train half and dev.
Never reads a test movie, nor Courtship ids 17-31. Output: data/processed_fly/eyes2/. Resumes: a finished recording is kept.

Run from the repo root:  python scripts/extract_fly_eyes2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from etholive import model as etholive                                       # noqa: E402
from etholive import flyvsfly as fv                                 # noqa: E402
import train_fly_negatives as tn                                        # noqa: E402

OUT = ROOT / "data/processed_fly/eyes2"
WEIGHTS = ROOT / "models/pose/fly_pose2_best.pt"
RAW = ROOT / "data/raw/flyvsfly"


def points(video, session, frames=None):
    """Eye points of the whole video, or of the given raw frames, renumbered 0, 1, ..."""
    eyes = etholive.eyes_from_yolo(WEIGHTS, imgsz=(160, 160), conf=0.25, hold=3, kind="fly")
    eyes.network.to("mps")
    cap = cv2.VideoCapture(str(video))
    rows, new, raw = [], 0, 0
    wanted = None if frames is None else iter(frames)
    target = None if wanted is None else next(wanted, None)
    while True:
        if wanted is not None:
            if target is None:
                break
            while raw < target:
                cap.grab()
                raw += 1
        ok, image = cap.read()
        raw += 1
        if not ok:
            break
        p = eyes.find_mice(image)
        if p is not None:
            rows.append(p.assign(frame=new))
        new += 1
        if wanted is not None:
            target = next(wanted, None)
    grid = pd.MultiIndex.from_product([range(new), list(fv.FLIES), list(fv.KEYPOINTS)], names=["frame", "mouse", "keypoint"])
    full = pd.concat(rows, ignore_index=True).set_index(["frame", "mouse", "keypoint"]).reindex(grid).reset_index()
    full = full.assign(session=session, time_s=full["frame"] / fv.FPS)
    full["score"] = full["score"].fillna(0.0)
    return full[["session", "mouse", "keypoint", "frame", "time_s", "x", "y", "score"]], eyes.scale


def save(name, video, session, frames=None):
    path = OUT / f"{name}.parquet"
    if path.is_file():
        return
    poses, scale = points(video, session, frames)
    poses.to_parquet(path, index=False)
    print(f"  {name}: {poses['frame'].max() + 1} frames, eye scale {scale}", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for movie in (1, 2, 3):
        n = len(fv.load_movie(RAW / "BoyMeetsBoy", movie, part="boymeetsboy")[1])
        keep = np.unique(np.round(np.arange(0, (n - 1) / fv.BMB_FPS, 1 / fv.FPS) * fv.BMB_FPS).astype(int))
        save(f"bmb{movie}", RAW / f"new_bmb_video/movie{movie}.avi", fv.session_id(movie, "boymeetsboy"), keep)
    for movie in fv.SPLIT["train"] + fv.SPLIT["validation"]:
        save(f"agg{movie}", RAW / f"Aggression/movie{movie}/movie{movie}.mp4", fv.session_id(movie))
    for group, movie in tn.TRAIN + tn.DEV:
        save(f"{group}{movie}", tn.DATA / group / f"movie{movie}" / f"movie{movie}.mp4", tn.sid(group, movie))


if __name__ == "__main__":
    main()
