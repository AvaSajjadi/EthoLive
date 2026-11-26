"""Fly pose dataset for EthoLive's fly eyes (Section 1), from Fly vs. Fly Aggression videos + tracks (new file).

The labels are the FlyTracker tracks that come with the data set (automatic, not hand-marked): 5 points per fly
(head, thorax, abdomen, left and right wing tip, as in flyvsfly.py; head and abdomen put in order with the ScopeScore
head-first rule). One class, "fly": the two flies look alike, so identity is kept by tracking (etholive.FlyEyes).

  train       movies 1, 2, 4, 5      validation  movie 3      (test movies 6-10 are not read)
  every STEP-th frame, and every ACTION_STEP-th frame during a labelled action (training movies only): spread wings
  and fights are rare, and with calm frames alone the eyes learned that wings are always folded (wing threat F1 on
  validation fell from 0.66 with tracker points to 0.01 with the first eyes). A fly with a missing track is left out.

Output: data/fly_pose_yolo/{images,labels}/{train,val}/*, dataset.yaml (YOLO pose format).
Run from the repo root:  python scripts/fly_pose_dataset.py [--check]   (--check also writes check.png with labels drawn)
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from etholive import flyvsfly as fv                                # noqa: E402
from etholive import scopescore_fly as sf                          # noqa: E402

DATA = ROOT / "data/raw/flyvsfly/Aggression"
OUT = ROOT / "data/fly_pose_yolo"
STEP = 60
ACTION_STEP = 5
MARGIN = 3.0
SPLITS = {"train": [1, 2, 4, 5], "val": [3]}
FLIP_IDX = [0, 1, 2, 4, 3]                                  # a mirrored picture swaps the two wings


def label_lines(rows, w, h):
    lines = []
    for _, fly in rows.groupby("mouse"):
        pts = fly.set_index("keypoint").loc[list(fv.KEYPOINTS), ["x", "y"]].to_numpy(float)
        if not np.isfinite(pts[:3]).all():                   # no body: no fly in this picture
            continue
        ok = np.isfinite(pts).all(axis=1)
        x0, y0 = np.nanmin(pts[ok], axis=0) - MARGIN
        x1, y1 = np.nanmax(pts[ok], axis=0) + MARGIN
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        kp = []
        for (x, y), v in zip(pts, ok):
            kp += [x / w, y / h, 2] if v else [0, 0, 0]
        lines.append(" ".join(f"{v:.6f}" if isinstance(v, float) else str(v)
                              for v in [0, (x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h] + kp))
    return lines


def main():
    check = "--check" in sys.argv
    counts = {}
    sheet = []
    for split, movies in SPLITS.items():
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        n = 0
        for movie in movies:
            poses, lab = fv.load_movie(DATA, movie)
            poses = sf.fix_head_abdomen(poses)
            keep = set(range(0, len(lab), STEP))
            if split == "train":
                act = lab.loc[lab["label"].isin(fv.BEHAVIORS), "frame"]
                keep |= set(act[act % ACTION_STEP == 0])
            by_frame = dict(tuple(poses[poses["frame"].isin(keep)].groupby("frame")))
            cap = cv2.VideoCapture(str(DATA / f"movie{movie}" / f"movie{movie}.mp4"))
            f = 0
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                if f in by_frame:
                    h, w = image.shape[:2]
                    lines = label_lines(by_frame[f], w, h)
                    if lines:
                        stem = f"movie{movie}_{f:06d}"
                        cv2.imwrite(str(OUT / "images" / split / f"{stem}.jpg"), image)
                        (OUT / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")
                        n += 1
                        if check and split == "val" and len(sheet) < 12 and f % 600 == 0:
                            sheet.append(draw(image, by_frame[f]))
                f += 1
            print(f"  {split} movie{movie}: {f} frames read, {n} pictures so far", flush=True)
        counts[split] = n
    (OUT / "dataset.yaml").write_text(
        f"path: {OUT}\ntrain: images/train\nval: images/val\nkpt_shape: [5, 3]\nflip_idx: {FLIP_IDX}\n"
        f"names:\n  0: fly\n")
    if sheet:
        cv2.imwrite(str(OUT / "check.png"), np.hstack([cv2.resize(s, (288, 288), interpolation=cv2.INTER_NEAREST) for s in sheet[:6]]))
    print("pictures:", counts)


def draw(image, rows):
    img = image.copy()
    colors = {"head": (0, 0, 255), "thorax": (0, 255, 0), "abdomen": (255, 0, 0), "wing_l": (0, 255, 255), "wing_r": (255, 0, 255)}
    for _, r in rows.iterrows():
        if np.isfinite(r["x"]) and np.isfinite(r["y"]):
            cv2.circle(img, (int(round(r["x"])), int(round(r["y"]))), 2, colors[r["keypoint"]], -1)
    return img


if __name__ == "__main__":
    main()
