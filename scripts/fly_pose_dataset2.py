"""Fly pose dataset, second camera: the Aggression pictures of scripts/fly_pose_dataset.py plus crops of Boy meets boy pictures.

The eyes work at 160 px with flies about 21 px long. A Boy meets boy picture (640 x 480, flies 29 px) is first shrunk so its flies have
the training size (flyvsfly.calibrate), then cut to a 144 x 144 window around them, like an Aggression chamber. The network takes any
multiple of 32, so what it learns on crops holds on a whole picture of that size. Labels: the FlyTracker points, as before.

  train  Boy meets boy 1, 2 (raw frames, 200 fps)     val  Boy meets boy 3      (movies 4-6 are neither downloaded nor read)
  every STEP-th frame, every ACTION_STEP-th during a labelled action; only frames with both flies tracked (the second fly arrives late,
  and a fly with a missing track must not be taught as "no fly"); a window that cuts a fly is moved or the picture left out.

Output: data/fly_pose_yolo2/  (Aggression pictures are linked, not copied).
Run from the repo root:  python scripts/fly_pose_dataset2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from etholive import flyvsfly as fv                                # noqa: E402
from etholive import scopescore_fly as sf                          # noqa: E402
import fly_pose_dataset as d1                                          # noqa: E402

OLD, OUT = ROOT / "data/fly_pose_yolo", ROOT / "data/fly_pose_yolo2"
VIDEO = ROOT / "data/raw/flyvsfly/new_bmb_video"
SPLITS = {"train": [1, 2], "val": [3]}
STEP, ACTION_STEP, WINDOW = 200, 10, 144


def window(pts, scale, w, h):
    """Top-left of a 144 px window holding every point, or of the first fly alone if the pair does not fit and the other is clear of it."""
    p = pts * scale
    lo, hi = np.nanmin(p, axis=(0, 1)) - d1.MARGIN, np.nanmax(p, axis=(0, 1)) + d1.MARGIN
    if (hi - lo).max() > WINDOW:
        return None
    c = (lo + hi) / 2
    x0 = int(np.clip(c[0] - WINDOW / 2, 0, w - WINDOW))
    y0 = int(np.clip(c[1] - WINDOW / 2, 0, h - WINDOW))
    return x0, y0


def main():
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        for kind in ("images", "labels"):
            for p in (OLD / kind / split).iterdir():
                q = OUT / kind / split / p.name
                if not q.exists():
                    q.symlink_to(p)
    counts = {}
    for split, movies in SPLITS.items():
        n = far = 0
        for movie in movies:
            poses, lab = fv.load_movie(ROOT / "data/raw/flyvsfly/BoyMeetsBoy", movie, part="boymeetsboy")
            poses = sf.fix_head_abdomen(poses)
            scale = fv.calibrate(poses)["scale"]
            keep = set(range(0, len(lab), STEP))
            if split == "train":
                act = lab.loc[lab["label"].isin(fv.BEHAVIORS), "frame"]
                keep |= set(act[act % ACTION_STEP == 0])
            by_frame = dict(tuple(poses[poses["frame"].isin(keep)].groupby("frame")))
            cap = cv2.VideoCapture(str(VIDEO / f"movie{movie}.avi"))
            for f in range(max(keep) + 1):
                if f not in keep:
                    cap.grab()
                    continue
                ok, image = cap.read()
                if not ok:
                    break
                rows = by_frame.get(f)
                if rows is None:
                    continue
                pts = rows.set_index(["mouse", "keypoint"])[["x", "y"]].unstack("keypoint")
                body = np.stack([[rows[(rows.mouse == m) & (rows.keypoint == k)][["x", "y"]].to_numpy(float)[0] for k in fv.KEYPOINTS] for m in fv.FLIES])
                if not np.isfinite(body[:, :3]).all():
                    continue
                small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                h, w = small.shape[:2]
                at = window(body, scale, w, h)
                if at is None:
                    far += 1
                    continue
                x0, y0 = at
                crop = small[y0:y0 + WINDOW, x0:x0 + WINDOW]
                moved = rows.assign(x=rows["x"] * scale - x0, y=rows["y"] * scale - y0)
                lines = d1.label_lines(moved, WINDOW, WINDOW)
                stem = f"bmb{movie}_{f:06d}"
                cv2.imwrite(str(OUT / "images" / split / f"{stem}.jpg"), crop)
                (OUT / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")
                n += 1
            print(f"  {split} boy meets boy {movie}: scale {scale:.3f}, {n} crops so far, {far} left out (flies too far apart)", flush=True)
        counts[split] = n
    (OUT / "dataset.yaml").write_text(f"path: {OUT}\ntrain: images/train\nval: images/val\nkpt_shape: [5, 3]\n"
                                      f"flip_idx: {d1.FLIP_IDX}\nnames:\n  0: fly\n")
    print("crops:", counts)


if __name__ == "__main__":
    main()
