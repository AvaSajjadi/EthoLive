"""The mouse eyes on another lab's camera: CRIM13 frames hand-marked by five annotators (median), CC BY 4.0, Lightning Pose subset.
Uses the out-of-distribution half (new intruder mice). Black mouse = mouse a, white = mouse b, as in CalMS21. Never used for training.
Run from the repo root:  python scripts/check_mouse_eyes_crim13.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from etholive import calms21, etholive                               # noqa: E402

DATA = ROOT / "data/raw/datasets/crim13_subset"
SWAP_EARS = "--swap-ears" in sys.argv                                  # left and right may be defined from the picture, not the mouse
NAME = {"nose": "nose", "left_ear": "right_ear" if SWAP_EARS else "left_ear", "right_ear": "left_ear" if SWAP_EARS else "right_ear", "neck": "top_of_neck", "left_hip": "left_rear_hip", "right_hip": "right_rear_hip", "tail_base": "base_of_tail"}


def main():
    labels = pd.read_csv(DATA / "CollectedData_test.csv", header=[0, 1, 2], index_col=0)
    labels.columns = [f"{b}|{c}" for _, b, c in labels.columns]
    have = {c.split("|")[0] for c in labels.columns}
    print("label points:", sorted(have)[:16])
    eyes = etholive.load_eyes(ROOT / "models/etholive/mouse.pt", device="mps")
    rows = []
    for path, lab in labels.iterrows():
        image = cv2.imread(str(DATA / path))
        if image is None:
            continue
        h, w = image.shape[:2]
        eyes._memory.clear()
        found = eyes.find_mice(image)
        for mouse, coat in (("a", "black"), ("b", "white")):
            for k, n in NAME.items():
                col = f"{coat}_mouse_{n}"
                if f"{col}|x" not in labels.columns:
                    continue
                tx, ty = lab[f"{col}|x"], lab[f"{col}|y"]
                if not np.isfinite(tx):
                    continue
                p = None if found is None else found[(found["mouse"] == mouse) & (found["keypoint"] == k)]
                if p is None or p.empty:
                    rows.append((path, mouse, k, np.nan))
                    continue
                x = float(p["x"].iloc[0]) * w / calms21.IMAGE_SIZE[0]
                y = float(p["y"].iloc[0]) * h / calms21.IMAGE_SIZE[1]
                rows.append((path, mouse, k, float(np.hypot(x - tx, y - ty))))
    t = pd.DataFrame(rows, columns=["image", "mouse", "keypoint", "error_px"])
    body = []
    for path, lab in labels.iterrows():
        for coat in ("black", "white"):
            a, b = f"{coat}_mouse_nose", f"{coat}_mouse_base_of_tail"
            if f"{a}|x" in labels.columns and f"{b}|x" in labels.columns:
                body.append(np.hypot(lab[f"{a}|x"] - lab[f"{b}|x"], lab[f"{a}|y"] - lab[f"{b}|y"]))
    L = float(np.nanmedian(body))
    t.to_csv(ROOT / "outputs/etholive/crim13_eyes.csv", index=False)
    found = t["error_px"].notna().mean()
    e = t["error_px"].dropna()
    print(f"frames {t['image'].nunique()}, points found {found:.1%}; median error {e.median():.1f} px; mouse nose-to-tail {L:.0f} px")
    print(f"within 10% of body length: {(e < 0.1 * L).mean():.1%}; within 20%: {(e < 0.2 * L).mean():.1%}")
    print(t.groupby(["mouse", "keypoint"])["error_px"].median().unstack().round(1).to_string())


if __name__ == "__main__":
    main()
