"""Fly feature tables for EthoLive's fly pack: train and validation movies only (never test).

Writes data/processed_fly/features.parquet, labels.parquet, swap_rates.csv and splits/flyvsfly_split.json.
Run from the repo root:  python scripts/build_fly_features.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from etholive import flyvsfly as fv                                # noqa: E402
from etholive import scopescore_fly as sf                          # noqa: E402

DATA = ROOT / "data/raw/flyvsfly/Aggression"


def main():
    out = ROOT / "data/processed_fly"
    out.mkdir(parents=True, exist_ok=True)
    (ROOT / "splits/flyvsfly_split.json").write_text(json.dumps(fv.manifest(), indent=2) + "\n")
    feats, labels, rates = [], [], []
    for split in ("train", "validation"):
        for movie in fv.SPLIT[split]:
            t0 = time.time()
            poses, lab = fv.load_movie(DATA, movie)
            sw = sf.swapped_frames(poses)
            f = sf.session_features(poses)
            num = [c for c in f.columns if c not in ("session", "frame", "time_s", "tracking_ok")]
            feats.append(f.astype({c: np.float32 for c in num}))
            labels.append(lab)
            rates.append({"session": fv.session_id(movie), "split": split, "fly_frames": len(sw), "swapped": int(sw.sum())})
            print(f"movie{movie} ({split}): {len(f)} frames, {f.shape[1] - 4} features, labels "
                  f"{lab['label'].value_counts().to_dict()} ({time.time() - t0:.0f} s)", flush=True)
    pd.concat(feats, ignore_index=True).to_parquet(out / "features.parquet", index=False)
    pd.concat(labels, ignore_index=True).to_parquet(out / "labels.parquet", index=False)
    feats, labels = [], []
    for movie in fv.BMB_SPLIT["train"]:                       # Boy meets boy, official train movies only
        t0 = time.time()
        poses, lab = fv.load_movie(ROOT / "data/raw/flyvsfly/BoyMeetsBoy", movie, part="boymeetsboy")
        sw = sf.swapped_frames(poses)
        f = sf.session_features(poses)
        num = [c for c in f.columns if c not in ("session", "frame", "time_s", "tracking_ok")]
        feats.append(f.astype({c: np.float32 for c in num}))
        labels.append(lab)
        rates.append({"session": fv.session_id(movie, "boymeetsboy"), "split": "train", "fly_frames": len(sw),
                      "swapped": int(sw.sum())})
        print(f"boy meets boy movie{movie} (train): {len(f)} frames, labels {lab['label'].value_counts().to_dict()} "
              f"({time.time() - t0:.0f} s)", flush=True)
        del poses
    pd.concat(feats, ignore_index=True).to_parquet(out / "bmb_features.parquet", index=False)
    pd.concat(labels, ignore_index=True).to_parquet(out / "bmb_labels.parquet", index=False)
    pd.DataFrame(rates).to_csv(out / "swap_rates.csv", index=False)


if __name__ == "__main__":
    main()
