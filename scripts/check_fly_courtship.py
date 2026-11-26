"""False alarms on Fly vs. Fly Courtship (male and female, no fighting labelled; same chamber and clock as Aggression), which no
fly model has seen. Tracker points. Official split: ids 1-16 train, 17-31 test (wild 12-21, hyper 6-10).

  --half train   develop on it as needed
  --half test    fresh; run once, on a frozen model: refuses to run twice

Reports aggression calls per hour per model, with the counts by behavior.
Run from the repo root:  python scripts/check_fly_courtship.py --half train|test [--models name=path ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from etholive import model as etholive                                       # noqa: E402
from etholive import flyvsfly as fv                                 # noqa: E402

DATA = ROOT / "data/raw/flyvsfly/new_courtship/Courtship"
TRAIN = [("wild", m) for m in range(1, 12)] + [("hyper", m) for m in range(1, 6)]
TEST = [("wild", m) for m in range(12, 22)] + [("hyper", m) for m in range(6, 11)]
MODELS = {"fly 0.5": "backups/etholive_fly_0.5_with_eyes_2026-11-04/fly.pt", "fly 0.7": "backups/etholive_fly_0.7_2026-11-04/fly.pt",
          "fly 0.8": "models/etholive/fly.pt"}


def bouts(on):
    d = np.diff(np.r_[0, on.astype(int), 0])
    return int((d == 1).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--half", required=True, choices=("train", "test"))
    a = ap.parse_args()
    out = ROOT / f"outputs/etholive/fly_courtship_{a.half}.json"
    if a.half == "test" and out.exists():
        raise SystemExit("already run; not run again")
    behaviors = {n: etholive.load_behavior(ROOT / p) for n, p in MODELS.items()}
    rows = []
    for group, movie in (TRAIN if a.half == "train" else TEST):
        poses, _ = fv.load_movie(DATA / group, movie, part=f"courtship_{group}")
        for n, b in behaviors.items():
            f = b.session_features(poses)
            s = b.score_buffer(f, set(f["frame"]))
            p = s[list(fv.BEHAVIORS)].to_numpy(float)
            call = np.hstack([p, 1 - p.sum(1, keepdims=True)]).argmax(1)
            rows.append({"model": n, "group": group, "movie": movie, "hours": len(s) / fv.FPS / 3600,
                         **{f"{x}_bouts": bouts(call == i) for i, x in enumerate(fv.BEHAVIORS)},
                         **{f"{x}_frames": int((call == i).sum()) for i, x in enumerate(fv.BEHAVIORS)}})
        print(f"  {group} movie{movie}", flush=True)
    t = pd.DataFrame(rows)
    g = t.groupby(["model", "group"]).sum(numeric_only=True).drop(columns="movie")
    rate = g[[c for c in g if c.endswith("_bouts")]].div(g["hours"], axis=0).round(1)
    print(rate.assign(hours=g["hours"].round(1)).to_string())
    out.write_text(json.dumps({"rows": rows}, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
