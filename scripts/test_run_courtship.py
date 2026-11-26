"""ONE fresh false-alarm test: Fly vs. Fly Courtship test half (wild 12-21, hyper 6-10), which no model, choice or script has read.
Picture -> eyes of the shipped file -> fly 0.9, against fly 0.8 on the same points. Male and female pairs, no aggression labels, so this
measures false calls only, not whether fights are found. The model is frozen by the checksums written before any test data is read. Refuses to run twice.

Calls are counted outside copulation, per hour, and as the share of each labelled courtship behavior called a fight. "Unknown" bouts are
left out. A male may really attack a female, so not every call here is surely false.

Run from the repo root:  python scripts/test_run_courtship.py
"""

from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from etholive import model as etholive                                       # noqa: E402
from etholive import flyvsfly as fv                                 # noqa: E402
import extract_fly_eyes2 as ex                                          # noqa: E402
import train_fly_negatives as tn                                        # noqa: E402

OUT = ROOT / "outputs/test_run_courtship"
MOVIES = [("wild", m) for m in range(12, 22)] + [("hyper", m) for m in range(6, 11)]
MODELS = {"fly 0.8": "backups/etholive_fly_0.8_2026-11-05/fly.pt", "fly 0.9": "models/etholive/fly.pt"}


def summary(d):
    free = d[~d["mark"].isin(("copulation", "unknown"))]
    hours = len(free) / fv.FPS / 3600
    out = {"hours_outside_copulation": hours}
    for b in fv.BEHAVIORS:
        out[f"{b}_per_h"] = sum(int((np.diff(np.r_[0, (g["call"] == b).astype(int), 0]) == 1).sum()) for _, g in free.groupby("movie")) / hours
    for m in ("none", "copulation", "wing_extension", "circle", "copul_attempt"):
        g = d[d["mark"] == m]
        out[f"{m}_frames"] = len(g)
        out[f"{m}_called_fight"] = float((g["call"] != "other").mean()) if len(g) else np.nan
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "results.json").exists():
        raise SystemExit("already run; not run again")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (OUT / "frozen_checksums.txt").write_text(
        f"# frozen {datetime.datetime.now().isoformat(timespec='seconds')}, before reading test data; commit {commit}\n" +
        "".join(f"{hashlib.sha256((ROOT / p).read_bytes()).hexdigest()}  {p}\n" for p in MODELS.values()))
    ex.WEIGHTS = ROOT / "models/pose/fly_pose2_best.pt"
    behaviors = {n: etholive.load_behavior(ROOT / p) for n, p in MODELS.items()}
    parts = {n: [] for n in MODELS}
    for group, movie in MOVIES:
        path = OUT / f"{group}{movie}.parquet"
        if path.is_file():
            poses = pd.read_parquet(path)
        else:
            poses, _ = ex.points(tn.DATA / group / f"movie{movie}" / f"movie{movie}.mp4", tn.sid(group, movie))
            poses.to_parquet(path, index=False)
        for n, b in behaviors.items():
            f = b.session_features(poses)
            s = b.score_buffer(f, set(f["frame"])).sort_values("frame")
            mark = tn.true_marks(group, movie, int(poses["frame"].max()) + 1)
            p = s[list(fv.BEHAVIORS)].to_numpy(float)
            parts[n].append(pd.DataFrame({"movie": f"{group}{movie}", "group": group, "mark": mark[s["frame"].to_numpy()],
                                          "call": np.asarray(fv.BEHAVIORS + ("other",))[np.c_[p, 1 - p.sum(1)].argmax(1)]}))
        print(f"  {group} movie{movie}", flush=True)
    rows = []
    for n, pieces in parts.items():
        d = pd.concat(pieces, ignore_index=True)
        rows.append({"model": n, "group": "all", **summary(d)})
        for g, part in d.groupby("group"):
            rows.append({"model": n, "group": g, **summary(part)})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "results.csv", index=False)
    (OUT / "results.json").write_text(json.dumps({"date": datetime.date.today().isoformat(), "commit": commit, "rows": rows,
                                                  "note": "first and only look at the Courtship test half"}, indent=2, default=float) + "\n")
    print(t.set_index(["model", "group"]).T.round(3).to_string())


if __name__ == "__main__":
    main()
