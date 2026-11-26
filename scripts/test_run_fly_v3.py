"""ONE test run of EthoLive fly 0.7 against fly 0.6 and 0.5: Aggression 6-10 and Boy meets boy 4-6. THIRD look at both sets
(the first two: outputs/test_run_etholive/ + outputs/external_test/, outputs/test_run_fly_v2/); the first look at them for
0.6, 0.7 and the fly eyes. 0.7 was chosen on validation only and frozen in git before this run. Refuses to run twice.

  Aggression, from the video     the eyes of the shipped file (the real pipeline; points cached in the output folder)
  Aggression, tracker points     like-for-like with the earlier test runs
  Boy meets boy (no video)       tracker points, scale from flyvsfly.calibrate
All models score through score_buffer, which applies a model's lookahead. Intervals resample whole movies, paired across models.

Run from the repo root:  python scripts/test_run_fly_v3.py
"""

from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from etholive import model as etholive                                       # noqa: E402
from etholive import flyvsfly as fv                                 # noqa: E402

OUT = ROOT / "outputs/test_run_fly_v3"
MODELS = {"fly 0.5": "backups/etholive_fly_0.5_with_eyes_2026-11-04/fly.pt", "fly 0.6": "backups/etholive_fly_0.6_2026-11-04/fly.pt",
          "fly 0.7": "models/etholive/fly.pt"}
DATA = ROOT / "data/raw/flyvsfly"
RESAMPLES = 500


def eye_poses(movie):
    path = OUT / f"eyes_movie{movie}.parquet"
    if path.is_file():
        return pd.read_parquet(path)
    eyes = etholive.load_eyes(ROOT / MODELS["fly 0.7"], device="mps")
    cap = cv2.VideoCapture(str(DATA / f"Aggression/movie{movie}/movie{movie}.mp4"))
    rows, f = [], 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        p = eyes.find_mice(image)
        if p is not None:
            rows.append(p.assign(frame=f))
        f += 1
    grid = pd.MultiIndex.from_product([range(f), list(fv.FLIES), list(fv.KEYPOINTS)], names=["frame", "mouse", "keypoint"])
    full = pd.concat(rows, ignore_index=True).set_index(["frame", "mouse", "keypoint"]).reindex(grid).reset_index()
    full = full.assign(session=fv.session_id(movie), time_s=full["frame"] / fv.FPS)
    full["score"] = full["score"].fillna(0.0)
    full = full[["session", "mouse", "keypoint", "frame", "time_s", "x", "y", "score"]]
    full.to_parquet(path, index=False)
    print(f"  eyes on movie{movie}: {f} frames", flush=True)
    return full


def score(behaviors, poses, setup=None):
    out = {}
    for n, b in behaviors.items():
        f = b.session_features(poses, setup=setup)
        out[n] = b.score_buffer(f, set(f["frame"]))
    return out


def resample(rng, per_movie, labels, n):
    """Scores and labels of a draw of whole movies, each drawn movie its own session."""
    ids = rng.integers(0, len(labels), len(labels))
    s = {m: pd.concat([per_movie[m][i].assign(session=str(j)) for j, i in enumerate(ids)], ignore_index=True) for m in n}
    return s, pd.concat([labels[i].assign(session=str(j)) for j, i in enumerate(ids)], ignore_index=True)


def intervals(per_movie, labels, key="macro_f1"):
    rng = np.random.default_rng(0)
    names = list(per_movie)
    draws = {n: [] for n in names}
    for _ in range(RESAMPLES):
        s, lab = resample(rng, per_movie, labels, names)
        for n in names:
            draws[n].append(fv.metrics(s[n], lab)[key])
    d = {n: np.array(v) for n, v in draws.items()}
    q = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]     # noqa: E731
    return {**{n: q(v) for n, v in d.items()}, "0.7 minus 0.5": q(d["fly 0.7"] - d["fly 0.5"]),
            "0.7 minus 0.6": q(d["fly 0.7"] - d["fly 0.6"])}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "results.json").exists():
        raise SystemExit("already run; not run again")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (OUT / "frozen_checksums.txt").write_text(
        f"# frozen {datetime.datetime.now().isoformat(timespec='seconds')}, before reading test data; commit {commit}\n" +
        "".join(f"{hashlib.sha256((ROOT / p).read_bytes()).hexdigest()}  {p}\n" for p in MODELS.values()))
    behaviors = {n: etholive.load_behavior(ROOT / p) for n, p in MODELS.items()}
    sets = {"Aggression 6-10, video": {n: [] for n in MODELS}, "Aggression 6-10, tracker points": {n: [] for n in MODELS},
            "Boy meets boy 4-6, tracker points": {n: [] for n in MODELS}}
    labels = {k: [] for k in sets}
    for movie in fv.SPLIT["test"]:
        track, lab = fv.load_movie(DATA / "Aggression", movie)
        for name, poses in (("Aggression 6-10, video", eye_poses(movie)), ("Aggression 6-10, tracker points", track)):
            for n, s in score(behaviors, poses).items():
                sets[name][n].append(s)
            labels[name].append(lab)
        print(f"  aggression movie{movie}", flush=True)
    for movie in fv.BMB_SPLIT["test"]:
        poses, lab = fv.load_movie(DATA / "BoyMeetsBoy", movie, part="boymeetsboy")
        for n, s in score(behaviors, poses, fv.calibrate(poses)).items():
            sets["Boy meets boy 4-6, tracker points"][n].append(s)
        labels["Boy meets boy 4-6, tracker points"].append(lab)
        print(f"  boy meets boy movie{movie}", flush=True)
    rows, cis = [], {}
    for name, per_movie in sets.items():
        lab = pd.concat(labels[name], ignore_index=True)
        aggression = name.startswith("Aggression")
        keep = (["macro_f1", "lunge_f1", "wing_threat_f1", "tussle_f1", "hold_f1"] if aggression
                else ["lunge_f1", "wing_threat_f1"]) + ["lunge_bout_recall", "lunge_bout_precision"]
        for n in MODELS:
            m = fv.metrics(pd.concat(per_movie[n], ignore_index=True), lab)
            rows.append({"test set": name, "model": n, **{k: m[k] for k in keep}})
        cis[name] = intervals(per_movie, labels[name], "macro_f1" if aggression else "lunge_f1")
        print(f"  intervals done: {name}", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "results.csv", index=False)
    (OUT / "results.json").write_text(json.dumps(
        {"date": datetime.date.today().isoformat(), "commit": commit, "rows": rows, "intervals_95": cis,
         "note": "third look at both fly test sets; first for 0.6, 0.7 and the fly eyes"}, indent=2, default=float) + "\n")
    print(t.round(3).to_string(index=False))
    print(json.dumps(cis, indent=1, default=lambda v: round(v, 3)))


if __name__ == "__main__":
    main()
