"""ONE test run of EthoLive fly 0.5 (team of 3 from scripts/train_fly_v2.py) against fly 0.4, on the same movies:
Aggression 6-10 and Boy meets boy 4-6. This is the SECOND look at both (the first: outputs/test_run_etholive/ and
outputs/external_test/); 0.5 was chosen on validation only, before this run. Refuses to run twice.

Run from the repo root:  python scripts/test_run_fly_v2.py
"""

from __future__ import annotations

import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from etholive import model as etholive                                       # noqa: E402
from etholive import flyvsfly as fv                                 # noqa: E402
from etholive import scope_model as sm                              # noqa: E402
from etholive import scopescore_fly as sf                           # noqa: E402
from etholive import species as sp                                  # noqa: E402

OUT = ROOT / "outputs/test_run_fly_v2"
MODELS = {"EthoLive fly 0.4": "backups/etholive_fly_0.4_2026-11-02/fly.pt", "EthoLive fly 0.5": "models/etholive/fly.pt"}   # paths after the run (then: models/etholive/fly.pt and fly_v2.pt)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "results.json").exists():
        raise SystemExit("already run; not run again")
    (OUT / "frozen_checksums.txt").write_text(
        f"# frozen {datetime.datetime.now().isoformat(timespec='seconds')}, before reading test data\n" +
        "".join(f"{hashlib.sha256((ROOT / p).read_bytes()).hexdigest()}  {p}\n" for p in MODELS.values()))
    behaviors = {n: etholive.load_behavior(ROOT / p) for n, p in MODELS.items()}
    rows = []
    for part, folder, movies in (("aggression", "Aggression", fv.SPLIT["test"]), ("boymeetsboy", "BoyMeetsBoy", fv.BMB_SPLIT["test"])):
        scores, labels = {n: [] for n in MODELS}, []
        for movie in movies:
            poses, lab = fv.load_movie(ROOT / "data/raw/flyvsfly" / folder, movie, part=part)
            sid = fv.session_id(movie, part)
            setup = None if part == "aggression" else fv.calibrate(poses)
            for n, b in behaviors.items():
                f = sf.session_features(poses, sf.FlyParams(**b.methods), setup=setup)
                scores[n].append(sm.score_sessions(b.head, sm.session_arrays(f, None, [sid], b.feature_columns, b.norm),
                                                   species=sp.FLY))
            labels.append(lab)
            print(f"  {part} movie{movie}", flush=True)
        labels = pd.concat(labels, ignore_index=True)
        for n in MODELS:
            m = fv.metrics(pd.concat(scores[n], ignore_index=True), labels)
            keep = (["macro_f1", "lunge_f1", "wing_threat_f1", "tussle_f1", "hold_f1"] if part == "aggression"
                    else ["lunge_f1", "wing_threat_f1"]) + ["lunge_bout_recall", "lunge_bout_precision"]
            row = {"test set": part, "model": n, **{k: m[k] for k in keep}}
            if part == "boymeetsboy":
                row["lunge_wing_mean_f1"] = float(np.mean([m["lunge_f1"], m["wing_threat_f1"]]))
            rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "results.csv", index=False)
    (OUT / "results.json").write_text(json.dumps({"date": datetime.date.today().isoformat(), "rows": rows,
                                                  "note": "second look at both fly test sets"}, indent=2, default=float) + "\n")
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
