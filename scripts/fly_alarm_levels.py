"""Per-behavior alarm levels for a fly Section 2, on the validation movies only (never test).

A level is added to a behavior's log-probability before the classes are compared (scope_model.apply_levels); a
negative lunge level makes lunge harder to call. Search: coordinate ascent over the 4 behaviors (other fixed at 0),
levels -3 .. +3 in steps of 0.25, 3 passes, maximising the validation score of train_fly_v2.metrics.
Honest check: levels fitted on the first half of every validation movie are scored on the second half, and the other
way round; the final levels are then fitted on all of it.

Run from the repo root:  python scripts/fly_alarm_levels.py outputs/etholive/fly_v3
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from etholive import model as etholive                                       # noqa: E402
from etholive import scope_model as sm                              # noqa: E402
from etholive import species as sp                                  # noqa: E402

GRID = np.arange(-3.0, 3.01, 0.25)


def scores_with(p, table, levels):
    q = sm.apply_levels(p, levels)
    s = table.reset_index(drop=True).copy()
    for b in sp.FLY.behaviors:
        s[b] = q[:, sp.FLY.classes.index(b)]
    return s


def fit(p, table, labels, metric):
    levels = np.zeros(len(sp.FLY.classes))
    best = metric(scores_with(p, table, levels), labels)["macro_f1"]
    for _ in range(3):
        for c in range(len(sp.FLY.behaviors)):
            for v in GRID:
                trial = levels.copy()
                trial[c] = v
                m = metric(scores_with(p, table, trial), labels)["macro_f1"]
                if m > best + 1e-9:
                    best, levels = m, trial
    return levels, best


def main():
    out = Path(sys.argv[1])
    sys.argv = [sys.argv[0]] + (["--tip"] if "fly_v3" in str(out) else [])
    import train_fly_v2 as tf
    agg_f, agg_l = tf.agg_tables()
    bmb_f, bmb_l = tf.bmb_tables()
    feats, labels = pd.concat([agg_f, bmb_f], ignore_index=True), pd.concat([agg_l, bmb_l], ignore_index=True)
    manifest = json.loads((ROOT / "splits/flyvsfly_v2_split.json").read_text())
    nets, norm, info = [], None, None
    for s in (0, 1, 2):
        net, norm, info = etholive._behavior_from(torch.load(out / f"behavior_seed{s}.pt", map_location="cpu", weights_only=False))
        nets.append(net)
    team = etholive.Team(nets).eval()
    arrays = sm.session_arrays(feats, None, manifest["validation"], info["feature_columns"], norm)
    p = np.concatenate([sm.predict_proba(team, x) for x, _, _ in arrays.values()])
    table = pd.concat([t for _, _, t in arrays.values()], ignore_index=True)
    half = table.groupby("session")["frame"].transform(lambda f: f < f.median()).to_numpy()
    rows = []
    for fit_on, check_on in ((half, ~half), (~half, half)):
        lv, _ = fit(p[fit_on], table[fit_on], labels, tf.metrics)
        before = tf.metrics(scores_with(p[check_on], table[check_on], None), labels)
        after = tf.metrics(scores_with(p[check_on], table[check_on], lv), labels)
        rows.append({"levels": lv.round(2).tolist(), "held_out_before": before["macro_f1"], "held_out_after": after["macro_f1"],
                     "bmb_lunge_before": before["bmb_lunge_f1"], "bmb_lunge_after": after["bmb_lunge_f1"]})
    final, score = fit(p, table, labels, tf.metrics)
    result = {"halves": rows, "final_levels": dict(zip(sp.FLY.classes, final.round(2).tolist())),
              "validation_with_final_levels": tf.metrics(scores_with(p, table, final), labels)}
    (out / "alarm_levels.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("final levels:", result["final_levels"])


if __name__ == "__main__":
    main()
