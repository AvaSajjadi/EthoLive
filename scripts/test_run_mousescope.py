"""ONE held-out test run: model B vs MouseScope on the 19 CalMS21 Task 1 test sessions.

Rules (written before the run):
- the models are frozen: their SHA-256 must equal outputs/test_run/frozen_checksums.txt, recorded before any test
  data was read; nothing is trained, tuned or chosen on test;
- both models get the same 102 features (classify.build_feature_files on the test JSON -> data/processed_test/) and are
  judged with the same code as on validation: evaluate.summary_metrics and evaluate.monitor_event_metrics with the
  demo attack rule v1 (chosen for model B on validation);
- 95% intervals resample whole sessions (2000 times), because frames of one session are not independent.

Run from the repo root:  python scripts/test_run_mousescope.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from compare_mousescope import attack_rule                            # noqa: E402
from etholive import evaluate as ev                              # noqa: E402
from etholive import mousescope as ms                            # noqa: E402
from etholive.calms21 import check_manifest                      # noqa: E402
from etholive.classify import load_feature_set, score_frames     # noqa: E402

OUT = ROOT / "outputs/test_run"
FROZEN = ("models/mousescope.pt", "models/model_b.joblib", "models/model_b.json")


def check_frozen():
    recorded = {}
    for line in (OUT / "frozen_checksums.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and len(parts[0]) == 64:
            recorded[parts[1]] = parts[0]
    for rel in FROZEN:
        now = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        if recorded.get(rel) != now:
            raise SystemExit(f"{rel} changed after it was frozen: the test run is refused")


def per_session_counts(scores, labels):
    """Counts per session, so whole sessions can be resampled."""
    rows = []
    for sid, s in scores.groupby("session"):
        a = ev._align(s, labels)
        pred, true = ev.predict_labels(a), a["label"].to_numpy()
        row = {"session": sid}
        for c in ev.BEHAVIORS:
            row[f"{c}_tp"] = int(np.sum((pred == c) & (true == c)))
            row[f"{c}_pred"], row[f"{c}_true"] = int(np.sum(pred == c)), int(np.sum(true == c))
        e, _ = ev.monitor_event_metrics(s, labels, attack_rule())
        row.update(events=e["n_events"], true_events=round(e["event_precision"] * e["n_events"])
                   if e["n_events"] else 0, bouts=e["n_true_bouts"], found=e["detected_true"], frames=len(a))
        rows.append(row)
    return pd.DataFrame(rows)


def from_counts(c):
    f1 = [2 * c[f"{b}_tp"].sum() / (c[f"{b}_pred"].sum() + c[f"{b}_true"].sum())
          if c[f"{b}_pred"].sum() + c[f"{b}_true"].sum() else np.nan for b in ev.BEHAVIORS]
    ev_n, bouts = c["events"].sum(), c["bouts"].sum()
    return {"macro_f1": float(np.nanmean(f1)), "attack_f1": f1[0],
            "alert_precision": c["true_events"].sum() / ev_n if ev_n else np.nan,
            "alert_recall": c["found"].sum() / bouts if bouts else np.nan}


def intervals(c, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    draws = [from_counts(c.iloc[rng.integers(0, len(c), len(c))]) for _ in range(n)]
    return {k: [float(np.nanpercentile([d[k] for d in draws], q)) for q in (2.5, 97.5)] for k in draws[0]}


def main():
    check_frozen()
    manifest = check_manifest(json.loads((ROOT / "splits/split_manifest.json").read_text()))
    test = manifest["test"]
    feats = load_feature_set(ROOT / "data/processed_test", ["basic", "motion"])
    labels = pd.read_parquet(ROOT / "data/processed_test/labels.parquet")
    if set(feats["session"]) != set(test):
        raise SystemExit("the test features do not cover exactly the manifest's test sessions")

    behavior = ms.load_behavior(ROOT / "models/mousescope.pt")
    scores = {
        "model B": score_frames(joblib.load(ROOT / "models/model_b.joblib"), feats, test),
        ms.MODEL_NAME: ms.score_sessions(behavior.head, ms.session_arrays(feats, None, test, behavior.feature_columns,
                                                                         behavior.norm)),
    }
    result, table = {"sessions": len(test), "frames": int(len(labels)), "rule": "attack rule v1"}, {}
    for name, s in scores.items():
        frame = ev.summary_metrics(s, labels)
        events, bouts = ev.monitor_event_metrics(s, labels, attack_rule())
        short = bouts[bouts["dur_s"] <= 1.1]
        ci = intervals(per_session_counts(s, labels))
        table[name] = {
            "macro_f1": frame["macro_f1"], "attack_f1": frame["attack_f1"],
            "investigation_f1": frame["investigation_f1"], "mount_f1": frame["mount_f1"],
            "alerts": events["n_events"], "alert_precision": events["event_precision"],
            "bouts_found": f"{events['detected_true']} of {events['n_true_bouts']}",
            "alert_recall": events["event_recall"],
            "short_bouts_found": f"{int(short['start_offset_s'].notna().sum())} of {len(short)}",
            "false_alerts_per_hour": events["false_alerts_per_hour"],
            "fire_offset_median_s": events["fire_offset_median_s"],
            "ci95_macro_f1": ci["macro_f1"], "ci95_alert_precision": ci["alert_precision"],
            "ci95_alert_recall": ci["alert_recall"]}
    result["models"] = table
    (OUT / "test_results.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(f"HELD-OUT TEST: {len(test)} sessions, {len(labels)} frames, attack rule v1 for both\n")
    print(pd.DataFrame(table).to_string())
    print(f"\nwrote {OUT / 'test_results.json'}")


if __name__ == "__main__":
    main()
