"""Model B vs MouseScope's temporal head on the VALIDATION sessions (never test).

Both models get the same 102 features and are judged with the same code:
evaluate.summary_metrics (frame metrics) and evaluate.monitor_event_metrics with the
demo attack rule v1 (ask 0.6, at least 0.5 s), which was chosen for model B.
Rule v1 is kept fixed here, so the comparison does not favour MouseScope.

Run from the repo root:  python scripts/compare_mousescope.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from etholive import evaluate as ev                              # noqa: E402
from etholive import mousescope as ms                            # noqa: E402
from etholive.calms21 import check_manifest                      # noqa: E402
from etholive.classify import load_feature_set, score_frames     # noqa: E402
from etholive.rules import Rule                                  # noqa: E402


def attack_rule():
    r = ev.ATTACK_RULE_V1
    return Rule("V1", "attack", min_duration_s=r["min_duration_s"], alert_threshold=r["alert_threshold"],
                ask_threshold=r["ask_threshold"], alert_mode=r["alert_mode"], status="approved")


def report(name, scores, labels):
    frame = ev.summary_metrics(scores, labels)
    events, bouts = ev.monitor_event_metrics(scores, labels, attack_rule())
    short = bouts[bouts["dur_s"] <= 1.1]
    return {"model": name, "macro_f1": frame["macro_f1"],
            "attack_f1": frame["attack_f1"], "investigation_f1": frame["investigation_f1"],
            "mount_f1": frame["mount_f1"],
            "alerts": events["n_events"], "alert_precision": events["event_precision"],
            "bouts_found": f"{events['detected_true']} of {events['n_true_bouts']}",
            "alert_recall": events["event_recall"],
            "short_bouts_found": f"{int(short['start_offset_s'].notna().sum())} of {len(short)}",
            "false_alerts_per_hour": events["false_alerts_per_hour"],
            "fire_offset_median_s": events["fire_offset_median_s"]}


def main():
    manifest = check_manifest(json.loads((ROOT / "splits/split_manifest.json").read_text()))
    feats = load_feature_set(ROOT / "data/processed", ["basic", "motion"])
    labels = pd.read_parquet(ROOT / "data/processed/labels.parquet")
    val = manifest["validation"]

    model_b = joblib.load(ROOT / "models/model_b.joblib")
    b = score_frames(model_b, feats, val)

    head, norm, info = ms.load_head(ROOT / "models/mousescope_head.pt")
    m = ms.score_sessions(head, ms.session_arrays(feats, None, val, info["feature_columns"], norm))

    rows = [report("model B", b, labels), report(f"{ms.MODEL_NAME} head", m, labels)]
    table = pd.DataFrame(rows).set_index("model").T
    print(f"Validation, {len(val)} sessions, attack rule v1 for both:\n")
    print(table.to_string())
    out = ROOT / "outputs/mousescope_vs_model_b.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
