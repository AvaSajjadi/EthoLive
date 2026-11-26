"""ONE held-out test run of EthoLive 0.4: the mouse file on the 19 CalMS21 Task 1 test sessions, the fly file on
Fly vs. Fly Aggression movies 6-10. Nothing here is used to change either model.

Protocol:
  - refuses to run twice (outputs/test_run_etholive/test_results.json must not exist);
  - writes the SHA-256 of both model files and of the code that computes their features BEFORE reading any test data;
  - mouse: this is the second look at the CalMS21 test split (the first was MouseScope 0.1 on 2026-10-31,
    outputs/test_run/); MouseScope 0.1 and Model B are scored again on the same sessions for paired differences;
  - fly: the first look at movies 6-10; the frame-branch start (logistic regression) is scored as the baseline;
  - 95% intervals resample whole sessions (mouse, 19) or whole movies (fly, 5: wide by nature).

Run from the repo root:  python scripts/test_run_etholive.py
"""

from __future__ import annotations

import datetime
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

from test_run_mousescope import from_counts, per_session_counts      # noqa: E402
from compare_mousescope import attack_rule                            # noqa: E402
from etholive import model as etholive                                      # noqa: E402
from etholive import evaluate as ev                                # noqa: E402
from etholive import flyvsfly as fv                                # noqa: E402
from etholive import mousescope as ms                              # noqa: E402
from etholive import scope_model as sm                             # noqa: E402
from etholive import scopescore_fly as sf                          # noqa: E402
from etholive import scopescore_mouse as sc                        # noqa: E402
from etholive import species as sp                                 # noqa: E402
from etholive.calms21 import _read_json, _sessions, check_manifest, load_calms21    # noqa: E402
from etholive.classify import load_feature_set, score_frames      # noqa: E402

OUT = ROOT / "outputs/test_run_etholive"
# The recorded run (2026-11-02, outputs/test_run_etholive/frozen_checksums.txt) checksummed these files at their
# original paths in the LiveTracker source tree (src/live_tracker/<name>.py). In this repository they live in etholive/;
# model.py was etholive.py. Only import lines changed in the move, so the recorded checksums of the .py files do not match.
FROZEN = ("models/etholive/mouse.pt", "models/etholive/fly.pt", "etholive/model.py",
          "etholive/scope_model.py", "etholive/scopescore_mouse.py", "etholive/scopescore_fly.py",
          "etholive/flyvsfly.py", "etholive/features.py", "etholive/quality.py",
          "etholive/basic_features.py", "outputs/etholive/fly/frame_start.joblib")
N_BOOT = 2000


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "test_results.json").exists():
        raise SystemExit("the EthoLive test run was already done; it is not run again")
    lines = [f"# frozen {datetime.datetime.now().isoformat(timespec='seconds')}, before reading any test data"]
    lines += [f"{hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()}  {rel}" for rel in FROZEN]
    (OUT / "frozen_checksums.txt").write_text("\n".join(lines) + "\n")


def ci(values):
    return [float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))]


# --- mouse --------------------------------------------------------------------------------------

def mouse(result):
    manifest = check_manifest(json.loads((ROOT / "splits/split_manifest.json").read_text()))
    test = manifest["test"]
    labels = pd.read_parquet(ROOT / "data/processed_test/labels.parquet")
    behavior = etholive.load_behavior(ROOT / "models/etholive/mouse.pt")
    params = sc.ScopeParams(**behavior.methods)
    recs = _sessions(_read_json(ROOT / "data/raw/calms21/calms21_task1_test.json"))
    feats = []
    for i, sid in enumerate(test, 1):
        poses, _ = load_calms21({"g": {sid: recs[sid]}})
        feats.append(sc.session_features(poses, params))
        print(f"  mouse test [{i}/{len(test)}] {sid}", flush=True)
    del recs
    feats = pd.concat(feats, ignore_index=True)
    arrays = sm.session_arrays(feats, None, test, behavior.feature_columns, behavior.norm)
    old = load_feature_set(ROOT / "data/processed_test", ["basic", "motion"])
    ms_beh = ms.load_behavior(ROOT / "models/mousescope.pt")
    scores = {
        "EthoLive 0.4": sm.score_sessions(behavior.head, arrays),
        "MouseScope 0.1": ms.score_sessions(ms_beh.head, ms.session_arrays(old, None, test, ms_beh.feature_columns, ms_beh.norm)),
        "Model B": score_frames(joblib.load(ROOT / "models/model_b.joblib"), old, test),
    }
    table, counts = {}, {}
    for name, s in scores.items():
        frame = ev.summary_metrics(s, labels)
        events, _ = ev.monitor_event_metrics(s, labels, attack_rule())
        counts[name] = per_session_counts(s, labels).set_index("session").loc[test].reset_index()
        table[name] = {"macro_f1": frame["macro_f1"], "attack_f1": frame["attack_f1"],
                       "investigation_f1": frame["investigation_f1"], "mount_f1": frame["mount_f1"],
                       "alerts": events["n_events"], "alert_precision": events["event_precision"],
                       "alert_recall": events["event_recall"], "false_alerts_per_hour": events["false_alerts_per_hour"]}
    rng = np.random.default_rng(0)
    draws = {n: [] for n in scores}
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(test), len(test))                  # the same sessions for every model: paired
        for n in scores:
            draws[n].append(from_counts(counts[n].iloc[idx]))
    for n in scores:
        for k in ("macro_f1", "alert_precision", "alert_recall"):
            table[n][f"ci95_{k}"] = ci([d[k] for d in draws[n]])
    diffs = {f"EthoLive 0.4 - {other}": {k: ci([a[k] - b[k] for a, b in zip(draws["EthoLive 0.4"], draws[other])])
                                         for k in ("macro_f1", "attack_f1", "alert_precision", "alert_recall")}
             for other in ("MouseScope 0.1", "Model B")}
    result["mouse"] = {"sessions": len(test), "frames": int(len(labels)), "rule": "attack rule v1",
                       "note": "second look at the CalMS21 test split (the first: MouseScope 0.1, outputs/test_run/)",
                       "models": table, "paired_differences_ci95": diffs}
    print(pd.DataFrame(table).T[["macro_f1", "attack_f1", "investigation_f1", "mount_f1", "alert_precision",
                                 "alert_recall", "false_alerts_per_hour"]].round(3).to_string())
    print(json.dumps(diffs, indent=1, default=lambda v: round(v, 3)))


# --- fly ----------------------------------------------------------------------------------------

def fly_counts(scores, labels):
    """Per movie: frame tp / predicted / true per behavior, and lunge bouts (to resample whole movies)."""
    rows = []
    for sid, s in scores.groupby("session"):
        lab = labels[labels["session"] == sid]
        m = fv.metrics(s, lab)
        d = lab[["session", "frame", "label"]].merge(s, on=["session", "frame"])
        d = d[d["label"] != fv.IGNORE]
        p = d[list(fv.BEHAVIORS)].to_numpy(float)
        pred = np.asarray(fv.BEHAVIORS + ("other",))[np.hstack([p, 1 - p.sum(1, keepdims=True)]).argmax(1)]
        row = {"session": sid, "lunge_bouts": m["lunge_bouts"],
               "lunge_found": m["lunge_bout_recall"] * m["lunge_bouts"] if m["lunge_bouts"] else 0}
        for b in fv.BEHAVIORS:
            t = d["label"].to_numpy() == b
            row.update({f"{b}_tp": int((t & (pred == b)).sum()), f"{b}_pred": int((pred == b).sum()), f"{b}_true": int(t.sum())})
        rows.append(row)
    return pd.DataFrame(rows)


def fly_from_counts(c):
    f1 = [2 * c[f"{b}_tp"].sum() / (c[f"{b}_pred"].sum() + c[f"{b}_true"].sum())
          if c[f"{b}_pred"].sum() + c[f"{b}_true"].sum() else np.nan for b in fv.BEHAVIORS]
    return {"macro_f1": float(np.nanmean(f1)),
            "lunge_bout_recall": c["lunge_found"].sum() / c["lunge_bouts"].sum() if c["lunge_bouts"].sum() else np.nan}


def fly(result):
    behavior = etholive.load_behavior(ROOT / "models/etholive/fly.pt")
    params = sf.FlyParams(**behavior.methods)
    feats, labels = [], []
    for movie in fv.SPLIT["test"]:
        poses, lab = fv.load_movie(ROOT / "data/raw/flyvsfly/Aggression", movie)
        feats.append(sf.session_features(poses, params))
        labels.append(lab)
        print(f"  fly test movie{movie}", flush=True)
    feats, labels = pd.concat(feats, ignore_index=True), pd.concat(labels, ignore_index=True)
    test = [fv.session_id(m) for m in fv.SPLIT["test"]]
    arrays = sm.session_arrays(feats, None, test, behavior.feature_columns, behavior.norm)
    start = joblib.load(ROOT / "outputs/etholive/fly/frame_start.joblib")
    d = feats.sort_values(["session", "frame"])
    proba = start.predict_proba(d[behavior.feature_columns].astype("float32").to_numpy())
    order = [list(start.classes_).index(c) for c in sp.FLY.classes]
    base = d[["session", "frame", "time_s", "tracking_ok"]].reset_index(drop=True)
    for b in fv.BEHAVIORS:
        base[b] = proba[:, order][:, sp.FLY.classes.index(b)]
    scores = {"EthoLive 0.4 fly": sm.score_sessions(behavior.head, arrays, species=sp.FLY),
              "frame branch alone (logistic regression)": base}
    table, counts = {}, {}
    for name, s in scores.items():
        table[name] = fv.metrics(s, labels)
        counts[name] = fly_counts(s, labels)
    rng = np.random.default_rng(0)
    draws = {n: [] for n in scores}
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(test), len(test))
        for n in scores:
            draws[n].append(fly_from_counts(counts[n].iloc[idx]))
    for n in scores:
        table[n]["ci95_macro_f1"] = ci([x["macro_f1"] for x in draws[n]])
    names = list(scores)
    diff = ci([a["macro_f1"] - b["macro_f1"] for a, b in zip(draws[names[0]], draws[names[1]])])
    result["fly"] = {"movies": len(test), "frames": int((labels["label"] != fv.IGNORE).sum()),
                     "labels": labels["label"].value_counts().to_dict(),
                     "note": "first look at Fly vs. Fly Aggression movies 6-10", "models": table,
                     "paired_difference_macro_f1_ci95": {f"{names[0]} - {names[1]}": diff}}
    keep = ["macro_f1"] + [f"{b}_f1" for b in fv.BEHAVIORS] + ["lunge_bout_recall", "lunge_bout_precision"]
    print(pd.DataFrame(table).T[keep].astype(float).round(3).to_string())
    print("macro F1 difference, 95% interval:", [round(v, 3) for v in diff])


def main():
    freeze()
    result = {"date": datetime.date.today().isoformat(), "frozen": "outputs/test_run_etholive/frozen_checksums.txt"}
    print("MOUSE (CalMS21 Task 1 test, 19 sessions)")
    mouse(result)
    print("\nFLY (Fly vs. Fly Aggression, movies 6-10)")
    fly(result)
    (OUT / "test_results.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(f"\nwrote {(OUT / 'test_results.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
