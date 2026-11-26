"""External test of EthoLive 0.4 on videos no model has seen. Nothing here changes a model.

Mouse: CalMS21 Task 2 ("annotation style transfer"), train and test files: 77 videos, 5 annotators, the same three
behaviors as Task 1. None of these videos is in Task 1 (keypoint fingerprints, outputs/external_test/
calms21_task2_overlap.csv). The labels come from 5 annotators with their own styles, so part of any drop is the
annotators, not the model. EthoLive, MouseScope 0.1 and Model B are scored on the same videos.

Fly: Fly vs. Fly Boy meets boy, movies 4-6 (its official test half; never used). This set labels lunge and wing threat
but not tussle or hold, so only lunge and wing threat are scored. It was filmed with another setup (a bigger
arena, flies about 29 px long instead of 19.5): the poses are rescaled with flyvsfly.calibrate (chosen on Boy meets boy
movies 1-3, never these). Movies 4-6 also have a second human annotation:
second annotator vs first annotator is the human agreement on the same frames.

Run from the repo root:  python scripts/external_test.py mouse   (or: fly)
"""

from __future__ import annotations

import datetime
import gc
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

from compare_mousescope import attack_rule                             # noqa: E402
from test_run_mousescope import from_counts, per_session_counts       # noqa: E402
from etholive import model as etholive                                       # noqa: E402
from etholive import evaluate as ev                                 # noqa: E402
from etholive import flyvsfly as fv                                 # noqa: E402
from etholive import mousescope as ms                               # noqa: E402
from etholive import scope_model as sm                              # noqa: E402
from etholive import scopescore_fly as sf                           # noqa: E402
from etholive import scopescore_mouse as sc                         # noqa: E402
from etholive import species as sp                                  # noqa: E402
from etholive.calms21 import _read_json, _sessions, load_calms21   # noqa: E402
from etholive.classify import score_frames, session_features       # noqa: E402

OUT = ROOT / "outputs/external_test"
FROZEN = ("models/etholive/mouse.pt", "models/etholive/fly.pt", "models/mousescope.pt", "models/model_b.joblib")


def freeze(part):
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / f"results_{part}.json").exists():
        raise SystemExit(f"the external {part} test was already done; it is not run again")
    lines = [f"# frozen {datetime.datetime.now().isoformat(timespec='seconds')}, before reading the external {part} data"]
    lines += [f"{hashlib.sha256((ROOT / r).read_bytes()).hexdigest()}  {r}" for r in FROZEN]
    (OUT / f"frozen_checksums_{part}.txt").write_text("\n".join(lines) + "\n")


def ci(v):
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


# --- mouse -------------------------------------------------------------------------------------

def mouse_scores():
    ethol = etholive.load_behavior(ROOT / "models/etholive/mouse.pt")
    params = sc.ScopeParams(**ethol.methods)
    mscope = ms.load_behavior(ROOT / "models/mousescope.pt")
    model_b = joblib.load(ROOT / "models/model_b.joblib")
    parts = {"EthoLive 0.4": [], "MouseScope 0.1": [], "Model B": []}
    labels = []
    for split in ("train", "test"):
        recs = _sessions(_read_json(ROOT / f"data/raw/calms21_task2/calms21_task2_{split}.json"))
        for i, sid in enumerate(sorted(recs), 1):
            poses, lab = load_calms21({"g": {sid: recs[sid]}})
            f = sc.session_features(poses, params)
            parts["EthoLive 0.4"].append(sm.score_sessions(
                ethol.head, sm.session_arrays(f, None, [sid], ethol.feature_columns, ethol.norm)))
            basic, motion = session_features(poses)
            old = basic.merge(motion[[c for c in motion.columns if c not in basic.columns or c in ("session", "frame")]],
                              on=["session", "frame"])
            parts["MouseScope 0.1"].append(ms.score_sessions(
                mscope.head, ms.session_arrays(old, None, [sid], mscope.feature_columns, mscope.norm)))
            parts["Model B"].append(score_frames(model_b, old, [sid]))
            labels.append(lab)
            print(f"  mouse {split} [{i}/{len(recs)}] {sid}: {len(lab)} frames", flush=True)
        del recs
        gc.collect()
    return {n: pd.concat(p, ignore_index=True) for n, p in parts.items()}, pd.concat(labels, ignore_index=True)


def mouse(result):
    scores, labels = mouse_scores()
    labels["annotator"] = labels["session"].str.split("/").str[1]
    out = {}
    for group, lab in [("all 5 annotators", labels)] + list(labels.groupby("annotator")):
        sids = sorted(lab["session"].unique())
        row = {}
        counts = {}
        for name, s in scores.items():
            s = s[s["session"].isin(sids)]
            frame = ev.summary_metrics(s, lab)
            events, _ = ev.monitor_event_metrics(s, lab, attack_rule())
            row[name] = {"videos": len(sids), "frames": int(len(lab)), "macro_f1": frame["macro_f1"],
                         "attack_f1": frame["attack_f1"], "investigation_f1": frame["investigation_f1"],
                         "mount_f1": frame["mount_f1"], "alert_precision": events["event_precision"],
                         "alert_recall": events["event_recall"], "false_alerts_per_hour": events["false_alerts_per_hour"]}
            counts[name] = per_session_counts(s, lab).set_index("session").loc[sids].reset_index()
        if group == "all 5 annotators":
            rng = np.random.default_rng(0)
            draws = {n: [] for n in scores}
            for _ in range(2000):
                idx = rng.integers(0, len(sids), len(sids))
                for n in scores:
                    draws[n].append(from_counts(counts[n].iloc[idx]))
            for n in scores:
                row[n]["ci95_macro_f1"] = ci([d["macro_f1"] for d in draws[n]])
            row["paired_differences_ci95"] = {
                f"EthoLive 0.4 - {o}": {k: ci([a[k] - b[k] for a, b in zip(draws["EthoLive 0.4"], draws[o])])
                                        for k in ("macro_f1", "attack_f1", "alert_precision")}
                for o in ("MouseScope 0.1", "Model B")}
        out[group] = row
    result["mouse_calms21_task2"] = out
    flat = [{"annotators": g, "model": n, **{k: v for k, v in m.items() if not k.startswith("ci")}}
            for g, r in out.items() for n, m in r.items() if n != "paired_differences_ci95"]
    t = pd.DataFrame(flat)
    t.to_csv(OUT / "mouse_calms21_task2.csv", index=False)
    print(t[["annotators", "model", "videos", "macro_f1", "attack_f1", "investigation_f1", "mount_f1",
             "alert_precision", "alert_recall", "false_alerts_per_hour"]].round(3).to_string(index=False))
    print(json.dumps(out["all 5 annotators"]["paired_differences_ci95"], default=lambda v: round(v, 3)))
    print({n: [round(x, 3) for x in out["all 5 annotators"][n]["ci95_macro_f1"]] for n in scores})


# --- fly ---------------------------------------------------------------------------------------

def fly_scores_vs(scores, labels):
    """Lunge and wing threat only (this set does not label tussle or hold)."""
    m = fv.metrics(scores, labels)
    return {k: m[k] for k in ("lunge_f1", "wing_threat_f1", "lunge_precision", "lunge_recall", "wing_threat_precision",
                              "wing_threat_recall", "lunge_bouts", "lunge_bout_recall", "lunge_bout_precision")}


def as_scores(lab):
    """A human annotation as if it were model scores (1 for its label), to compare two annotators."""
    s = lab[["session", "frame"]].copy()
    for b in fv.BEHAVIORS:
        s[b] = (lab["label"] == b).astype(float).to_numpy()
    return s


def fly(result):
    behavior = etholive.load_behavior(ROOT / "models/etholive/fly.pt")
    params = sf.FlyParams(**behavior.methods)
    folder = ROOT / "data/raw/flyvsfly/BoyMeetsBoy"
    rows = []
    allp, alll, all2 = [], [], []
    for movie in fv.BMB_SPLIT["test"]:
        poses, lab = fv.load_movie(folder, movie, part="boymeetsboy")
        sid = fv.session_id(movie, "boymeetsboy")
        setup = fv.calibrate(poses)                          # chosen on Boy meets boy movies 1-3: scale + edge cap
        f = sf.session_features(poses, params, setup=setup)
        s = sm.score_sessions(behavior.head, sm.session_arrays(f, None, [sid], behavior.feature_columns, behavior.norm),
                              species=sp.FLY)
        rows.append({"movie": movie, "frames": len(lab), "who": "EthoLive 0.4 fly vs annotator 1", **fly_scores_vs(s, lab)})
        second = folder / f"movie{movie}" / f"movie{movie}_second_actions.mat"
        if second.is_file():
            lab2 = fv.load_labels(second, sid, len(lab))
            rows.append({"movie": movie, "frames": len(lab), "who": "annotator 2 vs annotator 1 (humans)",
                         **fly_scores_vs(as_scores(lab2), lab)})
            all2.append(lab2)
        allp.append(s)
        alll.append(lab)
        print(f"  fly boy meets boy movie{movie}: {len(lab)} frames", flush=True)
    s, lab = pd.concat(allp, ignore_index=True), pd.concat(alll, ignore_index=True)
    rows.append({"movie": "all", "frames": len(lab), "who": "EthoLive 0.4 fly vs annotator 1", **fly_scores_vs(s, lab)})
    if all2:
        lab2 = pd.concat(all2, ignore_index=True)
        rows.append({"movie": "all", "frames": len(lab), "who": "annotator 2 vs annotator 1 (humans)",
                     **fly_scores_vs(as_scores(lab2), lab)})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "fly_boymeetsboy.csv", index=False)
    result["fly_boymeetsboy"] = rows
    print(t[["movie", "who", "lunge_f1", "wing_threat_f1", "lunge_bout_recall", "lunge_bout_precision"]]
          .round(3).to_string(index=False))


def main():
    part = sys.argv[1] if len(sys.argv) > 1 else ""
    if part not in ("mouse", "fly"):
        raise SystemExit("usage: python scripts/external_test.py mouse|fly")
    freeze(part)
    result = {"date": datetime.date.today().isoformat()}
    (mouse if part == "mouse" else fly)(result)
    (OUT / f"results_{part}.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(f"\nwrote {(OUT / f'results_{part}.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
