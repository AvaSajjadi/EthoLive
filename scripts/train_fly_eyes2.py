"""EthoLive fly: the final brains, on the two-camera eyes' own points (scripts/extract_fly_eyes2.py). Never reads a test movie.

Everything the brain sees here is eye points, as it is live, so a hard-negative gain cannot be a tracker-versus-eyes shortcut.
Declared before running. Recipe = fly 0.8 (K 6, true clock, sqrt weights, partial labels, 3 seeds, team), folds 3, 2, 5.
  --arm plain   Aggression + Boy meets boy, eye points
  --arm neg     the same + Courtship wild 1-8, hyper 1-3 as "other", eye points
--report scores on eye points: held-out Aggression, held-out Boy meets boy (from its video), Courtship dev movies (wild 9-11, hyper 4-5),
for both arms, fly 0.8's brain, and the tracker-point hard negatives (the shortcut check: if its false calls rise on eye points, it
was a shortcut). Keep the arm with fewer false calls unless Aggression macro F1 or Boy meets boy lunge fall by more than seed spread.

--arm swap: fly 0.9's recipe (neg) plus each training movie shown with flies a and b exchanged, which raised the courtship model from 0.641 to 0.689. Which fly is "a" is arbitrary in fights too.
Declared before running: kept only if mean Aggression macro F1 and Boy meets boy lunge both beat the neg arm (0.587, 0.342) and false calls do not rise by more than the seed spread.

Run from the repo root:  python scripts/train_fly_eyes2.py --arm plain|neg|swap --fold 3      then --report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from etholive import model as etholive                                       # noqa: E402
from etholive import flyvsfly as fv                                 # noqa: E402
from etholive import scope_model as sm                              # noqa: E402
from etholive import scopescore_fly as sf                           # noqa: E402
from etholive import species as sp                                  # noqa: E402
from etholive.classify import feature_columns                      # noqa: E402
import train_fly_lookahead as tl                                        # noqa: E402
import train_fly_negatives as tn                                        # noqa: E402
import train_fly_v2 as v2                                               # noqa: E402

EYES = ROOT / "data/processed_fly/eyes2"
OUT = ROOT / "outputs/etholive/fly_eyes2"
LOOK = 6
PICTURE = (640, 480)


def features(name):
    path = EYES / f"{name}_features.parquet"
    if path.is_file():
        return pd.read_parquet(path)
    poses = pd.read_parquet(EYES / f"{name}.parquet")
    setup = fv.calibrate_live(poses, PICTURE) if name.startswith("bmb") else None
    f = sf.session_features(poses, v2.PARAMS, setup=setup)
    f = f.astype({c: "float32" for c in f.columns if c not in ("session", "frame", "time_s", "tracking_ok")})
    f.to_parquet(path, index=False)
    print(f"  features {name}" + (f": scale {setup['scale']:.3f}" if setup else ""), flush=True)
    return f


def fold_data(fold, neg):
    agg_held, bmb_held = tl.FOLDS[fold]
    agg_f, agg_l = v2.agg_tables()
    _, bmb_l = tl.bmb30_tables()
    train = [m for m in tl.AGG if m != agg_held]
    bmb_train = [m for m in tl.BMBS if m != bmb_held]
    feats = [features(f"agg{m}") for m in tl.AGG] + [features(f"bmb{m}") for m in tl.BMBS]
    labels = [agg_l, bmb_l]
    extra = []
    if neg:
        feats += [features(f"{g}{m}") for g, m in tn.TRAIN]
        labels.append(pd.concat([neg_labels(g, m) for g, m in tn.TRAIN], ignore_index=True))
        extra = [tn.sid(g, m) for g, m in tn.TRAIN]
    manifest = {"train": [fv.session_id(m) for m in train] + [fv.session_id(m, "boymeetsboy") for m in bmb_train] + extra,
                "validation": [fv.session_id(agg_held), fv.session_id(bmb_held, "boymeetsboy")],
                "test": [fv.session_id(m) for m in fv.SPLIT["test"]] + [fv.session_id(m, "boymeetsboy") for m in fv.BMB_SPLIT["test"]]}
    return (pd.concat(feats, ignore_index=True), pd.concat(labels, ignore_index=True), manifest, agg_f, agg_l,
            [fv.session_id(m) for m in train])


def neg_labels(group, movie):
    f = features(f"{group}{movie}")
    mark = tn.true_marks(group, movie, int(f["frame"].max()) + 1)
    return pd.DataFrame({"session": tn.sid(group, movie), "frame": f["frame"].to_numpy(), "time_s": f["time_s"].to_numpy(),
                         "label": np.where(mark[f["frame"].to_numpy()] == "unknown", fv.IGNORE, "other")})


def exchange(feats, labels, sessions):
    """The training movies again, with every per-fly column of a and b exchanged."""
    f = feats[feats["session"].isin(sessions)]
    names = {c: c[:-1] + ("b" if c.endswith("_a") else "a") for c in f.columns if c.endswith(("_a", "_b"))}
    f = f.rename(columns=names)[list(feats.columns)].assign(session=lambda d: d["session"] + "/swap")
    lab = labels[labels["session"].isin(sessions)].assign(session=lambda d: d["session"] + "/swap")
    return pd.concat([feats, f], ignore_index=True), pd.concat([labels, lab], ignore_index=True)


def train(arm, fold):
    out = OUT / f"{arm}_fold{fold}"
    out.mkdir(parents=True, exist_ok=True)
    feats, labels, manifest, agg_f, agg_l, agg_train = fold_data(fold, arm in ("neg", "swap"))
    cols = feature_columns(feats)
    if arm == "swap":
        feats, labels = exchange(feats, labels, manifest["train"])
        manifest = {**manifest, "train": manifest["train"] + [s + "/swap" for s in manifest["train"]]}
    labels = tl.moved_labels(labels, LOOK)
    start = sm.linear_start_from(tl.frame_start(fold, agg_f, agg_l, agg_train, cols, out), cols, classes=sp.FLY.classes)
    for seed in tl.SEEDS:
        path = out / f"behavior_seed{seed}.pt"
        if path.is_file():
            continue
        print(f"== {arm} fold {fold} seed {seed}", flush=True)
        net, norm, info = etholive.train(feats, labels, manifest, start, species=sp.FLY, seed=seed, metrics=v2.metrics,
                                         partial={v2.REST: ("other", "tussle", "hold")}, balance="sqrt", log=lambda *_: None)
        info["features"] = f"train_fly_eyes2.py --arm {arm} --fold {fold}"
        info["lookahead_frames"] = LOOK
        etholive.save_behavior(path, net, norm, info)


def scores(team, norm, info, f, session):
    return sm.score_sessions(team, sm.session_arrays(f, None, [session], info["feature_columns"], norm), species=sp.FLY)


def false_calls(team, norm, info):
    rows = []
    for g, m in tn.DEV:
        s = scores(team, norm, info, features(f"{g}{m}"), tn.sid(g, m)).sort_values("frame")
        mark = tn.true_marks(g, m, int(s["frame"].max()) + 1 + LOOK)
        p = s[list(fv.BEHAVIORS)].to_numpy(float)
        rows.append(pd.DataFrame({"mark": mark[np.maximum(s["frame"].to_numpy() - LOOK, 0)],
                                  "call": np.asarray(fv.BEHAVIORS + ("other",))[np.c_[p, 1 - p.sum(1)].argmax(1)]}))
    d = pd.concat(rows, ignore_index=True)
    free = d[d["mark"] != "copulation"]
    hours = len(free) / fv.FPS / 3600
    out = {f"{b}_per_h": int((np.diff(np.r_[0, (free["call"] == b).astype(int), 0]) == 1).sum()) / hours for b in fv.BEHAVIORS}
    for m in ("none", "copulation", "wing_extension", "circle"):
        g = d[d["mark"] == m]
        out[f"{m}_called_fight"] = float((g["call"] != "other").mean()) if len(g) else np.nan
    return out


def report():
    _, bmb_l = tl.bmb30_tables()
    bmb_l = bmb_l.assign(label=bmb_l["label"].replace({v2.REST: "other"}))
    _, agg_l = v2.agg_tables()
    arms = {"fly 0.8 brain": ROOT / "outputs/etholive/fly_look30/k6_fold{}", "tracker hard negatives": ROOT / "outputs/etholive/fly_neg/fold{}",
            "eyes2 plain": OUT / "plain_fold{}", "eyes2 neg": OUT / "neg_fold{}", "eyes2 swap": OUT / "swap_fold{}"}
    rows = []
    for name, base in arms.items():
        for fold, (am, bm) in tl.FOLDS.items():
            folder = Path(str(base).format(fold))
            if not (folder / "behavior_seed2.pt").is_file():
                continue
            team, norm, info = tn.team_from(folder)
            a_id, b_id = fv.session_id(am), fv.session_id(bm, "boymeetsboy")
            a = fv.metrics(scores(team, norm, info, features(f"agg{am}"), a_id), tl.moved_labels(agg_l[agg_l["session"] == a_id], LOOK))
            b = fv.metrics(scores(team, norm, info, features(f"bmb{bm}"), b_id), tl.moved_labels(bmb_l[bmb_l["session"] == b_id], LOOK))
            rows.append({"model": name, "fold": fold, "agg_macro": a["macro_f1"], "agg_lunge": a["lunge_f1"], "agg_wing": a["wing_threat_f1"],
                         "agg_tussle": a["tussle_f1"], "agg_hold": a["hold_f1"], "bmb_lunge": b["lunge_f1"], "bmb_wing": b["wing_threat_f1"],
                         "bmb_bouts_found": b["lunge_bout_recall"], "bmb_bouts_right": b["lunge_bout_precision"], **false_calls(team, norm, info)})
            print(f"  {name} fold {fold}", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "report.csv", index=False)
    print(t.groupby("model", sort=False).mean(numeric_only=True).drop(columns="fold").T.round(3).to_string())
    print(t[["model", "fold", "agg_macro", "bmb_lunge", "lunge_per_h"]].round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("plain", "neg", "swap"))
    ap.add_argument("--fold", type=int, choices=sorted(tl.FOLDS))
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    report() if a.report else train(a.arm, a.fold)


if __name__ == "__main__":
    main()
