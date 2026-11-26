"""The cost of being live: one model, the same 63-frame (2.1 s) window, moved from all-past to centred. VALIDATION only, never test.

For the paper (2026-11-19). Declared before running:
  arms     K = 0 (live: 62 past frames), 3, 6, 10, 15, 31 (centred: 31 past, 31 future = offline with the same context and weights).
           The label of frame t is predicted at frame t + K. Unlike the earlier lookahead scripts, the frame branch reads frame t itself
           (ShiftNet), so only the memory branch's window moves.
  data     mouse: CalMS21 Task 1, 56 train / 14 validation sessions (mouse 0.4 recipe).
           fights: Fly vs. Fly Aggression + Boy meets boy at 30 fps + Courtship hard negatives + a/b exchange (fly 0.10 recipe),
                   eye points, folds (Aggression, Boy meets boy held out) = (3, 3), (2, 1), (5, 2); scored on the held-out Aggression movie.
           courtship: Courtship train half, eye points, a/b exchange (courtship 0.2 recipe), folds A and B.
  seeds    0, 1, 2; the team (mean probability) is scored. Each net keeps its best validation epoch, the same rule for every arm.
  measure  per behavior: team F1 at each K, mean over folds; gain(K) = F1(K) - F1(0).
           duration: median bout length (s) of each behavior in the training labels.
  question does the gain from future frames shrink as behaviors get longer? Spearman correlation across the 11 behaviors between
           median duration and gain at K = 6 and at K = 31, with an exact permutation p-value. 11 points: a hint, not a law.

Run from the repo root:  python scripts/lookahead_sweep.py --data mouse|fights|courtship [--fold F]     then --report
"""

from __future__ import annotations

import argparse
import itertools
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
from etholive import flyvsfly as fv                                 # noqa: E402
from etholive import scope_model as sm                              # noqa: E402
from etholive import species as sp                                  # noqa: E402

LOOKS = (0, 3, 6, 10, 15, 31)
SEEDS = (0, 1, 2)
OUT = ROOT / "outputs/paper/lookahead"
MOUSE_B = ("attack", "investigation", "mount")


class ShiftNet(etholive.BehaviorNet):
    """BehaviorNet whose frame branch reads frame t - shift: with labels moved `shift` frames later, it sees the labelled frame."""
    shift = 0

    def forward(self, x):
        f = self.frame(x)
        k = self.shift
        if k:
            k = min(k, f.shape[1])
            f = torch.cat([f[:, :1].expand(-1, k, -1), f[:, :f.shape[1] - k]], dim=1)
        return f + self.memory(x)


def shift_class(k):
    return type(f"ShiftNet{k}", (ShiftNet,), {"shift": k})


def train_arm(folder, k, feats, labels, manifest, start, **kw):
    """Three seeds at lookahead k; returns the team. Saved nets are reused."""
    folder.mkdir(parents=True, exist_ok=True)
    cls = shift_class(k)
    nets, norm, info = [], None, None
    for seed in SEEDS:
        path = folder / f"k{k}_seed{seed}.pt"
        if not path.is_file():
            print(f"== {folder.name} K {k} seed {seed}", flush=True)
            plain = etholive.BehaviorNet
            etholive.BehaviorNet = cls
            try:
                net, norm, info = etholive.train(feats, labels, manifest, start, seed=seed, log=lambda *_: None, **kw)
            finally:
                etholive.BehaviorNet = plain
            info["lookahead_frames"] = k
            info["frame_branch_shift"] = k
            etholive.save_behavior(path, net, norm, info)
        blob = torch.load(path, map_location="cpu", weights_only=False)
        info, p = blob["info"], blob["info"]["params"]
        net = cls(info["n_features"], len(info["classes"]), p["width"], p["dropout"], p.get("dilations", sm.DILATIONS))
        net.load_state_dict(blob["state_dict"])
        nets.append(net.eval())
        norm = {a: v.numpy() for a, v in blob["norm"].items()}
    return etholive.Team(nets).eval(), norm, info


def save_row(row):
    path = OUT / "results.csv"
    old = pd.read_csv(path) if path.is_file() else pd.DataFrame()
    if len(old):
        old = old[~((old["data"] == row["data"]) & (old["fold"].astype(str) == str(row["fold"])) & (old["look"] == row["look"]))]
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)
    print({a: (round(b, 3) if isinstance(b, float) else b) for a, b in row.items()}, flush=True)


# --- mouse --------------------------------------------------------------------------------------

def run_mouse():
    import mouse_lookahead as ml
    from etholive.calms21 import check_manifest
    from etholive.evaluate import summary_metrics
    manifest = check_manifest(json.loads((ROOT / "splits/split_manifest.json").read_text()))
    keep = set(manifest["train"]) | set(manifest["validation"])
    base = pd.read_parquet(ROOT / "data/processed_scope/geometry_swap_smooth.parquet")
    labels0 = pd.read_parquet(ROOT / "data/processed/labels.parquet")
    base, labels0 = base[base["session"].isin(keep)], labels0[labels0["session"].isin(keep)]
    part = sm.model_b_part(ROOT / "models/model_b.joblib", ROOT / "models/model_b.json")
    for k in LOOKS:
        feats, labels = ml.moved(base, labels0, k)
        team, norm, info = train_arm(OUT / "mouse", k, feats, labels, manifest, part, width=64, dilations=(1, 2, 4, 8, 16))
        va = sm.session_arrays(feats, labels, manifest["validation"], info["feature_columns"], norm)
        m = summary_metrics(sm.score_sessions(team, va), labels)
        save_row({"data": "mouse", "fold": "val", "look": k, **{f"{b}_f1": m[f"{b}_f1"] for b in MOUSE_B}})


# --- fly fights ---------------------------------------------------------------------------------

def run_fights(fold):
    import train_fly_eyes2 as te
    import train_fly_lookahead as tl
    import train_fly_v2 as v2
    from etholive.classify import feature_columns
    feats, labels, manifest, agg_f, agg_l, agg_train = te.fold_data(fold, True)
    cols = feature_columns(feats)
    feats, labels = te.exchange(feats, labels, manifest["train"])
    manifest = {**manifest, "train": manifest["train"] + [s + "/swap" for s in manifest["train"]]}
    folder = OUT / f"fights_fold{fold}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "start").mkdir(exist_ok=True)
    start = sm.linear_start_from(tl.frame_start(fold, agg_f, agg_l, agg_train, cols, folder / "start"), cols, classes=sp.FLY.classes)
    am, bm = tl.FOLDS[fold]
    a_id, b_id = fv.session_id(am), fv.session_id(bm, "boymeetsboy")
    _, bmb_l = tl.bmb30_tables()
    bmb_l = bmb_l.assign(label=bmb_l["label"].replace({v2.REST: "other"}))
    for k in LOOKS:
        team, norm, info = train_arm(folder, k, feats, tl.moved_labels(labels, k), manifest, start, species=sp.FLY,
                                     metrics=v2.metrics, partial={v2.REST: ("other", "tussle", "hold")}, balance="sqrt")
        a = fv.metrics(te.scores(team, norm, info, te.features(f"agg{am}"), a_id), tl.moved_labels(agg_l[agg_l["session"] == a_id], k))
        b = fv.metrics(te.scores(team, norm, info, te.features(f"bmb{bm}"), b_id), tl.moved_labels(bmb_l[bmb_l["session"] == b_id], k))
        save_row({"data": "fights", "fold": fold, "look": k, **{f"{x}_f1": a[f"{x}_f1"] for x in fv.BEHAVIORS},
                  "bmb_lunge_f1": b["lunge_f1"], "bmb_wing_threat_f1": b["wing_threat_f1"]})


# --- fly courtship ------------------------------------------------------------------------------

def run_courtship(fold):
    import train_fly_courtship as tc
    import train_fly_lookahead as tl
    from etholive.classify import feature_columns
    feats, labels = tc.tables()
    man = tc.manifest(fold)
    cols = feature_columns(feats)
    feats, labels = tc.swapped(feats, labels, man["train"])
    man = {**man, "train": man["train"] + [s + "/swap" for s in man["train"]]}
    start = sm.linear_start_from(tc.start_from(feats, labels, man["train"], cols), cols, classes=tc.PACK.classes)
    for k in LOOKS:
        team, norm, info = train_arm(OUT / f"courtship_fold{fold}", k, feats, tl.moved_labels(labels, k), man, start,
                                     species=tc.PACK, metrics=tc.metrics, balance="sqrt")
        s = sm.score_sessions(team, sm.session_arrays(feats, None, man["validation"], info["feature_columns"], norm), species=tc.PACK)
        m = tc.metrics(s, tl.moved_labels(labels[labels["session"].isin(man["validation"])], k))
        save_row({"data": "courtship", "fold": fold, "look": k, **{f"{b}_f1": m[f"{b}_f1"] for b in tc.PACK.behaviors}})


# --- report -------------------------------------------------------------------------------------

def bout_seconds(labels, names, fps=30.0):
    """Median bout length in seconds of each behavior, over the given label table (one label per frame)."""
    out = {}
    for b in names:
        lens = []
        for _, g in labels.sort_values(["session", "frame"]).groupby("session"):
            d = np.diff(np.r_[0, (g["label"].to_numpy() == b).astype(int), 0])
            lens += list(np.where(d == -1)[0] - np.where(d == 1)[0])
        out[b] = float(np.median(lens)) / fps if lens else np.nan
    return out


def durations():
    import train_fly_courtship as tc
    import train_fly_v2 as v2
    import mouse_lookahead  # noqa: F401
    from etholive.calms21 import check_manifest
    manifest = check_manifest(json.loads((ROOT / "splits/split_manifest.json").read_text()))
    ml = pd.read_parquet(ROOT / "data/processed/labels.parquet")
    d = bout_seconds(ml[ml["session"].isin(manifest["train"])], MOUSE_B)
    _, agg_l = v2.agg_tables()
    d.update(bout_seconds(agg_l[agg_l["session"].isin([fv.session_id(m) for m in (1, 2, 4, 5)])], fv.BEHAVIORS))
    _, cl = tc.tables()
    d.update(bout_seconds(cl, tc.PACK.behaviors))
    return d


def spearman_exact(x, y):
    """Spearman rho and a two-sided permutation p-value (exact for n <= 11, else 200,000 random permutations).

    Before 2026-11-26 the cut was n <= 9, so the recorded stats.json values for the 11 behaviors are Monte Carlo
    estimates (0.0146, 0.0872); the exact values are 0.0144 and 0.0876 (results/paper/lookahead/stats_exact.json)."""
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    rho = float(np.corrcoef(rx, ry)[0, 1])
    n = len(rx)
    if n <= 11:                       # exact: every permutation (11! = 39.9 M, a few seconds vectorized)
        rxc, ryc = rx - rx.mean(), ry - ry.mean()
        denom = np.sqrt((rxc ** 2).sum() * (ryc ** 2).sum())
        hits = total = 0
        it = itertools.permutations(range(n))
        while True:
            chunk = np.fromiter(itertools.chain.from_iterable(itertools.islice(it, 2_000_000)), dtype=np.int8)
            if chunk.size == 0:
                break
            r = (ryc[chunk.reshape(-1, n)] * rxc).sum(axis=1) / denom
            hits += int((np.abs(r) >= abs(rho) - 1e-12).sum())
            total += len(r)
        return rho, hits / total
    else:
        rng = np.random.default_rng(0)
        perms = [np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(200_000)]
    return rho, float(np.mean(np.abs(perms) >= abs(rho) - 1e-12))


def report():
    r = pd.read_csv(OUT / "results.csv")
    names = {"mouse": MOUSE_B, "fights": fv.BEHAVIORS, "courtship": ("wing_extension", "circle", "copul_attempt", "copulation")}
    rows = []
    for data, bs in names.items():
        g = r[r["data"] == data].groupby("look").mean(numeric_only=True)
        if not len(g):
            continue
        for b in bs:
            col = f"{b}_f1"
            if col in g:
                rows.append({"data": data, "behavior": b, **{f"K{k}": g.loc[k, col] for k in LOOKS if k in g.index}})
    t = pd.DataFrame(rows)
    dur = durations()
    t["median_bout_s"] = t["behavior"].map(dur)
    for k in LOOKS[1:]:
        if f"K{k}" in t:
            t[f"gain_K{k}"] = t[f"K{k}"] - t["K0"]
    t.to_csv(OUT / "summary.csv", index=False)
    print(t.round(3).to_string(index=False))
    stats = {}
    for k in (6, 31):
        c = f"gain_K{k}"
        if c in t and t[c].notna().sum() >= 4:
            ok = t[[c, "median_bout_s"]].dropna()
            rho, p = spearman_exact(ok["median_bout_s"], ok[c])
            stats[c] = {"n": len(ok), "spearman_rho": rho, "p_permutation": p}
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=("mouse", "fights", "courtship"))
    ap.add_argument("--fold")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.report:
        return report()
    if a.data == "mouse":
        run_mouse()
    elif a.data == "fights":
        run_fights(int(a.fold))
    else:
        run_courtship(a.fold)


if __name__ == "__main__":
    main()
