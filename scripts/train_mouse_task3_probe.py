"""EthoLive mouse, CalMS21 Task 3 again: build on mouse 0.4 instead of training from scratch (scripts/train_mouse_task3.py scored test mean F1 0.228).

Each frame is passed through mouse 0.4's three trained memory branches; their hidden layer (3 x 64 numbers per frame, which already describe mouse movement learned from
many sessions) feeds one logistic regression per behavior. Only that small classifier learns from Task 3. Same time split as before: first 75% of each training video to fit,
last 25% to choose the regularisation C.
Declared before running: the test file is read only if the validation mean F1 beats 0.438 (the scratch models), and that read is the SECOND look at the Task 3 test set.

Run from the repo root:  python scripts/train_mouse_task3_probe.py    then  --test
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from etholive import model as etholive                                        # noqa: E402
from etholive import scope_model as sm                               # noqa: E402
import train_mouse_task3 as t3                                           # noqa: E402

OUT = ROOT / "outputs/etholive/mouse_task3_probe"
CS = (0.001, 0.01, 0.1, 1.0)


@torch.no_grad()
def hidden(feats, sessions):
    """{session: (frames table, (T, 192) array)} from mouse 0.4's memory branches."""
    b = etholive.load_behavior(ROOT / "models/etholive/mouse.pt")
    arrays = sm.session_arrays(feats, None, sessions, b.feature_columns, b.norm)
    out = {}
    for sid, (x, _, table) in arrays.items():
        hs = []
        for net in b.head.nets:
            m = net.memory
            h = m.inp(torch.from_numpy(x)[None]).transpose(1, 2)
            for blk in m.blocks:
                h = blk(h)
            hs.append(h[0].T.numpy())
        out[sid] = (table.reset_index(drop=True), np.hstack(hs))
    return out


def stack(h, labels, b, sessions):
    xs, ys, keys = [], [], []
    lab = labels.set_index(["session", "frame"])["label"]
    for s in sessions:
        table, x = h[s]
        y = lab.reindex(pd.MultiIndex.from_arrays([table["session"], table["frame"]])).to_numpy()
        xs.append(x)
        ys.append(y == b)
        keys.append(table[["session", "frame"]])
    return np.vstack(xs), np.concatenate(ys), pd.concat(keys, ignore_index=True)


def f1(t, q):
    return 2 * int((t & q).sum()) / max(1, int(t.sum() + q.sum()))


def fit(b):
    feats, labels = t3.split_in_time(*t3.tables("train", b))
    sessions = sorted(feats["session"].unique())
    h = hidden(feats, sessions)
    xtr, ytr, _ = stack(h, labels, b, [s for s in sessions if s.endswith("/early")])
    xva, yva, _ = stack(h, labels, b, [s for s in sessions if s.endswith("/late")])
    scale = StandardScaler().fit(xtr)
    best = None
    for c in CS:
        m = LogisticRegression(C=c, class_weight="balanced", max_iter=3000).fit(scale.transform(xtr), ytr)
        score = f1(yva, m.predict(scale.transform(xva)).astype(bool))
        if best is None or score > best[0]:
            best = (score, c, m)
    return {"behavior": b, "f1": best[0], "C": best[1], "frames": int(yva.sum())}, (scale, best[2])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    test_now = "--test" in sys.argv
    rows, models = [], {}
    for b in t3.BEHAVIORS:
        row, models[b] = fit(b)
        rows.append(row)
        print(row, flush=True)
    v = pd.DataFrame(rows)
    v.to_csv(OUT / "validation.csv", index=False)
    print(v.round(3).to_string(index=False), "\nmean validation F1", round(v["f1"].mean(), 3), flush=True)
    if not test_now:
        return
    if v["f1"].mean() <= 0.438:
        raise SystemExit("validation does not beat the scratch models: the test set is not read")
    if (OUT / "test.csv").exists():
        raise SystemExit("already run; not run again")
    out = []
    for b in t3.BEHAVIORS:
        feats, labels = t3.tables("test", b)
        sessions = sorted(feats["session"].unique())
        x, y, _ = stack(hidden(feats, sessions), labels, b, sessions)
        scale, m = models[b]
        q = m.predict(scale.transform(x)).astype(bool)
        out.append({"behavior": b, "videos": len(sessions), "f1": f1(y, q), "precision": int((y & q).sum()) / max(1, q.sum()), "recall": int((y & q).sum()) / max(1, y.sum())})
    t = pd.DataFrame(out)
    t.to_csv(OUT / "test.csv", index=False)
    print(t.round(3).to_string(index=False), "\nmean test F1 (second look at this test set)", round(t["f1"].mean(), 3))


if __name__ == "__main__":
    main()
