"""Robustness of the duration trend (manuscript Table 2), from the recorded lookahead summary. No training.

Spearman rho between median bout length and the gain at K = 6 after removing behaviors or datasets:
  - every subset of 1, 2 and 3 behaviors removed (the worst rho of each size, and how many subsets are weaker than -0.5);
  - each dataset removed (rho and exact permutation p);
  - within each dataset (rho only: 3 or 4 behaviors cannot support a test).

    python scripts/robustness_duration.py [--summary results/paper/lookahead/summary.csv] [--gain gain_K6]
"""
import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def rho_of(x, y):
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def p_exact(x, y):
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    rxc, ryc = rx - rx.mean(), ry - ry.mean()
    denom = np.sqrt((rxc ** 2).sum() * (ryc ** 2).sum())
    rho = float((rxc * ryc).sum() / denom)
    perms = np.array(list(itertools.permutations(range(len(rx)))), dtype=np.int8)
    r = (ryc[perms] * rxc).sum(axis=1) / denom
    return float(np.mean(np.abs(r) >= abs(rho) - 1e-12))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--summary", default="results/paper/lookahead/summary.csv")
    ap.add_argument("--gain", default="gain_K6")
    a = ap.parse_args(argv)
    s = pd.read_csv(ROOT / a.summary)
    x, y = s["median_bout_s"].to_numpy(), s[a.gain].to_numpy()
    n = len(s)
    out = {"all": {"n": n, "rho": rho_of(x, y)}}
    for k in (1, 2, 3):
        rhos = [rho_of(np.delete(x, list(c)), np.delete(y, list(c))) for c in itertools.combinations(range(n), k)]
        worst = int(np.argmax(rhos))
        out[f"drop_{k}"] = {"subsets": math.comb(n, k), "worst_rho": max(rhos), "weaker_than_-0.5": int(np.sum(np.array(rhos) > -0.5)),
                            "worst_removed": list(s.iloc[list(list(itertools.combinations(range(n), k))[worst])]["behavior"])}
    for d in s["data"].unique():
        keep = s["data"] != d
        out[f"without_{d}"] = {"n": int(keep.sum()), "rho": rho_of(x[keep], y[keep]), "p_exact": p_exact(x[keep], y[keep])}
        inside = s["data"] == d
        out[f"within_{d}"] = {"n": int(inside.sum()), "rho": rho_of(x[inside], y[inside])}
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
