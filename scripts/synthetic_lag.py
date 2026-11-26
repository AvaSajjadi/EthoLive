"""Is the duration effect a property of the behavior or of our network? The Bayes-optimal answer on simulated behavior.

For the paper (2026-11-19). Declared before running:
  source   one behavior that switches on and off (alternating renewal process), 30 fps, 200,000 frames per setting.
           Bout lengths are phase-type so the exact posterior is cheap:
             geometric   (1 phase, memoryless, the shape behind classic HMM smoothing results)
             erlang4     (4 phases, regular bouts)
             mixture     (geometric mixture, 80% short + 20% 5x longer: heavy right tail)
           all three matched on MEDIAN bout length m; gaps are geometric with mean 4x the mean bout (behavior ~20% of frames).
  frames   each frame gives a noisy reading y_t ~ N(d * on_t, 1), d = 1 (one standard deviation between on and off).
  decoder  exact fixed-lag posterior P(on_t | y_1..y_{t+K}) (forward filter + K-step backward), K = 0, 3, 6, 10, 15, 31.
           F1 at the best threshold for each K (the network also picks its epoch for F1).
  question does the gain F1(K) - F1(0) of the BEST POSSIBLE decoder shrink as bouts get longer, as our network's did,
           and does a short wait capture most of the centred-window gain? Does the bout shape matter at the same median?
  medians  2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64 frames (0.07 to 2.1 s).

Run:  python scripts/synthetic_lag.py   -> outputs/paper/synthetic_lag/results.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/paper/synthetic_lag"
LOOKS = (0, 3, 6, 10, 15, 31)
MEDIANS = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
SHAPES = ("geometric", "erlang4", "mixture")
T = 200_000
D = 1.0


def geometric_p(median):
    """Exit probability per frame so that a geometric bout (>= 1 frame) has this median."""
    return 1 - 0.5 ** (1 / median)


def on_phases(shape, median):
    """Return (start probabilities, within-bout transition matrix, exit probabilities) for the on bout."""
    if shape == "geometric":
        p = geometric_p(median)
        return np.array([1.0]), np.array([[1 - p]]), np.array([p])
    if shape == "erlang4":
        # sum of 4 geometric stages; find the stage exit probability that gives this median by bisection
        lo, hi = 1e-4, 1.0
        for _ in range(60):
            p = (lo + hi) / 2
            med = erlang_median(p)
            lo, hi = (p, hi) if med > median else (lo, p)
        p = (lo + hi) / 2
        a = np.zeros((4, 4))
        for i in range(4):
            a[i, i] = 1 - p
            if i < 3:
                a[i, i + 1] = p
        exit_ = np.array([0, 0, 0, p])
        return np.array([1.0, 0, 0, 0]), a, exit_
    if shape == "mixture":
        lo, hi = 0.2, 3.0
        for _ in range(60):
            s = (lo + hi) / 2
            med = mixture_median(median * s)
            lo, hi = (lo, s) if med > median else (s, hi)
        short = median * (lo + hi) / 2
        ps, pl = geometric_p(short), geometric_p(5 * short)
        return np.array([0.8, 0.2]), np.diag([1 - ps, 1 - pl]), np.array([ps, pl])
    raise ValueError(shape)


def erlang_median(p, n=200_000):
    rng = np.random.default_rng(1)
    return np.median(rng.geometric(p, size=(n, 4)).sum(1))


def mixture_median(short, n=200_000):
    rng = np.random.default_rng(1)
    long_ = rng.random(n) < 0.2
    return np.median(np.where(long_, rng.geometric(geometric_p(5 * short), n), rng.geometric(geometric_p(short), n)))


def chain(shape, median):
    """Full chain: state 0 = off (geometric gap), states 1.. = on phases. Returns (start, A, label per state)."""
    s0, a_on, ex = on_phases(shape, median)
    r = len(s0)
    mean_on = mean_duration(s0, a_on)
    q = 1 / (4 * mean_on)                                  # gap mean = 4x bout mean
    a = np.zeros((r + 1, r + 1))
    a[0, 0] = 1 - q
    a[0, 1:] = q * s0
    a[1:, 1:] = a_on
    a[1:, 0] = ex
    start = np.full(r + 1, 1 / (r + 1))
    return start, a, np.r_[0, np.ones(r)]


def mean_duration(s0, a_on):
    return float(s0 @ np.linalg.solve(np.eye(len(s0)) - a_on, np.ones(len(s0))))


def simulate(start, a, rng):
    cum = np.cumsum(a, 1)
    x = np.empty(T, dtype=np.int64)
    x[0] = rng.choice(len(start), p=start)
    u = rng.random(T)
    for t in range(1, T):
        x[t] = min(np.searchsorted(cum[x[t - 1]], u[t]), len(start) - 1)
    return x


def fixed_lag(y, start, a, label):
    """P(on_t | y_1..t+K) for every K in LOOKS."""
    e = np.exp(-0.5 * (y[:, None] - D * label[None, :]) ** 2)
    alpha = np.empty((T, len(start)))
    f = start * e[0]
    alpha[0] = f / f.sum()
    for t in range(1, T):
        f = (alpha[t - 1] @ a) * e[t]
        alpha[t] = f / f.sum()
    out = {}
    b = np.ones_like(alpha)
    for k in range(max(LOOKS) + 1):
        if k:
            nb = np.ones_like(b)
            nb[:-1] = (e[1:] * b[1:]) @ a.T
            b = nb / nb.sum(1, keepdims=True)
        if k in LOOKS:
            p = alpha * b
            p /= p.sum(1, keepdims=True)
            out[k] = p @ label
    return out


def best_f1(truth, prob):
    best = 0.0
    for th in np.linspace(0.05, 0.95, 37):
        pred = prob >= th
        tp = np.sum(pred & truth)
        if tp:
            best = max(best, 2 * tp / (pred.sum() + truth.sum()))
    return best


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for shape in SHAPES:
        for m in MEDIANS:
            rng = np.random.default_rng(m * 7 + SHAPES.index(shape))
            start, a, label = chain(shape, m)
            x = simulate(start, a, rng)
            truth = label[x].astype(bool)
            y = D * truth + rng.standard_normal(T)
            post = fixed_lag(y, start, a, label)
            row = {"shape": shape, "median_frames": m, "median_s": m / 30,
                   "on_share": round(float(truth.mean()), 3)}
            for k in LOOKS:
                row[f"K{k}"] = round(best_f1(truth, post[k]), 4)
            rows.append(row)
            print(row, flush=True)
    d = pd.DataFrame(rows)
    for k in LOOKS[1:]:
        d[f"gain_K{k}"] = (d[f"K{k}"] - d["K0"]).round(4)
    d["share_of_centred_gain_at_K6"] = (d["gain_K6"] / d["gain_K31"]).round(3)
    d.to_csv(OUT / "results.csv", index=False)
    print(d[["shape", "median_frames", "K0", "K6", "K31", "gain_K6", "gain_K31", "share_of_centred_gain_at_K6"]].to_string())


if __name__ == "__main__":
    main()
