"""Figures for the paper from saved results (no training). Run: python scripts/paper_figures.py -> paper/figures/*.pdf"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper/figures"
real = pd.read_csv(ROOT / "results/paper/lookahead/summary.csv")
syn = pd.read_csv(ROOT / "results/paper/synthetic_lag/results.csv")
LOOKS = [0, 3, 6, 10, 15, 31]
COL = {"mouse": "#1f77b4", "fights": "#d62728", "courtship": "#2ca02c"}
NAME = {"mouse": "Mouse (CalMS21)", "fights": "Fly aggression", "courtship": "Fly courtship"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})

fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.2), sharey=True)
for ax, ds in zip(axes, ["mouse", "fights", "courtship"]):
    for _, r in real[real.data == ds].iterrows():
        ax.plot([k / 30 for k in LOOKS], [r[f"K{k}"] for k in LOOKS], marker="o", ms=3, label=r.behavior.replace("_", " "))
    ax.set_title(NAME[ds]); ax.set_xlabel("wait before labelling (s)")
    ax.legend(frameon=False, fontsize=6)
axes[0].set_ylabel("F1 (validation, 3-seed team)")
fig.tight_layout(); fig.savefig(FIG / "f1_vs_wait.pdf"); fig.savefig(FIG / "f1_vs_wait.png", dpi=150)

fig, ax = plt.subplots(figsize=(3.4, 2.5))
for shape, ls in [("geometric", "-"), ("erlang4", "--"), ("mixture", ":")]:
    s = syn[syn["shape"] == shape]
    ax.plot(s.median_s, s.gain_K6, ls, color="gray", lw=1, label=f"Bayes-optimal, {shape}")
for ds in COL:
    r = real[(real.data == ds) & (real.median_bout_s < 60)]
    ax.scatter(r.median_bout_s, r.gain_K6, color=COL[ds], s=14, label=NAME[ds], zorder=3)
    for _, row in r.iterrows():
        ax.annotate(row.behavior.replace("_", " "), (row.median_bout_s, row.gain_K6), fontsize=5, xytext=(2, 2), textcoords="offset points")
ax.set_xscale("log"); ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("median bout length (s)"); ax.set_ylabel("F1 gain from a 0.2 s wait")
ax.legend(frameon=False, fontsize=5.5, loc="upper right")
fig.tight_layout(); fig.savefig(FIG / "gain_vs_bout.pdf"); fig.savefig(FIG / "gain_vs_bout.png", dpi=150)
print("saved", sorted(p.name for p in FIG.glob("*.pdf")))
