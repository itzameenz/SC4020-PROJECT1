"""
make_report_figures.py -- Booking.com figures for the written report.

Reads only saved artefacts (results.csv, results/tuning/*.json, *-per-seed.csv)
plus data/processed/processed.pkl, so it needs numpy / pandas / matplotlib and
no GPU. Writes PNGs to results/figures/.

Usage:  python scripts/make_report_figures.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")                            # write files only, no GUI window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.evaluate import load_results  # noqa: E402

TUNE = os.path.join(ROOT, "results", "tuning")
OUT = os.path.join(ROOT, "results", "figures")
os.makedirs(OUT, exist_ok=True)

# colour follows the model family (fixed categorical order)
FAMILY_COLOR = {"Markov": "#2a78d6", "FPMC": "#eb6834", "GRU4Rec": "#1baf7a", "SASRec": "#6250d6"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

VARIANTS = [  # (results.csv row, family, tuning json tag)
    ("Markov (no recursion)", "Markov", None), ("Markov", "Markov", None),
    ("FPMC-user", "FPMC", "fpmc-user"), ("FPMC-persona", "FPMC", "fpmc-persona"),
    ("GRU4Rec-no-side", "GRU4Rec", "gru4rec-no-side"), ("GRU4Rec-side", "GRU4Rec", "gru4rec-side"),
    ("SASRec-side-concat", "SASRec", "sasrec-side-concat"), ("SASRec-no-side", "SASRec", "sasrec-no-side"),
    ("SASRec-side-sum", "SASRec", "sasrec-side-sum"),
]

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.dpi": 100, "savefig.dpi": 220, "savefig.bbox": "tight",
})

means, stds = load_results(os.path.join(ROOT, "results.csv"))


def val_score(row, tag):
    if tag is None:
        return json.load(open(os.path.join(TUNE, "markov.json")))[row]["best_val_NDCG@10"]
    return json.load(open(os.path.join(TUNE, f"{tag}.json")))["best_val_NDCG@10"]


# ---------------------------------------------------------------------------
# 1. Ablation dot plot: every Booking.com variant, test NDCG@10 mean +/- std
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.2))
y = np.arange(len(VARIANTS))[::-1]
for yi, (row, fam, _) in zip(y, VARIANTS):
    m, s = means.loc[row, "NDCG@10"], stds.loc[row, "NDCG@10"]
    s = 0.0 if np.isnan(s) else s
    ax.errorbar(m, yi, xerr=s, fmt="o", ms=8, color=FAMILY_COLOR[fam], ecolor=INK2,
                elinewidth=1.5, capsize=4, mec="white", mew=1.5, zorder=3)
    label = f"{m:.4f}" + (f" ± {s:.4f}" if s else "  (deterministic)")
    ax.text(m + 0.0035, yi, label, va="center", fontsize=8.5, color=INK2)
ax.set_yticks(y, [v[0] for v in VARIANTS])
ax.set_xlim(0.34, 0.48)
ax.set_xlabel("Test NDCG@10 (mean ± std over 5 seeds)")
ax.set_title("Booking.com: every ablation variant", loc="left", fontweight="bold", color=INK)
for b in (1.5, 3.5, 5.5):                      # separate the four families
    ax.axhline(y[0] - b, color=GRID, lw=1)
ax.grid(axis="y", visible=False)
fig.savefig(os.path.join(OUT, "fig_booking_ablation.png"))
plt.close(fig)

# ---------------------------------------------------------------------------
# 2. Validation vs test NDCG@10 per variant (dumbbell)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.2))
for yi, (row, fam, tag) in zip(y, VARIANTS):
    v, t = val_score(row, tag), means.loc[row, "NDCG@10"]
    ax.plot([t, v], [yi, yi], color=GRID, lw=3, zorder=1, solid_capstyle="round")
    ax.scatter(v, yi, s=60, facecolor="white", edgecolor=FAMILY_COLOR[fam], lw=2, zorder=3)
    ax.scatter(t, yi, s=60, color=FAMILY_COLOR[fam], edgecolor="white", lw=1.5, zorder=3)
    ax.text(max(v, t) + 0.003, yi, f"gap {v - t:+.4f}", va="center", fontsize=8.5, color=INK2)
ax.set_yticks(y, [v[0] for v in VARIANTS])
ax.set_xlim(0.34, 0.50)
ax.set_xlabel("NDCG@10")
ax.scatter([], [], s=60, facecolor="white", edgecolor=INK2, lw=2, label="validation (tuning)")
ax.scatter([], [], s=60, color=INK2, label="test (5-seed mean)")
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False)
ax.set_title("Validation vs test NDCG@10 per variant", loc="left", fontweight="bold", color=INK, pad=28)
ax.grid(axis="y", visible=False)
fig.savefig(os.path.join(OUT, "fig_val_vs_test.png"))
plt.close(fig)

# ---------------------------------------------------------------------------
# 3. Round trips in the test set, by visible-prefix length
# ---------------------------------------------------------------------------
from src.preprocessing import load_processed  # noqa: E402

test = load_processed(os.path.join(ROOT, "data", "processed"))["test"]
buckets = [("≤3 visible", 0, 3), ("4–5 visible", 4, 5), ("6+ visible", 6, 10 ** 6)]
first, anyprev, ns = [], [], []
for _, lo, hi in buckets:
    sub = [t for t in test if lo <= len(t["cities"]) <= hi]
    ns.append(len(sub))
    first.append(np.mean([t["target_city"] == t["cities"][0] for t in sub]))
    anyprev.append(np.mean([t["target_city"] in t["cities"] for t in sub]))
fig, ax = plt.subplots(figsize=(6.4, 3.8))
x = np.arange(len(buckets))
w = 0.36
for off, vals, col, lab in [(-w / 2 - 0.01, first, "#2a78d6", "destination = first city (origin)"),
                            (w / 2 + 0.01, anyprev, "#eb6834", "destination = any earlier city")]:
    bars = ax.bar(x + off, vals, w, color=col, label=lab, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.1%}", ha="center", fontsize=9, color=INK)
ax.set_xticks(x, [f"{b[0]}\n(n = {n:,})" for b, n in zip(buckets, ns)])
ax.set_ylim(0, max(anyprev) * 1.25)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_ylabel("Share of test trips")
ax.legend(frameon=False, loc="upper right")
ax.grid(axis="x", visible=False)
ax.set_title("How often the final city revisits an earlier stop", loc="left", fontweight="bold", color=INK)
fig.savefig(os.path.join(OUT, "fig_round_trips.png"))
plt.close(fig)

# ---------------------------------------------------------------------------
# 4. Accuracy vs model size for the four headline (base) models
# ---------------------------------------------------------------------------
HEADLINE = [("Markov", "Markov", None), ("FPMC-user", "FPMC", "fpmc-user"),
            ("GRU4Rec-no-side", "GRU4Rec", "gru4rec-no-side"), ("SASRec-no-side", "SASRec", "sasrec-no-side")]
MARKOV_COUNTS = 1_104_652   # stored (context, next) counts of the tuned order-3 model, from 07_comparison
fig, ax = plt.subplots(figsize=(6.4, 3.8))
for row, fam, tag in HEADLINE:
    n = MARKOV_COUNTS if tag is None else pd.read_csv(os.path.join(TUNE, f"{tag}-per-seed.csv"))["n_params"].mean()
    m = means.loc[row, "NDCG@10"]
    ax.scatter(n, m, s=90, color=FAMILY_COLOR[fam], edgecolor="white", lw=2, zorder=3)
    ax.annotate(f"{row}\n{n / 1e6:.1f}M · {m:.4f}", (n, m), xytext=(8, -4), textcoords="offset points",
                fontsize=8.5, color=INK)
ax.set_xscale("log")
ax.set_xlim(6e5, 2e8)
ax.set_ylim(0.35, 0.47)
ax.set_xlabel("Parameters / stored counts (log scale)")
ax.set_ylabel("Test NDCG@10")
ax.set_title("Accuracy vs model size (headline models)", loc="left", fontweight="bold", color=INK)
fig.savefig(os.path.join(OUT, "fig_size_vs_accuracy.png"))
plt.close(fig)

print("wrote:", sorted(os.listdir(OUT)))
