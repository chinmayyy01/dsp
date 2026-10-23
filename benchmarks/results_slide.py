"""Render a single 16:9 results panel for the presentation.

Run:  .venv/bin/python benchmarks/results_slide.py
Out:  assets/results-slide.png
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.run_benchmark import anc_learning_curves, erle_curve, smooth  # noqa: E402

ROWS = list(csv.DictReader((ROOT / "results" / "results.csv").open()))
NOISES = ["white", "pink", "brown", "hum", "fan", "babble"]
INK = "#141414"
GREY = "#8a8a8a"
BLUE = "#1d5fbf"
GRID = "#e6e6e6"
MUTED = "#6b6b6b"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 13,
    "axes.edgecolor": "#c9c9c9",
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.titlesize": 15,
    "axes.titleweight": 600,
})


def mean(method: str, noise: str, key: str) -> float:
    values = [float(r[key]) for r in ROWS if r["method"] == method and r["noise"] == noise]
    return float(np.mean(values))


fig, axes = plt.subplots(2, 2, figsize=(13.33, 7.5), dpi=200)
fig.patch.set_facecolor("white")

ax = axes[0, 0]
gain = [mean("adaptive_wiener", n, "si_sdr") - mean("noisy", n, "si_sdr") for n in NOISES]
order = np.argsort(gain)
ax.barh([NOISES[i] for i in order], [gain[i] for i in order], color=BLUE, height=0.6)
for i, idx in enumerate(order):
    ax.text(gain[idx] + 0.2, i, f"+{gain[idx]:.1f}", va="center", color=INK, fontsize=13)
ax.set_title("SI-SDR gain over noisy input (dB)")
ax.set_xlim(0, max(gain) + 2)
ax.grid(axis="x", color=GRID)
ax.set_axisbelow(True)

ax = axes[0, 1]
methods = [("noisy", "noisy"), ("spectral_subtraction", "spec-sub"), ("adaptive_wiener", "adaptive")]
x = np.arange(len(NOISES))
width = 0.26
for i, (key, label) in enumerate(methods):
    values = [mean(key, n, "pesq") for n in NOISES]
    ax.bar(x + (i - 1) * width, values, width, label=label,
           color=[GREY, "#b8b8b8", INK][i])
ax.set_xticks(x, NOISES, fontsize=11)
ax.set_ylim(1.0, 2.1)
ax.set_title("PESQ wideband (1.0 bad – 4.5 excellent)")
ax.grid(axis="y", color=GRID)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=11, loc="upper left", ncol=3)

ax = axes[1, 0]
for i, (key, label) in enumerate(methods):
    values = [mean(key, n, "stoi") for n in NOISES]
    ax.bar(x + (i - 1) * width, values, width, label=label,
           color=[GREY, "#b8b8b8", INK][i])
ax.set_xticks(x, NOISES, fontsize=11)
ax.set_ylim(0.7, 1.0)
ax.set_title("STOI intelligibility")
ax.grid(axis="y", color=GRID)
ax.set_axisbelow(True)

ax = axes[1, 1]
curves, desired = anc_learning_curves()
for (name, result), colour in zip(curves.items(), [INK, "#8a8a8a", BLUE]):
    ax.plot(smooth(erle_curve(desired, result["output"]), 5), label=name, color=colour, linewidth=2)
ax.axhline(20, color="#c9c9c9", linestyle="--", linewidth=1)
ax.set_title("ANC convergence (ERLE per 256 samples)")
ax.set_xlabel("block")
ax.set_ylabel("dB", rotation=0, labelpad=14)
ax.set_ylim(0, 60)
ax.grid(color=GRID)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=11, loc="lower right")

for ax in axes.flat:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.tight_layout(pad=1.4)
target = ROOT / "assets" / "results-slide.png"
fig.savefig(target, facecolor="white")
print(f"[ok] {target.relative_to(ROOT)}")
