# -*- coding: utf-8 -*-
"""
Regenerate Fig. 3 (working-point collision proxy with bootstrap 95% CIs) as a
PDF vector sized for the IEEE single column.

Differences from the thesis version (thesis/figures/scripts/plot_results.py):
  * PDF vector instead of 200-dpi PNG  -> IEEE resolution rule does not apply
  * figsize 3.5 in wide (IEEE single column) so fonts land at ~8-9 pt at
    final size, instead of a 6.4 in figure shrunk to 3.5 in (fonts -> 5.5 pt)
  * hatching on the control bars so colour and texture carry the same
    information -> survives a greyscale printout (IEEE accessibility)

Data source is the same committed result JSON; nothing is hard-coded.

Run:  uv run --with matplotlib python make_fig3.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
WP_JSON = Path(r"D:\Research\Thesis\KnowledgeBase\03_实验结果\phase2_7b\stats_wp.json")

# Okabe-Ito colourblind-safe palette (same as the thesis figure)
BLUE = "#0072B2"      # full PA-Loss
ORANGE = "#D55E00"    # stop-gradient control
GREY = "#999999"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8.5,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

CELLS = [
    ("oracle", "global", "Oracle\nGlobal"),
    ("oracle", "high_conflict", "Oracle\nHigh-conf."),
    ("perceived", "global", "Perceived\nGlobal"),
    ("perceived", "high_conflict", "Perceived\nHigh-conf."),
]


def asym(mean, lo, hi):
    return max(mean - lo, 0.0), max(hi - mean, 0.0)


data = json.loads(WP_JSON.read_text(encoding="utf-8"))["domains"]

full_v, sg_v, full_e, sg_e, labels = [], [], [], [], []
print("Working-point collision proxy   mean [lo, hi]")
for domain, subset, label in CELLS:
    vc = data[domain]["variant_cis"]
    f = vc["full"][subset]["collision_proxy"]
    s = vc["stopgrad"][subset]["collision_proxy"]
    full_v.append(f[0]); sg_v.append(s[0])
    full_e.append(asym(*f)); sg_e.append(asym(*s))
    labels.append(label)
    sep = "non-overlap" if f[2] < s[1] else "*** OVERLAP ***"
    print("  %-9s %-13s full %.3f [%.3f, %.3f] | sg %.3f [%.3f, %.3f]  (%s)"
          % (domain, subset, f[0], f[1], f[2], s[0], s[1], s[2], sep))

x = list(range(len(labels)))
w = 0.38
# Height chosen so the tight-bbox output keeps the thesis figure's ~1.68 aspect.
# A taller figure costs ~0.4 in of a 10-page budget for no extra information.
fig, ax = plt.subplots(figsize=(3.5, 2.05))

ax.bar([i - w / 2 for i in x], full_v, w, yerr=list(zip(*full_e)), capsize=2.5,
       color=BLUE, edgecolor="black", linewidth=0.5, label="Full PA-Loss",
       error_kw={"elinewidth": 0.9, "ecolor": "black"})
ax.bar([i + w / 2 for i in x], sg_v, w, yerr=list(zip(*sg_e)), capsize=2.5,
       color=ORANGE, edgecolor="black", linewidth=0.5, hatch="///",
       label="Stop-gradient control",
       error_kw={"elinewidth": 0.9, "ecolor": "black"})

ax.axvline(1.5, color=GREY, linewidth=0.7, linestyle=(0, (4, 4)))
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Collision proxy at working point")
ax.set_ylim(0, max(sg_v) * 1.22)
ax.legend(frameon=False, loc="upper left", handlelength=1.4, borderpad=0.2)
ax.margins(x=0.04)

out = HERE / "guo3.pdf"
fig.savefig(out)
plt.close(fig)
print("\n-> wrote %s  (%.1f KB, vector)" % (out.name, out.stat().st_size / 1024))
