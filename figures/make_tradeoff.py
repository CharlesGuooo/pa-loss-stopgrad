# -*- coding: utf-8 -*-
"""
Fig. 3 -- the accuracy/safety trade-off and the guard-cap working point.

Why this figure exists: Section III-D defines the working point as a constrained
argmin, and Section V-A reports the numbers, but nothing shows the selection
happening. A reviewer's natural objection to any "we picked epoch 3" claim is
that the checkpoint was chosen to flatter the result. Plotting both arms' full
six-epoch trajectories in (minADE, collision proxy) space answers that directly:
the cap is drawn, every epoch is visible, and the control's trajectory is seen
to go nowhere at all.

Data: outputs/phase2_3/{full,stopgrad}/summary.json -> results/trainpath.json
(oracle domain, Gaussian surrogate, the configuration of Table I).

IEEE single column (3.5 in), vector PDF. Colour and marker/line style carry the
same distinction so the figure survives the greyscale print edition.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SRC = Path(os.environ.get("PA_LOSS_TRAINPATH", HERE.parent / "results" / "trainpath.json"))

d = json.loads(io.open(SRC, encoding="utf-8").read())
full, sg = d["full"], d["stopgrad"]
CAP = d["guard_cap"]
WP = d["working_point_epoch"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,          # embed as TrueType, not Type 3
    "ps.fonttype": 42,
})

BLUE, ORANGE = "#1f77b4", "#d95f02"

fig, ax = plt.subplots(figsize=(3.5, 2.7))

# region excluded by the guard-cap
ax.axvspan(CAP, 3.5, color="0.90", zorder=0)
ax.axvline(CAP, color="0.45", ls=":", lw=1.1, zorder=1)
ax.text(CAP + 0.02, 0.79, "guard-cap\n(1.5$\\times$ control minADE)",
        fontsize=6.5, color="0.35", va="top", ha="left", linespacing=1.25)

# full model: gradient reaches the predictor
ax.plot(full["minADE"], full["collision"], "-o", color=BLUE, lw=1.4, ms=4.2,
        mfc=BLUE, mec="black", mew=0.5, label="Full (gradient reaches $\\theta$)",
        zorder=3)
# stop-gradient control: identical objective, gradient severed
ax.plot(sg["minADE"], sg["collision"], "--s", color=ORANGE, lw=1.4, ms=4.2,
        mfc="white", mec=ORANGE, mew=1.2, label="Stop-gradient control", zorder=3)

# the selected working point -- annotated into the empty upper-middle so it
# collides with neither the curve nor the axis
i = full["epoch"].index(WP)
ax.plot(full["minADE"][i], full["collision"][i], "o", ms=11, mfc="none",
        mec="black", mew=1.3, zorder=4)
ax.annotate("working point:\nlowest proxy inside the cap",
            xy=(full["minADE"][i], full["collision"][i]),
            xytext=(2.30, 0.475), fontsize=6.8, ha="left", linespacing=1.3,
            arrowprops=dict(arrowstyle="->", lw=0.8, color="black",
                            shrinkA=2, shrinkB=8))

# epoch direction: label the endpoints instead of drawing a second arrow that
# would cross the curve and the working-point annotation
for idx, lab, dx, dy in ((0, "epoch 1", 24, -5), (len(full["epoch"]) - 1, "6", 5, 6)):
    ax.annotate(lab, xy=(full["minADE"][idx], full["collision"][idx]),
                xytext=(dx, dy), textcoords="offset points",
                fontsize=6.5, color=BLUE, ha="center")

ax.set_xlabel("Mean minADE (m)   —   accuracy cost $\\rightarrow$")
ax.set_ylabel("Collision proxy   —   safer $\\downarrow$")
ax.set_xlim(1.85, 3.5)
ax.set_ylim(0.12, 0.85)
ax.legend(loc="upper left", frameon=False, handlelength=2.2, borderpad=0.2)
ax.grid(alpha=0.25, lw=0.5)
ax.set_axisbelow(True)

fig.tight_layout(pad=0.3)
out = HERE / "guo3.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(HERE / "tradeoff_preview.png", dpi=200, bbox_inches="tight")

print("-> wrote %s (vector)" % out.name)
print("   guard-cap            : %.4f" % CAP)
print("   working point        : epoch %d, minADE %.4f, collision %.4f"
      % (WP, full["minADE"][i], full["collision"][i]))
print("   control minADE range : %.3f - %.3f  (moves %.3f m)"
      % (min(sg["minADE"]), max(sg["minADE"]),
         max(sg["minADE"]) - min(sg["minADE"])))
print("   control collision    : %.4f - %.4f" % (min(sg["collision"]), max(sg["collision"])))
