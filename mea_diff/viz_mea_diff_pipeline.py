# -*- coding: utf-8 -*-
"""Visualize (left) where MEA sits in the NETWORK/pipeline vs the EquiDiff baseline, and
(right) the APPLIED TASK + the keyed-vs-unkeyed control that isolates the thesis.

  source bash/init.sh && python mea_diff/viz_mea_diff_pipeline.py
"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

GRN, RED, BLU, ORN, PUR, GRY = "#16a34a", "#dc2626", "#0ea5e9", "#f59e0b", "#a855f7", "#6b7280"


def box(ax, x, y, w, h, t, fc, ec="#111827", fs=8.5, lw=1.3, tc="#111827"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.6",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=tc, zorder=3)


def arr(ax, x0, y0, x1, y1, color="#374151", lw=1.8):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13,
                                 color=color, lw=lw, zorder=1))


fig = plt.figure(figsize=(15, 7.4))
gsL = fig.add_axes([0.02, 0.06, 0.52, 0.86]); gsL.set_xlim(0, 100); gsL.set_ylim(0, 100); gsL.axis("off")
gsR = fig.add_axes([0.58, 0.06, 0.40, 0.86]); gsR.set_xlim(0, 100); gsR.set_ylim(0, 100); gsR.axis("off")

# ================= LEFT: pipeline =================
gsL.text(50, 97, "NETWORK / pipeline change", fontsize=12, weight="bold", ha="center")
# baseline row
gsL.text(3, 82, "BASELINE", fontsize=9, weight="bold", color=GRY, rotation=90, va="center")
box(gsL, 10, 74, 22, 14, "MimicGen\ndemos\n(robomimic hdf5)", "#e5e7eb", fs=8)
box(gsL, 40, 74, 26, 14, "EquiDiff\nequivariant U-Net\n(global SO(2))", "#dbeafe", ec=BLU, fs=8, lw=1.8)
box(gsL, 74, 74, 18, 14, "policy\nπ(a|o)", "#e5e7eb", fs=8)
arr(gsL, 32, 81, 40, 81); arr(gsL, 66, 81, 74, 81)

# MEA row
gsL.text(3, 42, "MEA v2", fontsize=9, weight="bold", color=GRN, rotation=90, va="center")
box(gsL, 10, 34, 22, 14, "MimicGen\ndemos", "#e5e7eb", fs=8)
box(gsL, 37, 33, 22, 16, "MEA phase-aug\n(NEW)\nlow-dim/pc poses\n+ action", "#dcfce7", ec=GRN, fs=7.6, lw=2)
box(gsL, 63, 34, 26, 14, "EquiDiff\nequivariant U-Net\n(UNCHANGED)", "#dbeafe", ec=BLU, fs=7.8, lw=1.8)
arr(gsL, 32, 41, 37, 41); arr(gsL, 59, 41, 63, 41, color=GRN)
gsL.text(48, 26, "v1 = DATA-side only → the network is IDENTICAL to EquiDiff\n"
                 "(no arch change; a config flag selects the aug)", ha="center", fontsize=7.6, color=GRN)

# optional context channel
box(gsL, 40, 8, 40, 10, "optional v2: feed phase/keyed context c\ninto a layer (partial-equi) — LATER ablation",
    "#faf5ff", ec=PUR, fs=7.3, lw=1.2)
arr(gsL, 60, 18, 72, 34, color=PUR)
gsL.text(50, 2, "MEA is an augmentation, not a new network. The comparison holds the backbone fixed.",
         ha="center", fontsize=7.6, color="#111827")

# ================= RIGHT: task + control =================
gsR.text(50, 97, "APPLIED TASK + keyed-vs-unkeyed control", fontsize=11.5, weight="bold", ha="center")


def peg_hole(ax, cx, cy, keyed, color):
    """glyph only: base bar with a keyed(square)/unkeyed(round) hole + geometry label."""
    ax.add_patch(Rectangle((cx - 16, cy - 3.5), 32, 7, fc="#9ca3af", ec="#4b5563", zorder=1))
    if keyed:
        ax.add_patch(Rectangle((cx - 4.5, cy - 3.5), 9, 7, fc="white", ec=color, lw=2, zorder=2))
        ax.text(cx + 26, cy, "square hole", ha="left", va="center", fontsize=7.5, color=color)
    else:
        ax.add_patch(Circle((cx, cy), 4.2, fc="white", ec=color, lw=2, zorder=2))
        ax.text(cx + 26, cy, "round hole", ha="left", va="center", fontsize=7.5, color=color)


# phases strip
gsR.text(50, 90, "phases:  APPROACH (free SO(2) angle)  →  GRASP  →  ENGAGE/INSERT", ha="center", fontsize=8.2, color="#111827")

# Square (keyed) — MEA should WIN
gsR.add_patch(FancyBboxPatch((6, 50), 88, 32, boxstyle="round,pad=0.4", fc="#f0fdf4", ec=GRN, lw=1.6))
gsR.text(50, 78, "SQUARE  (MimicGen 'square', download-ready)", ha="center", fontsize=9, weight="bold", color=GRN)
peg_hole(gsR, 44, 66, keyed=True, color=GRN)
gsR.text(50, 58, "engage group = C4 (4 keyed orientations)", ha="center", fontsize=8, color=GRN)
gsR.text(50, 53, "→ MEA keyed aug adds real info → should BEAT EquiDiff", ha="center", fontsize=8, color=GRN, weight="bold")

# Round (unkeyed) — control
gsR.add_patch(FancyBboxPatch((6, 13), 88, 32, boxstyle="round,pad=0.4", fc="#fef2f2", ec=RED, lw=1.6))
gsR.text(50, 41, "ROUND  (control — GENERATE: robosuite NutAssemblyRound)", ha="center", fontsize=8.6, weight="bold", color=RED)
peg_hole(gsR, 44, 29, keyed=False, color=RED)
gsR.text(50, 21, "engage group = SO(2) (continuous, no reduction)", ha="center", fontsize=8, color=RED)
gsR.text(50, 16, "→ no reduction → MEA should ≈ TIE EquiDiff", ha="center", fontsize=8, color=RED, weight="bold")

gsR.text(50, 8, "'win on square, tie on round' = clean thesis (answers GIC: it's the keyed\n"
                "point-group stabilizer, not contact). Also run Threading (≈C1).",
         ha="center", fontsize=7.6, color="#111827")

fig.suptitle("MEA v2 on the diffusion route — network change (left) & applied task (right)", fontsize=12, y=0.99)
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "context/plan/mea_diff_pipeline.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=125, bbox_inches="tight")
print("saved ->", out)
