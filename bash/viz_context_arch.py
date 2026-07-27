# -*- coding: utf-8 -*-
"""Schematic of the MEA v2 context-conditioned equivariant policy/critic:
belief-inferred phase routes among sub-networks with DIFFERENT equivariance.
Annotates which branches ESCAPE the extrinsic-equivariance floor. No mathtext.

  source bash/init.sh && python bash/viz_context_arch.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

C_EQUI, C_FREE, C_ANCH, C_LOAD = "#22c55e", "#ef4444", "#0ea5e9", "#a855f7"
C_BOX = "#e5e7eb"


def box(ax, x, y, w, h, text, fc, ec="#111827", fs=8.5, tc="#111827", lw=1.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, zorder=3)


def arrow(ax, x0, y0, x1, y1, color="#6b7280", lw=1.6):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13,
                                 color=color, lw=lw, zorder=1))


fig, ax = plt.subplots(figsize=(14.5, 8.4))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

# inputs
box(ax, 1, 46, 13, 9, "occupancy obs o_t\n(gripper-centered)", C_BOX, fs=8)
box(ax, 1, 30, 13, 9, "RNN belief\nb_t = P(s | o<=t)", C_BOX, fs=8)
box(ax, 17, 30, 14, 9, "phase / gauge\nestimator c(b_t)\n-> p(c|.)  [C4-invariant]", "#fef9c3", fs=7.4)
box(ax, 17, 46, 14, 9, "shared stem\nlift -> C4 steerable\nfeature field", "#dcfce7", fs=7.6)
arrow(ax, 14, 50.5, 17, 50.5)
arrow(ax, 14, 34.5, 17, 34.5)

# router
box(ax, 35, 38, 10, 12, "ROUTER\n(argmax /\nsoft alpha_c)", "#fde68a", fs=8)
arrow(ax, 31, 34.5, 35, 42)
arrow(ax, 31, 50.5, 35, 47)

# experts
ex_x, ex_w = 49, 31
box(ax, ex_x, 70, ex_w, 12, "c1 APPROACH  .  free / unloaded\nC4 equivariant, anchor = OBJECT",
    "#dcfce7", ec=C_EQUI, fs=8, lw=2)
box(ax, ex_x, 44, ex_w, 12, "c2 CONTACT / GRASP  .  rotation broken\nC4->C1 restriction + free (unconstrained)",
    "#fee2e2", ec=C_FREE, fs=8, lw=2)
box(ax, ex_x, 18, ex_w, 12, "c3 MANIPULATE  .  loaded\nC4 equivariant, anchor = GOAL + loaded head",
    "#e0f2fe", ec=C_ANCH, fs=8, lw=2)
arrow(ax, 45, 46, ex_x, 76, C_EQUI)
arrow(ax, 45, 44, ex_x, 50, C_FREE)
arrow(ax, 45, 42, ex_x, 24, C_ANCH)

# escape-floor tags
ax.text(ex_x + ex_w + 1.2, 76, "stays\nequivariant\n(bias OK)", ha="left", va="center",
        fontsize=7.3, color=C_EQUI)
ax.text(ex_x + ex_w + 1.2, 39, "* ESCAPES the\nextrinsic-equi\nFLOOR here", ha="left", va="center",
        fontsize=7.8, color=C_FREE, weight="bold")
ax.text(ex_x + ex_w + 1.2, 24, "re-anchor = bias;\nloaded head =\ndynamics split\n(not a break)",
        ha="left", va="center", fontsize=7.3, color=C_ANCH)

# fuse + heads
box(ax, 89, 44, 9.5, 12, "restrict to\ncommon field\n+ mix alpha_c", "#f3e8ff", ec=C_LOAD, fs=7.4)
for yy in (76, 50, 24):
    arrow(ax, ex_x + ex_w, yy, 89, 50)
box(ax, 72, 2, 26.5, 9, "shared invariant Q-head / steerable Gaussian actor-head", "#f3f4f6", fs=8)
arrow(ax, 93.7, 44, 90, 11)

# Curie no-op ablation callout
box(ax, 1, 5, 29, 16,
    "ABLATION (B2a): C4-invariant scalar -> C4 net\n"
    "stays GLOBALLY C4-equivariant (Curie: G_phi(x) contains G_x)\n"
    "-> CANNOT break symmetry -> should be ~ baseline.\n"
    "Escape needs GROUP DEGENERATION (c2), not a scalar.",
    "#fff1f2", ec=C_FREE, fs=7.3, tc="#7f1d1d", lw=1.4)

fig.suptitle("MEA v2 - context-conditioned equivariant policy/critic:  belief-inferred phase routes among "
             "sub-networks of DIFFERENT equivariance", fontsize=11, y=0.975)
ax.text(50, 90.5,
        "Novelty (unoccupied conjunction): K>2 group ladder C4/C2/C1  +  belief-phase routing in a POMDP  +  "
        "contact-phase-transition schedule  +  object->goal re-anchor  +  paired MEA data aug.\n"
        "Differentiate hard from PE-SAC (arXiv:2512.00915, ICLR 2026 - G-vs-trivial gate, fully-observed, "
        "dynamics-disagreement).  Only the RED (group-degeneration / free) branch escapes the floor.",
        ha="center", va="center", fontsize=8, color="#374151")

plt.tight_layout(rect=[0, 0, 1, 0.93])
out = "context/plan/context_to_network_arch.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=125, bbox_inches="tight")
print("saved ->", out)
