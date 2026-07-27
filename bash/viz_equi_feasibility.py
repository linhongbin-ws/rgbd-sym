# -*- coding: utf-8 -*-
"""Explainer: why the per-context equivariant net is FEASIBLE, and the honest
"feasible != beneficial" distinction (where the extrinsic-equivariance FLOOR appears).

Top row  — the orbit mechanism:
  A) symmetry PRESENT: rotating the scene rotates the optimal action (pi*(g.o)=g.pi*(o))
     -> C4 net folds the orbit exactly, floor = 0  -> C4 feasible & sample-efficient.
  B) symmetry BROKEN: a C4-symmetric observation but an orientation-specific optimal
     action -> Curie (G_out >= G_in) forces a C4-symmetric (directionless) output =
     orbit-average -> floor > 0  -> C4 is EXTRINSIC/harmful; relax C4->C1.
Bottom row — the three contexts, each with the largest valid group + an honest
"is global C4 actually broken here (=> gain), or already correct (=> no net-side gain)".

  source bash/init.sh && python bash/viz_equi_feasibility.py
"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrow, Arc, FancyBboxPatch

GRN, RED, BLU, PUR, GRY = "#16a34a", "#dc2626", "#0ea5e9", "#a855f7", "#6b7280"


def block(ax, cx, cy, s, yaw, fc="#3b82f6", ec="#1e3a8a", alpha=1.0):
    th = np.deg2rad(yaw)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    pts = (R @ (np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]).T * s / 2)).T + [cx, cy]
    ax.add_patch(plt.Polygon(pts, closed=True, fc=fc, ec=ec, lw=1.4, alpha=alpha, zorder=2))


def gripper(ax, cx, cy, yaw, color="#111827", alpha=1.0, size=0.9):
    th = np.deg2rad(yaw)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    for dx in (-0.45, 0.45):
        a = R @ np.array([dx * size, -0.5 * size]); b = R @ np.array([dx * size, 0.5 * size])
        ax.plot([cx + a[0], cx + b[0]], [cy + a[1], cy + b[1]], color=color, lw=2.6, alpha=alpha, zorder=3)
    a = R @ np.array([-0.45 * size, 0.5 * size]); b = R @ np.array([0.45 * size, 0.5 * size])
    ax.plot([cx + a[0], cx + b[0]], [cy + a[1], cy + b[1]], color=color, lw=2.6, alpha=alpha, zorder=3)


def arrow(ax, x, y, dx, dy, color, lw=2.4, alpha=1.0, hw=0.32):
    ax.add_patch(FancyArrow(x, y, dx, dy, width=0.02, head_width=hw, head_length=0.3,
                            length_includes_head=True, color=color, alpha=alpha, zorder=4))


def badge(ax, x, y, text, ok):
    fc = "#dcfce7" if ok else "#fee2e2"; ec = GRN if ok else RED
    ax.add_patch(FancyBboxPatch((x, y), 4.7, 0.9, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc=fc, ec=ec, lw=1.2, zorder=5))
    ax.text(x + 0.15, y + 0.45, ("✓ " if ok else "✗ ") + text, fontsize=7.4, va="center",
            ha="left", color="#111827", zorder=6)


def setup(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal"); ax.axis("off")


fig = plt.figure(figsize=(15, 9.6))
gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.05], hspace=0.18, wspace=0.35)

# ============ TOP A: symmetry present -> C4 correct ============
axA = fig.add_subplot(gs[0, 0:3]); setup(axA)
axA.add_patch(FancyBboxPatch((0.2, 0.2), 9.6, 9.6, boxstyle="round,pad=0.02", fc="#f0fdf4", ec=GRN, lw=2))
axA.set_title("A.  symmetry PRESENT (e.g. free-space approach)", fontsize=10, color=GRN, weight="bold")
# original
block(axA, 3.0, 6.6, 1.3, 20); gripper(axA, 3.0, 8.4, 20, alpha=0.9); arrow(axA, 3.0, 8.0, -0.0, -1.0, GRN)
axA.text(3.0, 5.4, "o,  π*(o)", ha="center", fontsize=8, color=GRN)
# rotated copy g.o
block(axA, 7.0, 6.6, 1.3, 20 + 70); gripper(axA, 7.0, 8.4, 20 + 70, alpha=0.9)
arrow(axA, 7.0, 8.0, np.sin(np.deg2rad(70)) * 1.0, -np.cos(np.deg2rad(70)) * 1.0, GRN)
axA.text(7.0, 5.4, "g·o,  π*(g·o)=g·π*(o)", ha="center", fontsize=8, color=GRN)
axA.add_patch(Arc((5.0, 7.2), 3.2, 3.2, angle=0, theta1=25, theta2=150, color=GRY, lw=1.4))
axA.annotate("", xy=(6.1, 8.35), xytext=(6.4, 8.15), arrowprops=dict(arrowstyle="-|>", color=GRY))
axA.text(5.0, 9.0, "rotate scene by g", ha="center", fontsize=7.5, color=GRY)
axA.text(5.0, 3.7, "obs AND optimal action lie on the SAME orbit\n"
                   "→ C4 net folds the orbit exactly → error floor = 0\n"
                   "→ full C4 is FEASIBLE and maximally sample-efficient",
         ha="center", va="center", fontsize=8.2, color="#065f46")

# ============ TOP B: symmetry broken -> extrinsic -> floor ============
axB = fig.add_subplot(gs[0, 3:6]); setup(axB)
axB.add_patch(FancyBboxPatch((0.2, 0.2), 9.6, 9.6, boxstyle="round,pad=0.02", fc="#fef2f2", ec=RED, lw=2))
axB.set_title("B.  symmetry BROKEN (orientation-specific contact, e.g. needle)", fontsize=10, color=RED, weight="bold")
# C4-symmetric observation (a 4-fold shape) but a single true optimal direction
for a in (0, 90, 180, 270):
    r = np.deg2rad(a)
    axB.add_patch(Circle((3.0 + 0.9 * np.cos(r), 7.0 + 0.9 * np.sin(r)), 0.32, fc="#fca5a5", ec="#991b1b", lw=1))
axB.text(3.0, 5.15, "obs is C4-symmetric\n(G_x = C4)", ha="center", fontsize=7.6, color="#991b1b")
arrow(axB, 3.0, 7.0, 0.0, 1.35, "#111827")  # true optimal = one fixed direction
axB.text(3.0, 8.75, "true optimal:\none fixed grasp dir", ha="center", fontsize=7.2, color="#111827")
# C4-equivariant forced output = symmetric (directionless)
for a in (0, 90, 180, 270):
    r = np.deg2rad(a); arrow(axB, 7.0, 7.0, 0.85 * np.cos(r), 0.85 * np.sin(r), RED, lw=1.8, alpha=0.85, hw=0.26)
axB.text(7.0, 5.15, "C4 net MUST output\nC4-symmetric action\n= orbit-average (no dir)", ha="center", fontsize=7.4, color=RED)
axB.text(5.0, 3.4, "Curie: equivariant map cannot lower symmetry (G_out ⊇ G_in)\n"
                   "symmetric obs + asymmetric optimal → forced orbit-average → FLOOR > 0\n"
                   "→ full C4 is EXTRINSIC / harmful here → relax C4 → C1 (or free head)",
         ha="center", va="center", fontsize=8.2, color="#7f1d1d")

# ============ BOTTOM: three contexts ============
def ctx_axes(col):
    ax = fig.add_subplot(gs[1, col:col + 2]); setup(ax); return ax

# c1 approach
ax1 = ctx_axes(0)
ax1.add_patch(FancyBboxPatch((0.2, 0.2), 9.6, 9.6, boxstyle="round,pad=0.02", fc="#f8fafc", ec=GRN, lw=1.8))
ax1.set_title("c1 APPROACH  ·  free / unloaded", fontsize=9.5, weight="bold")
block(ax1, 5.0, 4.2, 1.3, 15); gripper(ax1, 3.2, 7.4, 40); arrow(ax1, 3.6, 6.9, 1.0, -1.6, GRN)
ax1.text(5.0, 9.05, "group: full C4  (anchor = object)", ha="center", fontsize=8, color="#111827")
badge(ax1, 0.7, 2.1, "FEASIBLE: free-space isotropy", True)
badge(ax1, 0.7, 1.0, "global C4 already correct → net gain≈0", False)
ax1.text(5.0, 0.55, "(gain here is DATA-side: approach-angle aug)", ha="center", fontsize=6.8, color=GRY)

# c2 contact (two sub-cases)
ax2 = ctx_axes(2)
ax2.add_patch(FancyBboxPatch((0.2, 0.2), 9.6, 9.6, boxstyle="round,pad=0.02", fc="#f8fafc", ec=RED, lw=1.8))
ax2.set_title("c2 CONTACT / GRASP", fontsize=9.5, weight="bold")
block(ax2, 2.7, 5.6, 1.1, 0, fc="#93c5fd"); gripper(ax2, 2.7, 5.6, 0, size=0.8)
ax2.text(2.7, 7.2, "symmetric block", ha="center", fontsize=7, color=GRN)
badge(ax2, 0.6, 3.0, "C4 still correct → no gain", False)
# asymmetric object (needle)
ax2.plot([6.4, 7.9], [5.2, 6.0], color="#b45309", lw=3)  # needle
gripper(ax2, 7.15, 5.6, 28, size=0.8)
ax2.text(7.15, 7.2, "needle (asym.)", ha="center", fontsize=7, color=RED)
badge(ax2, 5.2, 3.0, "C4 BROKEN → C1 gains", True)
ax2.text(5.0, 9.05, "group: C4 → C1  (only if orientation-specific)", ha="center", fontsize=7.8, color="#111827")
ax2.text(5.0, 1.4, "★ the ONLY phase that escapes the floor —\nand only for orientation-specific contact",
         ha="center", va="center", fontsize=7.6, color=RED, weight="bold")

# c3 manipulate
ax3 = ctx_axes(4)
ax3.add_patch(FancyBboxPatch((0.2, 0.2), 9.6, 9.6, boxstyle="round,pad=0.02", fc="#f8fafc", ec=BLU, lw=1.8))
ax3.set_title("c3 MANIPULATE  ·  loaded", fontsize=9.5, weight="bold")
ax3.add_patch(Circle((7.2, 4.2), 1.2, fc="none", ec=PUR, lw=1.8, ls="--")); ax3.text(7.2, 4.2, "goal", ha="center", va="center", fontsize=7, color=PUR)
block(ax3, 3.2, 6.0, 1.2, 10, fc="#3b82f6"); gripper(ax3, 3.2, 6.0, 10, size=0.85)
arrow(ax3, 3.9, 5.7, 2.6, -1.2, BLU)
ax3.text(5.0, 9.05, "group: C4 about GOAL  +  loaded head", ha="center", fontsize=8, color="#111827")
badge(ax3, 0.7, 2.3, "C4-about-goal correct (feasible)", True)
badge(ax3, 0.7, 1.15, "loaded≠unloaded hidden ctx → feed c → gain", True)

# bottom conclusion strip
fig.text(0.5, 0.012,
         "FEASIBLE = each phase's LARGEST true-symmetry group holds (theory: per-context Group-Invariant POMDP).   "
         "BENEFICIAL (beats global-C4 baseline) = ONLY where global C4 is actually BROKEN: orientation-specific "
         "contact (needle) / loaded-vs-unloaded / partial observability.\n"
         "Symmetric sim blocks + full obs → global C4 holds everywhere → no floor → group degeneration valid but "
         "NO gain (this is why this session's results were null).  Pick a task with a REAL break.",
         ha="center", va="bottom", fontsize=8.1, color="#111827",
         bbox=dict(fc="#fef9c3", ec="#ca8a04", lw=1, boxstyle="round,pad=0.4"))

fig.suptitle("Why each context's equivariant net is FEASIBLE — and why feasible ≠ beneficial (where the "
             "extrinsic-equivariance FLOOR appears)", fontsize=11.5, y=0.985)
plt.tight_layout(rect=[0, 0.055, 1, 0.96])
out = "context/plan/equi_feasibility_explainer.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=125, bbox_inches="tight")
print("saved ->", out)
