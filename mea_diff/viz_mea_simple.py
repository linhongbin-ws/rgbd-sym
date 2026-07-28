# -*- coding: utf-8 -*-
"""Plain-language picture of the C4-engage augmentation: 1 real demo -> 4 demos.
  source bash/init.sh && python mea_diff/viz_mea_simple.py
"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Arc, FancyArrowPatch

BLUE, BROWN, GRN, ORN, RED, GRY, DK = "#60a5fa", "#b45309", "#16a34a", "#f59e0b", "#dc2626", "#9ca3af", "#111827"


def rot(pts, deg, c):
    th = np.deg2rad(deg); R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return (R @ np.asarray(pts).T).T + np.asarray(c)


def _seg(ax, p0, p1, yaw, c, **kw):
    a, b = rot([p0, p1], yaw, c)
    ax.plot([a[0], b[0]], [a[1], b[1]], **kw)


def nut(ax, cx, cy, yaw, r=4.2, ec=DK, plate=BLUE, inserted=True):
    """a square nut with a short handle, grasped by a compact 2-finger gripper (fingers
    straddle the handle) — one rigid unit rotated by yaw about (cx,cy)."""
    c = np.array([cx, cy])
    # plate + square hole
    ax.add_patch(plt.Polygon(rot([[-r, -r], [r, -r], [r, r], [-r, r]], yaw, c), closed=True,
                             fc=plate, ec=ec, lw=1.6, zorder=2))
    h = r * 0.42
    ax.add_patch(plt.Polygon(rot([[-h, -h], [h, -h], [h, h], [-h, h]], yaw, c), closed=True,
                             fc="white", ec=ec, lw=1.2, zorder=3))
    if inserted:                                            # peg filling the hole = inserted
        ax.add_patch(plt.Circle((cx, cy), r * 0.22, fc="#6b7280", ec=DK, lw=0.8, zorder=4))
    # short handle bar sticking out +x in the nut frame
    hb = r * 0.24; hl0, hl1 = r * 0.95, r + r * 0.55
    ax.add_patch(plt.Polygon(rot([[hl0, -hb], [hl1, -hb], [hl1, hb], [hl0, hb]], yaw, c),
                             closed=True, fc=BROWN, ec="#7c2d12", lw=1, zorder=3))
    # compact 2-finger gripper straddling the handle
    gap = hb + r * 0.16
    f0, f1 = r * 1.0, hl1 + r * 0.1
    for s in (-1, 1):
        _seg(ax, [f0, s * gap], [f1, s * gap], yaw, c, color=DK, lw=3.5, zorder=5, solid_capstyle="round")
    _seg(ax, [f1, -gap], [f1, gap], yaw, c, color=DK, lw=3.5, zorder=5, solid_capstyle="round")


fig, ax = plt.subplots(figsize=(15, 6.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off"); ax.set_aspect("equal")

ax.text(50, 96, "The augmentation in one picture:  1 real demo  →  4 demos", fontsize=15, weight="bold", ha="center")

# --- tiny "why" inset top-left: a square peg fits a square hole 4 ways ---
ax.add_patch(FancyBboxPatch((2, 60), 20, 30, boxstyle="round,pad=0.4", fc="#f8fafc", ec=GRY, lw=1))
ax.text(12, 86, "Why 4?", fontsize=10, weight="bold", ha="center")
ax.add_patch(plt.Rectangle((7, 68), 10, 10, fc="white", ec=RED, lw=2))       # square hole
ax.add_patch(plt.Rectangle((9, 70), 6, 6, fc=BLUE, ec=DK, lw=1.4))           # square peg
ax.add_patch(Arc((12, 73), 16, 16, theta1=20, theta2=290, color=DK, lw=1.6))
ax.annotate("", xy=(19.5, 74.5), xytext=(19.5, 72), arrowprops=dict(arrowstyle="-|>", color=DK))
ax.text(12, 63, "a square peg fits a\nsquare hole 4 ways", fontsize=8, ha="center", color=DK)

# --- the 4 insertion scenes ---
xs = [34, 52, 70, 88]; y = 45
labels = ["① REAL demo", "② MEA", "③ MEA", "④ MEA"]
for i, (x, yaw) in enumerate(zip(xs, [0, 90, 180, 270])):
    ec = GRN if i == 0 else ORN
    ax.add_patch(FancyBboxPatch((x - 8.5, y - 11), 17, 26, boxstyle="round,pad=0.3",
                                fc="#f0fdf4" if i == 0 else "#fff7ed", ec=ec, lw=2, zorder=1))
    nut(ax, x, y + 2, yaw)
    ax.text(x, y + 12.5, f"{yaw}°", fontsize=9, ha="center", color=ec, weight="bold")
    ax.text(x, y - 9, labels[i], fontsize=9.5, ha="center", color=ec, weight="bold")
    ax.text(x, y - 13.5, "✓ inserted", fontsize=8, ha="center", color=GRN)

# "+90°" rotation arrows between the MEA nuts (reads as a rotation, not a teleport)
for i in [1, 2]:
    xm = (xs[i] + xs[i + 1]) / 2
    ax.add_patch(Arc((xm, y + 2), 9, 6, theta1=200, theta2=340, color=GRY, lw=1.4))
    ax.annotate("", xy=(xm + 3.4, y + 1.2), xytext=(xm + 2.2, y - 0.6),
                arrowprops=dict(arrowstyle="-|>", color=GRY, lw=1.2))
    ax.text(xm, y - 3.2, "+90°", fontsize=7.5, ha="center", color=GRY)

# bracket over the 3 MEA ones
ax.plot([xs[1] - 9, xs[3] + 9], [y + 18.5, y + 18.5], color=ORN, lw=1.6)
ax.plot([xs[1] - 9, xs[1] - 9], [y + 16.5, y + 18.5], color=ORN, lw=1.6)
ax.plot([xs[3] + 9, xs[3] + 9], [y + 16.5, y + 18.5], color=ORN, lw=1.6)
ax.text((xs[1] + xs[3]) / 2, y + 21, "+3 more, FREE — same task, just the nut rotated (all valid)",
        fontsize=10, ha="center", color=ORN, weight="bold")
# big arrow from the real one to the MEA ones
ax.annotate("", xy=(xs[1] - 10, y + 2), xytext=(xs[0] + 9, y + 2),
            arrowprops=dict(arrowstyle="-|>", color=DK, lw=2.5))
ax.text((xs[0] + xs[1]) / 2, y + 6, "MEA", fontsize=10, ha="center", weight="bold", color=DK)

# --- bottom: the point ---
ax.add_patch(FancyBboxPatch((6, 6), 88, 15, boxstyle="round,pad=0.5", fc="#fef9c3", ec="#ca8a04", lw=1.2))
ax.text(50, 16.5, "The robot's demos only EVER show orientation ①.", fontsize=11, ha="center", weight="bold", color=DK)
ax.text(50, 11.5, "Without MEA the policy never sees ② ③ ④ — those 3 orientations are the extra information MEA adds.",
        fontsize=10, ha="center", color=DK)
ax.text(50, 7.6, "(EquiDiff's built-in symmetry rotates the WHOLE scene together — it can't rotate just the nut, so it can't add these.)",
        fontsize=8.3, ha="center", color=GRY)

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "context/plan/mea_diff_simple.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=125, bbox_inches="tight")
print("saved ->", out)
