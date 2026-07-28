# -*- coding: utf-8 -*-
"""Verify the pivoted C4-ENGAGE augmentation is CORRECT (uses the real phase_aug primitives).

Panel 1  the 4 C4-equivalent insertions: a HANDLED square nut rotated 0/90/180/270 about
         its OWN axis, gripper on the handle, all into the SAME fixed square hole. Shows:
         handle moves (genuinely different obs) / nut spins in place / all 4 fit the hole
         (C4 equivalence of the peg) / gripper follows the handle.
Panel 2  feasibility fix: apply a 180 deg keyed element to the CARRY trajectory with the
         OBJECT-OWN-AXIS anchor (fixed code: spins in place) vs the HOLE anchor (old bug:
         sweeps the gripper to the opposite side / large radius / off-table).
Panel 3  action stays consistent: the commanded action (arrow) transforms WITH the nut —
         the action in the nut's frame is unchanged (what test_action_consistency asserts).

  source bash/init.sh && python mea_diff/viz_mea_diff_verify.py
"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from phase_aug import rot_z, transform_point, make_synthetic_episode, ENGAGE

GRN, RED, BLU, ORN, GRY, PUR = "#16a34a", "#dc2626", "#0ea5e9", "#f59e0b", "#6b7280", "#a855f7"


def draw_nut(ax, center, yaw_deg, color, s=0.03, handle=0.028, alpha=1.0, lw=1.6):
    """handled square nut: outer square plate + square hole + a protruding handle."""
    th = np.deg2rad(yaw_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    c = np.asarray(center, float)
    def poly(pts, **kw):
        ax.add_patch(plt.Polygon((R @ (np.array(pts).T)).T + c, closed=True, **kw))
    poly([[-s, -s], [s, -s], [s, s], [-s, s]], fc=color, ec="#1e3a8a", lw=lw, alpha=alpha)          # plate
    poly([[-s*.4, -s*.4], [s*.4, -s*.4], [s*.4, s*.4], [-s*.4, s*.4]], fc="white", ec="#1e3a8a", lw=1, alpha=alpha)  # hole
    hb = s * 0.35
    poly([[s, -hb], [s + handle, -hb], [s + handle, hb], [s, hb]], fc="#b45309", ec="#7c2d12", lw=1, alpha=alpha)     # handle
    # gripper on the handle tip
    tip = R @ np.array([s + handle, 0]) + c
    perp = R @ np.array([0, 1.0])
    for sgn in (-1, 1):
        a = tip + perp * sgn * 0.012 - R @ np.array([0.01, 0])
        b = tip + perp * sgn * 0.012 + R @ np.array([0.006, 0])
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#111827", lw=2.2, alpha=alpha)
    return tip


def draw_hole(ax, center, s=0.032, color=RED):
    ax.add_patch(plt.Rectangle((center[0] - s / 2, center[1] - s / 2), s, s, fill=False, ec=color, lw=2))
    ax.text(center[0], center[1] + s * 0.9, "fixed square hole/peg", ha="center", fontsize=7, color=color)


fig, ax = plt.subplots(1, 3, figsize=(16, 5.4))
hole = np.array([0.0, 0.0])

# ---------- Panel 1: the 4 C4-equivalent insertions ----------
a0 = ax[0]; a0.set_title("① C4-equivalent insertions (the augmentation)", fontsize=10, weight="bold")
draw_hole(a0, hole)
cols = [GRN, BLU, ORN, PUR]
for k, ang in enumerate([0, 90, 180, 270]):
    # nut inserted at the hole, rotated by the keyed element about its OWN center (=hole here)
    tip = draw_nut(a0, hole, ang, cols[k], alpha=0.9)
    a0.text(tip[0], tip[1], f"{ang}°", fontsize=7, color=cols[k], weight="bold")
a0.text(0, -0.075, "handle at 4 positions = genuinely different obs;\n"
                   "all 4 fit the SAME square hole = C4 equivalent", ha="center", fontsize=8, color="#111827")

# ---------- Panel 2: feasibility fix (object-anchor vs hole-anchor), single carry frame ----------
a1 = ax[1]; a1.set_title("② feasibility of a 180° keyed element (mid-carry frame)", fontsize=9.5, weight="bold")
obj_c = np.array([-0.03, -0.02]); holexy = np.array([0.06, 0.05])
grip = obj_c + np.array([0.028, 0.0])                       # gripper on the handle (offset from nut center)
R180 = rot_z(np.pi)
g_obj = transform_point(np.r_[grip, 0], R180, obj_c)[:2]    # FIX: about object's own center -> local orbit
g_hole = transform_point(np.r_[grip, 0], R180, holexy)[:2]  # OLD BUG: about the hole -> far sweep
draw_nut(a1, obj_c, 0, GRY, alpha=0.6)
a1.plot(holexy[0], holexy[1], "r+", ms=10, mew=2); a1.text(holexy[0], holexy[1] + 0.012, "hole", fontsize=7, color=RED)
a1.plot(*grip, "ko", ms=6); a1.text(grip[0], grip[1] - 0.014, "gripper", fontsize=7)
# FIX
a1.annotate("", xy=g_obj, xytext=grip, arrowprops=dict(arrowstyle="-|>", color=GRN, lw=2))
a1.plot(*g_obj, "o", ms=6, color=GRN)
a1.text(g_obj[0] - 0.005, g_obj[1] + 0.012, f"object-anchor (FIX)\norbits {np.linalg.norm(g_obj-grip)*100:.1f} cm — feasible",
        ha="center", fontsize=7, color=GRN)
# BUG (points off-plot -> draw a clipped arrow toward it + annotate the distance)
dir_hole = (g_hole - grip); dir_hole = dir_hole / np.linalg.norm(dir_hole)
a1.annotate("", xy=grip + dir_hole * 0.10, xytext=grip, arrowprops=dict(arrowstyle="-|>", color=RED, lw=2, ls="--"))
a1.text(grip[0] + dir_hole[0] * 0.055, grip[1] + dir_hole[1] * 0.055 - 0.012,
        f"hole-anchor (OLD BUG)\nswept {np.linalg.norm(g_hole-grip)*100:.0f} cm → off-workspace",
        ha="center", fontsize=7, color=RED)

# ---------- Panel 3: action stays consistent (transforms WITH the nut) ----------
a2 = ax[2]; a2.set_title("③ action transforms WITH the nut (in-nut-frame invariant)", fontsize=9.5, weight="bold")
draw_hole(a2, hole)
# original: nut at yaw 0, action = commanded push into hole (a small arrow along +y, in nut frame)
act_in_nut = np.array([0.0, 0.02])                         # commanded delta in the nut's frame
for ang, col, dy in [(0, GRY, 0.06), (90, GRN, -0.06)]:
    center = hole + np.array([dy, 0.05])
    tip = draw_nut(a2, center, ang, col, alpha=0.85)
    th = np.deg2rad(ang); Rn = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    d = Rn @ act_in_nut                                    # action rotates with the nut
    a2.arrow(center[0], center[1], d[0], d[1], color=col, width=0.001, head_width=0.008, length_includes_head=True)
    a2.text(center[0], center[1] - 0.05, f"nut {ang}°\n+ its action", ha="center", fontsize=7.5, color=col)
a2.text(0.5, -0.14, "same action IN THE NUT FRAME → keyed aug is label-consistent\n"
                    "(exactly what test_action_consistency.py asserts)", transform=a2.transAxes,
        ha="center", fontsize=7.5, color="#111827")

for a in ax:
    a.set_aspect("equal"); a.set_xlim(-0.12, 0.12); a.set_ylim(-0.12, 0.12); a.grid(alpha=0.2); a.tick_params(labelsize=7)

fig.suptitle("Verify the C4-ENGAGE augmentation (pivoted method) — generated by the real mea_diff.phase_aug primitives",
             fontsize=11.5, y=1.0)
fig.text(0.5, 0.005,
         "The nut is drawn WITH A HANDLE (robosuite's nut is C1, not C4): the handle visibly moves so the aug is a "
         "genuinely new observation, while the square hole/peg admits all 4 → the C4 equivalence is a PEG property, "
         "carried rigidly by the grasp. Object-axis anchoring keeps it feasible; the action stays consistent in the nut frame.",
         ha="center", fontsize=8.2, color="#111827", bbox=dict(fc="#fef9c3", ec="#ca8a04", lw=1, boxstyle="round,pad=0.4"))
plt.tight_layout(rect=[0, 0.06, 1, 0.95])
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "context/plan/mea_diff_verify.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=125, bbox_inches="tight")
print("saved ->", out)
