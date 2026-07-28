# -*- coding: utf-8 -*-
"""Visualize what MEA-v2 changes on the DATA side vs the EquiDiff baseline, using the
REAL augmentor (mea_diff.phase_aug) on a synthetic nut-insertion episode.

3 panels (top-down x-y):
  BASELINE (off)          the demo as-is (what EquiDiff trains on)
  GLOBAL (redundant)      one whole-scene SO(2) rotation — what EquiDiff's equivariance
                          ALREADY gives for free -> augmenting with it adds nothing
  MEA phase (non-redund.) APPROACH: gripper approaches the (fixed) nut from many angles
                          alpha; ENGAGE: the grasped nut+gripper inserts in the keyed C4
                          orientations into the FIXED hole -> information global
                          equivariance cannot express.

  source bash/init.sh && python mea_diff/viz_mea_diff_aug.py
"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from phase_aug import (MEAPhaseAug, make_synthetic_episode, sixd_to_mat,
                       rot_z, transform_point, APPROACH, ENGAGE)

GRN, BLU, RED, GRY, ORN = "#16a34a", "#0ea5e9", "#dc2626", "#9ca3af", "#f59e0b"


def draw_scene(ax, ep, nut_c="#3b82f6", traj=True, alpha=1.0, label_hole=True):
    hole = ep["hole_xy"]
    # hole (fixed target): square outline
    ax.add_patch(plt.Rectangle((hole[0] - 0.02, hole[1] - 0.02), 0.04, 0.04,
                               fill=False, ec=RED, lw=2, alpha=alpha))
    if label_hole:
        ax.text(hole[0], hole[1] + 0.035, "hole (fixed)", ha="center", fontsize=7, color=RED)
    if traj:
        appr = ep["phase"] == APPROACH
        eng = ep["phase"] == ENGAGE
        ax.plot(ep["eef_pos"][appr, 0], ep["eef_pos"][appr, 1], "-o", ms=3, color=GRN, alpha=alpha)
        ax.plot(ep["eef_pos"][eng, 0], ep["eef_pos"][eng, 1], "-s", ms=3, color=BLU, alpha=alpha)
    # nut at its approach-phase resting pose (a small square, oriented)
    op = ep["obj_pos"][0]
    ax.add_patch(plt.Rectangle((op[0] - 0.018, op[1] - 0.018), 0.036, 0.036,
                               fc=nut_c, ec="#1e3a8a", lw=1.2, alpha=alpha))


def keyed_glyph(ax, center, ang_deg, color):
    """a square nut + a short gripper tick at the hole, rotated by ang_deg (C4 variant)."""
    th = np.deg2rad(ang_deg); R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    corners = (R @ (np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]).T * 0.02)).T + center
    ax.add_patch(plt.Polygon(corners, closed=True, fc="none", ec=color, lw=1.6))
    tick = R @ np.array([0, 0.03])
    ax.plot([center[0], center[0] + tick[0]], [center[1], center[1] + tick[1]], color=color, lw=1.6)


def setup(ax, title):
    ax.set_xlim(-0.22, 0.22); ax.set_ylim(-0.20, 0.20); ax.set_aspect("equal")
    ax.set_title(title, fontsize=10, weight="bold"); ax.grid(alpha=0.2)
    ax.tick_params(labelsize=7)


ep = make_synthetic_episode(T=16, k_grasp=6, seed=0)
fig, ax = plt.subplots(1, 3, figsize=(15, 5.2))

# ---- BASELINE ----
setup(ax[0], "BASELINE (EquiDiff trains on this)")
draw_scene(ax[0], ep)
ax[0].plot([], [], "-o", color=GRN, label="approach")
ax[0].plot([], [], "-s", color=BLU, label="engage / insert")
ax[0].legend(fontsize=7, loc="lower left")
ax[0].text(0, -0.185, "nut → grasp → insert into hole", ha="center", fontsize=8, color=GRY)

# ---- GLOBAL (redundant) ----
setup(ax[1], "GLOBAL SO(2) aug  (REDUNDANT w/ EquiDiff)")
for i, seed in enumerate([11, 22, 33]):
    g = MEAPhaseAug(mode="global", rng=np.random.default_rng(seed)).augment(
        {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ep.items()})
    draw_scene(ax[1], g, nut_c=["#93c5fd", "#60a5fa", "#3b82f6"][i], alpha=0.6, label_hole=(i == 0))
ax[1].text(0, -0.185, "whole scene (nut+hole+path) rotates rigidly\n"
                      "= what the equivariant net already covers → no new info",
           ha="center", fontsize=7.6, color=GRY)

# ---- MEA phase (non-redundant) ----
setup(ax[2], "MEA phase aug  (NON-redundant)")
draw_scene(ax[2], ep, traj=False)                     # fixed nut + hole reference
# APPROACH fan: rotate the gripper path about the FIXED nut by a range of angles alpha,
# decaying to 0 at grasp (uses the real transform primitives -> converges at the nut).
nut0 = ep["obj_pos"][0]
appr = np.where(ep["phase"] == APPROACH)[0]
k = ep["k_grasp"]
for a_deg in [-75, -45, -20, 0, 20, 45, 75]:
    path = []
    for t in appr:
        R = rot_z(np.deg2rad(a_deg) * max(0.0, (k - t) / k))   # decaying schedule
        path.append(transform_point(ep["eef_pos"][t], R, nut0[:2])[:2])
    path = np.array(path)
    ax[2].plot(path[:, 0], path[:, 1], "-", lw=1.1, color=GRN, alpha=0.6)
    ax[2].plot(path[0, 0], path[0, 1], "o", ms=3, color=GRN, alpha=0.7)   # start
ax[2].plot(nut0[0], nut0[1], "*", ms=13, color="#065f46")                 # grasp convergence
ax[2].text(nut0[0] - 0.02, nut0[1] - 0.055, "approach angle α\n(nut FIXED, converge to grasp)",
           ha="center", fontsize=7.2, color=GRN)
# ENGAGE keyed C4: 4 equivalent insertion orientations at the FIXED hole
for j, ang in enumerate([0, 90, 180, 270]):
    keyed_glyph(ax[2], ep["hole_xy"], ang, ORN)
ax[2].text(ep["hole_xy"][0], ep["hole_xy"][1] - 0.055, "keyed C4 insert\n(hole FIXED)",
           ha="center", fontsize=7.4, color=ORN)

fig.suptitle("MEA v2 — DATA-side change vs baseline  (generated by the real mea_diff.phase_aug on a "
             "synthetic nut-insertion episode)", fontsize=11.5, y=1.0)
fig.text(0.5, 0.005,
         "APPROACH: rotate GRIPPER about the fixed nut by α (green fan) → new relative approach angles.   "
         "ENGAGE: rotate the grasped nut+gripper about the fixed hole by the object POINT-GROUP (square→C4, "
         "orange) → the keyed equivalent insertions.   Neither is a whole-scene rotation → neither is covered "
         "by EquiDiff's global SO(2) equivariance.",
         ha="center", fontsize=8.3, color="#111827",
         bbox=dict(fc="#fef9c3", ec="#ca8a04", lw=1, boxstyle="round,pad=0.4"))
plt.tight_layout(rect=[0, 0.06, 1, 0.96])
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "context/plan/mea_diff_aug.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=125, bbox_inches="tight")
print("saved ->", out)
