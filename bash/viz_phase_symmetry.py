# -*- coding: utf-8 -*-
"""Task-selection explainer: WHERE the valid symmetry group is phase-dependent.

Row 1 TABLETOP (object stays on table: pull/push/pick/drawer):
  every phase has the SAME group SE(2) (yaw + planar translation). pitch/roll is
  ALWAYS broken (table+gravity, in approach too); z-rotation ALWAYS valid.
  -> no phase change -> no contextual degeneration -> baseline correct everywhere.
Row 2 PHASE-CHANGING (peg-in-hole / in-air 6-DoF reorientation):
  free-space approach has free 3D orientation (SO(3)/SE(3)); contact/insertion
  CONSTRAINS it (-> about the hole axis / SE(2)). The group genuinely DEGENERATES
  -> this is where context-conditioned equivariance has real footing.

  source bash/init.sh && python bash/viz_phase_symmetry.py
"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrow, Arc, Circle

GRN, RED, BLU, GRY, ORN = "#16a34a", "#dc2626", "#0ea5e9", "#6b7280", "#b45309"


def setup(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal"); ax.axis("off")


def table(ax, y=2.2, x0=0.6, x1=9.4, color="#9ca3af"):
    ax.plot([x0, x1], [y, y], color=color, lw=3, zorder=1)
    for xx in np.linspace(x0 + 0.3, x1 - 0.3, 8):
        ax.plot([xx, xx - 0.3], [y, y - 0.5], color=color, lw=1, zorder=1)


def block(ax, cx, cy, s, yaw, fc="#3b82f6", ec="#1e3a8a"):
    th = np.deg2rad(yaw)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    pts = (R @ (np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]).T * s / 2)).T + [cx, cy]
    ax.add_patch(plt.Polygon(pts, closed=True, fc=fc, ec=ec, lw=1.4, zorder=3))


def gripper(ax, cx, cy, yaw, color="#111827", size=0.8):
    th = np.deg2rad(yaw); R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    for dx in (-0.45, 0.45):
        a = R @ np.array([dx * size, -0.5 * size]); b = R @ np.array([dx * size, 0.5 * size])
        ax.plot([cx + a[0], cx + b[0]], [cy + a[1], cy + b[1]], color=color, lw=2.4, zorder=4)
    a = R @ np.array([-0.45 * size, 0.5 * size]); b = R @ np.array([0.45 * size, 0.5 * size])
    ax.plot([cx + a[0], cx + b[0]], [cy + a[1], cy + b[1]], color=color, lw=2.4, zorder=4)


def yaw_arc(ax, cx, cy, r=0.95, color=GRN):
    ax.add_patch(Arc((cx, cy), 2 * r, 2 * r, theta1=20, theta2=320, color=color, lw=2))
    ax.annotate("", xy=(cx + r * np.cos(np.deg2rad(20)), cy + r * np.sin(np.deg2rad(20))),
                xytext=(cx + r * np.cos(np.deg2rad(35)), cy + r * np.sin(np.deg2rad(35))),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2))


def pitch_cross(ax, cx, cy, r=0.9):
    # curved pitch/roll arrow, struck out red
    ax.add_patch(Arc((cx, cy), 2 * r, 1.0 * r, theta1=200, theta2=340, color=GRY, lw=1.8))
    ax.plot([cx - 0.7, cx + 0.7], [cy - 0.7, cy + 0.7], color=RED, lw=2.4)
    ax.plot([cx - 0.7, cx + 0.7], [cy + 0.7, cy - 0.7], color=RED, lw=2.4)


fig = plt.figure(figsize=(15, 8.6))
gs = fig.add_gridspec(2, 3, hspace=0.16, wspace=0.12)

# ---------- ROW 1: TABLETOP (SE(2) throughout) ----------
titles1 = ["APPROACH (free)", "CONTACT / GRASP", "MANIPULATE (on table)"]
for j in range(3):
    ax = fig.add_subplot(gs[0, j]); setup(ax)
    ax.add_patch(FancyBboxPatch((0.2, 0.2), 9.6, 9.6, boxstyle="round,pad=0.02", fc="#f0fdf4", ec=GRN, lw=1.8))
    table(ax)
    block(ax, 5.0, 2.9, 1.2, 0)
    if j == 0:
        gripper(ax, 3.2, 6.6, 30); ax.add_patch(FancyArrow(3.5, 6.1, 1.1, -2.4, width=0.02, head_width=0.3, color=GRN, length_includes_head=True))
    elif j == 1:
        gripper(ax, 5.0, 4.0, 0)
    else:
        gripper(ax, 5.0, 4.0, 0); ax.add_patch(FancyArrow(5.6, 2.9, 2.2, 0, width=0.02, head_width=0.3, color=GRN, length_includes_head=True))
    yaw_arc(ax, 5.0, 2.9)
    pitch_cross(ax, 8.1, 6.6)
    ax.text(8.1, 8.0, "pitch/roll\nX (table)", ha="center", fontsize=6.8, color=RED)
    ax.text(2.0, 8.4, "yaw ✓", ha="center", fontsize=8, color=GRN, weight="bold")
    ax.text(5.0, 9.15, titles1[j], ha="center", fontsize=9.2, weight="bold")
    ax.text(5.0, 0.75, "group = SE(2)", ha="center", fontsize=8.5, color=GRN, weight="bold")

# ---------- ROW 2: PHASE-CHANGING (peg-in-hole) ----------
titles2 = ["APPROACH in air (free 3D orient.)", "ALIGN to hole", "INSERT (constrained)"]
ecs = [GRN, ORN, RED]
for j in range(3):
    ax = fig.add_subplot(gs[1, j]); setup(ax)
    ax.add_patch(FancyBboxPatch((0.2, 0.2), 9.6, 9.6, boxstyle="round,pad=0.02", fc="#eff6ff", ec=ecs[j], lw=1.8))
    # base with a hole
    ax.add_patch(Rectangle((2.2, 2.0), 5.6, 1.4, fc="#9ca3af", ec="#4b5563", zorder=1))
    ax.add_patch(Rectangle((4.6, 2.0), 0.8, 1.4, fc="#eff6ff", ec="#4b5563", zorder=2))  # hole
    if j == 0:  # in air, free orientation
        block(ax, 4.0, 7.2, 1.0, 35, fc="#f59e0b", ec="#92400e")  # peg tilted (allowed in air)
        gripper(ax, 4.0, 7.2, 35, size=0.7)
        # SO(3) freedom: show multiple orientation ghosts
        for a in (-25, 55):
            th = np.deg2rad(a)
            ax.add_patch(Arc((4.0, 7.2), 2.6, 2.6, theta1=a - 8, theta2=a + 8, color=BLU, lw=1.4))
        ax.text(7.1, 7.4, "SO(3):\nany 3D\norientation ✓", ha="center", fontsize=7.2, color=BLU, weight="bold")
    elif j == 1:
        block(ax, 5.0, 5.6, 1.0, 8, fc="#f59e0b", ec="#92400e"); gripper(ax, 5.0, 5.6, 8, size=0.7)
        ax.add_patch(FancyArrow(5.0, 5.0, 0, -1.2, width=0.02, head_width=0.28, color=ORN, length_includes_head=True))
        ax.text(7.4, 6.0, "orientation\nnarrowing", ha="center", fontsize=7.2, color=ORN)
    else:  # inserted, constrained
        block(ax, 5.0, 3.4, 1.0, 0, fc="#f59e0b", ec="#92400e"); gripper(ax, 5.0, 4.6, 0, size=0.7)
        pitch_cross(ax, 7.6, 6.4)
        ax.text(7.6, 7.9, "pitch/roll X\n(hole axis)", ha="center", fontsize=7, color=RED)
        ax.text(3.0, 6.6, "only\nabout-axis\n(SE(2)/1-DoF)", ha="center", fontsize=7.2, color=RED, weight="bold")
    ax.text(5.0, 9.15, titles2[j], ha="center", fontsize=8.8, weight="bold")
    grp = ["group = SE(3) / SO(3)", "SO(3) → narrowing", "group = about-axis (reduced)"][j]
    ax.text(5.0, 0.75, grp, ha="center", fontsize=8.3, color=ecs[j], weight="bold")

fig.text(0.5, 0.965, "Where is the valid symmetry group PHASE-DEPENDENT?  (task selection for MEA v2)",
         ha="center", fontsize=12.5, weight="bold")
fig.text(0.012, 0.72, "TABLETOP\n(pull/push/\npick/drawer)", fontsize=10, weight="bold", color=GRN,
         rotation=90, va="center", ha="center")
fig.text(0.012, 0.28, "PHASE-CHANGING\n(peg-in-hole /\nin-air 6-DoF)", fontsize=10, weight="bold", color=RED,
         rotation=90, va="center", ha="center")
fig.text(0.5, 0.015,
         "TABLETOP: object on the table in EVERY phase → yaw always valid, pitch/roll always broken → group = SE(2) "
         "throughout → NO phase change → baseline correct everywhere → contextual degeneration has no footing "
         "(this session's null root cause).\n"
         "PHASE-CHANGING: free 3D approach orientation (SO(3)) genuinely COLLAPSES at insertion → the group changes "
         "across phases → contextual-equivariant net has real footing. Pick a task from HERE.",
         ha="center", va="bottom", fontsize=8.4, color="#111827",
         bbox=dict(fc="#fef9c3", ec="#ca8a04", lw=1, boxstyle="round,pad=0.4"))

plt.tight_layout(rect=[0.03, 0.055, 1, 0.95])
out = "context/plan/phase_symmetry_task_selection.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=125, bbox_inches="tight")
print("saved ->", out)
