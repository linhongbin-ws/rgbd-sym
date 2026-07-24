# -*- coding: utf-8 -*-
"""Visualize the AUGMENTATION MECHANISM: baseline (seq_rot) vs mea_v2.

A 2x4 matrix on one representative frame:
    rows  = [ baseline (seq_rot, 2D raster warp) | mea_v2 (pc rotate + re-voxelize) ]
    cols  = [ original | rotation theta | reflection | rotation+reflection ]

The rotation column of both rows is angle-matched (seq_rot's theta is searched to
best overlap mea_v2's rendered rotation) so the eye sees they are the SAME
symmetry -- redundant. The reflection columns of the baseline row are struck out:
seq_rot is a pure rotation (utils.helpers.perturb, no det=-1 term) and the C4 net
has flip_symmetry=false, so reflection is the ONE symmetry only mea_v2 supplies.

Both paths run their REAL training code:
  baseline -> utils.helpers.perturb (the seq_rot buffer's per-frame op)
  mea_v2   -> generate_sym_v2 global (pc re-render through Occup(DummyEnv))
Action overlays reuse the validated convention from viz_mea_v2_compare.py:
  red + = image center (= gripper), white o = gripper centroid,
  arrow = action in the pc frame via S @ a[1:3], S = [[0,1],[1,0]].

  source bash/init.sh
  python bash/viz_aug_mechanism.py --ep $TMPDIR/ep_v2img.pkl \
      --out context/plan/viz_aug_mechanism.png
"""
import argparse
import os
import numpy as np

PC_MIN, PC_RANGE, RES = -0.2, 0.4, 200
THETA = np.pi / 3
S = np.array([[0.0, 1.0], [1.0, 0.0]])
GREEN, RED, GREY = "#22c55e", "#ef4444", "#9ca3af"


def to_px(xy):
    p = (np.asarray(xy, float) - PC_MIN) / PC_RANGE * RES
    return p[0], p[1]


def grip_xy(obs_frame):
    pc = obs_frame.get("pc", {}).get("gripper")
    if pc is None or len(pc) == 0:
        return None
    return np.asarray(pc)[:, :2].mean(axis=0)


def seq_rot_frame(img, action, theta, perturb):
    """One frame through the seq_rot buffer op: rotate the raster about center
    by theta (translation zeroed, as in SeqRotBuffer._augment_and_add_episodes),
    and rotate the action's (dx, dy). Returns (rot_img, new_action)."""
    pivot = (img.shape[1] / 2, img.shape[0] / 2)
    rot_img, _, rot_dxy, _ = perturb(
        img.copy(), None, np.asarray(action, float)[1:3].copy(),
        theta, [0.0, 0.0], pivot, set_trans_zero=True)
    new_a = np.asarray(action, float).copy()
    new_a[1:3] = rot_dxy
    return rot_img, new_a


def match_seq_theta(base_img, target_img, action, perturb):
    """Find the seq_rot theta whose rotated raster best overlaps target_img
    (mea_v2's rendered rotation), so the two rotation panels are angle-aligned.
    Foreground L1 over object pixels; pure geometry, no convention assumed."""
    tgt = np.asarray(target_img, float)
    fg = tgt < np.median(tgt) - 1e-3
    best = (None, np.inf)
    for th in np.linspace(-np.pi, np.pi, 145):     # 2.5 deg steps
        r, _ = seq_rot_frame(base_img, action, th, perturb)
        err = np.abs(r - tgt)[fg].mean() if fg.any() else np.abs(r - tgt).mean()
        if err < best[1]:
            best = (th, err)
    return best[0]


def draw(ax, img, obs_frame, action, arrow_col, struck=False):
    import matplotlib.pyplot as plt
    if struck:
        ax.imshow(img, cmap="gray", alpha=0.25)
        ax.plot([0, RES], [0, RES], color=RED, lw=2)
        ax.plot([0, RES], [RES, 0], color=RED, lw=2)
        ax.text(RES / 2, RES / 2, "seq_rot\nhas no\nreflection",
                ha="center", va="center", fontsize=8, color=RED, weight="bold")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(0, RES); ax.set_ylim(RES, 0)
        return
    ax.imshow(img, cmap="viridis")
    ax.plot(RES / 2, RES / 2, "r+", ms=9, mew=1.6)
    if obs_frame is not None:
        g = grip_xy(obs_frame)
        if g is not None:
            c, r = to_px(g)
            ax.plot(c, r, "o", ms=7, mfc="none", mec="w", mew=1.3)
            if action is not None:
                d = S @ np.asarray(action, float)[1:3] / PC_RANGE * RES * 2.5
                if np.hypot(*d) > 1:
                    ax.arrow(c, r, d[0], d[1], color=arrow_col, width=0.7,
                             head_width=6, length_includes_head=True)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", default="ep_v2img.pkl")
    ap.add_argument("--task", default="block_pull")
    ap.add_argument("--out", default="context/plan/viz_aug_mechanism.png")
    ap.add_argument("--frame", type=int, default=None)
    args = ap.parse_args()

    import joblib
    from rgbd_sym.tool.sym_v2 import generate_sym_v2
    from rgbd_sym.env.embodied.dummy.env import DummyEnv
    from rgbd_sym.env.wrapper.occup import Occup
    from utils.helpers import perturb

    data = joblib.load(args.ep)
    obs, actions = data["obs"], data["actions"]
    dummy = Occup(DummyEnv(task=args.task))

    # pick a frame with a clear action and gripper present
    t = args.frame
    if t is None:
        mags = [np.hypot(*np.asarray(a, float)[1:3]) for a in actions]
        t = int(np.argmax(mags))
    print(f"frame t={t}/{len(obs)-1}, action={np.asarray(actions[t],float).round(3)}")

    frame_obs = obs[t]
    frame_act = actions[t] if t < len(actions) else np.zeros(5)
    base_img = np.asarray(frame_obs["occup_image"], float)

    # ---- mea_v2 (pc re-render): rotation, reflection, rotation+reflection ----
    def v2(theta, reflect):
        o, a = generate_sym_v2(obs, actions, dummy_env=dummy, mode="global",
                               anchor="origin", theta_global=theta, reflect=reflect)
        return o[t], (a[t] if t < len(a) else np.zeros(5))
    o_rot, a_rot = v2(THETA, False)
    o_ref, a_ref = v2(0.0, True)
    o_rr, a_rr = v2(THETA, True)

    # ---- baseline (seq_rot): rotation only; match angle to mea_v2's rotation ----
    th_seq = match_seq_theta(base_img, o_rot["occup_image"], frame_act, perturb)
    b_rot_img, b_rot_a = seq_rot_frame(base_img, frame_act, th_seq, perturb)
    print(f"matched seq_rot theta = {np.degrees(th_seq):+.1f} deg "
          f"(mea_v2 theta = {np.degrees(THETA):+.1f} deg)")

    # quantify the redundancy claim: baseline-rotation vs mea_v2-rotation overlap
    A = np.asarray(b_rot_img, float); B = np.asarray(o_rot["occup_image"], float)
    fa, fb = A < np.median(A) - 1e-3, B < np.median(B) - 1e-3
    iou = (fa & fb).sum() / (fa | fb).sum() if (fa | fb).any() else 1.0
    print(f"baseline-rot vs mea_v2-rot foreground IoU = {iou:.3f} "
          f"(-> the rotation is the SAME symmetry; only mechanism differs)")

    # ---- render 2x4 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 4, figsize=(13, 6.8))
    cols = ["original\n(real demo)", f"ROTATION  θ≈{np.degrees(THETA):+.0f}°",
            "REFLECTION", f"ROTATION+REFLECTION"]

    # baseline row
    draw(ax[0, 0], base_img, frame_obs, frame_act, GREEN)
    draw(ax[0, 1], b_rot_img, None, None, GREEN)
    # baseline arrow for rotation: overlay gripper-centered rotated action
    ax[0, 1].plot(RES / 2, RES / 2, "r+", ms=9, mew=1.6)
    d = S @ np.asarray(b_rot_a, float)[1:3] / PC_RANGE * RES * 2.5
    if np.hypot(*d) > 1:
        ax[0, 1].arrow(RES / 2, RES / 2, d[0], d[1], color=GREEN, width=0.7,
                       head_width=6, length_includes_head=True)
    draw(ax[0, 2], base_img, None, None, GREY, struck=True)
    draw(ax[0, 3], base_img, None, None, GREY, struck=True)

    # mea_v2 row
    draw(ax[1, 0], base_img, frame_obs, frame_act, GREEN)
    draw(ax[1, 1], o_rot["occup_image"], o_rot, a_rot, GREEN)
    draw(ax[1, 2], o_ref["occup_image"], o_ref, a_ref, GREEN)
    draw(ax[1, 3], o_rr["occup_image"], o_rr, a_rr, GREEN)

    for j, c in enumerate(cols):
        ax[0, j].set_title(c, fontsize=10)
    ax[0, 0].set_ylabel("BASELINE\nseq_rot\n(2D raster warp)", fontsize=10)
    ax[1, 0].set_ylabel("mea_v2\npc rotate +\nre-voxelize", fontsize=10)

    fig.suptitle(
        "Augmentation mechanism: baseline (seq_rot) vs mea_v2  —  block_pull, one frame\n"
        "ROTATION is the SAME symmetry in both (redundant); only mea_v2 adds REFLECTION "
        "(the C4 net's missing flip). Baseline warps the raster; mea_v2 re-renders the cloud.",
        fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=125)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
