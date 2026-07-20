# -*- coding: utf-8 -*-
"""Three SEPARATE baseline-vs-augmentation trajectory comparisons, one per
mea_v2 change (see viz_mea_v2_trajs.py for the combined view):

  1_rotation.png    BASELINE vs continuous global rotation (theta=+60, no mirror)
  2_reflection.png  BASELINE vs pure mirror (theta=0, reflect=True)
  3_action.png      the action-LABEL transform in isolation: the same rotated
                    clone drawn twice -- red arrows = labels NOT transformed
                    (what naive copying would train on; mismatch the scene),
                    green arrows = conjugate-transformed labels (v2 fix).

All clones go through the real training code path (generate_sym_v2, global
mode, origin anchor). Overlays: red + = image center (= gripper in every real
obs, camera is gripper-centered); white o = gripper centroid; arrow = action
drawn in the pc frame via M @ a[1:3], M = [[0,1],[1,0]] (measured world->pc
axis swap, check_pc_action_frame.py). The camera rides on the gripper, so the
blocks stream OPPOSITE a consistent arrow.

  source bash/init.sh
  python bash/viz_mea_v2_compare.py --ep ep_v2.pkl --outdir context/plan
"""
import argparse
import os
import numpy as np

PC_MIN, PC_RANGE, RES = -0.2, 0.4, 200   # Occup wrapper window (block_pull)
THETA = np.pi / 3                        # the rotation used in figs 1 and 3
GREEN, RED = "#22c55e", "#ef4444"


def to_px(xy):
    p = (np.asarray(xy, float) - PC_MIN) / PC_RANGE * RES
    return p[0], p[1]


def grip_xy(obs_frame):
    pc = obs_frame.get("pc", {}).get("gripper")
    if pc is None or len(pc) == 0:
        return None
    return np.asarray(pc)[:, :2].mean(axis=0)


def plot_rows(rows, out, title):
    """rows: list of (label, obs_list, action_list, arrow_color)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = len(rows[0][1])
    K = 2.5  # arrow exaggeration
    S = np.array([[0.0, 1.0], [1.0, 0.0]])   # world->pc swap
    fig, ax = plt.subplots(len(rows), T, figsize=(1.55 * T, 1.8 * len(rows)),
                           squeeze=False)
    for ri, (label, o, a, col) in enumerate(rows):
        for t in range(T):
            axc = ax[ri, t]
            axc.imshow(o[t]["occup_image"], cmap="viridis")
            axc.plot(RES / 2, RES / 2, "r+", ms=8, mew=1.5)
            g = grip_xy(o[t])
            if g is not None:
                c, r = to_px(g)
                axc.plot(c, r, "o", ms=7, mfc="none", mec="w", mew=1.3)
                if t < len(a):
                    d = S @ np.asarray(a[t], float)[1:3] / PC_RANGE * RES * K
                    if np.hypot(*d) > 1:
                        axc.arrow(c, r, d[0], d[1], color=col,
                                  width=0.6, head_width=5,
                                  length_includes_head=True)
            axc.set_xticks([]); axc.set_yticks([])
            if ri == 0:
                axc.set_title(f"t={t}", fontsize=8)
            if t == 0:
                axc.set_ylabel(label, fontsize=8)
    fig.suptitle(title, fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(out, dpi=115)
    plt.close(fig)
    print(f"Saved -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", default="ep_v2.pkl")
    ap.add_argument("--task", default="block_pull")
    ap.add_argument("--outdir", default="context/plan")
    args = ap.parse_args()

    import joblib
    from rgbd_sym.tool.sym_v2 import generate_sym_v2
    from rgbd_sym.env.embodied.dummy.env import DummyEnv
    from rgbd_sym.env.wrapper.occup import Occup

    assert os.path.exists(args.ep), f"missing {args.ep} (run diagnose_mea_v2.py first)"
    data = joblib.load(args.ep)
    obs, actions = data["obs"], data["actions"]
    dummy = Occup(DummyEnv(task=args.task))

    o_rot, a_rot = generate_sym_v2(obs, actions, dummy_env=dummy,
                                   mode="global", theta_global=THETA, reflect=False)
    o_mir, a_mir = generate_sym_v2(obs, actions, dummy_env=dummy,
                                   mode="global", theta_global=0.0, reflect=True)

    deg = f"{np.degrees(THETA):+.0f}"
    plot_rows(
        [("BASELINE\n(real demo)", obs, actions, GREEN),
         (f"AUG rotation\nθ={deg}°", o_rot, a_rot, GREEN)],
        os.path.join(args.outdir, "viz_cmp_1_rotation.png"),
        "Change 1 / 3 -- CONTINUOUS ROTATION: scene + action rotate together about the pc origin\n"
        "(= gripper = image center).  red + center | white o gripper | arrow = action in pc frame")

    plot_rows(
        [("BASELINE\n(real demo)", obs, actions, GREEN),
         ("AUG mirror\n(θ=0)", o_mir, a_mir, GREEN)],
        os.path.join(args.outdir, "viz_cmp_2_reflection.png"),
        "Change 2 / 3 -- REFLECTION: scene mirrored (pc x-flip = world y-mirror), action a[2] and\n"
        "dyaw flip with it. The one symmetry the C4 net lacks (flip_symmetry=false).")

    plot_rows(
        [("BASELINE\n(real demo)", obs, actions, GREEN),
         ("AUG obs +\nRAW labels\n(wrong)", o_rot, actions, RED),
         ("AUG obs +\ntransformed\nlabels (v2)", o_rot, a_rot, GREEN)],
        os.path.join(args.outdir, "viz_cmp_3_action.png"),
        f"Change 3 / 3 -- ACTION-LABEL TRANSFORM (same θ={deg}° clone twice): red = labels left\n"
        "as-is, arrows mismatch the rotated scene; green = conjugated by M (rotate -θ), consistent.")


if __name__ == "__main__":
    main()
