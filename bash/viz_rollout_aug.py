# -*- coding: utf-8 -*-
"""Rollout-level augmentation comparison: baseline (seq_rot) vs mea_v2, across
every frame of one expert rollout. Default task: block_pick.

Rows (all frames of the SAME rollout):
  real demo                 the captured expert trajectory
  baseline  (seq_rot)       2D raster rotation, ONE angle for the whole episode
                            (utils.helpers.perturb, the SeqRotBuffer op)
  mea_v2  rotation          pc rotate about the vertical axis + re-voxelize,
                            SAME visible angle as the baseline row (redundant)
  mea_v2  rotation+reflect  + the mirror the C4 net structurally lacks
                            (flip_symmetry=false) -- mea_v2's only NEW symmetry
Columns = time steps. Overlays (validated convention, viz_mea_v2_compare.py):
  red + = image center (= gripper, camera is gripper-centered),
  white o = gripper centroid, arrow = action in the pc frame via S @ a[1:3],
  S = [[0,1],[1,0]].

  source bash/init.sh
  python bash/viz_rollout_aug.py --task block_pick \
      --ep $TMPDIR/ep_block_pick.pkl --out context/plan/viz_rollout_aug_block_pick.png
Collects the rollout with query_expert if the pickle is absent.
"""
import argparse
import os
import numpy as np

PC_MIN, PC_RANGE, RES = -0.2, 0.4, 200
THETA = np.pi / 3
S = np.array([[0.0, 1.0], [1.0, 0.0]])
GREEN, BLUE, PURPLE = "#22c55e", "#0ea5e9", "#a855f7"


def to_px(xy):
    p = (np.asarray(xy, float) - PC_MIN) / PC_RANGE * RES
    return p[0], p[1]


def grip_xy(obs_frame):
    pc = obs_frame.get("pc", {}).get("gripper")
    if pc is None or len(pc) == 0:
        return None
    return np.asarray(pc)[:, :2].mean(axis=0)


def collect_episode(task, seed, max_steps=100):
    from rgbd_sym.env.embodied import PomdpEnv
    from rgbd_sym.env.wrapper.occup import Occup
    env = Occup(PomdpEnv(task=task))
    env.seed = seed
    obs = env.reset()
    obss, actions = [obs], []
    done = False
    while not done and len(actions) < max_steps:
        a = np.asarray(env.query_expert(0), float)
        obs, r, d, info = env.step(a)
        done = d
        actions.append(a); obss.append(obs)
    return obss, actions


def seq_rot_frame(img, action, theta, perturb):
    pivot = (img.shape[1] / 2, img.shape[0] / 2)
    r, _, rot_dxy, _ = perturb(img.copy(), None, np.asarray(action, float)[1:3].copy(),
                               theta, [0.0, 0.0], pivot, set_trans_zero=True)
    a = np.asarray(action, float).copy(); a[1:3] = rot_dxy
    return r, a


def match_seq_theta(base_img, target_img, perturb):
    tgt = np.asarray(target_img, float)
    fg = tgt < np.median(tgt) - 1e-3
    best = (0.0, np.inf)
    for th in np.linspace(-np.pi, np.pi, 145):
        r, _ = seq_rot_frame(base_img, np.zeros(5), th, perturb)
        err = np.abs(r - tgt)[fg].mean() if fg.any() else np.abs(r - tgt).mean()
        if err < best[1]:
            best = (th, err)
    return best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="block_pick")
    ap.add_argument("--ep", default="ep_block_pick.pkl")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="context/plan/viz_rollout_aug_block_pick.png")
    ap.add_argument("--maxcols", type=int, default=8)
    args = ap.parse_args()

    import joblib
    from rgbd_sym.tool.sym_v2 import generate_sym_v2
    from rgbd_sym.env.embodied.dummy.env import DummyEnv
    from rgbd_sym.env.wrapper.occup import Occup
    from utils.helpers import perturb

    if os.path.exists(args.ep):
        data = joblib.load(args.ep); obs, actions = data["obs"], data["actions"]
    else:
        obs, actions = collect_episode(args.task, args.seed)
        joblib.dump({"obs": obs, "actions": actions}, args.ep)
    print(f"{args.task}: {len(obs)} frames, {len(actions)} actions")

    dummy = Occup(DummyEnv(task=args.task))

    # mea_v2 clones (whole episode, one theta)
    o_rot, a_rot = generate_sym_v2(obs, actions, dummy_env=dummy, mode="global",
                                   anchor="origin", theta_global=THETA, reflect=False)
    o_rr, a_rr = generate_sym_v2(obs, actions, dummy_env=dummy, mode="global",
                                 anchor="origin", theta_global=THETA, reflect=True)
    # baseline seq_rot: match the visible angle of the mea_v2 rotation on a mid frame
    mid = len(obs) // 2
    th_seq = match_seq_theta(obs[mid]["occup_image"], o_rot[mid]["occup_image"], perturb)
    print(f"matched seq_rot theta={np.degrees(th_seq):+.1f} (mea_v2 {np.degrees(THETA):+.0f})")
    b_imgs, b_acts = [], []
    for i, o in enumerate(obs):
        act = actions[i] if i < len(actions) else np.zeros(5)
        bi, ba = seq_rot_frame(np.asarray(o["occup_image"], float), act, th_seq, perturb)
        b_imgs.append(bi); b_acts.append(ba)

    # ---- assemble rows: (label, img_getter, obs_for_gripper, actions, color) ----
    T = min(len(obs), args.maxcols)
    idx = np.linspace(0, len(obs) - 1, T).astype(int)

    rows = [
        ("real demo", [obs[i]["occup_image"] for i in idx],
         [obs[i] for i in idx], [actions[i] if i < len(actions) else None for i in idx], GREEN),
        ("baseline\nseq_rot\n(raster)", [b_imgs[i] for i in idx],
         [None] * T, [b_acts[i] if i < len(actions) else None for i in idx], BLUE),
        (f"mea_v2\nrotation\nθ≈{np.degrees(THETA):+.0f}°", [o_rot[i]["occup_image"] for i in idx],
         [o_rot[i] for i in idx], [a_rot[i] if i < len(actions) else None for i in idx], GREEN),
        ("mea_v2\nrotation\n+REFLECT", [o_rr[i]["occup_image"] for i in idx],
         [o_rr[i] for i in idx], [a_rr[i] if i < len(actions) else None for i in idx], PURPLE),
    ]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(rows), T, figsize=(1.55 * T, 1.75 * len(rows)), squeeze=False)
    for ri, (label, imgs, ofr, acts, col) in enumerate(rows):
        for c in range(T):
            axc = ax[ri, c]
            axc.imshow(imgs[c], cmap="viridis")
            axc.plot(RES / 2, RES / 2, "r+", ms=8, mew=1.4)
            g = grip_xy(ofr[c]) if ofr[c] is not None else np.zeros(2)  # center if no pc
            px, py = to_px(g)
            if ofr[c] is not None:
                axc.plot(px, py, "o", ms=6, mfc="none", mec="w", mew=1.2)
            if acts[c] is not None:
                d = S @ np.asarray(acts[c], float)[1:3] / PC_RANGE * RES * 2.5
                if np.hypot(*d) > 1:
                    axc.arrow(px, py, d[0], d[1], color=col, width=0.6,
                              head_width=5, length_includes_head=True)
            axc.set_xticks([]); axc.set_yticks([])
            if ri == 0:
                axc.set_title(f"t={idx[c]}", fontsize=8)
            if c == 0:
                axc.set_ylabel(label, fontsize=8)
    fig.suptitle(
        f"{args.task}: augmentation on one expert rollout -- baseline (seq_rot) vs mea_v2\n"
        "rows 2-3 are the SAME rotation (baseline raster vs mea_v2 re-render); row 4 adds the "
        "REFLECTION only mea_v2 can do.  red+ center | white-o gripper | arrow = action (pc frame)",
        fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=125)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
