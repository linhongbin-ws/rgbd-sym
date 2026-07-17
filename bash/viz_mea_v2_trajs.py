# -*- coding: utf-8 -*-
"""Visualize one REAL block_pull expert trajectory vs several mea_v2-augmented
clones -- exactly what generate_sym_v2 (mode=global) feeds the buffer as
synthetic demos, via the same code path used in training.

Figure: rows = ORIG + one row per (theta, reflect) draw; columns = every frame.
Overlays per cell:
  red +      image center (where the gripper sits in ALL real obs, since the
             depth camera is gripper-centered)
  white o    gripper centroid of THIS obs (real or augmented)
  green ->   the (transformed) action's (dx, dy), exaggerated for visibility

Also prints, per augmented row, the measured gripper off-center displacement
and the closed-form prediction |(I - F R) q0| where q0 = frame-0 scene centroid
(the anchor the ACTUAL code rotates about, sym_v2.py mode='global').
NB: diagnose_mea_v2.py's T2 rotated about the PER-FRAME centroid, which the
real code does NOT do -- this script measures the true training-time behavior.

  source bash/init.sh
  python bash/viz_mea_v2_trajs.py --ep ep_v2.pkl --out viz_mea_v2_trajs.png
"""
import argparse
import os
import numpy as np

PC_MIN, PC_RANGE, RES = -0.2, 0.4, 200   # Occup wrapper window (block_pull)


def to_px(xy):
    """world xy -> (col, row) pixel coords of the occup image (z transposed)."""
    p = (np.asarray(xy, float) - PC_MIN) / PC_RANGE * RES
    return p[0], p[1]


def grip_xy(obs_frame):
    pc = obs_frame.get("pc", {}).get("gripper")
    if pc is None or len(pc) == 0:
        return None
    return np.asarray(pc)[:, :2].mean(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", default="ep_v2.pkl")
    ap.add_argument("--task", default="block_pull")
    ap.add_argument("--out", default="viz_mea_v2_trajs.png")
    ap.add_argument("--q0resets", type=int, default=6,
                    help="also measure |q0| over N fresh env resets (0=skip)")
    args = ap.parse_args()

    import joblib
    from rgbd_sym.tool.sym_v2 import generate_sym_v2, scene_centroid_xy, rot2d
    from rgbd_sym.env.embodied.dummy.env import DummyEnv
    from rgbd_sym.env.wrapper.occup import Occup

    assert os.path.exists(args.ep), f"missing {args.ep} (run diagnose_mea_v2.py first)"
    data = joblib.load(args.ep)
    obs, actions = data["obs"], data["actions"]
    T = len(obs)
    dummy = Occup(DummyEnv(task=args.task))

    q0 = scene_centroid_xy(obs[0])
    print(f"anchor q0 (frame-0 scene centroid) = {np.round(q0, 4)}  "
          f"|q0| = {np.linalg.norm(q0):.4f} m = {np.linalg.norm(q0)/PC_RANGE*RES:.1f} px")

    # the three augmented clones to show (theta_global, reflect)
    AUGS = [(np.pi / 3, False), (2.4, True), (-np.pi / 2, True)]

    rows = [("ORIG", obs, actions, None, None)]
    for th, rf in AUGS:
        n_obs, n_act = generate_sym_v2(obs, actions, dummy_env=dummy,
                                       mode="global", theta_global=th, reflect=rf)
        label = f"AUG θ={np.degrees(th):+.0f}°" + ("\n+mirror" if rf else "")
        rows.append((label, n_obs, n_act, th, rf))

    # ---- gripper off-center: measured vs closed-form prediction ----
    print(f"\n{'row':>18} | grip off-center px (mean / max) | predicted |(I-FR)q0| px")
    for label, o, a, th, rf in rows:
        offs = []
        for t in range(T):
            g = grip_xy(o[t])
            if g is None:
                continue
            c, r = to_px(g)
            offs.append(np.hypot(c - RES / 2, r - RES / 2))
        pred = "-"
        if th is not None:
            FR = rot2d(th)
            if rf:
                FR = np.diag([-1.0, 1.0]) @ FR
            pred = f"{np.linalg.norm((np.eye(2) - FR) @ q0) / PC_RANGE * RES:.1f}"
        lab = label.replace("\n", " ")
        print(f"{lab:>18} | {np.mean(offs):>13.1f} / {np.max(offs):<13.1f} | {pred:>8}")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    K = 2.5  # action-arrow exaggeration
    fig, ax = plt.subplots(len(rows), T, figsize=(1.55 * T, 1.75 * len(rows)))
    for ri, (label, o, a, th, rf) in enumerate(rows):
        for t in range(T):
            axc = ax[ri, t]
            axc.imshow(o[t]["occup_image"], cmap="viridis")
            axc.plot(RES / 2, RES / 2, "r+", ms=8, mew=1.5)          # image center
            g = grip_xy(o[t])
            if g is not None:
                c, r = to_px(g)
                axc.plot(c, r, "o", ms=7, mfc="none", mec="w", mew=1.3)  # gripper
                if t < len(a):                                        # action arrow
                    d = np.asarray(a[t], float)[1:3] / PC_RANGE * RES * K
                    if np.hypot(*d) > 1:
                        axc.arrow(c, r, d[0], d[1], color="#22c55e",
                                  width=0.6, head_width=5, length_includes_head=True)
            axc.set_xticks([]); axc.set_yticks([])
            if ri == 0:
                axc.set_title(f"t={t}", fontsize=8)
            if t == 0:
                axc.set_ylabel(label, fontsize=9)
    fig.suptitle("real expert trajectory (top) vs mea_v2 global-mode augmented clones\n"
                 "red + = image center   white o = gripper   green arrow = action (dx,dy)",
                 fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(args.out, dpi=115)
    print(f"\nSaved -> {args.out}")

    # ---- |q0| across fresh resets (how big can the anchor offset get?) ----
    if args.q0resets > 0:
        from rgbd_sym.env.embodied.pomdp.env import PomdpEnv
        env = Occup(PomdpEnv(task=args.task))
        vals = []
        print(f"\n== |q0| over {args.q0resets} fresh resets ==")
        for i in range(args.q0resets):
            o0 = env.reset()
            v = float(np.linalg.norm(scene_centroid_xy(o0)))
            vals.append(v)
            print(f"  reset {i}: |q0| = {v:.4f} m  ({v / PC_RANGE * RES:.1f} px)")
        wc = 2 * np.max(vals) / PC_RANGE * RES
        print(f"  mean {np.mean(vals):.4f} m, max {np.max(vals):.4f} m "
              f"-> worst-case gripper off-center = 2*max = {wc:.0f} px (at θ=180°)")


if __name__ == "__main__":
    main()
