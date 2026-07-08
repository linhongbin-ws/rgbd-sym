# -*- coding: utf-8 -*-
"""Offline validation for mea_v2 `generate_sym_v2` -- no pybullet rollout needed.

It (1) infers the action<->world-xy convention from ONE real episode and prints
the `mea_v2_action_sign` you should use, (2) runs generate_sym_v2 in both modes
with structural asserts, and (3) saves an occupancy before/after figure to eyeball.

--------------------------------------------------------------------------------
HOW TO CAPTURE ONE EPISODE (drop-in, one line, then run one expert rollout):

  In rgbd_sym/env/wrapper/sym.py, top of `_on_end_gt_eps`, temporarily add:

      import joblib, os
      if not os.path.exists('ep_v2.pkl'):
          joblib.dump({'obs': obss_origin, 'actions': actions_origin}, 'ep_v2.pkl')

  Run any training/prefill briefly so one GT episode finishes, then remove the
  line. (obss_origin/actions_origin are exactly what generate_sym_v2 consumes.)
--------------------------------------------------------------------------------
RUN (in the rgbd-sym env, which has cv2/scipy/numpy):

      source bash/init.sh
      python bash/test_sym_v2.py --ep ep_v2.pkl --out sym_v2_check.png
"""
import argparse
import numpy as np


def infer_action_world_map(obs, actions, dpos=0.05):
    """Fit the signed-permutation phi mapping action[1:3]*dpos -> gripper world-xy
    displacement (from obs['gripper_pos']). Returns (rel_err, phi 2x2, det)."""
    G = np.array([np.asarray(o["gripper_pos"], float)[:2] for o in obs])
    dG = np.diff(G, axis=0)                                  # (T,2) world deltas
    A = np.array([np.asarray(a, float)[1:3] for a in actions]) * dpos
    n = min(len(dG), len(A))
    dG, A = dG[:n], A[:n]
    best = None
    for perm in [(0, 1), (1, 0)]:
        for s0 in (1, -1):
            for s1 in (1, -1):
                P = np.zeros((2, 2))
                P[0, perm[0]] = s0
                P[1, perm[1]] = s1
                pred = (P @ A.T).T
                err = np.linalg.norm(pred - dG) / (np.linalg.norm(dG) + 1e-9)
                if best is None or err < best[0]:
                    best = (err, P, float(round(np.linalg.det(P))))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", required=True, help="joblib pkl: {'obs':[...], 'actions':[...]}")
    ap.add_argument("--task", default="block_pull")
    ap.add_argument("--out", default="sym_v2_check.png")
    ap.add_argument("--frames", type=int, default=4, help="how many frames to plot")
    args = ap.parse_args()

    import joblib
    from rgbd_sym.env.embodied.dummy.env import DummyEnv
    from rgbd_sym.env.wrapper.occup import Occup
    from rgbd_sym.tool.sym_v2 import (generate_sym_v2, segment_grasp_step,
                                      nearest_object_key, scene_centroid_xy)

    data = joblib.load(args.ep)
    obs, actions = data["obs"], data["actions"]
    print(f"episode: {len(obs)} obs frames, {len(actions)} actions")
    print(f"pc entities: {sorted(obs[0].get('pc', {}).keys())}")

    # (1) action <-> world convention -> recommended action_sign
    if all("gripper_pos" in o for o in obs):
        err, phi, det = infer_action_world_map(obs, actions)
        print("\n=== action<->world-xy map (from gripper_pos vs action[1:3]) ===")
        print(f"  phi =\n{phi}")
        print(f"  fit rel_err = {err:.3f}  (small = clean convention)")
        print(f"  det(phi) = {det:+.0f}  ->  RECOMMEND  mea_v2_action_sign = {det:+.0f}")
        if err > 0.3:
            print("  WARNING: high fit error; action frame may not be a pure signed permutation.")
    else:
        det = 1.0
        print("\n(no gripper_pos in obs; cannot infer action_sign -- defaulting to +1)")

    # (2) phase segmentation report
    k = segment_grasp_step(obs)
    print("\n=== conditional segmentation ===")
    print(f"  grasp_step k = {k}")
    if k is not None and 0 < k < len(obs) - 1:
        tk = nearest_object_key(obs[k])
        print(f"  target(grasped) object = {tk}")
        print(f"  approach anchor (target xy) = {np.round(scene_centroid_xy(obs[k]), 4)}")

    # (3) run generate_sym_v2 in both modes with structural asserts
    dummy_env = Occup(DummyEnv(task=args.task))
    results = {}
    for mode in ("global", "conditional"):
        n_obs, n_act = generate_sym_v2(
            obs, actions, dummy_env=dummy_env, mode=mode,
            action_sign=float(det), theta_global=np.pi / 3, theta_approach=np.pi / 3)
        assert len(n_obs) == len(obs), f"{mode}: obs len {len(n_obs)} != {len(obs)}"
        assert len(n_act) == len(actions), f"{mode}: act len mismatch"
        # occupancy actually changed on a rotated frame?
        changed = [float(np.mean(np.abs(n_obs[t]["occup_image"] - obs[t]["occup_image"])))
                   for t in range(len(obs)) if "occup_image" in obs[t]]
        results[mode] = (n_obs, n_act, changed)
        print(f"\n[{mode}] lengths OK; mean |occup delta| over frames = "
              f"{np.mean(changed):.4f} (max {np.max(changed):.4f})")

    # (4) figure: original vs global vs conditional occupancy for a few frames
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        T = len(obs)
        idxs = np.linspace(0, T - 1, args.frames).astype(int)
        rows = ["orig", "global", "conditional"]
        fig, ax = plt.subplots(len(rows), len(idxs), figsize=(3 * len(idxs), 8))
        for ci, t in enumerate(idxs):
            imgs = [obs[t].get("occup_image"),
                    results["global"][0][t].get("occup_image"),
                    results["conditional"][0][t].get("occup_image")]
            for ri, im in enumerate(imgs):
                a = ax[ri, ci]
                if im is not None:
                    a.imshow(im, cmap="viridis")
                a.set_title(f"{rows[ri]} t={t}", fontsize=8)
                a.axis("off")
        plt.tight_layout()
        plt.savefig(args.out, dpi=110)
        print(f"\nSaved occupancy comparison -> {args.out}")
    except Exception as e:
        print(f"plot skipped: {e}")

    print("\nDONE. If action_sign recommendation != current config, update "
          "mea_v2_action_sign in configs/block_pull/mea_v2-rnn-equi-all.yml.")


if __name__ == "__main__":
    main()
