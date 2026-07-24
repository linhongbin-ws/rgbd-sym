# -*- coding: utf-8 -*-
"""Numerical parity check: generate_sym_v2_img (image-space, fork-free) vs
generate_sym_v2 (pc re-render) on ONE real expert episode, with the SAME
injected (theta, reflect) per case.

Expectations (see sym_v2_img.py docstring):
  - pure reflection: near-EXACT (voxel-floor quantization only, <=1 px at
    object edges; flip itself is lossless);
  - rotations: match inside the inscribed disc up to interpolation/quantization;
    borders legitimately differ (pc path CLIPS out-of-grid points onto the
    boundary, img path fills corners with background);
  - actions: bit-identical (both paths call the same transform_action_se2).

RUN (fork env):
    source bash/init.sh
    python bash/check_sym_v2_img.py --ep $TMPDIR/ep_v2img.pkl \
        --out context/plan/check_sym_v2_img.png
Collects the episode via query_expert if the pickle does not exist.
"""
import argparse
import os
import numpy as np


def collect_episode(task, seed, max_steps=100):
    """One expert episode through PomdpEnv+Occup -- exactly the obs dicts
    (with 'pc' and 'occup_image') that the Sym wrapper buffers."""
    from rgbd_sym.env.embodied import PomdpEnv
    from rgbd_sym.env.wrapper.occup import Occup
    env = Occup(PomdpEnv(task=task))
    env.seed = seed
    obs = env.reset()
    obss, actions = [obs], []
    done = False
    while not done and len(actions) < max_steps:
        a = np.asarray(env.query_expert(0), dtype=float)
        obs, reward, done, info = env.step(a)
        actions.append(a)
        obss.append(obs)
    print(f"collected episode: {len(obss)} frames, {len(actions)} actions, "
          f"done={done}")
    return obss, actions


def compare_case(obs, actions, dummy_env, theta, reflect, tol):
    from rgbd_sym.tool.sym_v2 import generate_sym_v2
    from rgbd_sym.tool.sym_v2_img import generate_sym_v2_img

    obs_pc, act_pc = generate_sym_v2(
        obs, actions, dummy_env=dummy_env, mode="global", anchor="origin",
        theta_global=theta, reflect=reflect)
    obs_im, act_im = generate_sym_v2_img(
        obs, actions, theta_global=theta, reflect=reflect)

    # actions must be bit-identical (same transform_action_se2 call)
    for t, (a, b) in enumerate(zip(act_pc, act_im)):
        assert np.allclose(a, b, atol=0), f"action mismatch at t={t}: {a} vs {b}"

    h, w = np.asarray(obs[0]["occup_image"]).shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    disc = (yy - (h - 1) / 2) ** 2 + (xx - (w - 1) / 2) ** 2 <= ((h - 1) / 2) ** 2

    match_all, match_disc, mae_disc = [], [], []
    for t in range(len(obs)):
        A = np.asarray(obs_pc[t]["occup_image"], dtype=float)
        B = np.asarray(obs_im[t]["occup_image"], dtype=float)
        d = np.abs(A - B)
        match_all.append(np.mean(d <= tol))
        match_disc.append(np.mean(d[disc] <= tol))
        mae_disc.append(np.mean(d[disc]))
    return (float(np.mean(match_all)), float(np.mean(match_disc)),
            float(np.mean(mae_disc)), obs_pc, obs_im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", default="ep_v2img.pkl")
    ap.add_argument("--task", default="block_pull")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="context/plan/check_sym_v2_img.png")
    ap.add_argument("--frame", type=int, default=None,
                    help="frame to plot (default: mid-episode)")
    args = ap.parse_args()

    import joblib
    if os.path.exists(args.ep):
        data = joblib.load(args.ep)
        obs, actions = data["obs"], data["actions"]
        print(f"loaded {args.ep}: {len(obs)} frames, {len(actions)} actions")
    else:
        obs, actions = collect_episode(args.task, args.seed)
        joblib.dump({"obs": obs, "actions": actions}, args.ep)
        print(f"saved -> {args.ep}")

    from rgbd_sym.env.embodied import DummyEnv
    from rgbd_sym.env.wrapper.occup import Occup
    dummy_env = Occup(DummyEnv(task=args.task))

    res = getattr(dummy_env, "_occup_res", 200)
    rng = getattr(dummy_env, "_pc_range", 0.4)
    ztol = 1.5 * rng / (res - 1)     # 1.5 voxel depth steps
    print(f"tolerance: {ztol:.4f} (1.5 z-voxel steps)")

    cases = [
        ("reflect only         ", 0.0, True),
        ("rot +pi/3            ", np.pi / 3, False),
        ("rot +pi/3 + reflect  ", np.pi / 3, True),
        ("rot -2.1             ", -2.1, False),
        ("rot +pi/2 (C4 grid)  ", np.pi / 2, False),
    ]
    print(f"\n{'case':<22} | match(all) | match(disc) | MAE(disc)")
    keep = None
    for name, th, rf in cases:
        m_all, m_disc, mae, obs_pc, obs_im = compare_case(
            obs, actions, dummy_env, th, rf, ztol)
        print(f"{name} | {m_all:>9.4f}  | {m_disc:>10.4f} | {mae:.5f}")
        if rf and th != 0.0:
            keep = (name, obs_pc, obs_im)
    print("actions: identical in every case (asserted).")

    # ---- figure: orig | pc-v2 | img-v2 | diff on one frame ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        name, obs_pc, obs_im = keep
        t = args.frame if args.frame is not None else len(obs) // 2
        A0 = np.asarray(obs[t]["occup_image"], dtype=float)
        A = np.asarray(obs_pc[t]["occup_image"], dtype=float)
        B = np.asarray(obs_im[t]["occup_image"], dtype=float)
        fig, axes = plt.subplots(1, 4, figsize=(15, 4))
        for ax, im, ttl in zip(
                axes, [A0, A, B, np.abs(A - B)],
                [f"original (t={t})", f"pc re-render ({name.strip()})",
                 "image warp (v2img)", "|pc - img|"]):
            h = ax.imshow(im, cmap="viridis")
            ax.set_title(ttl, fontsize=10)
            ax.axis("off")
            fig.colorbar(h, ax=ax, fraction=0.046)
        fig.suptitle("mea_v2 pc path vs fork-free image path -- same injected transform",
                     fontsize=11)
        fig.tight_layout()
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        fig.savefig(args.out, dpi=130)
        print(f"figure -> {args.out}")
    except Exception as e:
        print(f"plot skipped: {e}")


if __name__ == "__main__":
    main()
