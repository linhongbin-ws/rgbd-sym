# -*- coding: utf-8 -*-
"""Quantify occlusion holes introduced by each augmentation arm.

An occlusion hole = a background pixel ENCLOSED inside the object footprint
(binary_fill_holes(fg) & ~fg) -- a spot that "should" be object but is empty,
i.e. exactly the "occluded-then-rotated-into-view, no data" artifact. This is
distinct from FOV-boundary loss (which erodes from the edge, not enclosed) and
so isolates the occlusion effect the user asked about.

Arms (block_pull), averaged over frames x several rotation angles:
  original        raw captured frames (baseline holeyness of the sim itself)
  global_seqrot   baseline aug: seq_rot 2D raster rotation (utils.helpers.perturb)
  global_v2img    mea_v2img: image-warp rotation (occlusion-free by construction)
  global_v2       mea_v2: pc rotate about vertical axis + re-voxelize
  conditional_v2  mea_v2 conditional: gripper-only rotation about target (blocks
                  fixed) -- the mode that MOVES the occluder relative to the scene

Prediction: top-down camera + vertical-axis global rotation commute -> global
arms ~0 enclosed holes; conditional separates occluder from occluded.

  source bash/init.sh
  python bash/quantify_occlusion_holes.py --ep $TMPDIR/ep_v2img.pkl \
      --out context/plan/occlusion_holes.png
"""
import argparse
import os
import numpy as np
from scipy.ndimage import binary_fill_holes


def frame_metrics(img, tol):
    """(fg_frac, hole_frac_of_footprint, hole_px) for one occup image."""
    img = np.asarray(img, float)
    fg = img < np.median(img) - tol
    if not fg.any():
        return 0.0, 0.0, 0
    filled = binary_fill_holes(fg)
    holes = filled & ~fg
    footprint = filled.sum()
    return (fg.sum() / fg.size,
            holes.sum() / footprint if footprint else 0.0,
            int(holes.sum()))


def episode_metrics(obs_list, tol):
    ms = [frame_metrics(o["occup_image"], tol) for o in obs_list]
    return (np.mean([m[0] for m in ms]),
            np.mean([m[1] for m in ms]),
            np.mean([m[2] for m in ms]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", default="ep_v2img.pkl")
    ap.add_argument("--task", default="block_pull")
    ap.add_argument("--out", default="context/plan/occlusion_holes.png")
    args = ap.parse_args()

    import joblib
    from rgbd_sym.tool.sym_v2 import generate_sym_v2
    from rgbd_sym.tool.sym_v2_img import generate_sym_v2_img
    from rgbd_sym.env.embodied.dummy.env import DummyEnv
    from rgbd_sym.env.wrapper.occup import Occup
    from utils.helpers import perturb

    data = joblib.load(args.ep)
    obs, actions = data["obs"], data["actions"]
    dummy = Occup(DummyEnv(task=args.task))
    res = getattr(dummy, "_occup_res", 200)
    rng = getattr(dummy, "_pc_range", 0.4)
    tol = 1.5 * rng / (res - 1)

    angles = np.deg2rad([30, 60, 90, 120, 150, -45, -90, -135])

    def seqrot_ep(theta):
        new = []
        pivot = (res / 2, res / 2)
        for o in obs:
            img = np.asarray(o["occup_image"], float)
            r, _, _, _ = perturb(img.copy(), None, np.zeros(2), theta,
                                 [0.0, 0.0], pivot, set_trans_zero=True)
            new.append({"occup_image": r})
        return new

    arms = {}
    # original (angle-independent)
    arms["original"] = [episode_metrics(obs, tol)]
    for key in ["global_seqrot", "global_v2img", "global_v2", "conditional_v2"]:
        arms[key] = []
    seg_reports = []
    for th in angles:
        arms["global_seqrot"].append(episode_metrics(seqrot_ep(th), tol))
        o_img, _ = generate_sym_v2_img(obs, actions, theta_global=th, reflect=False)
        arms["global_v2img"].append(episode_metrics(o_img, tol))
        o_v2, _ = generate_sym_v2(obs, actions, dummy_env=dummy, mode="global",
                                  anchor="origin", theta_global=th, reflect=False)
        arms["global_v2"].append(episode_metrics(o_v2, tol))
        o_c, _ = generate_sym_v2(obs, actions, dummy_env=dummy, mode="conditional",
                                 anchor="origin", theta_global=th, theta_approach=th,
                                 reflect=False)
        arms["conditional_v2"].append(episode_metrics(o_c, tol))

    print(f"\ntolerance={tol:.4f} | {len(angles)} angles x {len(obs)} frames | "
          f"hole = background pixel ENCLOSED in object footprint\n")
    print(f"{'arm':<16} | {'fg area %':>9} | {'HOLE % of footprint':>20} | {'hole px/frame':>13}")
    print("-" * 70)
    summary = {}
    for k, rows in arms.items():
        fg = np.mean([r[0] for r in rows]) * 100
        hf = np.mean([r[1] for r in rows]) * 100
        hp = np.mean([r[2] for r in rows])
        summary[k] = (fg, hf, hp)
        print(f"{k:<16} | {fg:>8.2f}% | {hf:>18.3f}% | {hp:>13.1f}")

    # ---- figure: bar of hole% + an example conditional frame with holes marked ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ks = ["original", "global_seqrot", "global_v2img", "global_v2", "conditional_v2"]
        vals = [summary[k][1] for k in ks]
        colors = ["#6b7280", "#22c55e", "#0ea5e9", "#f59e0b", "#dc2626"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6),
                                       gridspec_kw={"width_ratios": [1.1, 1]})
        ax1.bar(range(len(ks)), vals, color=colors)
        ax1.set_xticks(range(len(ks)))
        ax1.set_xticklabels(ks, rotation=20, ha="right", fontsize=9)
        ax1.set_ylabel("enclosed hole  (% of object footprint)")
        ax1.set_title("occlusion holes by augmentation arm\n"
                      "(mean over 8 angles x %d frames)" % len(obs), fontsize=10)
        for i, v in enumerate(vals):
            ax1.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        ax1.grid(axis="y", alpha=0.3)

        # worst conditional frame, holes in red
        o_c, _ = generate_sym_v2(obs, actions, dummy_env=dummy, mode="conditional",
                                 anchor="origin", theta_global=np.pi / 2,
                                 theta_approach=np.pi / 2, reflect=False)
        worst = max(range(len(o_c)),
                    key=lambda t: frame_metrics(o_c[t]["occup_image"], tol)[2])
        img = np.asarray(o_c[worst]["occup_image"], float)
        fg = img < np.median(img) - tol
        holes = binary_fill_holes(fg) & ~fg
        ax2.imshow(img, cmap="viridis")
        ys, xs = np.where(holes)
        ax2.scatter(xs, ys, s=2, c="red", marker="s")
        ax2.set_title(f"conditional_v2, worst frame t={worst} (θ=90°)\n"
                      f"red = enclosed holes ({holes.sum()} px)", fontsize=10)
        ax2.set_xticks([]); ax2.set_yticks([])
        fig.suptitle("Occlusion holes: global rotation (top-down, commutes) vs "
                     "conditional (moves the occluder)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        fig.savefig(args.out, dpi=125)
        print(f"\nsaved -> {args.out}")
    except Exception as e:
        print(f"plot skipped: {e}")


if __name__ == "__main__":
    main()
