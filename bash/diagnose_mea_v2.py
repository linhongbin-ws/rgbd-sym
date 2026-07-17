# -*- coding: utf-8 -*-
"""Empirical diagnosis: WHY did mea_v2 augmentation fail to help / hurt?

Captures one clean block_pull expert episode, then measures concrete artifacts
the global-rotation+reflect augmentation injects into the re-rendered occupancy:

  T1 IDENTITY ROUND-TRIP  theta=0, reflect=False -> should be a NO-OP. Any pixel
     difference vs the original occup_image means the re-render pipeline itself is
     lossy/non-idempotent (corrupts data before any rotation).
  T2 GRIPPER OFF-CENTERING  the depth camera is gripper-centered, so the gripper
     sits at the image center in ALL real obs. generate_sym_v2 rotates about the
     SCENE centroid, not the gripper -> measure how far (px) the gripper leaves
     center under 60/90/180 deg. Off-center gripper = obs the net never sees for real.
  T3 WINDOW CLIPPING  occupancy is binned in a FIXED world window [-0.2,0.2]^2.
     Rotating about the scene centroid can push scene points outside -> clipped.
     Measure fraction of points that leave the window.
  T4 OCCUPIED-AREA (dis)conservation  a rigid rotation should ~conserve occupied
     pixels. Compare orig vs identity/90/60/180/reflect. Big drops = clipping or
     interpolation holes. 90deg is a symmetry the C4 net ALREADY has (clean ref).

Saves a figure with orig / identity / 90 / 60 / reflect + identity-diff map.
CPU only; no GPU, no network. Run:
  source bash/init.sh
  python bash/diagnose_mea_v2.py --out diag_mea_v2.png
"""
import argparse
import os
import numpy as np


def occupied_mask(img, bg_tol=1e-4):
    """Non-background pixels. Background is the constant max fill (max+0.07)."""
    bg = img.max()
    return np.abs(img - bg) > bg_tol


def gripper_center_offset_px(pc_gripper, pc_x_min=-0.2, pc_range=0.4, res=200):
    """Pixel offset of the gripper centroid from the image center."""
    if pc_gripper is None or len(pc_gripper) == 0:
        return None
    c = np.asarray(pc_gripper)[:, :2].mean(axis=0)          # world xy
    center = pc_x_min + pc_range / 2.0                      # window center (=0.0)
    off_world = c - center
    off_px = off_world / pc_range * res
    return c, np.linalg.norm(off_px)


def frac_outside_window(pc_all, lo=-0.2, hi=0.2):
    xy = pc_all[:, :2]
    inside = (xy[:, 0] >= lo) & (xy[:, 0] <= hi) & (xy[:, 1] >= lo) & (xy[:, 1] <= hi)
    return 1.0 - inside.mean()


def all_points(pc_dict):
    pts = [v for v in pc_dict.values() if v is not None and len(v)]
    return np.concatenate(pts, axis=0) if pts else np.zeros((0, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", default="ep_v2.pkl")
    ap.add_argument("--task", default="block_pull")
    ap.add_argument("--out", default="diag_mea_v2.png")
    args = ap.parse_args()

    import joblib
    from rgbd_sym.env.embodied.dummy.env import DummyEnv
    from rgbd_sym.env.wrapper.occup import Occup
    from rgbd_sym.tool.sym_v2 import (generate_sym_v2, scene_centroid_xy,
                                      se2_about, transform_pc)

    # ---- capture one clean episode if not cached ----
    if not os.path.exists(args.ep):
        from rgbd_sym.env.embodied.pomdp.env import PomdpEnv
        env = Occup(PomdpEnv(task=args.task))
        env.seed = 0
        obs = env.reset()
        obss, actions = [obs], []
        done, steps = False, 0
        while not done and steps < 50:
            a = np.array(env.query_expert(1), dtype=float)   # ep 1 -> pull_movable
            obs, r, done, info = env.step(a)
            obss.append(obs); actions.append(a); steps += 1
        joblib.dump({"obs": obss, "actions": actions}, args.ep)
        print(f"captured {len(obss)} frames -> {args.ep}")

    data = joblib.load(args.ep)
    obs, actions = data["obs"], data["actions"]
    T = len(obs)
    print(f"episode: {T} frames; pc entities = {sorted(obs[0]['pc'].keys())}")
    has_img = "occup_image" in obs[0]
    print(f"occup_image present in captured obs: {has_img}  "
          f"shape={obs[0]['occup_image'].shape if has_img else None}")

    dummy_env = Occup(DummyEnv(task=args.task))

    def render(transform_dict_per_frame):
        out = []
        for t in range(T):
            dummy_env.set_current_points(obs[t]["pc"])
            dummy_env.apply_transform(transform_dict_per_frame(t))
            out.append(dummy_env.reset()["occup_image"])
        return out

    def const_tf(T4):
        return lambda t: {k: T4 for k in obs[t]["pc"].keys()}

    I4 = np.eye(4)

    # ======== T1: IDENTITY ROUND-TRIP ========
    print("\n== T1  identity round-trip (theta=0, reflect=False) ==")
    ident = render(lambda t: {k: I4 for k in obs[t]["pc"].keys()})
    if has_img:
        diffs = [np.mean(np.abs(ident[t] - obs[t]["occup_image"])) for t in range(T)]
        maxd = [np.max(np.abs(ident[t] - obs[t]["occup_image"])) for t in range(T)]
        fracdiff = [np.mean(np.abs(ident[t] - obs[t]["occup_image"]) > 1e-6) for t in range(T)]
        print(f"  vs captured occup_image: mean|Δ|={np.mean(diffs):.5f}  "
              f"max|Δ|={np.max(maxd):.5f}  mean frac pixels changed={np.mean(fracdiff):.4f}")
        print("  (identity SHOULD be exact; nonzero => re-render is not idempotent)")

    # ======== T2 + T3: rotation about SCENE CENTROID (what the aug does) ========
    print("\n== T2/T3  global rotation about SCENE centroid (as generate_sym_v2 does) ==")
    print(f"{'angle':>8} | grip_off_px(center=100) | frac_pts_clipped | occ_area_ratio_vs_orig")
    frame0_imgs = {}
    for name, theta, refl in [("orig", None, False), ("ident", 0.0, False),
                              ("90deg", np.pi/2, False), ("60deg", np.pi/3, False),
                              ("180deg", np.pi, False), ("reflect", 0.0, True)]:
        # per-frame occupied-area ratio + grip offset + clip, averaged over frames
        offs, clips, area_ratios = [], [], []
        imgs = []
        for t in range(T):
            pcd = obs[t]["pc"]
            q = scene_centroid_xy(obs[t])
            if name == "orig":
                Tm = I4
            else:
                Tm = se2_about(q, theta, reflect=refl)
            # transform pc entity-wise
            new_pc = {k: (transform_pc(v, Tm) if v is not None and len(v) else v)
                      for k, v in pcd.items()}
            # gripper offset
            go = gripper_center_offset_px(new_pc.get("gripper"))
            if go is not None:
                offs.append(go[1])
            # clipping
            ap_ = all_points(new_pc)
            if len(ap_):
                clips.append(frac_outside_window(ap_))
            # render for area
            dummy_env.set_current_points(pcd)
            dummy_env.apply_transform({k: Tm for k in pcd.keys()} if name != "orig" else {})
            im = dummy_env.reset()["occup_image"]
            imgs.append(im)
            a_new = occupied_mask(im).sum()
            # orig area baseline from captured image if present else identity render
            base_img = obs[t].get("occup_image", ident[t])
            a_base = occupied_mask(base_img).sum()
            area_ratios.append(a_new / max(a_base, 1))
        frame0_imgs[name] = imgs
        print(f"{name:>8} | {np.mean(offs):>21.1f} | {np.mean(clips):>16.4f} | "
              f"{np.mean(area_ratios):>21.3f}")

    # ======== figure ========
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        show = ["orig", "ident", "90deg", "60deg", "reflect"]
        tshow = [T // 2, T - 2]                      # a mid frame and a late (pull) frame
        nrow = len(tshow) + 1
        fig, ax = plt.subplots(nrow, len(show), figsize=(2.6 * len(show), 2.6 * nrow))
        for ci, name in enumerate(show):
            for ri, t in enumerate(tshow):
                a = ax[ri, ci]
                a.imshow(frame0_imgs[name][t], cmap="viridis")
                a.set_title(f"{name} t={t}", fontsize=9); a.axis("off")
                # mark image center
                a.plot(100, 100, "r+", ms=9, mew=1.5)
        # bottom row: identity-diff map (|ident - orig|) at the two frames + spacer
        for ci, name in enumerate(show):
            a = ax[nrow - 1, ci]
            if name in ("ident",) and has_img:
                d = np.abs(frame0_imgs["ident"][tshow[-1]] - obs[tshow[-1]]["occup_image"])
                im = a.imshow(d, cmap="magma"); a.set_title("|ident-orig| diff", fontsize=9)
            elif name == "orig":
                a.imshow(obs[tshow[-1]]["occup_image"], cmap="viridis")
                a.set_title("orig (ref)", fontsize=9)
            else:
                a.axis("off"); continue
            a.axis("off")
        plt.tight_layout(); plt.savefig(args.out, dpi=115)
        print(f"\nSaved -> {args.out}")
    except Exception as e:
        print(f"plot skipped: {e}")


if __name__ == "__main__":
    main()
