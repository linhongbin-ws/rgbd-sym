# -*- coding: utf-8 -*-
"""Contrast the WEAK vs the STRONG phase-conditioned augmentation, to check the
author's (arXiv:2508.11204) correction:

  row A  real demo
  row B  RIGID phase-gated rotation (my earlier row3): rotate the WHOLE scene
         (gripper + objects together) by theta0 on approach frames, identity on
         contact. gripper<->target RELATIVE pose is UNCHANGED -> equivalent to the
         camera rotating once mid-trajectory. WEAK / rare.
  row C  APPROACH-ANGLE augmentation (the paper): on approach frames rotate the
         NON-gripper entities about the gripper (origin) by a DECAYING angle
         alpha(t)=alpha0*(k-t)/k, keeping the gripper fixed; identity on contact.
         The target sweeps in from a NEW approach angle alpha and converges to the
         SAME grasp at contact -> gripper<->target RELATIVE pose VARIES. STRONG.

Overlay: red+ = gripper (image center); white line = gripper->target bearing;
green tick = gripper facing (fixed ref). The angle between them = the relative
approach angle: identical in A & B, shifted by alpha(t) in C. Titles annotate it.

  source bash/init.sh && python bash/viz_approach_angle.py --task block_pick
"""
import argparse, os
import numpy as np

A0 = np.deg2rad(55)      # augmentation angle
Z_THRES, HOLD = 0.15, 0.02
S = np.array([[0.0, 1.0], [1.0, 0.0]])


def cxy(pc):
    return np.asarray(pc)[:, :2].mean(0) if pc is not None and len(pc) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="block_pick")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"context/plan/approach_angle_{args.task}.png"
    ep = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"ep_{args.task}.pkl")

    import joblib
    from rgbd_sym.tool.sym_v2 import se2_about, segment_grasp_step
    from rgbd_sym.env.embodied.dummy.env import DummyEnv
    from rgbd_sym.env.wrapper.occup import Occup

    data = joblib.load(ep)
    obs, actions = data["obs"], data["actions"]
    T = len(obs)
    RES = np.asarray(obs[0]["occup_image"]).shape[0]
    dummy = Occup(DummyEnv(task=args.task))

    # first contact frame k
    k = segment_grasp_step(obs)
    if k is None or k < 1:
        # fallback: first z<0.15
        k = next((t for t in range(T) if obs[t].get("gripper_pos") is not None
                  and float(obs[t]["gripper_pos"][2]) < Z_THRES), max(1, T // 3))
    # target = object nearest gripper over contact frames (visible only)
    keys = [kk for kk in obs[0]["pc"] if kk.startswith("object")]
    med = {}
    for kk in keys:
        ds = []
        for t in range(k, T):
            pc, g = obs[t]["pc"].get(kk), cxy(obs[t]["pc"].get("gripper"))
            if pc is not None and len(pc) >= 50 and g is not None:
                ds.append(np.linalg.norm(cxy(pc) - g))
        med[kk] = np.median(ds) if ds else np.inf
    tkey = min(med, key=med.get)
    non_grip = [kk for kk in obs[0]["pc"] if kk != "gripper"]

    def render(pc, tf):
        dummy.set_current_points(pc)
        dummy.apply_transform(tf)
        r = dummy.reset()
        return r["occup_image"], r["pc"]

    def alpha(t):                       # decaying approach-angle schedule
        return A0 * (k - t) / k if t < k else 0.0

    rowB, rowC, tgtA, tgtB, tgtC = [], [], [], [], []
    for t in range(T):
        pc = obs[t]["pc"]
        tgtA.append(cxy(pc.get(tkey)))
        # B: rigid whole-scene rotation on approach, identity on contact
        tfB = {kk: se2_about([0, 0], A0) for kk in pc.keys()} if t < k else {}
        imgB, pcB = render(pc, tfB); rowB.append(imgB); tgtB.append(cxy(pcB.get(tkey)))
        # C: non-gripper rotated by decaying alpha about origin, gripper fixed
        tfC = {kk: se2_about([0, 0], alpha(t)) for kk in non_grip} if t < k else {}
        imgC, pcC = render(pc, tfC); rowC.append(imgC); tgtC.append(cxy(pcC.get(tkey)))

    def to_px(xy):
        if xy is None:
            return None
        p = S @ np.asarray(xy, float)                 # pc->image axis swap
        return (p[0] + 0.2) / 0.4 * RES, (p[1] + 0.2) / 0.4 * RES

    # egocentric relative approach angle = target abs bearing - gripper facing
    def bearing(xy):
        return np.degrees(np.arctan2(xy[1], xy[0])) if xy is not None else np.nan
    ego_real = np.array([bearing(tgtA[t]) for t in range(T)])
    ego_rigid = np.array([bearing(tgtB[t]) - (np.degrees(A0) if t < k else 0.0) for t in range(T)])
    ego_appr = np.array([bearing(tgtC[t]) - 0.0 for t in range(T)])
    # unwrap relative to real for readability
    def rel_to_real(a):
        d = a - ego_real
        return (d + 180) % 360 - 180
    d_rigid, d_appr = rel_to_real(ego_rigid), rel_to_real(ego_appr)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    N = T
    fig = plt.figure(figsize=(1.55 * N, 7.4))
    gs = fig.add_gridspec(4, N, height_ratios=[1, 1, 1, 0.7], hspace=0.14, wspace=0.05)
    ax = np.array([[fig.add_subplot(gs[r, c]) for c in range(N)] for r in range(3)])
    C = RES / 2

    def draw(axc, img, tgt, grip_face_deg, rel_deg, is_contact):
        axc.imshow(img, cmap="viridis")
        axc.plot(C, C, "r+", ms=8, mew=1.5)
        # gripper facing tick (fixed ref rotated by grip_face_deg)
        fa = np.deg2rad(grip_face_deg)
        axc.plot([C, C + 26 * np.cos(fa)], [C, C + 26 * np.sin(fa)],
                 color="#22c55e", lw=2)
        p = to_px(tgt)
        if p is not None:
            axc.plot([C, p[0]], [C, p[1]], "-", color="w", lw=1.3)
            axc.plot(p[0], p[1], "o", ms=5, mfc="none", mec="w", mew=1.2)
        axc.set_xticks([]); axc.set_yticks([])
        for sp in axc.spines.values():
            sp.set_edgecolor("#f59e0b" if is_contact else "#22c55e"); sp.set_linewidth(2.2)

    rows = [
        ("real demo", [obs[t]["occup_image"] for t in range(T)], tgtA, 0.0, "A"),
        ("RIGID phase-gated\n(scene+gripper rotate\ntogether → WEAK)", rowB, tgtB, "B", "B"),
        ("APPROACH-ANGLE\n(only target bearing\nrotates → STRONG)", rowC, tgtC, "C", "C"),
    ]
    for ri, (label, imgs, tgts, kind, tag) in enumerate(rows):
        for t in range(T):
            is_c = t >= k
            # relative approach angle & gripper facing per row
            if tag == "A":
                gf, rel = 0.0, 0.0
            elif tag == "B":
                gf = np.degrees(A0) if t < k else 0.0
                rel = 0.0                                   # unchanged (rigid)
            else:
                gf = 0.0
                rel = np.degrees(alpha(t))                  # target bearing shift
            draw(ax[ri, t], imgs[t], tgts[t], gf, rel, is_c)
            if ri == 0:
                ax[ri, t].set_title(f"t={t}" + ("  (contact)" if is_c else ""),
                                    fontsize=8, color="#f59e0b" if is_c else "#22c55e")
            if tag != "A" and t < k:
                ax[ri, t].text(0.03, 0.03, f"Δα={rel:+.0f}°", transform=ax[ri, t].transAxes,
                               fontsize=8, color="w", va="bottom",
                               bbox=dict(fc="k", alpha=0.4, pad=1))
            if t == 0:
                ax[ri, t].set_ylabel(label, fontsize=8)

    # decisive panel: egocentric relative approach angle vs frame
    axp = fig.add_subplot(gs[3, :])
    tt = np.arange(T)
    axp.axhline(0, color="#9ca3af", lw=0.8)
    axp.plot(tt, np.zeros(T), "-o", ms=4, color="#22c55e", label="real (egocentric α)")
    axp.plot(tt, d_rigid, "--s", ms=5, color="#0ea5e9",
             label="RIGID (rotate together) → overlaps real")
    axp.plot(tt, d_appr, "-D", ms=5, color="#ef4444",
             label="APPROACH-ANGLE → new α, decays to 0 at contact")
    axp.axvspan(k - 0.5, T - 0.5, color="#f59e0b", alpha=0.12)
    axp.text(k - 0.4, axp.get_ylim()[1] * 0.8 if False else 5, "  contact→", fontsize=8, color="#b45309")
    axp.set_xlim(-0.3, T - 0.7); axp.set_xlabel("frame", fontsize=8)
    axp.set_ylabel("egocentric relative\napproach angle Δα (°)", fontsize=8)
    axp.legend(fontsize=7.5, loc="upper right", ncol=1)
    axp.tick_params(labelsize=7)
    axp.set_title("Δα = (target bearing) − (gripper facing).  RIGID overlaps REAL (it is only a viewpoint "
                  "rotation); APPROACH-ANGLE injects a genuinely new relative approach angle.", fontsize=8)

    fig.suptitle(
        f"{args.task}: RIGID vs APPROACH-ANGLE phase augmentation  "
        f"(k={k} first contact | target={tkey})\n"
        "green tick=gripper facing, white line=gripper→target bearing.  ROW B: scene+gripper rotate together → "
        "egocentric Δα=0 (≡ a mid-trajectory camera rotation, WEAK).  "
        "ROW C: only target bearing rotates, Δα(t)→0 at contact → a NEW approach angle (STRONG, the arXiv:2508.11204 lever).",
        fontsize=8.5)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"{args.task}: T={T} k={k} target={tkey} -> {out}")


if __name__ == "__main__":
    main()
