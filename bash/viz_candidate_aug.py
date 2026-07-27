# -*- coding: utf-8 -*-
"""Visualize the candidate augmentations that COULD beat baseline, per task, on a
real expert rollout, so their plausibility can be eyeballed.

Rows (columns = subsampled time steps of ONE rollout):
  1 real demo              captured occupancy; border GREEN=free-space / ORANGE=contact
  2 baseline seq_rot       uniform rotation theta on EVERY frame incl. contact
                           (the redundant-with-C4 recipe that measured no gain)
  3 CANDIDATE phase-gated  rotate free-space frames by theta, keep CONTACT frames
                           canonical (theta=0, trivial group) -- the arXiv:2508.11204
                           winning recipe; border colored by phase
  4 task-specific lever    pull  -> global REFLECTION (corrected action label)
                           pick/push/drawer -> DISTRACTOR decorrelation (in-place yaw
                           of the reward-irrelevant object; orthogonal to ALL symmetry)
Bottom: phase signals (gripper z, gripper<->target xy dist) with thresholds; contact
frames shaded; subsampled columns marked. Contact proxy = z<0.15 OR holding(d<0.02).

  source bash/init.sh
  python bash/viz_candidate_aug.py --task block_pick
"""
import argparse, os
import numpy as np

THETA = np.pi / 3            # visible rotation angle
PSI = np.deg2rad(60)         # distractor in-place yaw
Z_THRES, HOLD_THRES = 0.15, 0.02
FREE, CONTACT = "#22c55e", "#f59e0b"
S = np.array([[0.0, 1.0], [1.0, 0.0]])


def centroid_xy(pc):
    return np.asarray(pc)[:, :2].mean(0) if pc is not None and len(pc) else None


def contact_mask(obs):
    """per-frame interaction flag from observable signals."""
    m = []
    for o in obs:
        z = float(o["gripper_pos"][2]) if o.get("gripper_pos") is not None else 1.0
        g = centroid_xy(o["pc"].get("gripper"))
        d = np.inf
        for k in o["pc"]:
            if k in ("gripper", "goal"):
                continue
            c = centroid_xy(o["pc"].get(k))
            if c is not None and g is not None:
                d = min(d, np.linalg.norm(c - g))
        m.append((z < Z_THRES) or (d < HOLD_THRES))
    return np.array(m, bool)


def distractor_key(obs, contact):
    """Robustly split target vs distractor. The per-entity pc is NOISY (occlusion ->
    0-point frames, gripper-frame drift), so a single-frame nearest test is
    unreliable. TARGET = object the gripper actually engages = min MEDIAN gripper<->
    object xy distance over the CONTACT frames (skipping frames where the object is
    occluded, <50 pts). DISTRACTOR (reward-irrelevant) = the farthest such object."""
    keys = [k for k in obs[0]["pc"] if k.startswith("object")]
    med = {}
    cidx = [t for t in range(len(obs)) if contact[t]] or list(range(len(obs)))
    for k in keys:
        ds = []
        for t in cidx:
            pc = obs[t]["pc"].get(k)
            g = centroid_xy(obs[t]["pc"].get("gripper"))
            if pc is not None and len(pc) >= 50 and g is not None:
                ds.append(np.linalg.norm(centroid_xy(pc) - g))
        med[k] = np.median(ds) if ds else np.inf
    target = min(med, key=med.get)
    distractor = max(med, key=med.get)
    return distractor, target, med


def z_signals(obs):
    z = [float(o["gripper_pos"][2]) if o.get("gripper_pos") is not None else np.nan for o in obs]
    dist = []
    for o in obs:
        g = centroid_xy(o["pc"].get("gripper"))
        d = np.inf
        for k in o["pc"]:
            if k in ("gripper", "goal"):
                continue
            c = centroid_xy(o["pc"].get(k))
            if c is not None and g is not None:
                d = min(d, np.linalg.norm(c - g))
        dist.append(d if np.isfinite(d) else np.nan)
    return np.array(z), np.array(dist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="block_pick")
    ap.add_argument("--ncols", type=int, default=9)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"context/plan/candidate_aug_{args.task}.png"
    ep = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"ep_{args.task}.pkl")

    import joblib
    from utils.helpers import perturb
    from rgbd_sym.tool.sym_v2 import generate_sym_v2, se2_about
    from rgbd_sym.env.embodied.dummy.env import DummyEnv
    from rgbd_sym.env.wrapper.occup import Occup

    data = joblib.load(ep)
    obs, actions = data["obs"], data["actions"]
    T = len(obs)
    RES = np.asarray(obs[0]["occup_image"]).shape[0]
    contact = contact_mask(obs)
    dkey, tkey, med = distractor_key(obs, contact)
    z, dist = z_signals(obs)
    dummy = Occup(DummyEnv(task=args.task))

    def to_px(xy):
        p = (np.asarray(xy, float) + 0.2) / 0.4 * RES
        return p[0], p[1]

    # --- row 2/3: raster rotation via the real seq_rot op (perturb) ---
    pivot = (RES / 2, RES / 2)

    def seqrot(img, act, theta):
        r, _, rd, _ = perturb(np.asarray(img, float).copy(), None,
                              np.asarray(act, float)[1:3].copy(), theta, [0., 0.],
                              pivot, set_trans_zero=True)
        return r, rd

    base_imgs, gate_imgs, gate_dxy = [], [], []
    for t in range(T):
        act = actions[t] if t < len(actions) else np.zeros(5)
        b, _ = seqrot(obs[t]["occup_image"], act, THETA)
        base_imgs.append(b)
        th = 0.0 if contact[t] else THETA           # phase gate
        g, rd = seqrot(obs[t]["occup_image"], act, th)
        gate_imgs.append(g); gate_dxy.append(rd)

    # --- row 4: global REFLECTION (whole scene incl. goal), corrected action label ---
    # The one symmetry the C4 net structurally lacks (flip_symmetry=false). Clean and
    # well-defined for every task; the distractor lever is shown separately (its
    # per-entity pc is too occlusion-noisy to render convincingly in a strip).
    row4_label = "REFLECTION\n(mirror whole\nscene incl. goal;\nfixed action label)"
    o_ref, _ = generate_sym_v2(obs, actions, dummy_env=dummy, mode="global",
                               anchor="origin", theta_global=0.0, reflect=True)
    row4_imgs = [o_ref[t]["occup_image"] for t in range(T)]
    row4_mark = None

    # ---------------- plot ----------------
    idx = np.unique(np.linspace(0, T - 1, min(args.ncols, T)).astype(int))
    N = len(idx)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(1.5 * N + 0.6, 9.2))
    gs = fig.add_gridspec(5, N, height_ratios=[1, 1, 1, 1, 0.72], hspace=0.12, wspace=0.05)

    rows = [
        ("real demo", [obs[i]["occup_image"] for i in idx], True, "demo"),
        ("baseline seq_rot\nUNIFORM θ\n(contact too)", [base_imgs[i] for i in idx], False, "base"),
        ("CANDIDATE\nphase-gated θ\n(contact=canonical)", [gate_imgs[i] for i in idx], True, "gate"),
        (row4_label, [row4_imgs[i] for i in idx], False, "lever"),
    ]
    for ri, (label, imgs, phase_color, tag) in enumerate(rows):
        for c, t in enumerate(idx):
            ax = fig.add_subplot(gs[ri, c])
            ax.imshow(imgs[c], cmap="viridis")
            ax.plot(RES / 2, RES / 2, "r+", ms=7, mew=1.3)
            # gripper marker
            g = centroid_xy(obs[t]["pc"].get("gripper"))
            if g is not None and tag in ("demo",):
                px, py = to_px(g); ax.plot(px, py, "o", ms=6, mfc="none", mec="w", mew=1.1)
            # action arrow on demo (real a) and gate free-frames (rotated dxy)
            if tag == "demo" and t < len(actions):
                d = S @ np.asarray(actions[t], float)[1:3] / 0.4 * RES * 2.5
                if np.hypot(*d) > 1 and g is not None:
                    px, py = to_px(g); ax.arrow(px, py, d[0], d[1], color="w", width=0.6,
                                                head_width=5, length_includes_head=True)
            if tag == "gate" and t < len(actions):
                d = S @ np.asarray(gate_dxy[t], float) / 0.4 * RES * 2.5
                if np.hypot(*d) > 1:
                    ax.arrow(RES/2, RES/2, d[0], d[1], color="w", width=0.6,
                             head_width=5, length_includes_head=True)
            # distractor ring
            if tag == "lever" and row4_mark is not None and row4_mark[c] is not None:
                px, py = to_px(row4_mark[c])
                ax.plot(px, py, "o", ms=15, mfc="none", mec="#ef4444", mew=1.8)
            ax.set_xticks([]); ax.set_yticks([])
            # phase-colored border
            col = (CONTACT if contact[t] else FREE) if phase_color else "#9ca3af"
            for sp in ax.spines.values():
                sp.set_edgecolor(col); sp.set_linewidth(2.4 if phase_color else 0.8)
            if ri == 0:
                ax.set_title(f"t={t}", fontsize=8,
                             color=(CONTACT if contact[t] else FREE))
            if c == 0:
                ax.set_ylabel(label, fontsize=7.5)

    # phase-signal panel
    axp = fig.add_subplot(gs[4, :])
    tt = np.arange(T)
    axp.plot(tt, z, "-o", ms=3, color="#0ea5e9", label="gripper z")
    axp.plot(tt, dist, "-s", ms=3, color="#a855f7", label="grip↔target dist")
    axp.axhline(Z_THRES, ls="--", lw=0.8, color="#0ea5e9")
    axp.axhline(HOLD_THRES, ls="--", lw=0.8, color="#a855f7")
    for t in range(T):
        if contact[t]:
            axp.axvspan(t - 0.5, t + 0.5, color=CONTACT, alpha=0.12)
    for t in idx:
        axp.axvline(t, color="k", lw=0.4, alpha=0.3)
    axp.set_xlim(-0.5, T - 0.5); axp.set_ylim(0, max(0.3, np.nanmax(z) * 1.05))
    axp.set_xlabel("frame", fontsize=8); axp.legend(fontsize=7, ncol=2, loc="upper right")
    axp.set_title("phase signals — shaded = CONTACT (z<0.15 or holding d<0.02);  "
                  "green/orange borders above mark free/contact frames", fontsize=8)
    axp.tick_params(labelsize=7)

    n_contact = int(contact.sum())
    fig.suptitle(
        f"{args.task}: candidate augmentations that could beat baseline  "
        f"(rollout {T} frames, {n_contact} contact frames)\n"
        "row2 uniform rotation = redundant w/ C4 (measured no gain);  row3 phase-gated "
        "(contact kept canonical) = arXiv:2508.11204 recipe;  row4 = REFLECTION (the coset C4 lacks)",
        fontsize=9)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"{args.task}: T={T} contact={n_contact} target={tkey} distractor={dkey} med={ {k:round(v*100,1) for k,v in med.items()} } -> {out}")


if __name__ == "__main__":
    main()
