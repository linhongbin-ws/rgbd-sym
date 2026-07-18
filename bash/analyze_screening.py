# -*- coding: utf-8 -*-
"""Summarize the MEA screening / mea_v2 A-B runs from wandb.

Pulls `metrics/success_rate_eval` vs env_steps for runs whose name contains
--tag, classifies each into an arm by its prefix token, averages across seeds,
prints a milestone table and saves a mean +/- std comparison plot.

Arms (token -> label, checked in order):
    v2gr_     -> V2+REFL  (v2 global rotation + reflection)   hypothesis arm
    v2g_      -> V2ROT    (v2 global rotation only)           ablation
    scr_mea   -> V1       (old generate_sym3 augmentation)
    scr_base  -> BASE     (no augmentation)

Run AFTER the seeds finish (wandb API only, no heavy compute):

    python bash/analyze_screening.py
    # or override:
    python bash/analyze_screening.py --project linhongbin/Symmetry_block_pull_e15 \
        --tag d15_s --out screening_arms.png
"""
import argparse
import re
import warnings
import bisect

warnings.filterwarnings("ignore")
import numpy as np

# (token, label) checked in order -- longer tokens MUST precede their prefixes.
# v2f* = post frame-fix reruns (correct action labels + origin anchor);
# v2g*/v2gr* = the 2026-07 runs whose action labels were WRONG (kept for record).
ARM_TOKENS = [
    ("v2fgr_", "V2FIX+REFL"),
    ("v2fg_", "V2FIX-ROT"),
    ("v2gr_", "V2+REFL"),
    ("v2g_", "V2ROT"),
    ("scr_mea", "V1"),
    ("scr_base", "BASE"),
]
ARM_ORDER = ["V2FIX+REFL", "V2FIX-ROT", "V2+REFL", "V2ROT", "V1", "BASE"]
COLORS = {"V2FIX+REFL": "#0f766e", "V2FIX-ROT": "#ea580c",
          "V2+REFL": "#16a34a", "V2ROT": "#2563eb", "V1": "#9333ea", "BASE": "#dc2626"}


def classify(name):
    """Return (arm, seed) parsed from the wandb run name, e.g.
    'r4-mea_e12_n0-iso_r4-s0_v2gr_d15_s0-occup' -> ('V2+REFL', 0)."""
    name = name or ""
    arm = None
    for tok, label in ARM_TOKENS:
        if tok in name:
            arm = label
            break
    s = re.search(r"-s(\d+)_", name)
    seed = int(s.group(1)) if s else None
    return arm, seed


def step_interp(xs, ys, grid):
    """Step-hold interpolation: value at each grid point = last y with x <= g."""
    row = []
    for g in grid:
        i = bisect.bisect_right(xs, g) - 1
        row.append(ys[i] if i >= 0 else np.nan)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="linhongbin/Symmetry_block_pull_e15")
    ap.add_argument("--tag", default="d15_s", help="substring selecting runs by name")
    ap.add_argument("--exclude", default="nrm_",
                    help="skip runs containing this token, unless it is part of --tag "
                         "(default keeps normal-net nrm_ runs out of the equi-net view)")
    ap.add_argument("--xkey", default="env_steps")
    ap.add_argument("--ykey", default="metrics/success_rate_eval")
    ap.add_argument("--out", default="screening_arms.png")
    args = ap.parse_args()

    import wandb
    api = wandb.Api(timeout=30)
    drop = (lambda n: args.exclude and args.exclude in n and args.exclude not in args.tag)
    runs = [r for r in api.runs(args.project)
            if args.tag in (r.name or "") and not drop(r.name or "")]
    if args.exclude and args.exclude not in args.tag:
        print(f"(excluding runs containing '{args.exclude}'; pass --exclude '' to keep)")
    if not runs:
        print(f"No runs whose name contains '{args.tag}' in {args.project}")
        return

    arms = {label: [] for label in ARM_ORDER}
    print("== runs ==")
    for r in runs:
        arm, seed = classify(r.name)
        if arm is None:
            print(f"  ?        {r.name}: cannot classify, skipped"); continue
        try:
            h = r.history(keys=[args.xkey, args.ykey], samples=5000, pandas=False)
            pts = sorted((p[args.xkey], p[args.ykey]) for p in h
                         if p.get(args.xkey) is not None and p.get(args.ykey) is not None)
        except Exception as e:
            print(f"  {arm:8s} seed={seed} {r.name}: history failed {e}"); continue
        if not pts:
            print(f"  {arm:8s} seed={seed} {r.name}: no eval points yet"); continue
        xs = np.array([x for x, _ in pts]); ys = np.array([y for _, y in pts])
        arms[arm].append((seed, xs, ys))
        print(f"  {arm:8s} seed={seed} points={len(pts)} final={ys[-1]:.2f} "
              f"AUC={ys.mean():.3f} ({r.state})")

    if not any(arms.values()):
        print("No usable eval curves."); return

    allx = np.concatenate([xs for lst in arms.values() for _, xs, _ in lst])
    grid = np.linspace(float(allx.min()), float(allx.max()), 60)

    # milestone table
    mile = [g for g in [5000, 10000, 15000, 20000, 25000, 30000, 40000, 50000] if g <= allx.max()]
    print("\n== success_rate_eval (mean over seeds) ==")
    print(f"{'env_steps':>14} | " + " | ".join(f"{m//1000:>4}k" for m in mile) + " |  AUC")
    curves = {}
    mile_means = {}
    for arm in ARM_ORDER:
        lst = arms[arm]
        if not lst:
            continue
        M = np.array([step_interp(xs, ys, grid) for _, xs, ys in lst], dtype=float)
        curves[arm] = (np.nanmean(M, axis=0), np.nanstd(M, axis=0), len(lst))
        Mm = np.array([step_interp(xs, ys, mile) for _, xs, ys in lst], dtype=float)
        mile_means[arm] = np.nanmean(Mm, axis=0)
        auc = np.mean([ys.mean() for _, _, ys in lst])
        print(f"{arm + f'(n={len(lst)})':>14} | "
              + " | ".join(f"{v:.2f} " for v in mile_means[arm]) + f" | {auc:.3f}")

    # key contrasts
    print()
    for a, b, why in [("V2FIX+REFL", "BASE", "FIXED hypothesis arm vs baseline"),
                      ("V2FIX-ROT", "BASE", "FIXED rotation-only vs baseline"),
                      ("V2FIX+REFL", "V2FIX-ROT", "FIXED reflection contribution"),
                      ("V2FIX+REFL", "V2+REFL", "label-fix effect (same aug geometry)"),
                      ("V2+REFL", "V2ROT", "reflection contribution"),
                      ("V2+REFL", "BASE", "hypothesis arm vs baseline"),
                      ("V2ROT", "BASE", "continuous rotation vs baseline")]:
        if a in mile_means and b in mile_means:
            d = mile_means[a] - mile_means[b]
            print(f"{a + '-' + b:>14} | " + " | ".join(f"{v:+.2f}" for v in d) + f"   <- {why}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7.5, 4.5))
        for arm in ARM_ORDER:
            if arm not in curves:
                continue
            mean, std, n = curves[arm]
            c = COLORS.get(arm, "#666666")
            plt.plot(grid, mean, color=c, lw=2, label=f"{arm}, n={n}")
            plt.fill_between(grid, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                             color=c, alpha=0.12)
        plt.xlabel("env steps"); plt.ylabel("eval success rate"); plt.ylim(-0.02, 1.05)
        plt.title("mea_v2 reflection A/B (data-scarce, mean +/- std over seeds)")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(args.out, dpi=130)
        print(f"\nSaved plot -> {args.out}")
    except Exception as e:
        print(f"plot skipped: {e}")


if __name__ == "__main__":
    main()
