# -*- coding: utf-8 -*-
"""Summarize the MEA-vs-baseline screening runs from wandb.

Pulls `metrics/success_rate_eval` vs env_steps for the screening runs
(name contains --tag), classifies each as MEA / BASELINE from its run name
(`mea_e<N>`), averages across seeds, prints a milestone table and saves a
mean +/- std comparison plot.

Run this AFTER the seeds finish (safe to run while training continues; it only
hits the wandb API, no heavy compute):

    python bash/analyze_screening.py
    # or override:
    python bash/analyze_screening.py --project linhongbin/Symmetry_block_pull_e15 \
        --tag scr_ --out screening_mea_vs_base.png
"""
import argparse
import re
import warnings
import bisect

warnings.filterwarnings("ignore")
import numpy as np


def classify(name):
    """Return (arm, seed) parsed from the wandb run name, e.g.
    'r4-mea_e12_n0-iso_r4-s0_scr_mea_d15_s0-occup' -> ('MEA', 0)."""
    m = re.search(r"mea_e(\d+)", name or "")
    s = re.search(r"-s(\d+)_", name or "")
    mea = int(m.group(1)) if m else None
    seed = int(s.group(1)) if s else None
    if mea is None:
        return None, seed
    return ("MEA" if mea > 0 else "BASE"), seed


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
    ap.add_argument("--tag", default="scr_", help="substring selecting screening runs by name")
    ap.add_argument("--xkey", default="env_steps")
    ap.add_argument("--ykey", default="metrics/success_rate_eval")
    ap.add_argument("--out", default="screening_mea_vs_base.png")
    args = ap.parse_args()

    import wandb
    api = wandb.Api(timeout=30)
    runs = [r for r in api.runs(args.project) if args.tag in (r.name or "")]
    if not runs:
        print(f"No runs whose name contains '{args.tag}' in {args.project}")
        return

    arms = {"MEA": [], "BASE": []}
    print("== runs ==")
    for r in runs:
        arm, seed = classify(r.name)
        if arm is None:
            print(f"  ?    {r.name}: cannot classify, skipped"); continue
        try:
            h = r.history(keys=[args.xkey, args.ykey], samples=5000, pandas=False)
            pts = sorted((p[args.xkey], p[args.ykey]) for p in h
                         if p.get(args.xkey) is not None and p.get(args.ykey) is not None)
        except Exception as e:
            print(f"  {arm:4s} seed={seed} {r.name}: history failed {e}"); continue
        if not pts:
            print(f"  {arm:4s} seed={seed} {r.name}: no eval points yet"); continue
        xs = np.array([x for x, _ in pts]); ys = np.array([y for _, y in pts])
        arms[arm].append((seed, xs, ys))
        print(f"  {arm:4s} seed={seed} points={len(pts)} final={ys[-1]:.2f} ({r.state})")

    if not any(arms.values()):
        print("No usable eval curves."); return

    allx = np.concatenate([xs for lst in arms.values() for _, xs, _ in lst])
    grid = np.linspace(float(allx.min()), float(allx.max()), 60)

    # milestone table
    mile = [g for g in [5000, 10000, 15000, 20000, 25000, 30000, 40000, 50000] if g <= allx.max()]
    print("\n== success_rate_eval (mean over seeds) ==")
    print(f"{'env_steps':>12} | " + " | ".join(f"{m//1000}k" for m in mile))
    curves = {}
    for arm, lst in arms.items():
        if not lst:
            continue
        M = np.array([step_interp(xs, ys, grid) for _, xs, ys in lst], dtype=float)
        curves[arm] = (np.nanmean(M, axis=0), np.nanstd(M, axis=0), len(lst))
        Mm = np.array([step_interp(xs, ys, mile) for _, xs, ys in lst], dtype=float)
        row = np.nanmean(Mm, axis=0)
        print(f"{arm+f'(n={len(lst)})':>12} | " + " | ".join(f"{v:.2f}" for v in row))

    if "MEA" in curves and "BASE" in curves:
        diff = np.nanmean(np.array([step_interp(xs, ys, mile) for _, xs, ys in arms["MEA"]]), axis=0) \
             - np.nanmean(np.array([step_interp(xs, ys, mile) for _, xs, ys in arms["BASE"]]), axis=0)
        print(f"{'MEA - BASE':>12} | " + " | ".join(f"{v:+.2f}" for v in diff))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4.5))
        colors = {"MEA": "#2563eb", "BASE": "#dc2626"}
        labels = {"MEA": "MEA (mea=12)", "BASE": "baseline (mea=0)"}
        for arm, (mean, std, n) in curves.items():
            plt.plot(grid, mean, color=colors[arm], lw=2, label=f"{labels[arm]}, n={n}")
            plt.fill_between(grid, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                             color=colors[arm], alpha=0.15)
        plt.xlabel("env steps"); plt.ylabel("eval success rate"); plt.ylim(-0.02, 1.05)
        plt.title("MEA vs baseline (data-scarce screening, mean +/- std over seeds)")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(args.out, dpi=130)
        print(f"\nSaved plot -> {args.out}")
    except Exception as e:
        print(f"plot skipped: {e}")


if __name__ == "__main__":
    main()
