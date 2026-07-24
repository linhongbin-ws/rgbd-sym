# -*- coding: utf-8 -*-
"""Offline demo-count ablation summary + figure for the ORIGINAL
equi-rl-for-pomdps baseline, read straight from the local .wandb binaries
(no network -- the wandb API is unreachable from the sandbox).

Groups runs by the --num_expert_episodes recorded in each run's
files/wandb-metadata.json, reads (env_steps, success_rate_eval) via
read_wandb_hist.read_history, and writes a 2-panel figure (AUC + plateau
vs demo count) plus a per-seed table.

    python bash/plot_ablation_offline.py \
        --root ext/equi-rl-for-pomdps-original/wandb \
        --out  context/plan/demo_ablation.png
"""
import argparse
import glob
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_wandb_hist import read_history, convergence


def run_meta(run_dir):
    meta = os.path.join(run_dir, "files", "wandb-metadata.json")
    if not os.path.exists(meta):
        return None
    a = json.load(open(meta)).get("args", [])

    def g(k):
        return a[a.index(k) + 1] if k in a else None

    demos = g("--num_expert_episodes")
    seed = g("--seed")
    prefix = g("--prefix") or ""
    if demos is None:
        return None
    return int(demos), (int(seed) if seed is not None else -1), prefix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="ext/equi-rl-for-pomdps-original/wandb")
    ap.add_argument("--out", default="context/plan/demo_ablation.png")
    ap.add_argument("--prefix", default="abl_",
                    help="only runs whose --prefix contains this token")
    args = ap.parse_args()

    per_demo = {}   # demos -> list of (seed, auc, plateau, conv_step, npts)
    for run_dir in sorted(glob.glob(os.path.join(args.root, "run-*")),
                          key=os.path.getmtime):
        m = run_meta(run_dir)
        if not m:
            continue
        demos, seed, prefix = m
        if args.prefix not in prefix:
            continue
        wf = glob.glob(os.path.join(run_dir, "*.wandb"))
        if not wf:
            continue
        pts = read_history(wf[0])
        if not pts:
            continue
        auc = float(np.mean([s for _, s in pts]))
        plateau = float(np.mean([s for _, s in pts[-5:]]))
        cenv = convergence(pts)[0] if convergence(pts) else None
        # keep the newest run per (demos, seed) -- glob is mtime-sorted asc
        per_demo.setdefault(demos, {})[seed] = (auc, plateau, cenv, len(pts))

    if not per_demo:
        print(f"no abl_ runs under {args.root}")
        return

    print(f"\n== original Equi-RSAC baseline -- demo-count ablation (COMPLETE) ==")
    print(f"{'demos':>5} | {'n':>1} | {'AUC mean+/-std':>16} | "
          f"{'plateau':>7} | per-seed  auc/plateau/conv")
    ds = sorted(per_demo)
    summ = {}
    for d in ds:
        rows = sorted(per_demo[d].items())
        aucs = np.array([v[0] for _, v in rows])
        plats = np.array([v[1] for _, v in rows])
        detail = "  ".join(
            f"s{s}={v[0]:.2f}/{v[1]:.2f}/{(v[2]/1000 if v[2] else 0):.0f}k"
            for s, v in rows)
        summ[d] = (aucs.mean(), aucs.std(), plats.mean(), plats.std())
        print(f"{d:>5} | {len(rows):>1} | {aucs.mean():>7.3f}+/-{aucs.std():<6.3f} | "
              f"{plats.mean():>7.2f} | {detail}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
        am = [summ[d][0] for d in ds]
        asd = [summ[d][1] for d in ds]
        pm = [summ[d][2] for d in ds]
        psd = [summ[d][3] for d in ds]
        ax1.errorbar(ds, am, yerr=asd, marker="o", capsize=4, color="#dc2626", lw=2)
        for d in ds:
            for s, v in per_demo[d].items():
                ax1.plot(d, v[0], ".", color="#dc2626", alpha=0.35)
        ax1.set_xscale("log"); ax1.set_xticks(ds); ax1.set_xticklabels([str(d) for d in ds])
        ax1.set_xlabel("expert demos"); ax1.set_ylabel("eval success rate AUC")
        ax1.set_ylim(-0.02, 1.02); ax1.grid(alpha=0.3)
        ax1.set_title("AUC (whole-run mean) vs demos")
        ax2.errorbar(ds, pm, yerr=psd, marker="s", capsize=4, color="#2563eb", lw=2)
        for d in ds:
            for s, v in per_demo[d].items():
                ax2.plot(d, v[1], ".", color="#2563eb", alpha=0.35)
        ax2.set_xscale("log"); ax2.set_xticks(ds); ax2.set_xticklabels([str(d) for d in ds])
        ax2.set_xlabel("expert demos"); ax2.set_ylabel("plateau (last-5 mean)")
        ax2.set_ylim(-0.02, 1.02); ax2.grid(alpha=0.3)
        ax2.set_title("converged plateau vs demos")
        fig.suptitle("original equi-rl-for-pomdps: demo-count ablation "
                     "(3 seeds each, 800 iters)", fontsize=11)
        fig.tight_layout()
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        fig.savefig(args.out, dpi=130)
        print(f"\nsaved -> {args.out}")
    except Exception as e:
        print(f"plot skipped: {e}")


if __name__ == "__main__":
    main()
