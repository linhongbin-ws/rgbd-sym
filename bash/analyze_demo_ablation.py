# -*- coding: utf-8 -*-
"""Plot the equi-baseline demo-count scaling curve (bash/bench_demo_ablation.sh).

wandb projects are split by demo count -- learner.py:365 builds
`project_name = f"Symmetry_{env_name}_e{num_expert_rollouts_pool}"` -- so this
walks one project per demo count instead of taking a single --project.

For each demo count it pulls `metrics/success_rate_eval` vs env_steps for every
BASE run (scr_base_ token), reports per-seed AUC and final value, and plots
mean +/- std vs demo count.

    source bash/init.sh
    python bash/analyze_demo_ablation.py
    python bash/analyze_demo_ablation.py --demos 5 10 15 30 80 --arm scr_base_
"""
import argparse
import re
import warnings

warnings.filterwarnings("ignore")
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="linhongbin")
    ap.add_argument("--env", default="block_pull")
    ap.add_argument("--demos", type=int, nargs="+", default=[5, 10, 15, 30, 80])
    ap.add_argument("--arm", default="scr_base_",
                    help="run-name token selecting the arm (default: BASE)")
    ap.add_argument("--xkey", default="env_steps")
    ap.add_argument("--ykey", default="metrics/success_rate_eval")
    ap.add_argument("--out", default="demo_ablation.png")
    args = ap.parse_args()

    import wandb
    api = wandb.Api(timeout=30)

    per_demo = {}   # demo -> list of (seed, auc, final, state)
    for d in args.demos:
        project = f"{args.entity}/Symmetry_{args.env}_e{d}"
        try:
            runs = list(api.runs(project))
        except Exception as e:
            print(f"d={d:<3} project {project}: not found / unreadable ({e})")
            continue
        # the prefix embeds the demo count, so d5 cannot leak into d15 etc.
        want = f"{args.arm}d{d}_s"
        rows = []
        for r in runs:
            name = r.name or ""
            if want not in name:
                continue
            try:
                h = r.history(keys=[args.xkey, args.ykey], samples=5000, pandas=False)
                pts = sorted((p[args.xkey], p[args.ykey]) for p in h
                             if p.get(args.xkey) is not None and p.get(args.ykey) is not None)
            except Exception as e:
                print(f"  d={d} {name}: history failed {e}")
                continue
            if not pts:
                print(f"  d={d} {name}: no eval points yet")
                continue
            ys = np.array([y for _, y in pts])
            m = re.search(r"-s(\d+)_", name)
            rows.append((int(m.group(1)) if m else -1, float(ys.mean()),
                         float(ys[-1]), r.state))
        if rows:
            per_demo[d] = sorted(rows)

    if not per_demo:
        print("No runs found. Has bench_demo_ablation.sh been launched?")
        return

    print(f"\n== {args.arm} scaling over demo count ({args.ykey}) ==")
    print(f"{'demos':>6} | {'n':>2} | {'AUC mean+/-std':>18} | {'final mean':>10} | per-seed AUC")
    for d in sorted(per_demo):
        rows = per_demo[d]
        aucs = np.array([a for _, a, _, _ in rows])
        fins = np.array([f for _, _, f, _ in rows])
        detail = ", ".join(f"s{s}={a:.3f}{'*' if st != 'finished' else ''}"
                           for s, a, _, st in rows)
        print(f"{d:>6} | {len(rows):>2} | {aucs.mean():>8.3f} +/- {aucs.std():<6.3f} | "
              f"{fins.mean():>10.3f} | {detail}")
    print("(* = run not finished yet)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ds = sorted(per_demo)
        mean = [np.mean([a for _, a, _, _ in per_demo[d]]) for d in ds]
        std = [np.std([a for _, a, _, _ in per_demo[d]]) for d in ds]
        plt.figure(figsize=(6.5, 4.2))
        plt.errorbar(ds, mean, yerr=std, marker="o", capsize=4,
                     color="#dc2626", lw=2, label=f"{args.arm} (AUC)")
        for d in ds:
            for _, a, _, _ in per_demo[d]:
                plt.plot(d, a, ".", color="#dc2626", alpha=0.35)
        plt.xscale("log")
        plt.xticks(ds, [str(d) for d in ds])
        plt.xlabel("expert demos"); plt.ylabel("eval success rate (AUC)")
        plt.ylim(-0.02, 1.02)
        plt.title("equi baseline: demo-count scaling (mean +/- std over seeds)")
        plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
        plt.savefig(args.out, dpi=130)
        print(f"\nSaved plot -> {args.out}")
    except Exception as e:
        print(f"plot skipped: {e}")


if __name__ == "__main__":
    main()
