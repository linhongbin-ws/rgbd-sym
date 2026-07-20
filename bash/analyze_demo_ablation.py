# -*- coding: utf-8 -*-
"""Plot the demo-count scaling curve of the ORIGINAL equi-rl-for-pomdps
baseline (bash/bench_demo_ablation.sh).

The original repo's wandb conventions differ from our fork's, so this cannot
reuse analyze_screening.py:
    project = Symmetry_BlockPulling-Symm     (learner.py:355 -- ONE project
              for every demo count; the fork splits per count instead)
    group   = <prefix>_sac_equi_equi_r4_e<demos>   (learner.py:357-361)
    name    = s<seed>                        (learner.py:373 -- the run NAME
              carries only the seed)
so runs are classified by GROUP (demo count parsed from its `_e<N>` tail),
not by name.

    source bash/init_equipomdp.sh
    python bash/analyze_demo_ablation.py
    python bash/analyze_demo_ablation.py --prefix abl_ --out demo_ablation.png
"""
import argparse
import re
import warnings

warnings.filterwarnings("ignore")
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="linhongbin/Symmetry_BlockPulling-Symm")
    ap.add_argument("--prefix", default="abl_",
                    help="group-name token selecting this ablation's runs")
    ap.add_argument("--xkey", default="env_steps")
    ap.add_argument("--ykey", default="metrics/success_rate_eval")
    ap.add_argument("--out", default="demo_ablation.png")
    args = ap.parse_args()

    import wandb
    api = wandb.Api(timeout=30)
    try:
        runs = list(api.runs(args.project))
    except Exception as e:
        print(f"cannot read project {args.project}: {e}")
        return

    per_demo = {}   # demos -> list of (seed, auc, final, state)
    skipped = 0
    for r in runs:
        group = r.group or ""
        if args.prefix not in group:
            skipped += 1
            continue
        m = re.search(r"_e(\d+)$", group)
        if not m:
            print(f"  ? group '{group}': no _e<demos> tail, skipped")
            continue
        demos = int(m.group(1))
        s = re.match(r"s(\d+)$", r.name or "")
        seed = int(s.group(1)) if s else -1
        try:
            h = r.history(keys=[args.xkey, args.ykey], samples=5000, pandas=False)
            pts = sorted((p[args.xkey], p[args.ykey]) for p in h
                         if p.get(args.xkey) is not None and p.get(args.ykey) is not None)
        except Exception as e:
            print(f"  d={demos} s{seed}: history failed {e}")
            continue
        if not pts:
            print(f"  d={demos} s{seed}: no eval points yet")
            continue
        ys = np.array([y for _, y in pts])
        per_demo.setdefault(demos, []).append(
            (seed, float(ys.mean()), float(ys[-1]), r.state))

    if not per_demo:
        print(f"No runs whose group contains '{args.prefix}' in {args.project} "
              f"({skipped} other runs present). Has bench_demo_ablation.sh been launched?")
        return

    print(f"\n== original Equi-RSAC baseline: demo-count scaling ({args.ykey}) ==")
    print(f"{'demos':>6} | {'n':>2} | {'AUC mean+/-std':>18} | {'final mean':>10} | per-seed AUC")
    for d in sorted(per_demo):
        rows = sorted(per_demo[d])
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
                     color="#dc2626", lw=2, label="Equi-RSAC (AUC)")
        for d in ds:
            for _, a, _, _ in per_demo[d]:
                plt.plot(d, a, ".", color="#dc2626", alpha=0.35)
        plt.xscale("log")
        plt.xticks(ds, [str(d) for d in ds])
        plt.xlabel("expert demos"); plt.ylabel("eval success rate (AUC)")
        plt.ylim(-0.02, 1.02)
        plt.title("original equi-rl-for-pomdps: demo-count scaling\n"
                  "(mean +/- std over seeds)", fontsize=10)
        plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
        plt.savefig(args.out, dpi=130)
        print(f"\nSaved plot -> {args.out}")
    except Exception as e:
        print(f"plot skipped: {e}")


if __name__ == "__main__":
    main()
