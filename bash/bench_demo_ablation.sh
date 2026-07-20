source bash/init.sh

# =====================================================================
# DEMO-COUNT ABLATION of the plain equi baseline (rnn-equi-all.yml only).
#
# Question: how does the C4-equivariant RSAC baseline scale with the number
# of real expert demos? This maps the headroom curve -- where the baseline
# is data-starved (aug has room to help) vs where it has saturated (aug can
# only dilute). At DEMOS=15 the fixed mea_v2 aug LOST to this baseline
# (ledger sec 8: BASE 0.388 vs V2FIX+REFL 0.244, 0/3 seeds); this ablation
# says whether 15 was already near saturation.
#
# ONE arm only: BASE (mea_expert=0, mea_normal=0), cfg rnn-equi-all.yml.
# ITERS=500 for every point -- internally consistent, and matches the
# existing d15 screening runs, which supply the 15-demo point for free.
#
# STAGED, low end first (that is where the aug hypothesis lives):
#   stage 1: DEMOS = 5, 10   (+ existing 15)  -> 6 runs
#   stage 2: DEMOS = 30, 80  (80 = readme protocol demo count)  -> 6 runs
#
# !! wandb projects are SPLIT BY DEMO COUNT: learner.py:365 builds
#    project_name = f"Symmetry_{env_name}_e{num_expert_rollouts_pool}",
#    so these land in Symmetry_block_pull_e5 / _e10 / _e30 / _e80, NOT in
#    the existing Symmetry_block_pull_e15. Analyze across all of them with:
#      python bash/analyze_demo_ablation.py
#
# CONFOUNDS to keep in mind when reading the curve (both are upstream
# behavior, present in the original repo too -- see
# context/plan/original_baseline_comparability.md sec 4):
#  1. num_init_rollouts_pool stays 20 regardless of demo count, so the
#     expert share of the buffer goes 5/25, 10/30, 15/35, 30/50, 80/100.
#     Part of any low-demo degradation is the random-rollout majority.
#  2. Warmup gradient updates scale with expert steps: learner.py:437-439
#     does update(int(_n_env_steps_total * num_updates_per_iter)) right
#     after expert collection, so DEMOS=5 gets ~1/3 the pretrain updates of
#     DEMOS=15 and ~1/16 of DEMOS=80. The ablation therefore varies data
#     quantity AND pretrain compute together -- it is a protocol-scaling
#     curve, not a pure data-quantity curve.
#     (Corollary worth noting for mea_v2: synthetic episodes do NOT
#     increment _n_env_steps_total, so a MEA run gets the SAME warmup
#     budget as its BASE twin despite holding 13x the expert data. The
#     augmented data only ever enters via online-phase replay sampling.)
#
# Sequential (each buffer ~10GB; no parallel runs on this machine).
# ~8h/run => stage 1 ~2 days, stage 2 ~2 days.
# =====================================================================

SEEDS="0 1 2"
ITERS=500

run_point () {   # $1 = demo count
  for s in $SEEDS; do
    python ./rgbd_sym/rl/main.py --cfg configs/block_pull/rnn-equi-all.yml \
      --algo sac --seed $s --cuda 0 --num_expert_episodes $1 --num_iters $ITERS \
      --prefix scr_base_d${1}_s${s} --mea_expert 0 --mea_normal 0
  done
}

# ---- stage 1: low end ----
run_point 5
run_point 10

# ---- stage 2: high end (uncomment after stage 1) ----
# run_point 30
# run_point 80
