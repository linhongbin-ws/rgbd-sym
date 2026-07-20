source bash/init_equipomdp.sh

# =====================================================================
# DEMO-COUNT ABLATION of the ORIGINAL equi-rl-for-pomdps baseline.
#
# Runs the UNMODIFIED ext/equi-rl-for-pomdps-original tree per its readme
# ("Training (RSAC, Equi-RSAC, ...)"), Equi-RSAC arm only:
#   python3 policies/main.py --cfg configs/block_pulling/rnn-equi-all.yml \
#       --algo sac --seed S --cuda 0 --num_expert_episodes N
#
# Question: how does the published C4-equivariant RSAC baseline scale with
# the number of real expert demos? That maps the headroom curve -- where the
# baseline is data-starved (augmentation has room to help) vs where it has
# saturated (augmentation can only dilute).
#
# !! PYTHONPATH HAZARD -- do not remove the export below.
#    init_equipomdp.sh puts the HACKED fork (ext/equi-rl-for-pomdps) on
#    PYTHONPATH, and BOTH trees contain policies/, utils/, torchkit/ and
#    buffers/ packages. `python policies/main.py` puts <repo>/policies at
#    sys.path[0] -- NOT the repo root -- so `import utils.helpers` etc.
#    would silently resolve to the HACKED fork. The readme's
#    `export PYTHONPATH=${PWD}:$PYTHONPATH`, run from the original repo
#    root, prepends the original tree so it wins. Keep it.
#
# !! Do NOT compare these numbers against our fork's curves. The fork feeds
#    the net a point-cloud occupancy render, the original a native depth
#    heightmap -- different observation modality. See
#    context/plan/original_baseline_comparability.md.
#
# wandb (original conventions, DIFFERENT from the fork):
#   project = Symmetry_BlockPulling-Symm      (learner.py:355, ONE project
#             for all demo counts -- the fork instead splits per count)
#   group   = <prefix>_sac_equi_equi_r4_e<demos>   (learner.py:357-361)
#   name    = s<seed>                         (learner.py:373 -- the run
#             NAME carries only the seed, so classification must use group)
# Analyze:  python bash/analyze_demo_ablation.py
#
# CONFOUNDS when reading the curve (upstream behavior, documented in
# context/plan/original_baseline_comparability.md sec 4):
#  1. num_init_rollouts_pool stays 20 regardless of demo count, so the
#     expert share of the buffer goes 5/25, 10/30, 30/50, 80/100. Part of
#     any low-demo degradation is the random-rollout majority.
#  2. Warmup gradient updates scale with expert steps (learner.py:435-439:
#     update(int(_n_env_steps_total * num_updates_per_iter)) right after
#     expert collection), so DEMOS=5 gets ~1/16 the pretrain updates of
#     DEMOS=80. This is a protocol-scaling curve, not a pure data-quantity
#     curve.
#
# Config default num_iters: 800 (readme protocol -- no --num_iters passed).
# Sequential. ~12h/run => 4 points x 3 seeds = 12 runs ~= 6 days.
# Stop after any point; the analyzer plots whatever has finished.
# =====================================================================

cd ext/equi-rl-for-pomdps-original
export PYTHONPATH=${PWD}:$PYTHONPATH   # readme step "Before Training"; see hazard note

SEEDS="0 1 2"

run_point () {   # $1 = demo count
  for s in $SEEDS; do
    python3 policies/main.py --cfg configs/block_pulling/rnn-equi-all.yml \
      --algo sac --seed $s --cuda 0 --num_expert_episodes $1 \
      --prefix abl_d$1
  done
}

# ---- low end: where the augmentation hypothesis lives ----
run_point 5
run_point 10

# ---- high end: 80 = the readme protocol demo count ----
run_point 30
run_point 80

# ---- optional: 15 = our fork screening's operating point ----
# run_point 15
