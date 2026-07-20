source bash/init_equipomdp_original.sh

# =====================================================================
# Reproduce the ORIGINAL equi-rl-for-pomdps baselines (block_pulling),
# readme protocol, in the isolated `equi-pomdp` env -- the reference
# numbers mea_v2 must be compared against.
#
# !! 2026-07-20 COMPARABILITY WARNING (context/plan/original_baseline_comparability.md)
# These numbers are NOT directly comparable to our fork's curves. The fork
# feeds the net a point-cloud OCCUPANCY render (obs_type: occup, via the Occup
# wrapper -> gym_regularizer.py:27-38), while the original feeds pybullet's
# native depth HEIGHTMAP. Different observation modality => different task
# difficulty. Depth encoding and gripper fill value differ too.
# For "mea_v2 vs baseline" use the FORK's own BASE arm (mea_expert=0), which
# the audit confirmed is behaviorally equivalent to the original learner path.
# This script answers a different question: "does the unmodified upstream
# reproduce its published numbers on this machine?"
#
# Runs the UNMODIFIED ext/equi-rl-for-pomdps-original tree (readme:
# 80 expert demos, config-default 800 iters). Two arms, 3 seeds each:
#   stage 1: Equi-RSAC (rnn-equi-all.yml)  <- the arch mea_v2 builds on;
#            headline reference, all seeds first
#   stage 2: RSAC      (rnn.yml)           <- non-equivariant baseline
#
# wandb: learner.py auto-names the project Symmetry_BlockPulling-Symm
# (separate from our Symmetry_block_pull_e15 -- no contamination).
# Analyze: python bash/analyze_screening.py \
#            --project linhongbin/Symmetry_BlockPulling-Symm --tag orig_
#   (NB: ARM_TOKENS has no orig_* entries yet; extend it, or read the
#    per-run AUC lines which classify-fail but still print names.)
#
# Sequential; ~800 iters/run on GPU. Setup: bash/setup_equipomdp_original.sh
# =====================================================================

SEEDS="0 1 2"
DEMOS=80          # readme protocol

cd ext/equi-rl-for-pomdps-original

# ---- stage 1: Equi-RSAC (the mea_v2 reference) on all seeds first ----
for s in $SEEDS; do
  python policies/main.py --cfg configs/block_pulling/rnn-equi-all.yml \
    --algo sac --seed $s --cuda 0 --num_expert_episodes $DEMOS \
    --prefix orig_equi_d${DEMOS}_s${s}
done

# ---- stage 2: plain RSAC afterwards ----
for s in $SEEDS; do
  python policies/main.py --cfg configs/block_pulling/rnn.yml \
    --algo sac --seed $s --cuda 0 --num_expert_episodes $DEMOS \
    --prefix orig_rsac_d${DEMOS}_s${s}
done

# ---- stage 3 (OPTIONAL, commented): matched-protocol variant for a
# like-for-like contrast with our screening runs (DEMOS=15, ITERS=500).
# Enable only if comparing original-vs-fork at the data-scarce operating
# point matters more than reproducing the published numbers.
# for s in $SEEDS; do
#   python policies/main.py --cfg configs/block_pulling/rnn-equi-all.yml \
#     --algo sac --seed $s --cuda 0 --num_expert_episodes 15 --num_iters 500 \
#     --prefix orig_equi_d15_s${s}
# done
