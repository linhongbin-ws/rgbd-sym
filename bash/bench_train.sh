source bash/init.sh

# =====================================================================
# mea_v2 reflection A/B experiment (isolate the reflection contribution).
#
# Hypothesis (design_mea_v2.md section 8): the C4-equivariant net already
# owns every rotation symmetry, so v1/rotation-only augmentation is
# self-redundant (confirmed negative result). REFLECTION is the symmetry
# the net structurally lacks (flip_symmetry=false) while block_pull's
# relational reward is mirror-invariant -> the non-redundant lever.
#
#   A: v2 global + reflect  (reflect_prob=0.5)   <- hypothesis arm
#   B: v2 global rotation-only (reflect_prob=0)  <- ablation
#   A - B  =  net contribution of reflection.
#
# baseline (mea=0) and v1 arms already exist from the scr_ screening runs
# (same protocol: DEMOS=15, ITERS=500, seeds 0/1/2) -- no need to rerun.
#
# Paired seeds, A then B per seed so the earliest signal is a full pair.
# Sequential (each buffer ~10GB; no parallel runs on this machine).
# Analyze with: python bash/analyze_screening.py   (classifies all 4 arms)
# =====================================================================

SEEDS="0 1 2"     # paired seeds across both arms
DEMOS=15          # data-scarce (matches scr_ screening)
ITERS=500         # shortened (matches scr_ screening)
MEA=12            # augmentations per real expert episode

for s in $SEEDS; do
  # ---- A: v2 global + reflect ----
  python ./rgbd_sym/rl/main.py --cfg configs/block_pull/mea_v2-rnn-equi-all.yml \
    --algo sac --seed $s --cuda 0 --num_expert_episodes $DEMOS --num_iters $ITERS \
    --prefix v2gr_d${DEMOS}_s${s} --mea_expert $MEA --mea_normal 0 \
    --mea_v2_reflect 0.5

  # ---- B: v2 global rotation-only (ablation) ----
  python ./rgbd_sym/rl/main.py --cfg configs/block_pull/mea_v2-rnn-equi-all.yml \
    --algo sac --seed $s --cuda 0 --num_expert_episodes $DEMOS --num_iters $ITERS \
    --prefix v2g_d${DEMOS}_s${s} --mea_expert $MEA --mea_normal 0 \
    --mea_v2_reflect 0
done
