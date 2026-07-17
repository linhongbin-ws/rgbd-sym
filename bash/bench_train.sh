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
# (same protocol: DEMOS=15, ITERS=500, seeds 0/1/2) -- no need to rerun:
# the frame fix only touched the v2 augmentation path, which baseline
# (mea_expert=0) never enters, so the old scr_base_ curves stay valid and
# seed-paired with the new arms.
#
# STAGED to put compute on the headline question first:
#   stage 1: all 3 seeds of A (V2FIX+REFL)  -> after ~1 day, compare vs the
#            existing BASE and answer "does fixed mea_v2 beat baseline?"
#   stage 2: all 3 seeds of B (V2FIX-ROT ablation) -> isolates reflection's
#            contribution; run later / skip if stage 1 already decides.
# Sequential (each buffer ~10GB; no parallel runs on this machine).
# Analyze anytime with: python bash/analyze_screening.py  (classifies all arms)
#
# 2026-07-17 RERUN (v2f* prefixes): the original v2gr_/v2g_ runs trained on
# WRONG action labels -- the pc/image frame is the world frame with x/y
# swapped (det=-1), so augmented actions must rotate by -theta (not +theta)
# and mirror must flip a[2] (not a[1]); also the rotation anchor must be the
# pc origin (= gripper = image center), not the scene centroid. All fixed
# (sym_v2.py, validated by bash/check_pc_action_frame.py). These reruns are
# the first CLEAN measurement of v2 augmentation vs the C4-equivariant net.
# =====================================================================

SEEDS="0 1 2"     # paired seeds across both arms
DEMOS=15          # data-scarce (matches scr_ screening)
ITERS=500         # shortened (matches scr_ screening)
MEA=12            # augmentations per real expert episode

# ---- stage 1: A (V2FIX+REFL, hypothesis arm) on all seeds first ----
for s in $SEEDS; do
  python ./rgbd_sym/rl/main.py --cfg configs/block_pull/mea_v2-rnn-equi-all.yml \
    --algo sac --seed $s --cuda 0 --num_expert_episodes $DEMOS --num_iters $ITERS \
    --prefix v2fgr_d${DEMOS}_s${s} --mea_expert $MEA --mea_normal 0 \
    --mea_v2_reflect 0.5
done

# ---- stage 2: B (V2FIX-ROT, rotation-only ablation) afterwards ----
for s in $SEEDS; do
  python ./rgbd_sym/rl/main.py --cfg configs/block_pull/mea_v2-rnn-equi-all.yml \
    --algo sac --seed $s --cuda 0 --num_expert_episodes $DEMOS --num_iters $ITERS \
    --prefix v2fg_d${DEMOS}_s${s} --mea_expert $MEA --mea_normal 0 \
    --mea_v2_reflect 0
done
