source bash/init.sh

# =====================================================================
# LOW-DEMO sweep: does mea_v2's value grow as real demos get scarcer?
#
# Motivation. At DEMOS=15 the fixed augmentation LOST to baseline
# (ledger sec 8: BASE 0.388 vs V2FIX+REFL 0.244, 0/3 seeds), and the
# diagnosis was redundancy + 13x dilution: the C4 net + seq_rot buffer
# already own everything the aug adds except reflection, while the
# expert pool becomes 92% synthetic clones.
#
# Both terms of that trade-off move with the demo count:
#   - REDUNDANCY is unchanged (architectural, demo-count independent)
#   - DILUTION is unchanged in RATIO (MEA=12 -> 12/13 synthetic either way)
#   - but the VALUE of each extra effective sample rises as demos shrink
# So if augmentation ever wins, it wins here. This sweep finds the
# crossover demo count -- or shows there isn't one.
#
# Design: paired arms at each demo count, same seeds, ITERS=500 (matches
# the d15 screening so the new points extend that curve rather than
# starting a new one). Existing d15 runs supply the third point for free.
#
#   BASE        mea_expert=0   (scr_base_ token -> analyzer arm BASE)
#   V2FIX+REFL  mea_expert=12, v2 global + reflect 0.5  (v2fgr_ token)
#
# STAGED, most-informative-first:
#   stage 1: DEMOS=5  -- the extreme; largest expected aug advantage, but
#            also the point where BOTH arms may flatline (no signal).
#            Baseline seeds run FIRST (per request) so the reference
#            exists before spending compute on the aug arm.
#   stage 2: DEMOS=10 -- interpolation point; run only if stage 1 shows
#            either (a) an aug win, or (b) usable learning signal.
#
# NB on protocol: num_init_rollouts_pool stays at the config default 20
# (rnn-equi-all.yml:29), so at DEMOS=5 the buffer holds 20 RANDOM
# rollouts vs 5 expert ones -- the expert fraction drops from 80/100 to
# 5/25. That is a real confound for "data scarcity": part of any effect
# is the random-rollout majority, not the demo count alone. Left at the
# default here so these runs stay comparable to the d15 screening; a
# cleaner (but non-comparable) variant would scale init rollouts too.
#
# Analyze: python bash/analyze_screening.py --tag d5_s   (then --tag d10_s)
# Sequential (each buffer ~10GB; no parallel runs on this machine).
# =====================================================================

SEEDS="0 1 2"
ITERS=500         # matches the d15 screening
MEA=12            # augmentations per real expert episode

# ---- stage 1a: BASE at 5 demos (reference first) ----
for s in $SEEDS; do
  python ./rgbd_sym/rl/main.py --cfg configs/block_pull/rnn-equi-all.yml \
    --algo sac --seed $s --cuda 0 --num_expert_episodes 5 --num_iters $ITERS \
    --prefix scr_base_d5_s${s} --mea_expert 0 --mea_normal 0
done

# ---- stage 1b: V2FIX+REFL at 5 demos ----
for s in $SEEDS; do
  python ./rgbd_sym/rl/main.py --cfg configs/block_pull/mea_v2-rnn-equi-all.yml \
    --algo sac --seed $s --cuda 0 --num_expert_episodes 5 --num_iters $ITERS \
    --prefix v2fgr_d5_s${s} --mea_expert $MEA --mea_normal 0 \
    --mea_v2_reflect 0.5
done

# ---- stage 2 (run only if stage 1 warrants it): DEMOS=10 ----
# for s in $SEEDS; do
#   python ./rgbd_sym/rl/main.py --cfg configs/block_pull/rnn-equi-all.yml \
#     --algo sac --seed $s --cuda 0 --num_expert_episodes 10 --num_iters $ITERS \
#     --prefix scr_base_d10_s${s} --mea_expert 0 --mea_normal 0
# done
# for s in $SEEDS; do
#   python ./rgbd_sym/rl/main.py --cfg configs/block_pull/mea_v2-rnn-equi-all.yml \
#     --algo sac --seed $s --cuda 0 --num_expert_episodes 10 --num_iters $ITERS \
#     --prefix v2fgr_d10_s${s} --mea_expert $MEA --mea_normal 0 \
#     --mea_v2_reflect 0.5
# done
