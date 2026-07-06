source bash/init.sh

# =====================================================================
# MEA screening experiment (GPU-scarce friendly).
# Question: does MEA help when data is SCARCE (baseline has headroom)?
#
# Design:
#   - data-scarce: few expert demos  -> augmentation matters most here
#   - shortened:   fewer iters       -> block_pull saturates early
#   - paired:      same seeds on both arms -> variance reduction
#   - MEA at full strength (12) to give it its best shot
#
# 3 seeds x {MEA, baseline} = 6 runs, sequential (each buffer ~GBs; no parallel).
# Tune the vars below. If baseline never learns at all, raise DEMOS (e.g. 20-30);
# if curves haven't plateaued, raise ITERS.
# =====================================================================

SEEDS="0 1 2"     # paired seeds across both arms
DEMOS=15          # data-scarce (full run used 80)
ITERS=500         # shortened (full run used 800)
MEA=12            # augmentations per real expert episode (max strength)

for s in $SEEDS; do
  # ---- ours: MEA ----
  python ./rgbd_sym/rl/main.py --cfg configs/block_pull/mea-rnn-equi-all.yml \
    --algo sac --seed $s --cuda 0 --num_expert_episodes $DEMOS --num_iters $ITERS \
    --prefix scr_mea_d${DEMOS}_s${s} --mea_expert $MEA --mea_normal 0

  # ---- baseline: no MEA ----
  python ./rgbd_sym/rl/main.py --cfg configs/block_pull/rnn-equi-all.yml \
    --algo sac --seed $s --cuda 0 --num_expert_episodes $DEMOS --num_iters $ITERS \
    --prefix scr_base_d${DEMOS}_s${s} --mea_expert 0 --mea_normal 0
done
