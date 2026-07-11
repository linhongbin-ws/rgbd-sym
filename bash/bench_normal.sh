source bash/init.sh

# =====================================================================
# Normal-network control: does augmentation SUBSTITUTE for architectural
# symmetry? (design_mea_v2.md / mea_screening_results.md section 6.)
#
# On the C4-equivariant net, NO augmentation beat baseline (rotation even
# hurt). Thesis: augmentation only helps when the network LACKS the symmetry.
# Test it by swapping the actor/critic to non-equivariant `normal` and asking
# whether reflection-MEA (V2+REFL) now beats baseline.
#
#   A: BASE-normal     normal net, mea=0
#   B: V2+REFL-normal  normal net, mea=12, v2 global + reflect 0.5
#   B > A  =>  aug substitutes for missing architectural symmetry (thesis).
#   B ~ A  =>  aug doesn't help even without the symmetry -> deeper limitation.
#
# CONFOUND (read before trusting): buffer_type=seq_rot ALSO does C4 rotation
# augmentation, so BOTH arms still get rotation invariance from the buffer.
# This isolates REFLECTION's value on a rotation-augmented non-equi net.
# For a STRICTER "net lacks ALL symmetry" test, also pass
#   --buffer_type seq_vanilla
# to both arms (risk: 15 demos may be too few to learn without any rotation
# help -> both arms could flatline and give no signal). Left as seq_rot here
# so the control is likely to produce a learning signal.
#
# Same data-scarce protocol (demo=15, 500 iters, 3 seeds). Sequential.
# Analyze with a fresh tag, e.g.:
#   python bash/analyze_screening.py --tag nrm_d15_s --out normal_ctrl.png
#   (note: analyze_screening classifies by v2gr_/scr_base tokens; the nrm_
#    prefixes below reuse those tokens so it still buckets A=BASE, B=V2+REFL)
# =====================================================================

SEEDS="0 1 2"
DEMOS=15
ITERS=500
MEA=12

for s in $SEEDS; do
  # ---- A: baseline on normal (non-equivariant) net ----
  python ./rgbd_sym/rl/main.py --cfg configs/block_pull/rnn-equi-all.yml \
    --algo sac --seed $s --cuda 0 --num_expert_episodes $DEMOS --num_iters $ITERS \
    --actor_type normal --critic_type normal \
    --prefix scr_base_nrm_d${DEMOS}_s${s} --mea_expert 0 --mea_normal 0

  # ---- B: V2+REFL on normal net ----
  python ./rgbd_sym/rl/main.py --cfg configs/block_pull/mea_v2-rnn-equi-all.yml \
    --algo sac --seed $s --cuda 0 --num_expert_episodes $DEMOS --num_iters $ITERS \
    --actor_type normal --critic_type normal \
    --prefix v2gr_nrm_d${DEMOS}_s${s} --mea_expert $MEA --mea_normal 0 \
    --mea_v2_reflect 0.5
done
