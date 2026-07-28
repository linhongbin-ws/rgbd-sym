# MEA v2 (diffusion) — experiment runbook (run on the 3090; Claude sandbox has no GPU)

Integration = **PATH A** (offline): the "arm" is *which hdf5 you train on* × *which host*.
No change to EquiDiff code. See `equidiff_integration.py` for why the dataloader hook is
wrong for the image policy. Authority on the pivot: `context/mea_v2_new_idea/falsification_and_pivot.md`.

## 0. Setup (once)
```
git clone https://github.com/pointW/equidiff && cd equidiff       # + its conda env
# download the keyed task (square) demos:
python mimicgen/scripts/download_datasets.py --dataset_type core --tasks square_d0
python equi_diffpo/scripts/robomimic_dataset_conversion.py \
    -i data/robomimic/datasets/square_d0/square_d0.hdf5 \
    -o data/robomimic/datasets/square_d0/square_d0_abs.hdf5 -n 12     # -> ABS 10-D actions
```

## 1. BLOCKING GATE (before any training)
```
python mea_diff/test_action_consistency.py --hdf5 .../square_d0_abs.hdf5 --demo demo_0
```
Verifies the rot6d/quat conjugation on REAL data. **FAIL → the aug math is mis-signed → do NOT train.**

## 2. Produce the dataset variants (the "arms")
| variant hdf5 | how to make it |
|---|---|
| `square_d0_abs` (BASELINE) | as downloaded/converted |
| `square_d0_keyedC4_abs` (**the method**) | MimicGen datagen with a **C4 rotation added to the insertion-subtask object-frame target** (mimicgen task/SubtaskConfig); re-solve + re-render; convert to abs. Uses `phase_aug.rot_z/transform_*` for the action math. |
| `square_d0_globalrot_abs` (CONTROL) | whole-scene SO(2) rotation of each demo (redundant w/ EquiDiff by design) — for the DP host mainly |
Round control (optional, NOT headline): generate `round_*` the SAME way with a genuinely
SO(2)-symmetric grasp (annular peg, centered grasp, no handle); predict `square_gain > round_gain`.

## 3. Headline ablation (within-Square) — the paper's core number
Hosts: EquiDiff = `train_equi_diffusion_unet_abs`; plain DP = `train_diffusion_unet`.
Run every cell at n_demo ∈ {50,100,200}, ≥3 seeds:
```
python train.py --config-name=<HOST> task_name=<VARIANT> n_demo=<N> training.seed=<S>
```
| host \ variant | baseline | keyed-C4 | global-rot |
|---|---|---|---|
| **EquiDiff** (C8) | ref | **should > baseline** | ≈ baseline (redundant) |
| **plain DP** (non-equi) | ref | should > baseline | should > baseline |

**The paper = the `(keyed − global)` delta on the EquiDiff host** (the non-equivariant info
keyed adds). **Cleanest falsification = the DP row**: if keyed helps DP but ties/hurts EquiDiff,
the destabilization hypothesis holds and the premise fails → run the DP row EARLY.

## 4. Positive-evidence artifacts (pre-empt "diffusion is already multimodal")
- histogram of raw `square_d0` insertion-yaw (nut yaw at the insertion subtask) → show **unimodal**;
- roll out un-augmented EquiDiff and show it **never produces the other 3 C4 orientations**.
(Claude can write these scripts offline once the hdf5 field layout is confirmed.)

## 5. Interpreting results
- keyed > global on EquiDiff, gain grows as n_demo shrinks, no reversal at 200 → thesis holds.
- keyed ≈ global on EquiDiff → the discrete-C4 info is not usable by BC diffusion → method dead; report honestly.
- Watch precision drop at n_demo=200 (extra modes split probability mass).
