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

## 2. Produce the dataset variants (the "arms")  — PATH A, verified against NVlabs/mimicgen
The keyed-C4 hook is a ONE-LINER (see `mea_diff/mimicgen_c4_hook.py`, math sandbox-tested):
in `DataGenerator.generate()`, right after
`cur_object_pose = cur_datagen_info.object_poses[subtask_object_name]`, post-multiply the
CURRENT peg pose by `Rz(k·90°)` **only** when `subtask_object_name=='square_peg'` (the
insertion subtask). Post-mult = peg's LOCAL frame → the insertion segment re-orients k·90°
about the peg axis, peg position fixed, grasp subtask untouched (no re-grasp). Valid because
the square nut is C4 about the peg and robosuite success (`objects_on_pegs`) is orientation-agnostic.

```
# on the GPU box (mimicgen + robosuite + equidiff installed):
python mimicgen/scripts/download_datasets.py --dataset_type source --tasks square      # 10 human seed demos
python mimicgen/scripts/prepare_src_dataset.py --dataset square.hdf5 \
       --env_interface MG_Square --env_interface_type robosuite -o square_src.hdf5      # adds DatagenInfo
# apply the patch in mea_diff/mimicgen_c4_hook.py::PATCH to data_generator.py, then:
python mimicgen/scripts/generate_dataset.py --config square_d0.json                     # _c4_key=0  -> BASELINE
python mimicgen/scripts/generate_dataset.py --config square_d0.json --keys 0 1 2 3      # loop k    -> KEYED-C4
#   (same experiment.seed for both; success-filter drops infeasible 180/270 carries)
python robomimic/scripts/dataset_states_to_obs.py --input <gen>.hdf5 --output <gen>_obs.hdf5 \
       --camera_names agentview robot0_eye_in_hand                                       # re-render valid RGB
python equi_diffpo/scripts/robomimic_dataset_conversion.py -i <gen>_obs.hdf5 -o <gen>_abs.hdf5 -n 12  # 10-D abs
```
| variant hdf5 | key | notes |
|---|---|---|
| `square_d0_abs` (BASELINE) | `_c4_key=0`/None | the standard pipeline; must byte-match |
| `square_d0_keyedC4_abs` (**the method**) | loop k∈{0,1,2,3} | 4 orientations per source demo |
| `square_d0_globalrot_abs` (CONTROL) | whole-scene SO(2) per demo | redundant w/ EquiDiff by design — mainly for the DP host |

Exact hooks/uncertainties (controller abs-mode, peg1==square peg, D0-vs-D1, k=180/270 IK yield):
see the `wnvooyj2y` workflow result. Round control (optional, not headline): same pipeline with a
genuinely SO(2)-symmetric grasp (annular peg, centered grasp, no handle); predict `square_gain > round_gain`.

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
