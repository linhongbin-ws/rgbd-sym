# MEA v2 (diffusion route) — install & run

Standalone setup for the MEA v2 experiments: **keyed-C4 augmentation for Equivariant
Diffusion Policy (EquiDiff) on MimicGen Square insertion**. Method/pivot: see
`falsification_and_pivot.md`; experiment matrix: `RUNBOOK.md`.

**Hardware**: a GPU is required for **training** (EquiDiff ≈ 22 GB at batch 128 → an
RTX 3090/24 GB fits; an 8 GB card can do reduced-scale dev, see §5). Data generation uses
MuJoCo rendering (GPU or CPU/osmesa). The augmentation *math/tests* run CPU-only.

---

# Download

```sh
# a working dir for the diffusion stack
mkdir -p ~/mea && cd ~/mea
git clone https://github.com/pointW/equidiff.git
git clone https://github.com/NVlabs/mimicgen_environments.git      # EquiDiff's envs (v0.1.0)
git clone https://github.com/NVlabs/mimicgen.git                   # datagen (data_generator.py)
# the MEA augmentation code (this repo) — only mea_diff/ is needed:
git clone https://github.com/linhongbin-ws/rgbd-sym.git -b meav2   # -> rgbd-sym/mea_diff/
```

---

# Install

## Conda install (EquiDiff)

- Install [miniconda](https://docs.anaconda.com/miniconda/) (or Mambaforge — faster solver).

- System deps for MuJoCo (Ubuntu):
    ```sh
    sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf gfortran
    ```
- Create the env from EquiDiff's spec (this pins torch/CUDA/diffusers/escnn/robomimic):
    ```sh
    cd ~/mea/equidiff
    conda env create -f conda_environment.yaml      # (mamba env create -f ... if using mamba)
    conda activate equidiff
    pip list | grep mujoco                           # MUST be mujoco==2.3.2
    ```
- Install MimicGen envs (the version EquiDiff was built against):
    ```sh
    cd ~/mea/mimicgen_environments
    git checkout 081f7dbbe5fff17b28c67ce8ec87c371f32526a9
    pip install -e .
    ```
- Install the MimicGen **datagen** package (provides `data_generator.py`, `generate_dataset.py`,
  `prepare_src_dataset.py` used to make the keyed dataset). ⚠️ verify these scripts import
  cleanly alongside the pinned envs; if the two mimicgen packages clash, generate the data in a
  separate env and only *train* in `equidiff`.
    ```sh
    cd ~/mea/mimicgen && pip install -e .
    ```
- Make the MEA augmentation code importable (it's pure numpy/scipy):
    ```sh
    cd ~/mea/rgbd-sym/mea_diff
    python test_phase_aug.py            # -> ALL PASS   (offline, no GPU)
    python test_action_consistency.py   # -> ALL PASS   (synthetic smoke)
    python mimicgen_c4_hook.py          # -> PASS       (C4 hook math)
    ```

---

# Data

## Baseline Square dataset (download-ready)

```sh
cd ~/mea/equidiff
python ~/mea/mimicgen/mimicgen/scripts/download_datasets.py --dataset_type core --tasks square_d0
# convert to the 10-D ABSOLUTE action format EquiDiff trains on:
python equi_diffpo/scripts/robomimic_dataset_conversion.py \
    -i data/robomimic/datasets/square_d0/square_d0.hdf5 \
    -o data/robomimic/datasets/square_d0/square_d0_abs.hdf5 -n 12
```

## Keyed-C4 dataset (the method) — PATH A

The keyed rotation is a **one-line hook** (verified; math in `mea_diff/mimicgen_c4_hook.py`).
Apply it in MimicGen's generator, then generate + re-render + convert:

```sh
# 1. source seed demos + datagen info
python ~/mea/mimicgen/mimicgen/scripts/download_datasets.py --dataset_type source --tasks square
python ~/mea/mimicgen/mimicgen/scripts/prepare_src_dataset.py \
    --dataset <square_source>.hdf5 --env_interface MG_Square --env_interface_type robosuite \
    -o square_src.hdf5

# 2. apply the C4 patch: in mimicgen/datagen/data_generator.py :: DataGenerator.generate(),
#    right after   cur_object_pose = cur_datagen_info.object_poses[subtask_object_name]
#    paste the block in mea_diff/mimicgen_c4_hook.py :: PATCH  (post-mult peg pose by Rz(k*90));
#    set data_generator._c4_key per demo in generate_dataset.py (loop k in {0,1,2,3}).

# 3. generate BASELINE (k=0) and KEYED-C4 (k looped), SAME experiment.seed
python ~/mea/mimicgen/mimicgen/scripts/generate_dataset.py --config square_d0.json           # baseline
python ~/mea/mimicgen/mimicgen/scripts/generate_dataset.py --config square_d0.json --keys 0 1 2 3

# 4. success-filter (drops infeasible 180/270 carries), re-render valid RGB, convert to abs
python ~/mea/mimicgen_environments/.../merge_hdf5.py ...                                       # keep successes
python robomimic/scripts/dataset_states_to_obs.py --input square_d0_keyedC4.hdf5 \
    --output square_d0_keyedC4_obs.hdf5 --camera_names agentview robot0_eye_in_hand
python equi_diffpo/scripts/robomimic_dataset_conversion.py \
    -i square_d0_keyedC4_obs.hdf5 -o data/robomimic/datasets/square_d0_keyedC4/square_d0_keyedC4_abs.hdf5 -n 12
```
Verify-on-repo (from the `wnvooyj2y` scout): controller must be OSC_POSE **absolute**
(`control_delta=False`) or use the delta→abs conversion; confirm `peg1` is the square peg;
choose Square_D0 (cleanest) vs D1/D2 (D1/D2 add init z-rot that partly overlaps C4).

---

# Run

## 1. Blocking gate — action conjugation on REAL data (do this first)
```sh
cd ~/mea/rgbd-sym/mea_diff
python test_action_consistency.py --hdf5 ~/mea/equidiff/data/robomimic/datasets/square_d0/square_d0_abs.hdf5 --demo demo_0
```
**If this FAILS the rot6d/quat conjugation is mis-signed — do NOT train.**

## 2. Reproduce baselines (EquiDiff + plain DP), Square
```sh
cd ~/mea/equidiff
python train.py --config-name=train_equi_diffusion_unet_abs task_name=square_d0 n_demo=100   # EquiDiff (C8)
python train.py --config-name=train_diffusion_unet          task_name=square_d0 n_demo=100   # plain DP
```

## 3. The headline ablation (within-Square) — host × variant × demo × seed
Point `task_name` at each dataset variant (baseline / keyed-C4 / global-rot); sweep
`n_demo ∈ {50,100,200}`, ≥3 seeds:
```sh
python train.py --config-name=train_equi_diffusion_unet_abs task_name=square_d0_keyedC4 n_demo=100 training.seed=0
```
| host \ variant | baseline | keyed-C4 | global-rot |
|---|---|---|---|
| **EquiDiff** | ref | **should > baseline** | ≈ baseline (redundant) |
| **plain DP** | ref | should > baseline | should > baseline |
**The paper = `(keyed − global)` on the EquiDiff host.  Run the DP row EARLY** (if keyed helps
DP but ties/hurts EquiDiff, the premise fails — cleanest falsification).

## 4. Positive-evidence artifact
```sh
python ~/mea/rgbd-sym/mea_diff/artifact_insertion_yaw.py --hdf5 .../square_d0.hdf5
# -> demos cover only 1 of the 4 C4 slots => the other 3 are out-of-support (what keyed adds)
```

---

# Notes

- **CPU render** (data gen without a GPU): prefix `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa`.
- **GPU memory**: default ~22 GB @ batch 128. On an 8 GB card reduce with
  `policy.enc_n_hidden=64 dataloader.batch_size=64` (dev/smoke only; full runs on the 3090).
- **Offline W&B**: `WANDB_MODE=offline` to avoid pushing throwaway runs; `HYDRA_FULL_ERROR=1`
  for readable config errors.
- **Open verify points** (see `RUNBOOK.md` / the `wnvooyj2y` scout): the two mimicgen packages'
  compatibility, the abs-action conversion route, k=180/270 IK success yield (success-filter),
  and EquiDiff's exact expected obs/action keys.
