# MEA v2 (diffusion route) — install & run

Standalone setup for the MEA v2 experiments: **keyed-C4 augmentation for Equivariant
Diffusion Policy (EquiDiff) on MimicGen Square insertion**. Method/pivot: see
`falsification_and_pivot.md`; experiment matrix: `RUNBOOK.md`.

**Hardware**: a GPU is required for **training** (EquiDiff ≈ 22 GB at batch 128 → an
RTX 3090/24 GB fits; an 8 GB card can do reduced-scale dev, see §5). Data generation uses
MuJoCo rendering (GPU or CPU/osmesa). The augmentation *math/tests* run CPU-only.

**⚠️ TWO separate conda envs are required** (verified via workflow `wlsj7ps2o`): NVlabs/mimicgen
datagen needs **ARISE robomimic `d0b37cf`** (has `env.base_env`, `experiment.logging`), while
EquiDiff pins a customized **pointW robomimic `8aad5b3`** (no `base_env`/`logging`). The two
robomimic forks are mutually incompatible → do NOT mix them (that is the `base_env`/`logging`
cascade). **The handoff is safe because robosuite (`b9d8d3de`==v1.4.1) and mujoco (`2.3.2`) are
IDENTICAL in both envs**, so the datagen HDF5 (states+xml) re-renders deterministically in the
`equidiff` env. Plan: **generate** in `mimicgen_datagen`, **render+convert+train** in `equidiff`.

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
    # Ubuntu <=22.04:
    sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf gfortran
    # Ubuntu 24.04+ (libgl1-mesa-glx was split -> use libgl1 + libglx-mesa0):
    sudo apt install -y libosmesa6-dev libgl1 libglx-mesa0 libglfw3 patchelf gfortran
    ```
- Create the env from EquiDiff's spec (this pins torch/CUDA/diffusers/escnn/robomimic):
    ```sh
    cd ~/mea/equidiff
    conda env create -f conda_environment.yaml      # (mamba env create -f ... if using mamba)
    conda activate equidiff
    pip list | grep mujoco                           # MUST be mujoco==2.3.2
    ```
## Conda install (MimicGen datagen — SEPARATE env)

Do NOT install MimicGen datagen into the `equidiff` env (its robomimic fork can't run it).
Create a dedicated env with version-matched robosuite/mujoco:
```sh
cd ~/mea                                   # a workdir for the ARISE checkouts
bash rgbd-sym/mea_diff/setup_datagen_env.sh    # -> conda env 'mimicgen_datagen'
# pins: python3.8, mujoco==2.3.2, robosuite b9d8d3de(v1.4.1), robomimic d0b37cf(ARISE), NVlabs/mimicgen
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
python ../mimicgen/mimicgen/scripts/download_datasets.py --dataset_type core --tasks square_d0
# convert to the 10-D ABSOLUTE action format EquiDiff trains on:
python equi_diffpo/scripts/robomimic_dataset_conversion.py \
    -i data/robomimic/datasets/square_d0/square_d0.hdf5 \
    -o data/robomimic/datasets/square_d0/square_d0_abs.hdf5 -n 12
```

## Keyed-C4 dataset (the method) — PATH A, in the `mimicgen_datagen` env

The keyed rotation is a **one-line hook** (math in `mea_diff/mimicgen_c4_hook.py`).

**A. GENERATE (in `conda activate mimicgen_datagen`, from `~/mea/mimicgen`):**
```sh
python mimicgen/scripts/download_datasets.py --dataset_type source --tasks square
python mimicgen/scripts/prepare_src_dataset.py \
    --dataset datasets/source/square.hdf5 --env_interface MG_Square --env_interface_type robosuite \
    --output square_src.hdf5
# apply mea_diff/mimicgen_c4_hook.py::PATCH to mimicgen/datagen/data_generator.py::generate()
#   (post-mult the square_peg pose by Rz(k*90)); set data_generator._c4_key per demo in generate_dataset.py.
python mimicgen/scripts/generate_core_configs.py            # writes /tmp/core_configs/demo_src_square_task_D0.json
#   -> edit that json: source.dataset_path = square_src.hdf5 ; generation.path = out dir
python mimicgen/scripts/generate_dataset.py --config /tmp/core_configs/demo_src_square_task_D0.json --auto-remove-exp  # baseline (_c4_key=0)
python mimicgen/scripts/generate_dataset.py --config /tmp/core_configs/demo_src_square_task_D0.json --auto-remove-exp  # keyed (loop k over 0..3)
```

**B. HANDOFF → `equidiff` env** (render + convert + train; robosuite/mujoco match, so states replay exactly):
```sh
conda activate equidiff
# copy the generated low-dim hdf5 into equidiff's data tree:
cp <gen>.hdf5 ~/mea/equidiff/data/robomimic/datasets/square_d0_keyedC4/square_d0_keyedC4.hdf5
# ⚠ CRITICAL: overwrite data.attrs['env_args'] with EquiDiff's downloaded square_d0.hdf5 env_args
#   (ARISE-mimicgen writes env kwargs the pointW loader rejects; robosuite is identical so states still replay):
python - <<'PY'
import h5py
ref=h5py.File("data/robomimic/datasets/square_d0/square_d0.hdf5"); ea=ref["data"].attrs["env_args"]; ref.close()
f=h5py.File("data/robomimic/datasets/square_d0_keyedC4/square_d0_keyedC4.hdf5","a")
f["data"].attrs["env_args"]=ea; f.close(); print("env_args substituted")
PY
# render obs IN THE EQUIDIFF ENV (not the datagen env), then convert to 10-D abs:
python equi_diffpo/scripts/dataset_states_to_obs.py --input .../square_d0_keyedC4.hdf5 \
    --output .../square_d0_keyedC4_obs.hdf5 --camera_names agentview robot0_eye_in_hand   # 84x84
python equi_diffpo/scripts/robomimic_dataset_conversion.py \
    -i .../square_d0_keyedC4_obs.hdf5 -o .../square_d0_keyedC4_abs.hdf5 -n 12
```
Gotchas (workflow `wlsj7ps2o`): **render inside `equidiff`, not datagen** (obs-container format
differs between the robomimic forks; values are fine since robosuite matches). Camera names +
84×84 must match EquiDiff's config. Use **Square_D0** (D1/D2 add init z-rot that overlaps C4).
Success-filter keeps physically-valid insertions (k=180/270 may miss IK).

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
