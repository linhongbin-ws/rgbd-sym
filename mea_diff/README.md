# mea_diff — MEA v2 phase-indexed augmentation for Equivariant Diffusion Policy

Host = **EquiDiff** (`github.com/pointW/equidiff`), data = **MimicGen** (robomimic hdf5).
MEA is a **data augmentation** (no network change).

> **⚠️ 2026-07-28 pivot after adversarial falsification** (`context/mea_v2_new_idea/falsification_and_pivot.md`):
> the **only** surviving contribution is the **ENGAGE C4 augmentation**. The `approach`
> arm is redundant + dominated by eye-in-hand/relative-action (arXiv:2505.13431) → DEMOTED
> to opt-in (`approach_aug=True`). The `global` arm is a redundant negative control only.

- **ENGAGE (the contribution)**: rotate the grasped object+gripper about the **object's own
  axis** by the object **point-group** (square peg → C4; round → SO(2)) → out-of-support
  keyed insertion yaws the C8 host / diffusion multimodality / canonicalization / relative
  frames cannot inject. (Keyed element sampled ONCE per episode; anchored at the object,
  not the hole, so the carry stays feasible; optional `workspace_radius` drops infeasible poses.)
- **APPROACH (demoted, opt-in)**: rotate the gripper about the fixed object by α.

## Files
- `phase_aug.py` — `MEAPhaseAug` + rot6d/quat math (pytorch3d rot6d convention verified)
  + `make_synthetic_episode` (toy data for tests/viz). Modes: `phase | global | off`.
  `keyed_order=4` (square) / `0` (round control).
- `test_phase_aug.py` — offline correctness tests (numpy only, **no GPU**):
  `source bash/init.sh && python mea_diff/test_phase_aug.py` → ALL PASS.
- `test_action_consistency.py` — **BLOCKING GATE**: closed-loop action-conjugation check
  (action-in-eef-frame invariant under aug; reflection mirrors by F). Synthetic smoke passes;
  run on real data before training: `python mea_diff/test_action_consistency.py --hdf5 square_d0_abs.hdf5`.
  **If it FAILS on real data, the rot6d/quat conjugation is mis-signed — do NOT train.**
- `equidiff_integration.py` — **how to integrate (corrected after reading the real repo)**:
  the dataloader hook is WRONG for the image policy (pre-rendered RGB; 2-frame obs window vs
  16-step action horizon; grasped object can't be re-oriented at load time). Use **PATH A =
  offline MimicGen datagen** with a C4 rotation on the insertion subtask. Low-dim dataloader
  is a scoped FALLBACK-B only.
- `RUNBOOK.md` — the on-GPU recipe: setup, blocking gate, the three dataset variants
  (baseline / keyed-C4 / global-rot), and the host×variant×demo×seed ablation matrix.
- `viz_mea_diff_aug.py` → `context/plan/mea_diff_aug.png` (data-side change, real augmentor).
- `viz_mea_diff_pipeline.py` → `context/plan/mea_diff_pipeline.png` (network + task + control).

## Integration (PATH A — no EquiDiff code change; see equidiff_integration.py + RUNBOOK.md)
The "arm" = which hdf5 you train on × which host. Produce `square_d0_keyedC4_abs` via MimicGen
datagen, then `train.py --config-name=train_equi_diffusion_unet_abs task_name=square_d0 ...`.

## Status
Augmentation math + tests DONE & passing offline. Not yet wired into a real EquiDiff
checkout (needs the repo on a GPU box). Design + risks: `context/mea_v2_new_idea/iclr_diffusion_plan.md`.
