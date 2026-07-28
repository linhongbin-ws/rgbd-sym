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
- `equidiff_hook.py` — adapter sketch: subclass `RobomimicReplayImageDataset`, apply the
  aug in `__getitem__` (low-dim poses + 10-dim abs action). RGB obs can't be pose-augmented
  → use low_dim / voxel / point_cloud, or re-render. Reuse MimicGen subtask boundaries as
  the phase index (fallback: gripper-close / distance heuristic).
- `viz_mea_diff_aug.py` → `context/plan/mea_diff_aug.png` (data-side change, real augmentor).
- `viz_mea_diff_pipeline.py` → `context/plan/mea_diff_pipeline.png` (network + task + control).

## Config toggle (in the EquiDiff task yaml)
```
dataset._target_: equidiff_hook.MEAAugImageDataset
dataset.mea_mode: phase     # phase | global(redundant) | off(baseline)
dataset.keyed_order: 4       # 4 = square(C4) ; 0 = round(SO(2) control)
dataset.reflect_prob: 0.0
```

## Status
Augmentation math + tests DONE & passing offline. Not yet wired into a real EquiDiff
checkout (needs the repo on a GPU box). Design + risks: `context/mea_v2_new_idea/iclr_diffusion_plan.md`.
