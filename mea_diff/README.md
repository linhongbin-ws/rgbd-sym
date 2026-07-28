# mea_diff — MEA v2 phase-indexed augmentation for Equivariant Diffusion Policy

Host = **EquiDiff** (`github.com/pointW/equidiff`), data = **MimicGen** (robomimic hdf5).
MEA is a **data augmentation** (no network change in v1); it injects what EquiDiff's
global SO(2) equivariance cannot express, indexed by manipulation phase:

- **APPROACH**: rotate the gripper about the (fixed) target object by α → new relative
  approach angles (decays to 0 at grasp).
- **ENGAGE**: rotate the grasped object+gripper about the (fixed) hole by the object
  **point-group** (square peg → C4; round → SO(2)) → the keyed equivalent insertions.

## Files
- `phase_aug.py` — `MEAPhaseAug` + rot6d/quat math (pytorch3d rot6d convention verified)
  + `make_synthetic_episode` (toy data for tests/viz). Modes: `phase | global | off`.
  `keyed_order=4` (square) / `0` (round control).
- `test_phase_aug.py` — offline correctness tests (numpy only, **no GPU**):
  `source bash/init.sh && python mea_diff/test_phase_aug.py` → ALL PASS.
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
