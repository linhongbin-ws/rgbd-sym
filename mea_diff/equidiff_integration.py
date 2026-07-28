# -*- coding: utf-8 -*-
"""How to integrate the MEA v2 C4-engage augmentation with EquiDiff — CORRECTED after
reading the real repo (pointW/equidiff).

FINDING (why the naive dataloader hook does NOT work for the IMAGE policy):
  1. obs are PRE-RENDERED RGB (agentview_image, robot0_eye_in_hand_image [3,84,84]);
     RobomimicReplayImageDataset.__getitem__ only does /255 — you cannot pose-augment
     pixels. A 90° whole-IMAGE rotation = whole-scene rotation = redundant with EquiDiff's
     C8; rotating only the object needs segmentation + inpainting.
  2. obs is a 2-frame window (n_obs_steps=2) but action is the full 16-step horizon
     (horizon=16) — different lengths; the phase index needs the WHOLE trajectory.
  3. The GRASPED object cannot be validly re-oriented at load time (fixed-finger grasp
     would slip) — a keyed re-orientation requires re-grasping from the start of the demo.

=> The C4-engage augmentation must be produced OFFLINE as new demos, not in __getitem__.

------------------------------------------------------------------------------------------
RECOMMENDED PATH (A): MimicGen data-generation with a keyed transform on the insertion subtask
------------------------------------------------------------------------------------------
MimicGen already generates demos by transforming object-centric subtask segments (each
anchored to an object frame) and RE-SOLVING + RE-RENDERING. The insertion subtask is
anchored on the nut/peg object frame. MEA v2 = bias that subtask's object-frame target
orientation to include the object POINT-GROUP {0,90,180,270} (square) — i.e. generate the
same insertion at C4-equivalent yaws. Output: valid images + valid 10-D abs actions +
consistent grasp, for free.

Where: MimicGen task/datagen config — the SubtaskConfig for the insertion subtask
(object_ref + selection_strategy). Add a C4 rotation to the sampled subtask transform.
`phase_aug.transform_sixd / transform_point / rot_z` give the exact action math to keep
labels consistent if you post-process instead of regenerate.

Cost: run MimicGen generation on the GPU box (MuJoCo render). Round control (SO(2)-symmetric
grasp) is generated the SAME way, changing only the peg geometry.

------------------------------------------------------------------------------------------
FALLBACK PATH (B): state-based / point-cloud EquiDiff, dataloader aug at load time
------------------------------------------------------------------------------------------
If (and only if) you train a LOW-DIM (object-pose) or POINT-CLOUD EquiDiff variant — no RGB —
the geometric transform IS analytic at load time and self-consistent:
  * low-dim: transform robot0_eef_pos/quat + object pose + the action (phase_aug does this).
  * point-cloud: rotate the {gripper+held-object} points (spatial crop near the gripper at
    engagement; a 90° C4 rotation is exact on points) + poses + action.
Caveats: (a) weaker/less comparable than EquiDiff's headline IMAGE result; (b) the point-
cloud per-entity crop is approximate; (c) obs(window) vs action(horizon) length mismatch
must be handled (transform the full action horizon with the per-sample element, and the
2-frame obs window consistently). Use ONLY as a cheap CPU pilot / ablation, not the headline.

The dataloader subclass below is the FALLBACK-B scaffold (low-dim path). For the headline
result use PATH A.
"""
import numpy as np
from phase_aug import MEAPhaseAug, APPROACH, ENGAGE

# from equi_diffpo.dataset.robomimic_replay_image_dataset import RobomimicReplayImageDataset
# verified signature:
#   __init__(shape_meta, dataset_path, horizon=1, pad_before=0, pad_after=0, n_obs_steps=None,
#            abs_action=False, rotation_rep='rotation_6d', use_legacy_normalizer=False,
#            use_cache=False, seed=42, val_ratio=0.0, n_demo=100)
#   __getitem__ -> {'obs': {rgb.../255, lowdim...}, 'action': (horizon,10)}  (obs sliced to n_obs_steps)


class MEALowdimAugDataset:  # (RobomimicReplayImageDataset)  <- real base for FALLBACK-B, low-dim only
    """FALLBACK-B ONLY. Valid when shape_meta has NO 'rgb' obs (low-dim / pose obs).
    Do NOT use with image obs (see FINDING above)."""

    def __init__(self, *args, mea_mode="phase", keyed_order=4, approach_aug=False,
                 reflect_prob=0.0, obj_key="object", **kwargs):
        # super().__init__(*args, **kwargs)
        self.aug = MEAPhaseAug(mode=mea_mode, keyed_order=keyed_order,
                               approach_aug=approach_aug, reflect_prob=reflect_prob)
        self.obj_key = obj_key

    def __getitem__(self, idx):
        data = super().__getitem__(idx)                 # obs window + full-horizon action
        obs = data["obs"]
        if any(getattr(v, "ndim", 0) == 4 for v in obs.values()):
            raise RuntimeError("MEALowdimAugDataset used with image obs — see FINDING; use PATH A")
        # NOTE: obs frames (n_obs_steps) and action frames (horizon) differ in length; sample
        # ONE augmentation element and apply it consistently to BOTH. Phase must come from the
        # full trajectory (precompute per-demo phase boundaries at init), not this window.
        # ... (left as scaffold: the honest recommendation is PATH A, not this) ...
        return data


# ------------------------------------------------------------------------------------------
# config guidance (no dataloader change for PATH A):
#   - PATH A: generate augmented demos -> data/.../square_d0_mea.hdf5 -> convert to _abs ->
#     train EquiDiff/DP on it exactly like the baseline. The "arm" = which hdf5 you train on
#     (baseline / mea-keyed / global-rotated). No code change to EquiDiff.
#   - Ablation arms are then just: task_name x {baseline_abs, keyedC4_abs, globalrot_abs} hdf5,
#     each on host in {equi_diffusion_unet_abs (EquiDiff), diffusion_unet (plain DP)}.
# ------------------------------------------------------------------------------------------
