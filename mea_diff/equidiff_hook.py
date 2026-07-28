# -*- coding: utf-8 -*-
"""Adapter sketch: how mea_diff.phase_aug plugs into EquiDiff's dataset.

Drop this next to equi_diffpo/dataset/ and select it in the task config
(`dataset._target_: equidiff_hook.MEAAugImageDataset`). It subclasses EquiDiff's
RobomimicReplayImageDataset and applies the phase-indexed augmentation ON THE FLY in
__getitem__, transforming the low-dim obs poses + the 10-dim absolute action.

IMPORTANT (verified): RGB obs are pre-rendered and CANNOT be pose-augmented here. Use a
geometry-bearing modality — low_dim (object poses) and/or the voxel/point_cloud dataset —
for the geometric augmentation. This file shows the low-dim pose + action path (which is
what EquiDiff's equivariant encoder consumes alongside images). For image obs you must
either train the low_dim/voxel variant, or precompute a re-rendered augmented hdf5.

Phase source: reuse MimicGen's native object-centric subtask segmentation
(pre-grasp subtask -> APPROACH, insertion subtask -> ENGAGE). If those boundaries are not
persisted in the distributed hdf5, fall back to a grasp-step heuristic (gripper_qpos
close-flip, or gripper<->object distance threshold) computed once per demo.
"""
import numpy as np

# In the real repo:
# from equi_diffpo.dataset.robomimic_replay_image_dataset import RobomimicReplayImageDataset
from phase_aug import MEAPhaseAug, APPROACH, ENGAGE


class MEAAugImageDataset:  # (RobomimicReplayImageDataset):  <- real base
    """Illustrative wrapper. In the repo, inherit RobomimicReplayImageDataset and only
    override __getitem__ to post-process the sample dict."""

    def __init__(self, *args, mea_mode="phase", keyed_order=4, reflect_prob=0.0,
                 obj_key="object", **kwargs):
        # super().__init__(*args, **kwargs)
        self.aug = MEAPhaseAug(mode=mea_mode, keyed_order=keyed_order, reflect_prob=reflect_prob)
        self.obj_key = obj_key

    def _phase_and_anchors(self, obs, action):
        """Return (phase[T], k_grasp, hole_xy). Prefer MimicGen subtask boundaries;
        fallback = first frame the gripper closes / is within contact distance."""
        # ---- fallback heuristic (replace with subtask boundaries when available) ----
        eef = obs["robot0_eef_pos"]; obj = obs[self.obj_key][..., 0:3]
        d = np.linalg.norm(eef[:, :2] - obj[:, :2], axis=1)
        gq = obs.get("robot0_gripper_qpos")
        k = None
        if gq is not None:
            closed = (gq[:, 0] - gq[:, 1]) if gq.shape[-1] >= 2 else gq[:, 0]
            flips = np.where(np.abs(np.diff(closed)) > 0.5 * np.ptp(closed) + 1e-9)[0]
            k = int(flips[0]) + 1 if len(flips) else None
        if k is None:
            below = np.where(d < 0.02)[0]
            k = int(below[0]) if len(below) else len(eef) // 2
        phase = np.array([APPROACH if t < k else ENGAGE for t in range(len(eef))], int)
        hole_xy = obj[k, :2]  # placeholder: real hole/peg anchor from the insertion subtask ref frame
        return phase, k, hole_xy

    def __getitem__(self, idx):
        data = super().__getitem__(idx)          # {'obs': {...}, 'action': (T,10)}
        obs = {k: np.asarray(v) for k, v in data["obs"].items()}
        action = np.asarray(data["action"])
        if "robot0_eef_pos" not in obs or self.obj_key not in obs:
            return data                          # no geometry to augment (pure-RGB) -> skip

        phase, k, hole_xy = self._phase_and_anchors(obs, action)
        s = dict(
            action=action.copy(),
            eef_pos=obs["robot0_eef_pos"].copy(),
            eef_quat=obs["robot0_eef_quat"].copy(),           # xyzw
            gripper=obs.get("robot0_gripper_qpos"),
            obj_pos=obs[self.obj_key][..., 0:3].copy(),
            obj_quat=obs[self.obj_key][..., 3:7].copy(),
            hole_xy=hole_xy, phase=phase, k_grasp=k,
        )
        out = self.aug.augment(s)
        # write back
        data["action"] = out["action"].astype(np.float32)
        obs["robot0_eef_pos"] = out["eef_pos"].astype(np.float32)
        obs["robot0_eef_quat"] = out["eef_quat"].astype(np.float32)
        obs[self.obj_key][..., 0:3] = out["obj_pos"]
        obs[self.obj_key][..., 3:7] = out["obj_quat"]
        data["obs"] = obs
        return data


# config toggle summary (put in the task yaml):
#   dataset._target_: equidiff_hook.MEAAugImageDataset
#   dataset.mea_mode: phase        # phase | global | off
#   dataset.keyed_order: 4         # 4 = square (C4) ; 0 = round (SO(2) control)
#   dataset.reflect_prob: 0.0
