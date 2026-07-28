# -*- coding: utf-8 -*-
"""MEA v2 — phase-indexed data augmentation for an equivariant diffusion policy
(host: Equivariant Diffusion Policy / EquiDiff, robomimic + MimicGen format).

WHAT IT ADDS OVER THE HOST (EquiDiff already has GLOBAL SO(2) equivariance, so a
whole-scene rotation is REDUNDANT). MEA injects the two things global equivariance
CANNOT express, indexed by the manipulation phase:

  - free APPROACH phase: rotate the GRIPPER (+ its action) about the target OBJECT by
    a continuous angle alpha (object + hole fixed), optionally decaying to 0 at grasp.
    -> new gripper<->object relative approach angle. NOT a whole-scene rotation.
  - ENGAGEMENT phase: rotate the grasped OBJECT+GRIPPER composite (+ action) about the
    HOLE axis by the object's POINT-GROUP element (square peg -> C4 {0,90,180,270};
    round -> continuous SO(2)), hole fixed. -> the keyed equivalent insertions.
    NOT a whole-scene rotation (the hole/rest of the scene stays put).

Conventions verified against pointW/equidiff:
  - action = 10-dim per arm = pos(3) + rotation_6d(6) + gripper(1), ABSOLUTE pose.
  - rotation_6d = pytorch3d convention = FIRST TWO ROWS of the rotation matrix, flat.
  - low-dim obs carry robot0_eef_pos(3), robot0_eef_quat(4, xyzw), robot0_gripper_qpos,
    and `object` (per-object pos+quat). These + the action are what we transform
    analytically (no render). RGB obs cannot be pose-augmented without a re-render.

This module is host-agnostic: it transforms a plain `Sample` dict of poses/action so it
can be unit-tested offline (numpy only) and then wired into EquiDiff's dataset
__getitem__ (see equidiff_hook.py).
"""
from __future__ import annotations
import numpy as np

try:
    from scipy.spatial.transform import Rotation as _R
    _HAVE_SCIPY = True
except Exception:                                    # pragma: no cover
    _HAVE_SCIPY = False


# --------------------------------------------------------------------------- #
# rotation_6d helpers (pytorch3d convention: first two ROWS of R, flattened)
# --------------------------------------------------------------------------- #
def sixd_to_mat(d6: np.ndarray) -> np.ndarray:
    """(...,6) -> (...,3,3) via Gram-Schmidt (matches pytorch3d rotation_6d_to_matrix)."""
    d6 = np.asarray(d6, float)
    a1, a2 = d6[..., 0:3], d6[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)           # rows b1,b2,b3


def mat_to_sixd(M: np.ndarray) -> np.ndarray:
    """(...,3,3) -> (...,6): first two ROWS flattened (matches matrix_to_rotation_6d)."""
    M = np.asarray(M, float)
    return np.concatenate([M[..., 0, :], M[..., 1, :]], axis=-1)


def rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def quat_to_mat(q: np.ndarray) -> np.ndarray:       # q = xyzw (robomimic/robosuite)
    if _HAVE_SCIPY:
        return _R.from_quat(np.asarray(q, float)).as_matrix()
    x, y, z, w = np.asarray(q, float)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def mat_to_quat(M: np.ndarray) -> np.ndarray:       # -> xyzw
    if _HAVE_SCIPY:
        return _R.from_matrix(np.asarray(M, float)).as_quat()
    M = np.asarray(M, float); t = np.trace(M)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2; w = 0.25 * s
        x = (M[2, 1] - M[1, 2]) / s; y = (M[0, 2] - M[2, 0]) / s; z = (M[1, 0] - M[0, 1]) / s
    else:
        i = int(np.argmax([M[0, 0], M[1, 1], M[2, 2]])); j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(M[i, i] - M[j, j] - M[k, k] + 1.0) * 2
        q = np.zeros(3); q[i] = 0.25 * s
        q[j] = (M[j, i] + M[i, j]) / s; q[k] = (M[k, i] + M[i, k]) / s
        w = (M[k, j] - M[j, k]) / s; x, y, z = q
    return np.array([x, y, z, w])


# --------------------------------------------------------------------------- #
# apply a world rotation R about a vertical axis through anchor (xy) to a pose
# --------------------------------------------------------------------------- #
def transform_point(p: np.ndarray, R: np.ndarray, anchor_xy: np.ndarray) -> np.ndarray:
    a = np.array([anchor_xy[0], anchor_xy[1], 0.0])
    return (R @ (np.asarray(p, float) - a)) + a


def transform_sixd(d6: np.ndarray, R: np.ndarray) -> np.ndarray:
    return mat_to_sixd(R @ sixd_to_mat(d6))


def transform_quat(q: np.ndarray, R: np.ndarray) -> np.ndarray:
    return mat_to_quat(R @ quat_to_mat(q))


# --------------------------------------------------------------------------- #
# the augmentor
# --------------------------------------------------------------------------- #
APPROACH, ENGAGE = 0, 1


class MEAPhaseAug:
    """Phase-indexed relative-pose augmentation.

    mode:
      'phase'  : APPROACH -> rotate gripper about object anchor by alpha (object fixed);
                 ENGAGE   -> rotate object+gripper about hole anchor by keyed element.
      'global' : one whole-scene Rz(theta) on every frame  (REDUNDANT w/ EquiDiff;
                 the ablation arm that should ~tie the equivariant baseline).
      'off'    : identity (host baseline).

    keyed_order: point-group order at engagement. 4 => square peg C4 (discrete
      {0,90,180,270}); 0/None => continuous SO(2) (round peg — no reduction; the
      unkeyed control arm, where MEA should add nothing over the SO(2)-equivariant net).
    """

    def __init__(self, mode="phase", keyed_order=4, approach_max_angle=np.pi,
                 approach_decay=True, reflect_prob=0.0, global_max_angle=np.pi, rng=None):
        assert mode in ("phase", "global", "off")
        self.mode = mode
        self.keyed_order = keyed_order
        self.approach_max_angle = float(approach_max_angle)
        self.approach_decay = bool(approach_decay)
        self.reflect_prob = float(reflect_prob)
        self.global_max_angle = float(global_max_angle)
        self.rng = rng or np.random

    # ---- keyed / approach angle sampling ----
    def _keyed_angle(self):
        if not self.keyed_order:                     # continuous SO(2) (round)
            return float(self.rng.uniform(-np.pi, np.pi))
        k = int(self.rng.integers(self.keyed_order)) if hasattr(self.rng, "integers") \
            else int(self.rng.randint(self.keyed_order))
        return 2.0 * np.pi * k / self.keyed_order

    def _approach_schedule(self, t, k_grasp, T):
        """alpha(t): sampled base angle, decaying to 0 at the grasp frame (approach_decay)."""
        base = float(self.rng.uniform(-self.approach_max_angle, self.approach_max_angle))
        if not self.approach_decay or k_grasp is None or k_grasp <= 0:
            return base
        return base * max(0.0, (k_grasp - t) / k_grasp)

    def augment(self, s: dict) -> dict:
        """s keys (all np arrays):
             action     (T,10)  pos3 + rot6d6 + grip1
             eef_pos    (T,3), eef_quat (T,4 xyzw), gripper (T,·)   [low-dim obs]
             obj_pos    (T,3), obj_quat (T,4)     [the manipulated object]
             hole_xy    (2,) or (T,2)             [fixed insertion anchor]
             phase      (T,)  in {APPROACH, ENGAGE}
             k_grasp    int    (first engage frame; for decay)
             points     optional (N,3), point_obj_mask optional (N,) bool  [pc obs]
        Returns a NEW dict with the same keys, transformed. RGB obs are NOT handled
        here (need re-render) — the caller passes low-dim / point-cloud fields only.
        """
        s = {k: (np.array(v) if isinstance(v, np.ndarray) else v) for k, v in s.items()}
        if self.mode == "off":
            return s
        T = len(s["action"])
        act, ep, eq = s["action"], s["eef_pos"], s["eef_quat"]
        op, oq = s.get("obj_pos"), s.get("obj_quat")
        phase = s.get("phase", np.zeros(T, int))
        k_grasp = s.get("k_grasp", None)
        hole = s.get("hole_xy", np.zeros(2))
        hole_t = (lambda t: hole[t]) if np.ndim(hole) == 2 else (lambda t: hole)

        do_reflect = self.rng.uniform() < self.reflect_prob

        if self.mode == "global":
            theta = float(self.rng.uniform(-self.global_max_angle, self.global_max_angle))
            R = rot_z(theta)
            anchor = np.zeros(2)                      # about world origin (whole scene)
            for t in range(T):
                self._apply_frame(s, t, R, anchor, move_obj=True)
            if do_reflect:
                self._reflect(s)
            return s

        # phase mode
        alpha = self._approach_schedule  # closure
        for t in range(T):
            if phase[t] == APPROACH:
                a = op[t, :2] if op is not None else np.zeros(2)      # object anchor
                R = rot_z(alpha(t, k_grasp, T))
                self._apply_frame(s, t, R, a, move_obj=False)        # gripper only
            else:                                                     # ENGAGE
                R = rot_z(self._keyed_angle())
                self._apply_frame(s, t, R, hole_t(t), move_obj=True)  # object+gripper
        if do_reflect:
            self._reflect(s)
        return s

    def _apply_frame(self, s, t, R, anchor, move_obj):
        # action (target pose) — the gripper's commanded pose
        s["action"][t, 0:3] = transform_point(s["action"][t, 0:3], R, anchor)
        s["action"][t, 3:9] = transform_sixd(s["action"][t, 3:9], R)
        # eef obs
        s["eef_pos"][t] = transform_point(s["eef_pos"][t], R, anchor)
        s["eef_quat"][t] = transform_quat(s["eef_quat"][t], R)
        # object obs (only when the object moves with the transform)
        if move_obj and s.get("obj_pos") is not None:
            s["obj_pos"][t] = transform_point(s["obj_pos"][t], R, anchor)
            s["obj_quat"][t] = transform_quat(s["obj_quat"][t], R)

    def _reflect(self, s):
        # planar mirror across the x-axis (world y -> -y): pos.y, and the rotation.
        F = np.diag([1.0, -1.0, 1.0])
        for t in range(len(s["action"])):
            s["action"][t, 0:3] = F @ s["action"][t, 0:3]
            s["action"][t, 3:9] = mat_to_sixd(F @ sixd_to_mat(s["action"][t, 3:9]) @ F)
            s["eef_pos"][t] = F @ s["eef_pos"][t]
            s["eef_quat"][t] = mat_to_quat(F @ quat_to_mat(s["eef_quat"][t]) @ F)
            if s.get("obj_pos") is not None:
                s["obj_pos"][t] = F @ s["obj_pos"][t]
                s["obj_quat"][t] = mat_to_quat(F @ quat_to_mat(s["obj_quat"][t]) @ F)


# --------------------------------------------------------------------------- #
# synthetic episode (for offline tests + visualization; NOT real MimicGen data)
# --------------------------------------------------------------------------- #
def make_synthetic_episode(T=16, k_grasp=6, seed=0):
    """A toy nut-insertion trajectory: gripper approaches a nut at `obj0`, grasps at
    k_grasp, carries it to a fixed hole and inserts. Poses are plausible, not physical.
    Returns a Sample dict compatible with MEAPhaseAug.augment."""
    rng = np.random.default_rng(seed)
    hole = np.array([0.10, 0.05])
    nut0 = np.array([-0.10, -0.08, 0.02])            # nut resting pose
    phase = np.array([APPROACH if t < k_grasp else ENGAGE for t in range(T)], int)

    eef_pos = np.zeros((T, 3)); eef_quat = np.zeros((T, 4)); grip = np.zeros((T, 1))
    obj_pos = np.zeros((T, 3)); obj_quat = np.zeros((T, 4))
    action = np.zeros((T, 10))
    id6 = mat_to_sixd(np.eye(3))
    for t in range(T):
        if phase[t] == APPROACH:                     # descend onto nut from above
            f = t / max(1, k_grasp)
            start = np.array([-0.02, 0.10, 0.18])
            eef_pos[t] = (1 - f) * start + f * (nut0 + [0, 0, 0.02])
            obj_pos[t] = nut0
            grip[t] = 0.0
        else:                                        # carry nut to hole and insert
            f = (t - k_grasp) / max(1, T - 1 - k_grasp)
            carry = (1 - f) * (nut0 + [0, 0, 0.06]) + f * np.array([hole[0], hole[1], 0.02])
            eef_pos[t] = carry + [0, 0, 0.02]
            obj_pos[t] = carry
            grip[t] = 1.0
        eef_quat[t] = np.array([0, 0, 0, 1.0])       # identity (xyzw)
        obj_quat[t] = np.array([0, 0, 0, 1.0])
        action[t, 0:3] = eef_pos[t]                  # abs pose target = next eef (toy)
        action[t, 3:9] = id6
        action[t, 9] = grip[t, 0]
    return dict(action=action, eef_pos=eef_pos, eef_quat=eef_quat, gripper=grip,
                obj_pos=obj_pos, obj_quat=obj_quat, hole_xy=hole, phase=phase,
                k_grasp=k_grasp)
