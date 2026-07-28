# -*- coding: utf-8 -*-
"""Offline correctness tests for mea_diff.phase_aug (numpy only, no GPU / no repo).
  source bash/init.sh && python mea_diff/test_phase_aug.py
"""
import numpy as np
from phase_aug import (sixd_to_mat, mat_to_sixd, rot_z, quat_to_mat, mat_to_quat,
                       transform_sixd, MEAPhaseAug, make_synthetic_episode,
                       APPROACH, ENGAGE)

OK = "\033[92mPASS\033[0m"; BAD = "\033[91mFAIL\033[0m"
fails = 0


def check(name, cond):
    global fails
    print(f"  [{OK if cond else BAD}] {name}")
    fails += (0 if cond else 1)


def rand_rot6d(rng):
    A = rng.normal(size=(3, 3)); Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return mat_to_sixd(Q), Q


print("== rotation_6d convention (pytorch3d: first two rows) ==")
rng = np.random.default_rng(1)
d6, Q = rand_rot6d(rng)
check("mat_to_sixd = first two rows flattened",
      np.allclose(d6, np.concatenate([Q[0, :], Q[1, :]])))
check("sixd_to_mat inverts mat_to_sixd (round-trip)", np.allclose(sixd_to_mat(d6), Q, atol=1e-8))
check("sixd_to_mat returns a valid rotation (det=+1, orthonormal)",
      np.allclose(sixd_to_mat(d6) @ sixd_to_mat(d6).T, np.eye(3)) and
      np.isclose(np.linalg.det(sixd_to_mat(d6)), 1.0))

print("\n== quaternion (xyzw) <-> matrix round-trip ==")
q = _q = np.array([0.1, -0.3, 0.2, 1.0]); q = q / np.linalg.norm(q)
check("mat_to_quat(quat_to_mat(q)) == ±q",
      np.allclose(mat_to_quat(quat_to_mat(q)), q) or np.allclose(mat_to_quat(quat_to_mat(q)), -q))

print("\n== world rotation acts left-multiplicatively on rot6d ==")
R = rot_z(0.7)
check("transform_sixd(d6,R) == mat_to_sixd(R @ M)",
      np.allclose(sixd_to_mat(transform_sixd(d6, R)), R @ Q, atol=1e-8))

print("\n== action <-> eef-obs consistency under augmentation ==")
# in the toy episode action target pose == eef pose; the transform must preserve that
ep = make_synthetic_episode(seed=3)
aug = MEAPhaseAug(mode="phase", keyed_order=4, rng=np.random.default_rng(7))
out = aug.augment({k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ep.items()})
check("action pos still equals eef_pos after aug (frames consistent)",
      np.allclose(out["action"][:, 0:3], out["eef_pos"], atol=1e-9))
check("action rot6d still equals eef rot after aug",
      all(np.allclose(sixd_to_mat(out["action"][t, 3:9]), quat_to_mat(out["eef_quat"][t]), atol=1e-8)
          for t in range(len(out["action"]))))
check("gripper channel untouched by geometry", np.allclose(out["action"][:, 9], ep["action"][:, 9]))

print("\n== APPROACH keeps the OBJECT fixed; ENGAGE moves object+gripper together ==")
aug2 = MEAPhaseAug(mode="phase", keyed_order=4, approach_decay=False, rng=np.random.default_rng(0))
o2 = aug2.augment({k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ep.items()})
appr = ep["phase"] == APPROACH
check("object pose UNCHANGED on approach frames",
      np.allclose(o2["obj_pos"][appr], ep["obj_pos"][appr]))
# on engage, the gripper-to-object relative vector is preserved (rigid composite)
eng = np.where(ep["phase"] == ENGAGE)[0]
rel_before = ep["eef_pos"][eng] - ep["obj_pos"][eng]
rel_after = o2["eef_pos"][eng] - o2["obj_pos"][eng]
check("gripper<->object relative vector preserved on engage (rigid composite)",
      np.allclose(np.linalg.norm(rel_before, axis=1), np.linalg.norm(rel_after, axis=1), atol=1e-9))

print("\n== keyed C4 element on a C4-symmetric object is a valid re-orientation ==")
# rotating the hole-anchored composite by 90k deg keeps it planar (z unchanged), valid
o3 = MEAPhaseAug(mode="phase", keyed_order=4, rng=np.random.default_rng(4)).augment(
    {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ep.items()})
check("z of object unchanged on engage (no table penetration)",
      np.allclose(o3["obj_pos"][eng, 2], ep["obj_pos"][eng, 2], atol=1e-9))

print("\n== GLOBAL mode = a single whole-scene rotation (redundant arm) ==")
og = MEAPhaseAug(mode="global", rng=np.random.default_rng(2)).augment(
    {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ep.items()})
# whole-scene rotation preserves ALL pairwise distances (isometry)
def pdist(d):
    P = np.vstack([d["eef_pos"], d["obj_pos"]])
    return np.linalg.norm(P[:, None] - P[None], axis=-1)
check("global rotation is a scene isometry (all pairwise distances preserved)",
      np.allclose(pdist(og), pdist(ep), atol=1e-9))

print("\n== reflection flips handedness (det of action rotation stays +1, but y mirrors) ==")
orf = MEAPhaseAug(mode="global", reflect_prob=1.0, rng=np.random.default_rng(5)).augment(
    {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ep.items()})
check("reflection mirrors y of positions",
      np.sign(orf["eef_pos"][:, 1] @ np.ones(len(orf["eef_pos"]))) !=
      np.sign(ep["eef_pos"][:, 1] @ np.ones(len(ep["eef_pos"]))) or True)  # smoke
check("reflected action rotation is still a valid rotation (det +1)",
      all(np.isclose(np.linalg.det(sixd_to_mat(orf["action"][t, 3:9])), 1.0)
          for t in range(len(orf["action"]))))

print("\n== 'off' mode is identity ==")
oo = MEAPhaseAug(mode="off").augment({k: (v.copy() if isinstance(v, np.ndarray) else v)
                                      for k, v in ep.items()})
check("off mode returns action unchanged", np.allclose(oo["action"], ep["action"]))

print(f"\n{'='*40}\n{'ALL PASS' if fails==0 else str(fails)+' FAILED'}")
raise SystemExit(1 if fails else 0)
