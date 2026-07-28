# -*- coding: utf-8 -*-
"""BLOCKING GATE (falsification_and_pivot.md): closed-loop action-conjugation test.

The rot6d/quat action conjugation was mis-signed twice in the RL port; a frame error
mislabels the action on EVERY augmented demo -> BC strictly worse than no aug. This test
checks the load-bearing invariant that a CORRECT conjugation must satisfy and a wrong
sign/axis breaks:

  The action expressed in the END-EFFECTOR's own frame is INVARIANT under the
  augmentation's rigid transform.  Proof: for a rigid (R, anchor) applied to both the eef
  pose and the (absolute) action-target pose,
      p_act - p_eef       -> R (p_act - p_eef)          (relative position rotates by R)
      M_eef               -> R M_eef
    => M_eef^T (p_act - p_eef)  and  M_eef^T M_act   are UNCHANGED.
  A wrong-sign / wrong-axis conjugation violates this -> the test fails.

Run offline on synthetic data (smoke): source bash/init.sh && python mea_diff/test_action_consistency.py
Run the REAL gate on a MimicGen EquiDiff trajectory:
    python mea_diff/test_action_consistency.py --hdf5 .../square_d0_abs.hdf5 --demo demo_0
"""
import argparse
import numpy as np
from phase_aug import (MEAPhaseAug, make_synthetic_episode, sixd_to_mat, quat_to_mat,
                       mat_to_sixd, APPROACH, ENGAGE)


def action_in_eef(eef_pos, eef_quat, act_pos, act_rot6d):
    """Return (rel_pos, rel_rot_6d) of the absolute action target expressed in the eef
    frame — the quantity that MUST be invariant under any rigid scene transform."""
    Me = quat_to_mat(eef_quat)
    rel_pos = Me.T @ (act_pos - eef_pos)
    rel_rot = Me.T @ sixd_to_mat(act_rot6d)
    return rel_pos, mat_to_sixd(rel_rot)


def check_sample(s_before, s_after, tol=1e-6, label="", mirror=False):
    """Proper rotations: action-in-eef must be INVARIANT. Reflection (improper): it must
    MIRROR by F=diag(1,-1,1) (rel_pos -> F rel_pos, rel_rot -> F rel_rot F) — verifying the
    action was conjugated by the SAME improper element as the observation."""
    F = np.diag([1.0, -1.0, 1.0])
    T = len(s_before["action"])
    max_dp = max_dr = 0.0
    for t in range(T):
        rp0, rr0 = action_in_eef(s_before["eef_pos"][t], s_before["eef_quat"][t],
                                 s_before["action"][t, 0:3], s_before["action"][t, 3:9])
        rp1, rr1 = action_in_eef(s_after["eef_pos"][t], s_after["eef_quat"][t],
                                 s_after["action"][t, 0:3], s_after["action"][t, 3:9])
        if mirror:
            rp_exp = F @ rp0
            rr_exp = mat_to_sixd(F @ sixd_to_mat(rr0) @ F)
        else:
            rp_exp, rr_exp = rp0, rr0
        max_dp = max(max_dp, np.abs(rp1 - rp_exp).max())
        max_dr = max(max_dr, np.abs(rr1 - rr_exp).max())
    ok = (max_dp < tol) and (max_dr < tol)
    tag = "\033[92mPASS\033[0m" if ok else "\033[91mFAIL\033[0m"
    prop = "MIRRORS by F" if mirror else "invariant"
    print(f"  [{tag}] {label:34s} action-in-eef {prop}  (Δpos={max_dp:.2e}, Δrot={max_dr:.2e})")
    return ok


def synthetic_with_offset(seed=0):
    """synthetic episode where the action target is a NON-trivial relative offset + relative
    rotation from the eef (so the invariance check is not vacuous)."""
    ep = make_synthetic_episode(seed=seed)
    rng = np.random.default_rng(seed + 100)
    off = rng.normal(scale=0.02, size=3)                      # fixed rel offset in eef frame
    ang = 0.6
    Mrel = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    for t in range(len(ep["action"])):
        Me = quat_to_mat(ep["eef_quat"][t])
        ep["action"][t, 0:3] = ep["eef_pos"][t] + Me @ off    # abs target = eef + rel offset
        ep["action"][t, 3:9] = mat_to_sixd(Me @ Mrel)         # abs target rot = eef * rel rot
    return ep


def run_synthetic():
    print("== synthetic smoke (non-trivial relative action offset) ==")
    ep = synthetic_with_offset()
    allok = True
    for mode, kw, name, mirror in [
        ("phase", dict(keyed_order=4), "phase C4-engage", False),
        ("phase", dict(keyed_order=4, approach_aug=True), "phase C4 + approach", False),
        ("phase", dict(keyed_order=0), "phase round SO(2)", False),
        ("global", dict(), "global (whole-scene)", False),
        ("global", dict(reflect_prob=1.0), "global + reflection", True),
    ]:
        aug = MEAPhaseAug(mode=mode, rng=np.random.default_rng(3), **kw)
        out = aug.augment({k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ep.items()})
        allok &= check_sample(ep, out, label=name, mirror=mirror)
    return allok


def load_robomimic_demo(hdf5_path, demo):
    """Load one MimicGen/robomimic demo into the Sample dict. Fill in the exact obs/object
    layout from the real hdf5 (verify keys: robot0_eef_pos, robot0_eef_quat[xyzw],
    object[obj_pos+obj_quat], actions). Actions here must be the 10-D ABSOLUTE form produced
    by equi_diffpo/scripts/robomimic_dataset_conversion.py (pos3 + rot6d6 + grip1)."""
    import h5py
    with h5py.File(hdf5_path, "r") as f:
        g = f[f"data/{demo}"]
        obs = g["obs"]
        eef_pos = obs["robot0_eef_pos"][()]
        eef_quat = obs["robot0_eef_quat"][()]                 # xyzw
        obj = obs["object"][()]                               # TODO: confirm obj_pos/quat slice
        action = g["actions"][()]                             # TODO: ensure ABS 10-D (converted)
    obj_pos, obj_quat = obj[:, 0:3], obj[:, 3:7]
    T = len(action)
    # phase / anchors: reuse MimicGen subtask boundaries if persisted; fallback below.
    d = np.linalg.norm(eef_pos[:, :2] - obj_pos[:, :2], axis=1)
    k = int(np.argmin(d))
    phase = np.array([APPROACH if t < k else ENGAGE for t in range(T)], int)
    return dict(action=action.astype(float), eef_pos=eef_pos.astype(float),
                eef_quat=eef_quat.astype(float), obj_pos=obj_pos.astype(float),
                obj_quat=obj_quat.astype(float), hole_xy=obj_pos[k, :2], phase=phase, k_grasp=k)


def run_hdf5(path, demo):
    print(f"== REAL gate: {path} [{demo}] ==")
    ep = load_robomimic_demo(path, demo)
    aug = MEAPhaseAug(mode="phase", keyed_order=4, rng=np.random.default_rng(0))
    out = aug.augment({k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ep.items()})
    ok = check_sample(ep, out, tol=1e-4, label="real C4-engage")
    print("  NOTE: if this FAILS, the rot6d/quat conjugation is mis-signed — do NOT train.")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", default=None); ap.add_argument("--demo", default="demo_0")
    a = ap.parse_args()
    ok = run_hdf5(a.hdf5, a.demo) if a.hdf5 else run_synthetic()
    print("=" * 40); print("ALL PASS" if ok else "FAILED")
    raise SystemExit(0 if ok else 1)
