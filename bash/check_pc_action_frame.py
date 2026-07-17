# -*- coding: utf-8 -*-
"""THE validation test_sym_v2.py should have had: measure the action <->
PC/IMAGE-frame map from real data, then check that generate_sym_v2's action
transform is exactly the M-conjugate of its observation transform.

Why the old validation missed the bug: it fitted action[1:3] against
obs['gripper_pos'] deltas, which are WORLD-frame -- but the augmentation
rotates the PC (camera) frame. The camera is gripper-centered, so the one
entity that could not reveal the world->pc axis map was... the gripper.

The map is observable from STATIC objects instead: in a gripper-centered
frame, a static object counter-moves by  d_obj_pc = -s * M @ a[1:3]
(s>0 scale). Discovered by the user noticing the viz action arrows matched
NEITHER the ground-truth nor the augmented apparent motion.

  source bash/init.sh
  python bash/check_pc_action_frame.py --ep ep_v2.pkl
"""
import argparse
import numpy as np


def fit_pc_map(obs, actions, k):
    """Fit signed permutation P (up to free positive scale) in
    d_obj_pc = P @ a[1:3], from APPROACH frames (t<k: both objects static in
    world) where the object is visible. Returns (rel_err, P, det, scale, n)."""
    A, D = [], []
    for t in range(min(k, len(actions))):
        a = np.asarray(actions[t], float)[1:3]
        if np.linalg.norm(a) < 5e-3:
            continue
        for key in ("object1", "object2"):
            p0, p1 = obs[t]["pc"].get(key), obs[t + 1]["pc"].get(key)
            if p0 is None or p1 is None or len(p0) == 0 or len(p1) == 0:
                continue
            A.append(a)
            D.append(np.asarray(p1)[:, :2].mean(0) - np.asarray(p0)[:, :2].mean(0))
    A, D = np.array(A), np.array(D)
    best = None
    for perm in [(0, 1), (1, 0)]:
        for s0 in (1, -1):
            for s1 in (1, -1):
                P = np.zeros((2, 2))
                P[0, perm[0]] = s0
                P[1, perm[1]] = s1
                PA = (P @ A.T).T
                den = float((PA * PA).sum())
                s = float((PA * D).sum() / den) if den > 1e-12 else 0.0
                if s <= 0:
                    continue
                err = np.linalg.norm(s * PA - D) / (np.linalg.norm(D) + 1e-9)
                if best is None or err < best[0]:
                    best = (err, P, float(round(np.linalg.det(P))), s, len(A))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", default="ep_v2.pkl")
    args = ap.parse_args()

    import joblib
    from rgbd_sym.tool.sym_v2 import (transform_action_se2, rot2d,
                                      segment_grasp_step)

    data = joblib.load(args.ep)
    obs, actions = data["obs"], data["actions"]
    k = segment_grasp_step(obs) or len(actions)

    # ---- 1. measure P = -s*M from static-object counter-motion ----
    err, P, detP, s, n = fit_pc_map(obs, actions, k)
    M = -P                                   # counter-motion: d = -sMa
    print("== measured pc-frame map (approach frames, static objects) ==")
    print(f"  d_obj_pc = P @ a[1:3],  P =\n{P}")
    print(f"  n={n} samples  rel_err={err:.3f}  scale={s:.3f}")
    print(f"  => M (action->pc gripper motion) =\n{M}\n  det(M) = {np.linalg.det(M):+.0f}")
    exp_M = np.array([[0., 1.], [1., 0.]])
    ok_M = np.allclose(M, exp_M)
    print(f"  matches DummyEnv.step convention [[0,1],[1,0]] (swap): {ok_M}")

    # ---- 2. closed-form: does transform_action_se2 conjugate through M? ----
    print("\n== consistency of transform_action_se2 (current code) ==")
    rng = np.random.RandomState(0)
    worst_new = worst_old = 0.0
    for _ in range(200):
        theta = rng.uniform(-np.pi, np.pi)
        refl = bool(rng.randint(2))
        a = np.concatenate([[rng.randint(2)], rng.uniform(-0.05, 0.05, 3),
                            [rng.uniform(-0.5, 0.5)]])
        # obs-side pc transform (linear part of se2_about): F ∘ R
        FR = (np.diag([-1.0, 1.0]) if refl else np.eye(2)) @ rot2d(theta)
        # required action: a'_xy = M^-1 FR M a_xy ; dyaw flips iff reflect
        req_xy = np.linalg.inv(M) @ FR @ M @ a[1:3]
        req_yaw = -a[4] if refl else a[4]
        got = transform_action_se2(a, theta, reflect=refl)          # new defaults
        # reproduce the OLD (buggy) transform: +theta, mirror flips a[1]
        old_xy = rot2d(+theta) @ a[1:3]
        if refl:
            old_xy[0] = -old_xy[0]
        worst_new = max(worst_new, np.abs(got[1:3] - req_xy).max(),
                        abs(got[4] - req_yaw))
        worst_old = max(worst_old, np.abs(old_xy - req_xy).max())
    print(f"  NEW transform vs required (200 random draws): max err = {worst_new:.2e}")
    print(f"  OLD transform vs required:                    max err = {worst_old:.3f} "
          f"(labels off by up to 2*theta / mirrored axis)")
    assert ok_M, "pc map is not the expected swap -- re-derive before trusting sym_v2"
    assert worst_new < 1e-9, "transform_action_se2 is NOT the M-conjugate of se2_about"
    print("\nPASS: observation and action transforms are consistent.")


if __name__ == "__main__":
    main()
