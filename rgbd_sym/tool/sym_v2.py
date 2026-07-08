# -*- coding: utf-8 -*-
"""mea_v2: context-conditioned (conditionally-equivariant) data augmentation.

See context/mea_v2_new_idea/design_mea_v2.md for the full design & rationale.

Why this exists (short version):
  - block_pull reward is RELATIONAL: success = objects[0].isTouching(objects[1]),
    reward = -||block0 - block1|| + 0.1. The only reward-invariant symmetry is a
    GLOBAL SE(2) rigid transform of the whole scene -- which the C4-equivariant
    network already has (discrete 90deg + conv translation). That redundancy is
    why v1 (generate_sym3) measured ~= baseline (see mea_screening_results.md).
  - The genuinely NON-redundant augmentation is a per-phase (context-conditioned)
    LOCAL gauge: during the free-space APPROACH, rotate the GRIPPER ONLY about the
    target object's anchor while both blocks stay fixed. The network never sees
    "same block layout, different approach angle" from any global symmetry.

Two modes:
  - 'global'      : whole-scene continuous SE(2). Strictly reward-invariant, and
                    the safe/correct baseline. Non-redundant only via continuous
                    angle (fills gaps between the 4 discrete C4 angles).
  - 'conditional' : APPROACH -> gripper-only rotation about target-object anchor;
                    CONTACT -> identity; PULL -> global scene rotation. The
                    research contribution (experimental; see caveats below).

Interface mirrors generate_sym3 so the Sym wrapper can swap them:
    new_sym_obs, new_sym_actions = generate_sym_v2(obs, origin_actions, dummy_env, ...)
    len(new_sym_obs) == len(obs)              (T+1 frames)
    len(new_sym_actions) == len(origin_actions)  (T actions)

CAVEATS (this is a scaffold -- validate on a real trajectory in the env):
  - transform_action_se2 rotates the (a1, a2) translation delta. The exact
    action-frame <-> camera-frame convention (axis swap/sign, and whether a
    world R_z(theta) maps to R_z(+theta) or R_z(-theta) in action space) MUST be
    checked against one saved rollout. `action_sign` exposes that choice.
  - APPROACH gripper-only rotation assumes obs['pc'] is in a static camera/world
    frame (verified: pomdp/env.py builds clouds with pose=I, no gripper subtract).
  - Yaw absolute-orientation offset during APPROACH is folded into a per-frame
    gripper yaw; validate the grasp hand-off frame.
"""
import numpy as np
from copy import deepcopy
from rgbd_sym.tool.common import getT, TxT


# --------------------------------------------------------------------------- #
# geometry primitives
# --------------------------------------------------------------------------- #
def rot2d(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def se2_about(anchor_xy, theta, reflect=False):
    """4x4 transform about the vertical axis through (anchor_xy): rotate by
    `theta`, then (if reflect) mirror across the x-axis through the anchor.

    Reflection is the O(2)\\SO(2) piece the C4-equivariant network (flip_symmetry
    =false) structurally LACKS, yet block_pull's relational reward is mirror-
    invariant -> a valid, non-redundant augmentation. See design_mea_v2.md.
    """
    ax, ay = float(anchor_xy[0]), float(anchor_xy[1])
    T_to = getT([-ax, -ay, 0.0], [0, 0, 0], rot_type="euler")
    R = getT([0.0, 0.0, 0.0], [0, 0, theta], rot_type="euler", euler_Degrees=False)
    T_from = getT([ax, ay, 0.0], [0, 0, 0], rot_type="euler")
    Ts = [T_from]
    if reflect:
        F = np.eye(4)
        F[0, 0] = -1.0                      # mirror x (about the anchor, since centered)
        Ts.append(F)
    Ts += [R, T_to]
    return TxT(Ts)


def transform_pc(pc, T):
    if pc is None or len(pc) == 0:
        return pc
    ones = np.ones((pc.shape[0], 1))
    P = np.concatenate((pc, ones), axis=1)
    return np.matmul(P, np.transpose(T))[:, :3]


def transform_action_se2(a, theta, action_sign=1.0, reflect=False):
    """Transform the action under a scene rotation by `theta` and optional x-mirror.

    Validated on a real rollout: action[1]->world x, action[2]->world y is the
    IDENTITY map (phi=I, det=+1) up to a positive scale, so action_sign=+1 and the
    world rotation applies directly to (a[1], a[2]). dz (a[3]), gripper (a[0]) are
    unchanged. Under reflection, a[1] flips and dyaw (a[4]) flips (handedness).
    """
    new_a = np.asarray(a, dtype=float).copy()
    xy = rot2d(action_sign * theta) @ new_a[1:3]
    if reflect:
        xy[0] = -xy[0]                      # mirror x (matches se2_about F[0,0]=-1)
        new_a[4] = -new_a[4]                # yaw handedness flips under reflection
    new_a[1:3] = xy
    return new_a


# --------------------------------------------------------------------------- #
# context / phase inference from obs signals
# --------------------------------------------------------------------------- #
def _object_keys(pc_dict):
    return [k for k in pc_dict.keys() if k.startswith("object")]


def _centroid_xy(pc):
    if pc is None or len(pc) == 0:
        return None
    return np.asarray(pc)[:, :2].mean(axis=0)


def segment_grasp_step(obs, z_thres=0.15):
    """Return the frame index k where the APPROACH ends and CONTACT/GRASP begins.

    Primary signal: first open->closed flip of gripper_close.
    Fallback: first frame with gripper_pos[2] < z_thres.
    Returns None if neither is available (caller falls back to 'global').
    """
    gc = [o.get("gripper_close", None) for o in obs]
    if all(g is not None for g in gc):
        for i in range(1, len(gc)):
            if float(gc[i]) != float(gc[i - 1]):
                return i
    for i, o in enumerate(obs):
        gp = o.get("gripper_pos", None)
        if gp is not None and float(gp[2]) < z_thres:
            return i
    return None


def nearest_object_key(obs_frame):
    """Key of the object closest (xy) to the gripper -- the presumed target/grasped
    object (block 0). Falls back to 'object1'."""
    pc = obs_frame.get("pc", {})
    g = _centroid_xy(pc.get("gripper"))
    best, best_d = None, np.inf
    for k in _object_keys(pc):
        c = _centroid_xy(pc.get(k))
        if c is None or g is None:
            continue
        d = np.linalg.norm(c - g)
        if d < best_d:
            best, best_d = k, d
    return best if best is not None else "object1"


def scene_centroid_xy(obs_frame):
    pc = obs_frame.get("pc", {})
    pts = [v for v in pc.values() if v is not None and len(v)]
    if not pts:
        return np.zeros(2)
    allp = np.concatenate(pts, axis=0)
    return allp[:, :2].mean(axis=0)


# --------------------------------------------------------------------------- #
# main entry
# --------------------------------------------------------------------------- #
def generate_sym_v2(obs, origin_actions, dummy_env,
                    mode="global",
                    max_angle=2 * np.pi,
                    approach_max_angle=None,
                    z_thres=0.15,
                    action_sign=1.0,
                    reflect_prob=0.0,
                    context_channel=False,
                    theta_global=None,
                    theta_approach=None,
                    reflect=None,
                    rng=None):
    """Generate ONE context-conditioned augmented episode.

    Args:
        obs:            list of original obs dicts (len T+1); each must carry
                        per-entity obs['pc'] and (for conditional) gripper_close/pos.
        origin_actions: list of original actions (len T).
        dummy_env:      Occup(DummyEnv) -- re-renders occupancy from transformed pc.
        mode:           'global' | 'conditional'.
        max_angle:      sampling range for the PULL/global rotation (rad).
        approach_max_angle: sampling range for the APPROACH gripper rotation
                        (defaults to max_angle).
        context_channel: if True, write the phase gauge c into obs['image'][1]
                        (the network's constant scalar plane). Off by default so
                        the network is unchanged (v2-aug-only).
        theta_global/theta_approach: optionally inject fixed angles (else sampled).
    Returns:
        (new_sym_obs, new_sym_actions) with the same lengths as (obs, origin_actions).
    """
    rng = rng or np.random
    if approach_max_angle is None:
        approach_max_angle = max_angle
    theta_g = float(theta_global) if theta_global is not None else float(rng.uniform(-max_angle, max_angle))
    theta_a = float(theta_approach) if theta_approach is not None else float(rng.uniform(-approach_max_angle, approach_max_angle))
    do_reflect = bool(reflect) if reflect is not None else bool(rng.uniform() < reflect_prob)

    T = len(origin_actions)
    k = segment_grasp_step(obs, z_thres=z_thres) if mode == "conditional" else None
    if mode == "conditional" and (k is None or k <= 0 or k >= len(obs) - 1):
        mode = "global"  # can't segment safely -> safe fallback

    if mode == "global":
        q = scene_centroid_xy(obs[0])
        T_all = se2_about(q, theta_g, reflect=do_reflect)

        def frame_tf(t):                      # every entity, every frame
            keys = obs[t].get("pc", {}).keys()
            return {kk: T_all for kk in keys}

        def action_tf(t):                     # (theta, reflect)
            return theta_g, do_reflect
    else:
        # conditional: approach gripper-only about target; contact identity; pull global
        target_key = nearest_object_key(obs[k])
        p0 = _centroid_xy(obs[k]["pc"].get(target_key))
        if p0 is None:
            p0 = scene_centroid_xy(obs[k])
        q = scene_centroid_xy(obs[k])
        T_grip = se2_about(p0, theta_a)       # APPROACH: gripper only, about target (no reflect)
        T_all = se2_about(q, theta_g, reflect=do_reflect)  # PULL: whole scene (reflect ok)

        def frame_tf(t):
            if t < k:                         # c1 APPROACH
                return {"gripper": T_grip}
            if t == k:                        # c2 CONTACT -> identity
                return {}
            keys = obs[t].get("pc", {}).keys()  # c3 PULL -> global
            return {kk: T_all for kk in keys}

        def action_tf(t):
            if t < k:
                return theta_a, False
            if t == k:
                return 0.0, False
            return theta_g, do_reflect

    # ---- re-render each frame's occupancy from the transformed clouds ----
    rendered = []
    for t in range(len(obs)):
        dummy_env.set_current_points(obs[t]["pc"])
        dummy_env.apply_transform(frame_tf(t))
        rendered.append(dummy_env.reset())     # {'pc', 'occup_image'}

    # ---- merge rendered keys into copies of the original obs ----
    new_sym_obs = deepcopy(obs)
    for t in range(len(obs)):
        new_sym_obs[t].update(rendered[t])
        if context_channel and "image" in new_sym_obs[t]:
            c = 0.0 if (mode == "global" or t < (k or 0)) else 1.0
            img = new_sym_obs[t]["image"]
            if img is not None and img.shape[0] >= 2:
                img[1, :, :] = c               # gauge -> network's layer1 scalar plane

    # ---- transform the actions consistently ----
    new_sym_actions = []
    for t in range(T):
        th, rf = action_tf(t)
        new_sym_actions.append(transform_action_se2(origin_actions[t], th, action_sign, reflect=rf))

    return new_sym_obs, new_sym_actions
