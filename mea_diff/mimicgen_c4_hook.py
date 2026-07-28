# -*- coding: utf-8 -*-
"""PATH A implementation: inject the C4 keyed rotation into MimicGen datagen for
NutAssemblySquare. Drop this into a mimicgen checkout (on the GPU box).

WHY THIS IS A ONE-LINER (verified against NVlabs/mimicgen):
  - Square task_spec (mimicgen/configs/robosuite.py, Square_Config) has 2 subtasks:
    GRASP  (object_ref='square_nut') and INSERTION (object_ref='square_peg', final).
  - In DataGenerator.generate() (mimicgen/datagen/data_generator.py), for each subtask:
        subtask_object_name = self.task_spec[subtask_ind]['object_ref']
        cur_object_pose     = cur_datagen_info.object_poses[subtask_object_name]   # 4x4
    then transform_source_data_segment_using_object_pose(cur_object_pose, ...) maps the
    source eef segment into the CURRENT object frame:  eef = cur_object_pose @ inv(src_obj) @ src_eef.
  - POST-multiplying cur_object_pose by Rz rotates in the PEG'S LOCAL frame -> the whole
    insertion segment (grasp->carry->insert) re-orients k*90deg about the peg axis, peg
    world position unchanged. Pre-multiplying would orbit the world origin (WRONG).
  - Valid because the square nut is C4-symmetric about the peg and robosuite's
    NutAssemblySquare success (objects_on_pegs) is orientation-agnostic -> all 4 keys succeed;
    the grasp subtask (square_nut) is untouched -> no re-grasp.
  - IMPORTANT: apply Rz ONLY to the CURRENT target pose (cur_object_pose), NOT to the source
    (src_obj_pose). Rotating both cancels (P Rz)(S Rz)^-1 = P S^-1. That is why a global
    get_object_poses() override does NOT work — use the generate() hook below.
"""
import numpy as np


def Rz4(theta):
    """4x4 homogeneous rotation about +z (matches mea_diff.phase_aug.rot_z on the 2x2 block)."""
    c, s = np.cos(theta), np.sin(theta)
    T = np.eye(4)
    T[0, 0], T[0, 1], T[1, 0], T[1, 1] = c, -s, s, c
    return T


def keyed_object_pose(cur_object_pose, subtask_object_name, c4_key,
                      keyed_object="square_peg", order=4):
    """Return the C4-keyed CURRENT object pose for the insertion subtask, else unchanged.
    cur_object_pose : 4x4 world pose of the current subtask object (as read in generate()).
    subtask_object_name : self.task_spec[subtask_ind]['object_ref'].
    c4_key : int in [0, order) — which of the `order` point-group orientations (None -> baseline).
    """
    if c4_key is None or subtask_object_name != keyed_object:
        return cur_object_pose
    return np.asarray(cur_object_pose) @ Rz4(2.0 * np.pi * (int(c4_key) % order) / order)


# --------------------------------------------------------------------------------------- #
# THE PATCH (drop into mimicgen/datagen/data_generator.py :: DataGenerator.generate())
# --------------------------------------------------------------------------------------- #
PATCH = r'''
# --- MEA v2 C4 keyed insertion --------------------------------------------------
# in DataGenerator.generate(), inside `for subtask_ind in range(len(self.task_spec)):`
# right AFTER:
#     subtask_object_name = self.task_spec[subtask_ind]['object_ref']
#     cur_object_pose = cur_datagen_info.object_poses[subtask_object_name]
# add:
    from mea_diff.mimicgen_c4_hook import keyed_object_pose      # or vendor the fn
    cur_object_pose = keyed_object_pose(
        cur_object_pose, subtask_object_name,
        getattr(self, "_c4_key", None))                          # peg-subtask only
# --------------------------------------------------------------------------------
# then set the per-demo key in mimicgen/scripts/generate_dataset.py before each generate():
#     data_generator._c4_key = k                                 # k in {0,1,2,3}
# for a balanced 4x paired set, loop k over {0,1,2,3} on the SAME seeded initial state
# (env.reset() is inside generate(): re-seed identically per key, or hoist the reset).
# baseline = _c4_key = 0 (or None). SUCCESS-FILTER the output (k=180/270 may miss IK).
'''


if __name__ == "__main__":                                       # sandbox-checkable math
    import sys
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from phase_aug import rot_z
    ok = True
    for th in [0, 0.3, np.pi / 2, np.pi, -1.2]:
        ok &= np.allclose(Rz4(th)[:2, :2], rot_z(th)[:2, :2])
    # keyed pose: peg at (0.1, -0.2, 0.8), any orientation; C4 keeps POSITION fixed, rotates frame
    P = np.eye(4); P[:3, 3] = [0.1, -0.2, 0.8]
    for k in range(4):
        Pk = keyed_object_pose(P, "square_peg", k)
        ok &= np.allclose(Pk[:3, 3], P[:3, 3])                   # peg position unchanged
        ok &= np.allclose(Pk[:3, :3], Rz4(k * np.pi / 2)[:3, :3])  # frame rotated by k*90
    ok &= np.allclose(keyed_object_pose(P, "square_nut", 1), P)  # grasp subtask untouched
    ok &= np.allclose(keyed_object_pose(P, "square_peg", None), P)  # baseline untouched
    print("Rz4 matches phase_aug.rot_z; keyed pose fixes position & rotates frame; "
          "grasp/baseline untouched:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
