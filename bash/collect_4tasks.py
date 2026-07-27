# -*- coding: utf-8 -*-
"""Collect one expert rollout per task + diagnose phase segmentation.
  source bash/init.sh && python bash/collect_4tasks.py
"""
import os, numpy as np, joblib

TASKS = ["block_pull", "block_pick", "block_push", "drawer_open"]
TMP = os.environ.get("TMPDIR", "/tmp")


def collect(task, seed=1, max_steps=60):
    from rgbd_sym.env.embodied import PomdpEnv
    from rgbd_sym.env.wrapper.occup import Occup
    env = Occup(PomdpEnv(task=task))
    env.seed = seed
    obs = env.reset()
    obss, actions = [obs], []
    done = False
    while not done and len(actions) < max_steps:
        a = np.asarray(env.query_expert(0), float)
        obs, r, d, info = env.step(a)
        done = bool(d)
        actions.append(a); obss.append(obs)
    return obss, actions


def diag(task, obs, actions):
    from rgbd_sym.tool.sym_v2 import segment_grasp_step, nearest_object_key
    ents = [k for k in obs[0].get("pc", {}).keys()]
    k = segment_grasp_step(obs)
    gc = [o.get("gripper_close", None) for o in obs]
    gz = [round(float(o["gripper_pos"][2]), 3) if o.get("gripper_pos") is not None else None for o in obs]
    # gripper<->nearest object xy distance per frame
    def cen(pc):
        return np.asarray(pc)[:, :2].mean(0) if pc is not None and len(pc) else None
    dists = []
    for o in obs:
        g = cen(o["pc"].get("gripper"))
        best = np.inf
        for kk in o["pc"]:
            if kk in ("gripper", "goal"): continue
            c = cen(o["pc"].get(kk))
            if c is not None and g is not None:
                best = min(best, np.linalg.norm(c - g))
        dists.append(round(best, 3) if np.isfinite(best) else None)
    tgt = nearest_object_key(obs[min(k or 1, len(obs)-1)])
    print(f"\n### {task}: {len(obs)} frames, entities={ents}")
    print(f"    grasp/contact split k={k}  target_key={tgt}")
    print(f"    gripper_close = {gc}")
    print(f"    gripper_z     = {gz}")
    print(f"    grip<->obj d  = {dists}")


for t in TASKS:
    try:
        obs, actions = collect(t)
        joblib.dump({"obs": obs, "actions": actions}, os.path.join(TMP, f"ep_{t}.pkl"))
        diag(t, obs, actions)
    except Exception as e:
        import traceback; print(f"\n### {t}: FAILED {e}"); traceback.print_exc()
print("\nsaved -> $TMPDIR/ep_<task>.pkl")
