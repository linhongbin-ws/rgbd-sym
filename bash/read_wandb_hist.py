# -*- coding: utf-8 -*-
"""Read (env_steps, success_rate_eval) history from a LOCAL wandb .wandb binary
(no network). Prints the convergence step: first step reaching >=90% of final
plateau, and a downsampled curve."""
import sys, json, glob, os
from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2 as pb


def read_history(wandb_file):
    ds = datastore.DataStore()
    ds.open_for_scan(wandb_file)
    pts = []
    while True:
        try:
            rec_bytes = ds.scan_data()
        except Exception:
            break
        if rec_bytes is None:
            break
        rec = pb.Record()
        try:
            rec.ParseFromString(rec_bytes)
        except Exception:
            continue
        if rec.WhichOneof("record_type") != "history":
            continue
        d = {}
        for item in rec.history.item:
            key = item.key or ("/".join(item.nested_key) if item.nested_key else "")
            try:
                d[key] = json.loads(item.value_json)
            except Exception:
                d[key] = item.value_json
        env = d.get("env_steps")
        sr = d.get("metrics/success_rate_eval")
        if env is not None and sr is not None:
            pts.append((float(env), float(sr)))
    pts.sort()
    return pts


def convergence(pts, frac=0.9):
    if not pts:
        return None
    finals = [sr for _, sr in pts[-5:]]
    plateau = sum(finals) / len(finals)
    thresh = frac * plateau
    for env, sr in pts:
        # first step where a 3-pt trailing window stays >= thresh
        pass
    # first crossing that then stays mostly above thresh
    for i, (env, sr) in enumerate(pts):
        window = [s for _, s in pts[i:i + 3]]
        if window and sum(window) / len(window) >= thresh:
            return env, plateau, thresh
    return pts[-1][0], plateau, thresh


if __name__ == "__main__":
    for wf in sys.argv[1:]:
        label = os.path.basename(os.path.dirname(wf))
        pts = read_history(wf)
        if not pts:
            print(f"\n{label}: NO history points"); continue
        auc = sum(s for _, s in pts) / len(pts)
        conv = convergence(pts)
        print(f"\n=== {label} ===")
        print(f"  eval points: {len(pts)}  last_env={pts[-1][0]:.0f}  "
              f"final_sr={pts[-1][1]:.2f}  AUC={auc:.3f}")
        if conv:
            cenv, plateau, thresh = conv
            print(f"  plateau(last5 mean)={plateau:.2f}  converge@>= {thresh:.2f}: "
                  f"env_step {cenv:.0f}  (= iter {cenv/50:.0f}, "
                  f"{100*cenv/pts[-1][0]:.0f}% of run)")
        # downsampled curve
        step = max(1, len(pts) // 12)
        curve = " ".join(f"{e/1000:.0f}k:{s:.1f}" for e, s in pts[::step])
        print(f"  curve: {curve}")
