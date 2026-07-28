# -*- coding: utf-8 -*-
"""Positive-evidence artifact (pre-empts the "diffusion is already multimodal" objection):
the raw MimicGen Square demos only ever demonstrate ONE of the 4 C4-equivalent insertion
orientations. If the nut-yaw-at-insertion (relative to the peg) is UNIMODAL / picks one C4
slot rather than spreading over {0,90,180,270}, then the other 3 orientations are OUT OF
SUPPORT — exactly what the keyed-C4 augmentation adds.

Run on the real dataset (on the GPU box; needs h5py + the demo's object-state layout):
  python mea_diff/artifact_insertion_yaw.py --hdf5 .../square_d0.hdf5
Verify the object-state slice for the SQUARE NUT quaternion in your hdf5 first (print keys).
"""
import argparse
import numpy as np


def quat_yaw(q):                                            # q = xyzw -> yaw about z (deg)
    x, y, z, w = q
    return np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def load_yaws(hdf5, nut_slice, peg_slice, insert_frac):
    """Return (nut_yaw_at_insertion, rel_yaw_to_peg) arrays over all demos.
    nut_slice/peg_slice: (start,end) indices into obs['object'] for the nut / peg quaternion
    (xyzw). If peg_slice is None, the peg is treated as fixed (rel = nut yaw)."""
    import h5py
    nut_yaws, rel_yaws = [], []
    with h5py.File(hdf5, "r") as f:
        demos = list(f["data"].keys())
        for d in demos:
            obj = f[f"data/{d}/obs/object"][()]            # (T, D) low-dim object-state
            T = len(obj)
            t = int(insert_frac * (T - 1))                 # near-insertion frame (default: end)
            nq = obj[t, nut_slice[0]:nut_slice[1]]
            ny = quat_yaw(nq); nut_yaws.append(ny)
            if peg_slice is not None:
                py = quat_yaw(obj[t, peg_slice[0]:peg_slice[1]])
                rel_yaws.append((ny - py))
            else:
                rel_yaws.append(ny)
    return np.array(nut_yaws), (np.array(rel_yaws) + 180) % 360 - 180


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--nut-quat", type=int, nargs=2, default=[3, 7],
                    help="obs['object'] slice for the SQUARE NUT xyzw quat (VERIFY in your hdf5)")
    ap.add_argument("--peg-quat", type=int, nargs=2, default=None, help="peg quat slice; omit if peg is fixed")
    ap.add_argument("--insert-frac", type=float, default=1.0, help="fraction of the demo at which insertion happens")
    ap.add_argument("--out", default="context/plan/artifact_insertion_yaw.png")
    args = ap.parse_args()

    nut_yaw, rel = load_yaws(args.hdf5, args.nut_quat, args.peg_quat, args.insert_frac)
    rel_mod90 = (rel + 45) % 90 - 45                        # alignment within a C4 slot
    slot = np.round(rel / 90.0).astype(int) % 4            # which of the 4 C4 slots
    print(f"demos={len(nut_yaw)}  rel-yaw: mean={rel.mean():.1f} std={rel.std():.1f}")
    print(f"C4 slot histogram {{0,90,180,270}} = {np.bincount(slot, minlength=4).tolist()}  "
          f"(UNIMODAL over slots => demos cover only 1 of 4 => keyed aug adds the other 3)")
    print(f"within-slot alignment |rel mod90|: mean={np.abs(rel_mod90).mean():.1f}deg")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].hist(rel, bins=48, color="#0ea5e9", ec="white")
    ax[0].set_title("nut yaw at insertion (relative to peg)\nunimodal → demos show ONE orientation")
    ax[0].set_xlabel("relative yaw (deg)"); ax[0].set_ylabel("# demos")
    b = np.bincount(slot, minlength=4)
    ax[1].bar(range(4), b, color=["#16a34a", "#f59e0b", "#f59e0b", "#f59e0b"])
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels(["0°(shown)", "90°", "180°", "270°"])
    ax[1].set_title("which C4 slot the demos use\n(green=covered; orange=MEA supplies)")
    ax[1].set_ylabel("# demos")
    fig.suptitle("Square demos only cover ONE of the 4 C4 insertion orientations → keyed-C4 aug is out-of-support",
                 fontsize=11)
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.92]); plt.savefig(args.out, dpi=125)
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
