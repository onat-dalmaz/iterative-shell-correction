"""Gradient entropy (R12) and normalised gradient energy (R13) for every
result volume: full 3D, and per slice (for the viewer).

  python compute_metrics.py --orig volume.npy [--results results]
Writes results/metrics.json:
  {"orig": {...}, "<method>": {"volume": {R12, R13, R12_max, R13_min, r12, r13, gain},
                               "slices": {"R12": [...], "R13": [...], "r12": [...], "r13": [...]}}}
Masks: one global Otsu mask on the original for the 3D values; a per-slice
Otsu mask on the original for the slice values (same convention as viewer).
"""
from __future__ import annotations

import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import tissue_mask, r12_r13, ratios  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--mask", default=None, help="bool volume [z,rows,cols] to use instead of the Otsu tissue mask (e.g. a breast mask)")
    args = ap.parse_args()
    orig = np.load(args.orig, mmap_mode="r")
    files = sorted(f for f in glob.glob(os.path.join(args.results, "*.npy"))
                   if not any(s in os.path.basename(f) for s in ("_fmap", "_selected_freq", "_patch")))
    omag = np.abs(np.asarray(orig)).astype(np.float32)
    if args.mask:
        mask3 = np.load(args.mask).astype(bool)
        masks2 = [mask3[z] for z in range(omag.shape[0])]
    else:
        mask3 = tissue_mask(omag)
        masks2 = [tissue_mask(omag[z]) for z in range(omag.shape[0])]
    base3 = r12_r13(omag, mask3)
    base2 = [r12_r13(omag[z], masks2[z]) for z in range(omag.shape[0])]
    out = {"orig": {"volume": {**base3, "r12": 1.0, "r13": 1.0, "gain": 1.0},
                    "slices": {"R12": [b["R12"] for b in base2], "R13": [b["R13"] for b in base2],
                               "r12": [1.0] * len(base2), "r13": [1.0] * len(base2)}}}
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        v = np.abs(np.asarray(np.load(f, mmap_mode="r"))).astype(np.float32)
        gain = float(np.median(v[mask3] / (omag[mask3] + 1e-9)))
        c3 = r12_r13(v, mask3); rr3 = ratios(base3, c3)
        R12s, R13s, r12s, r13s = [], [], [], []
        for z in range(v.shape[0]):
            c = r12_r13(v[z], masks2[z]); rr = ratios(base2[z], c)
            R12s.append(c["R12"]); R13s.append(c["R13"]); r12s.append(rr["r12"]); r13s.append(rr["r13"])
        out[name] = {"volume": {**c3, **rr3, "gain": gain},
                     "slices": {"R12": R12s, "R13": R13s, "r12": r12s, "r13": r13s}}
        print(f"{name:14s} gain {gain:.3f}  3D: gradient entropy {c3['R12']:.3f} (×{rr3['r12']:.3f})  "
              f"gradient energy {c3['R13']:.3e} (×{rr3['r13']:.3f})", flush=True)
        del v
    out["_mask"] = args.mask or "otsu"
    # merge into the existing file so entries written by other tools (e.g. compute_iqm_volume.py's
    # "iqm_volume") survive a recompute
    mp = os.path.join(args.results, "metrics.json")
    old = json.load(open(mp)) if os.path.exists(mp) else {}
    for k, v in out.items():
        if isinstance(v, dict) and isinstance(old.get(k), dict):
            old[k].update(v)
        else:
            old[k] = v
    with open(mp, "w") as fh:
        json.dump(old, fh)
    print(f"wrote {args.results}/metrics.json")


if __name__ == "__main__":
    main()
