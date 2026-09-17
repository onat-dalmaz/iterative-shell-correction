"""Minimal end-to-end example.

  python examples/run_example.py volume.npy            # [z, x, y] complex volume
  python examples/run_example.py sampledata.mat        # Brian's sample (source_image, [x,y,z])

Runs (1) the ring cascade 1:1 with ts_b0_corr.m (MATLAB kernel 9), (2) the
same with the Python full-Gaussian σ=4.5 that our earlier analyses used,
(3) time_seg_corr (N=16, overlap 8, kernel 7) if a time map is given, and
prints R12/R13 ratios vs the uncorrected volume.
"""
from __future__ import annotations

import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ring_b0_corr import ring_b0_corr_v2, ring_b0_corr_matlab   # noqa: E402
from time_seg_corr import time_seg_corr           # noqa: E402
from metrics import tissue_mask, r12_r13, ratios  # noqa: E402


def load(path):
    if path.endswith(".mat"):
        import scipy.io as sio
        d = sio.loadmat(path)
        key = [k for k in d if not k.startswith("__")][0]
        return np.ascontiguousarray(np.transpose(d[key], (2, 0, 1)))    # [x,y,z] → [z,x,y]
    return np.asarray(np.load(path, mmap_mode="r"))


def main():
    vol = load(sys.argv[1]).astype(np.complex64)
    tmap = load(sys.argv[2]).astype(np.float32) if len(sys.argv) > 2 else None
    prof = np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "tkr_profile_brian.npz"))
    mask = tissue_mask(vol)
    base = r12_r13(vol, mask)
    runs = []
    t0 = time.time()
    c1, f1, i1 = ring_b0_corr_matlab(vol, 8, prof["kr"], prof["t_s"], kernelsize=9)   # exact ts_b0_corr.m
    runs.append(("ring N=8, exact ts_b0_corr.m (kernel 9, literal fmap)", c1, time.time() - t0))
    t0 = time.time()
    c2, f2, i2 = ring_b0_corr_v2(vol, 8, prof["kr"], prof["t_s"], kr_edge=1.0, smooth_sigma=4.5)
    runs.append(("ring N=8, scipy Gaussian σ=4.5 (Python-as-run)", c2, time.time() - t0))
    if tmap is not None:
        t0 = time.time()
        c3, f3, i3 = time_seg_corr(vol, tmap, 16)
        runs.append(("time_seg_corr N=16 (overlap 8, kernel 7)", c3, time.time() - t0))
    print(f"ring shell times (ms): {np.round(i1['ring_times_s'] * 1e3, 2).tolist()}")
    for name, c, dt in runs:
        r = ratios(base, r12_r13(c, mask))
        g = np.median(np.abs(c)[mask] / (np.abs(vol)[mask] + 1e-9))
        print(f"{name:52s} {dt:6.1f}s  gain {g:.3f}  r12 {r['r12']:.3f}  r13 {r['r13']:.3f}")


if __name__ == "__main__":
    main()
