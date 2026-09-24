"""Edge width — the sharpness measure — and its paired protocol.

  detect_edges(vol, ...)   strong edges on |vol| (every 3rd slice, every 4th column: the row with the
                           largest |d/drow| in that column, kept if in the top 1 % of the slice's row
                           gradients and the 16-row window spans >= 15 % of the slice's 99th-percentile
                           intensity).  Returns (z, x, yc, sign) tuples.
  rise_at(vol, pts)        10-90 % width (px) of the 16-pixel profile through each edge, on ANY volume.
  paired_table(...)        median of per-edge differences A-B with a bootstrap 95 % CI, by |B0| bin.

The reference on which the edges are detected matters: detecting on a corrected image favours
that method by 0.1-0.3 px at high |B0|. The neutral choice is the UNCORRECTED image, which is
what `python edge_width.py --orig ... --results ...` uses to precompute results/edge_widths_neutral.npz
(per-edge widths for every method; the viewer reads it for paired A-vs-B tables).
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np


def detect_edges(ref_vol, slice_indices=None, x_step: int = 4, percentile_thr: float = 99.0, contrast_frac: float = 0.15):
    nz = ref_vol.shape[0]
    if slice_indices is None:
        slice_indices = list(range(2, nz, 3))
    pts = []
    for z in slice_indices:
        a = np.abs(np.asarray(ref_vol[z])).astype(np.float64)
        gy = np.gradient(a, axis=0); thr = np.percentile(np.abs(gy), percentile_thr); p99 = np.percentile(a, 99)
        H, W = a.shape
        for x in range(40, W - 40, x_step):
            gc = gy[:, x]; yc = int(np.argmax(np.abs(gc)))
            if yc < 10 or yc > H - 10 or abs(gc[yc]) < thr:
                continue
            seg = a[yc - 8:yc + 8, x]
            if seg.max() - seg.min() < contrast_frac * p99:
                continue
            pts.append((z, x, yc, int(np.sign(gc[yc]))))
    return pts


def _width(seg, sgn):
    seg = seg.astype(np.float64); lo, hi = seg.min(), seg.max()
    if hi - lo < 1e-6:
        return np.nan
    sn = (seg - lo) / (hi - lo)
    if sgn < 0:
        sn = sn[::-1]
    fx = np.linspace(0, sn.size - 1, 2000); fy = np.interp(fx, np.arange(sn.size), sn)
    fy = (fy - fy.min()) / (np.ptp(fy) + 1e-12)
    try:
        w = abs(fx[np.where(fy >= 0.9)[0][0]] - fx[np.where(fy >= 0.1)[0][0]])
    except IndexError:
        return np.nan
    return w if 0 < w < 16 else np.nan


def rise_at(vol, pts):
    """Per-edge widths (px, NaN where invalid) — same order as pts."""
    out = np.full(len(pts), np.nan); cache = {}
    for i, (z, x, yc, sgn) in enumerate(pts):
        if z not in cache:
            cache[z] = np.abs(np.asarray(vol[z])).astype(np.float64)
        a = cache[z]
        if yc - 8 < 0 or yc + 8 > a.shape[0]:
            continue
        out[i] = _width(a[yc - 8:yc + 8, x], sgn)
    return out


def boot_ci(d, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, (n, d.size)); med = np.median(d[idx], axis=1)
    return float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))


def paired_table(wa, wb, b0=None, bins=(0, 20, 40, 60, 90, 130, 200), select=None):
    """Rows of (bin label, n, median A-B, ci_lo, ci_hi) for finite pairs (optionally within `select`)."""
    ok = np.isfinite(wa) & np.isfinite(wb)
    if select is not None:
        ok &= select
    rows = []
    if b0 is None or not np.isfinite(b0).any():
        d = wa[ok] - wb[ok]
        if d.size >= 10:
            c = boot_ci(d); rows.append(("all", int(d.size), float(np.median(d)), c[0], c[1]))
        return rows
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = ok & (b0 >= lo) & (b0 < hi)
        if m.sum() < 25:
            continue
        d = wa[m] - wb[m]; c = boot_ci(d); rows.append((f"{lo}–{hi} Hz", int(m.sum()), float(np.median(d)), c[0], c[1]))
    d = wa[ok] - wb[ok]
    if d.size >= 10:
        c = boot_ci(d); rows.append(("all", int(d.size), float(np.median(d)), c[0], c[1]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True); ap.add_argument("--results", required=True)
    ap.add_argument("--b0map", default=None, help="optional |B0| estimate [z,x,y] (Hz) for stratification")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    orig = np.load(a.orig, mmap_mode="r")
    pts = detect_edges(orig)
    print(f"{len(pts)} edges detected on the uncorrected image", flush=True)
    widths = {"orig": rise_at(orig, pts)}
    for f in sorted(glob.glob(os.path.join(a.results, "*.npy"))):
        n = os.path.splitext(os.path.basename(f))[0]
        if "_fmap" in n or "_selected_freq" in n:
            continue
        widths[n] = rise_at(np.load(f, mmap_mode="r"), pts); print(f"  {n}: median {np.nanmedian(widths[n]):.3f} px", flush=True)
    P = np.array(pts, dtype=np.int32)
    b0 = np.full(len(pts), np.nan, np.float32)
    if a.b0map:
        m = np.load(a.b0map, mmap_mode="r"); b0 = np.array([abs(float(m[z, y, x])) for z, x, y, _ in pts], np.float32)
    out = a.out or os.path.join(a.results, "edge_widths_neutral.npz")
    np.savez(out, pts=P, b0=b0, methods=np.array(list(widths.keys())), widths=np.stack([widths[k] for k in widths]).astype(np.float32),
             note="edges detected on the UNCORRECTED image; widths = 10-90 % of 16-px profiles; pts = (z, x=col, yc=row, sign)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
