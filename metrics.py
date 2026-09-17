"""R12 and R13 — the two gradient-concentration metrics, 2D (per slice) and 3D.

  h   = |∇I| / Σ|∇I|      over a tissue mask (a distribution → gain-invariant)
  R12 = −Σ h·log2 h        gradient entropy      (lower = more concentrated); max log2(n)
  R13 = Σ h²               normalised gradient energy (higher = more concentrated); min 1/n

Ratios vs a reference image (1.0 = unchanged, > 1 = more concentrated):
  r12 = (R12_max − R12_cand) / (R12_max − R12_ref)
  r13 = (R13_cand − R13_min) / (R13_ref − R13_min)

Read these as what they are — gradient CONCENTRATION (≈ gradient SNR), the
focus criteria of the MRI autofocus literature (McGee 2000 family). They are
NOT edge-sharpness measures: any Gaussian blur of a noisy image raises both
(blur suppresses the many small noise gradients more than the few coherent
edge gradients). Use them to read artifacts alongside images, not to rank
methods by sharpness.

3D: gradient over all three axes (optionally with physical voxel spacing) and
one global Otsu mask — the honest version for a 3D correction.
"""
from __future__ import annotations

import numpy as np
from skimage.filters import threshold_otsu


def tissue_mask(reference: np.ndarray) -> np.ndarray:
    """Otsu threshold on log-magnitude (2D or 3D)."""
    log_m = np.log1p(np.abs(reference).astype(np.float32))
    try:
        return log_m > float(threshold_otsu(log_m))
    except Exception:
        return np.zeros(log_m.shape, dtype=bool)


def r12_r13(vol: np.ndarray, mask: np.ndarray, spacing=None) -> dict:
    """R12/R13 of |vol| over mask. 2D input → in-plane gradient; 3D → 3D gradient.
    Returns dict(R12, R13, R12_max, R13_min, n)."""
    m = np.abs(vol).astype(np.float32)
    if spacing is None:
        spacing = (1.0,) * m.ndim
    g = None
    for ax in range(m.ndim):
        if m.shape[ax] < 2:
            continue
        c = np.gradient(m, spacing[ax], axis=ax)
        g = c * c if g is None else g + c * c
    np.sqrt(g, out=g)
    g_in = g[mask].astype(np.float64)
    n = int(g_in.size)
    s = g_in.sum()
    if n == 0 or s <= 0:
        return {"R12": np.nan, "R13": np.nan, "R12_max": np.nan, "R13_min": np.nan, "n": n}
    h = g_in / s
    nz = h > 0
    return {"R12": float(-(h[nz] * np.log2(h[nz])).sum()), "R13": float((h * h).sum()),
            "R12_max": float(np.log2(n)), "R13_min": 1.0 / n, "n": n}


def ratios(ref: dict, cand: dict) -> dict:
    """r12, r13 of candidate vs reference (both from r12_r13 with the SAME mask)."""
    out = {"r12": np.nan, "r13": np.nan}
    d = ref["R12_max"] - ref["R12"]
    if np.isfinite(d) and d > 0:
        out["r12"] = (ref["R12_max"] - cand["R12"]) / d
    d = ref["R13"] - ref["R13_min"]
    if np.isfinite(d) and d > 0:
        out["r13"] = (cand["R13"] - ref["R13_min"]) / d
    return out


def per_slice(vol: np.ndarray, mask: np.ndarray) -> list[dict]:
    return [r12_r13(vol[z], mask[z]) for z in range(vol.shape[0])]


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        ref = np.load(sys.argv[1], mmap_mode="r"); cand = np.load(sys.argv[2], mmap_mode="r")
        m = tissue_mask(np.asarray(ref))
        a, b = r12_r13(np.asarray(ref), m), r12_r13(np.asarray(cand), m)
        print("3D:", {k: round(v, 4) for k, v in ratios(a, b).items()}, "| R12", round(a["R12"], 3), "→", round(b["R12"], 3),
              "| R13", f"{a['R13']:.3e} → {b['R13']:.3e}")
    else:
        print(__doc__)
