"""Reference-free sharpness measures for one 2D slice (or a box of it), computed for a
candidate against the uncorrected slice so the numbers are comparable across methods.

  edge width   median 10-90 % width (px) of 16-px profiles at strong edges detected on the
               UNCORRECTED slice (edge_width.detect_edges thresholds); lower = sharper
  Lim S        Lim et al., MRM 2019 (Sec. 3.5): CNR / d per edge, d = 20-80 % distance of the
               same profile, CNR = (I80 - I20) / sigma_air; median over edges; higher = sharper
  TG, AES      Marchetto & Eichhorn 2024 (arXiv 2412.18389, Table 1): Tenengrad = mean Sobel
               gradient² in the mask; Average Edge Strength = sqrt(sum gradient² at Canny edges)
               / number of edge pixels. Both after gain-matching to the uncorrected slice and
               percentile (1 / 99.9) normalisation taken from the uncorrected slice; higher = sharper
  NGS, GE, IE  normalised gradient square (= R13), gradient entropy, image entropy (same paper);
               NGS higher / GE, IE lower = more concentrated — shown for completeness only.
In our benchmark against visual scoring, AES and TG agreed best with the eye, NGS/GE/IE did not,
and the edge width needs many pooled edges (a single patch is noisy).
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage
from skimage.feature import canny
from edge_width import detect_edges

HALF = 8


def _sobel(x):
    return np.hypot(ndimage.sobel(x, axis=1), ndimage.sobel(x, axis=0))


def ref_free(x, m, lo, hi):
    xn = (x - lo) / (hi - lo + 1e-12)
    g = _sobel(xn); E = canny(xn, sigma=1.0) & m
    gi = g[m]; xi = xn[m]
    if gi.size == 0:
        return {k: np.nan for k in ("TG", "AES", "NGS", "GE", "IE")}
    h = gi / (gi.sum() + 1e-12)
    z = gi / (np.sqrt(np.sum(gi ** 2)) + 1e-12); zz = z[z > 0]
    y = np.abs(xi) / (np.sqrt(np.sum(xi ** 2)) + 1e-12); yy = y[y > 0]
    return {"TG": float(np.mean(gi ** 2)), "AES": float(np.sqrt(np.sum(g[E] ** 2)) / max(int(E.sum()), 1)),
            "NGS": float(np.sum(h ** 2)), "GE": float(-np.sum(zz * np.log(zz))), "IE": float(-np.sum(yy * np.log(yy)))}


def profiles(a, pts, sigma_air):
    """(median 10-90 width, median Lim S, n) at pts = (x=col, yc=row, sign) on a 2D magnitude image."""
    widths, S = [], []
    for x, yc, sgn in pts:
        if yc - HALF < 0 or yc + HALF > a.shape[0] or x >= a.shape[1]:
            continue
        seg = a[yc - HALF:yc + HALF, x].astype(np.float64)
        if sgn < 0:
            seg = seg[::-1]
        n = seg.size; fx = np.linspace(0, n - 1, n * 10); fy = np.interp(fx, np.arange(n), seg)
        lo, hi = fy.min(), fy.max()
        if hi - lo < 1e-9:
            continue
        sn = (fy - lo) / (hi - lo)
        try:
            i10 = np.where(sn >= 0.1)[0][0]; i90 = np.where(sn >= 0.9)[0][0]
            i20 = np.where(sn >= 0.2)[0][0]; i80 = np.where(sn >= 0.8)[0][0]
        except IndexError:
            continue
        w = abs(fx[i90] - fx[i10]); d = abs(fx[i80] - fx[i20])
        if not (0 < w < 16) or d <= 0:
            continue
        widths.append(w); S.append((fy[i80] - fy[i20]) / (sigma_air + 1e-12) / d)
    return (float(np.median(widths)) if widths else np.nan, float(np.median(S)) if S else np.nan, len(widths))


def sigma_air(orig2d, mask2d):
    """Robust noise level outside the tissue mask (1.4826·MAD of |I| where mask is False)."""
    air = np.abs(orig2d)[~mask2d]
    if air.size < 100:
        air = np.abs(orig2d).ravel()
    return float(1.4826 * np.median(np.abs(air - np.median(air))) + 1e-12)


def compare_iqms(orig2d, cand2d, mask2d, box=None, pts=None, sig=None):
    """All measures for a candidate slice vs the uncorrected slice, optionally inside box=(y0,y1,x0,x1).
    pts (edges on the uncorrected FULL slice) and sig (air noise) can be passed to reuse across methods."""
    o = np.abs(orig2d).astype(np.float64); c = np.abs(cand2d).astype(np.float64); m = mask2d.astype(bool)
    if pts is None:
        pts = [(x, yc, s) for (_, x, yc, s) in detect_edges(o[None], slice_indices=[0])]
    if sig is None:
        sig = sigma_air(o, m)
    if m.sum() > 0:
        c = c * (o[m].mean() / (c[m].mean() + 1e-12))              # gain match
    lo, hi = (np.percentile(o[m], [1, 99.9]) if m.sum() > 0 else (o.min(), o.max()))
    if box is not None:
        y0, y1, x0, x1 = box
        sl = (slice(y0, y1), slice(x0, x1)); o_, c_, m_ = o[sl], c[sl], m[sl]
        p_ = [(x - x0, yc - y0, s) for (x, yc, s) in pts if y0 <= yc < y1 and x0 <= x < x1]
    else:
        o_, c_, m_, p_ = o, c, m, pts
    ro, rc = ref_free(o_, m_, lo, hi), ref_free(c_, m_, lo, hi)
    wo, so, n = profiles(o_, p_, sig); wc, sc, _ = profiles(c_, p_, sig)
    out = {"n_edges": n, "edge_width_orig": wo, "edge_width": wc, "limS_orig": so, "limS": sc}
    for k in ("TG", "AES", "NGS", "GE", "IE"):
        out[k] = rc[k]; out[k + "_orig"] = ro[k]; out[k + "_ratio"] = (rc[k] / ro[k]) if ro[k] not in (0, np.nan) and np.isfinite(ro[k]) else np.nan
    return out
