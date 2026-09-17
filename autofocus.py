"""3D field-map-free autofocus (Noll 1992 family) — the multi-frequency
baseline, streamed so a 768×768×112 volume fits in memory.

  1. K = FT3(image). Low-resolution reference phase from |k| ≤ phase_ref_kr.
  2. For each candidate frequency f: m_f = IFT3(K · exp(+i·2π·f·t(k))) with the
     REAL sample-time map t(k) (3D map, or built from the radial profile),
     phase-referenced to the low-res image.
  3. Focus criterion per voxel: windowed mean of |Imag(m_f)| (the criterion
     that survived our ground-truth tests; gradient/entropy criteria pick
     noise). Per-voxel argmin over f, then Gaussian smoothing of the index map
     (unsmoothed per-voxel selection checkerboards).
  4. Second pass re-generates m_f and assembles the output from the smoothed
     index map (so the whole stack is never held in memory).

Honest note from the ground-truth work: per-voxel *selection* has a hard
ceiling (~+2 dB even with the true field) because off-resonance blur is
additive; on gridded data this typically ties or trails the ring cascade.
It is here as the standard comparison, not as a recommendation.
"""
from __future__ import annotations

import numpy as np
from scipy import fft as sfft
from scipy import ndimage
from scipy.interpolate import interp1d


def _fft3(x):
    return sfft.fftshift(sfft.fftn(sfft.ifftshift(x), norm="ortho", workers=-1))


def _ifft3(x):
    return sfft.fftshift(sfft.ifftn(sfft.ifftshift(x), norm="ortho", workers=-1))


def kr_norm_zxy(shape_zxy) -> np.ndarray:
    """Per-axis-normalised k-radius on a [z, x, y] grid: each axis' Nyquist → 1.0
    (the sampled cones sphere is inscribed, edge = 1.0)."""
    out = None
    for ax, n in enumerate(shape_zxy):
        f = ((np.arange(n) - n // 2) / (n / 2)).astype(np.float32) ** 2
        view = [None] * 3; view[ax] = slice(None)
        out = f[tuple(view)] if out is None else out + f[tuple(view)]
    return np.sqrt(out, dtype=np.float32)


def time_map_from_profile(shape_zxy, profile_npz: str, kr_edge: float = 1.0) -> np.ndarray:
    """t(k) for a [z, x, y] grid from a radial (kr, t) profile whose sampled
    sphere edge is at kr = kr_edge (1.0 for data/tkr_profile_brian.npz).
    Radial APPROXIMATION of the true 3D map (cones timing is not exactly
    radial: ±16 % within a shell); pass a real 3D map when you have one."""
    p = np.load(profile_npz); kr = p["kr"].astype(np.float64) / kr_edge; t = p["t_s"].astype(np.float64)
    o = np.argsort(kr)
    f = interp1d(kr[o], t[o], bounds_error=False, fill_value=(t[o][0], t[o][-1]))
    return f(kr_norm_zxy(shape_zxy)).astype(np.float32)


def autofocus_3d(vol_zxy: np.ndarray, tmap_zxy: np.ndarray, freqs_hz=None, window=(3, 5, 5),
                 idx_smooth_sigma: float = 5.0, phase_ref_kr: float = 0.1, verbose: bool = False):
    """Returns (corrected [z,x,y] complex64, selected_freq_hz [z,x,y] float32, info)."""
    if freqs_hz is None:
        freqs_hz = np.linspace(-150.0, 150.0, 21)
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
    vol = np.asarray(vol_zxy).astype(np.complex64)
    K = _fft3(vol).astype(np.complex64)
    del vol
    kr = kr_norm_zxy(K.shape)
    lowmask = (kr <= phase_ref_kr)
    w_low = np.abs(_ifft3(np.where(lowmask, K, 0))).astype(np.float32)   # signal weight for the index smoothing
    del kr
    t = np.asarray(tmap_zxy, dtype=np.float32)
    two_pi_t = (2 * np.pi * t).astype(np.float32)

    best_c = None; worst_c = None; best_idx = np.zeros(K.shape, np.int8)
    for i, f in enumerate(freqs_hz):
        if verbose:
            print(f"autofocus pass 1: {i + 1}/{len(freqs_hz)} ({f:+.0f} Hz)", flush=True)
        Kf = (K * np.exp(1j * (two_pi_t * np.float32(f))).astype(np.complex64)).astype(np.complex64)
        m = _ifft3(Kf).astype(np.complex64)
        # Each candidate is referenced to ITS OWN low-pass phase (Noll 1992):
        # this zeros the phase of flat tissue for every f, so only the
        # blur-induced (high-k) imaginary part at edges drives the criterion.
        # A single reference from the uncorrected image biases the choice to 0 Hz.
        m_low = _ifft3(np.where(lowmask, Kf, 0)).astype(np.complex64)
        del Kf
        m *= np.exp(-1j * np.angle(m_low)).astype(np.complex64)
        del m_low
        c = ndimage.uniform_filter(np.abs(m.imag), size=window, mode="nearest")
        del m
        if best_c is None:
            best_c = c.copy(); worst_c = c
        else:
            better = c < best_c
            best_c[better] = c[better]; best_idx[better] = i
            np.maximum(worst_c, c, out=worst_c)
            del better, c
    # Focus information lives where the criterion actually VARIES with f
    # (edges); in flat tissue and in the background every f gives |Imag|≈0
    # and the argmin is meaningless. Propagate the edge selections inward by
    # normalised-convolution smoothing of the index map with weight
    # (criterion range) × (low-res magnitude).
    focus_w = ((worst_c - best_c) * w_low).astype(np.float32)
    del best_c, worst_c, w_low
    idx_s = best_idx.astype(np.float32)
    if idx_smooth_sigma and idx_smooth_sigma > 0:
        num = ndimage.gaussian_filter(idx_s * focus_w, sigma=idx_smooth_sigma, mode="nearest")
        den = ndimage.gaussian_filter(focus_w, sigma=idx_smooth_sigma, mode="nearest")
        idx_s = num / (den + np.float32(1e-6) * float(focus_w.max()))
        del num, den
    idx_s = np.clip(np.round(idx_s), 0, len(freqs_hz) - 1).astype(np.int8)
    del best_idx, focus_w

    out = np.zeros(K.shape, np.complex64)
    for i, f in enumerate(freqs_hz):
        sel = idx_s == i
        if not sel.any():
            continue
        if verbose:
            print(f"autofocus pass 2: {i + 1}/{len(freqs_hz)} ({f:+.0f} Hz, {100 * sel.mean():.1f}% of voxels)", flush=True)
        m = _ifft3(K * np.exp(1j * (two_pi_t * np.float32(f))).astype(np.complex64)).astype(np.complex64)
        out[sel] = m[sel]
        del m, sel
    fsel = freqs_hz[idx_s.astype(np.intp)].astype(np.float32)
    info = {"freqs_hz": freqs_hz.tolist(), "window": list(window), "idx_smooth_sigma": idx_smooth_sigma,
            "phase_ref_kr": phase_ref_kr,
            "sign_note": "selected_freq f means the voxel was demodulated by exp(+i2πf·t(k)); "
                         "positive f cancels a k-space phase exp(−i2πf·t)"}
    return out, fsel, info


if __name__ == "__main__":
    print(__doc__)
