"""Ring cascade with a CORRECTLY CALIBRATED 3D field map (and optional temporal unwrap).

Why this exists (verified 2026-09-16, see memory/ring_construction_findings.md):

1. Timing convention. `data/tkr_profile_brian.npz` stores `kr` in units where
   the sampled sphere's edge is 1.0 (t reaches its 8.47 ms maximum there).
   `cones_recon_fcns.ring_b0_corr` puts that edge at k_radius = 0.5. Scripts
   were passing the raw profile unscaled, so every ring's time was read at
   half its true radius (N=8: 0.18…4.16 ms instead of 0.33…8.22 ms).
   Here `kr_edge` says what value of `sample_kradius` is the sphere edge, and
   the conversion (×0.5/kr_edge) happens inside.

2. Estimator. `ring_b0_corr` divides each ring's INCREMENTAL phase (the
   residual after all previous rings' phase was removed) by the time since
   ring 0 and averages. For a linear phase that returns
   mean_r (t_r − t_{r−1})/(t_r − t_0) × f ≈ 0.40 f. With the halved times the
   two errors nearly cancelled (≈1.1 f), so quoted Hz numbers were right only
   by accident, with rings 1–2 over-weighted ~3× and rings ≥5 contributing ~0.
   Here the increments are fitted as a least-squares slope of phase vs time
   (`ls_increment`, default), with the cumulative-phase fit (`ls_cumulative`)
   and Brian's estimator (`brian`) kept for comparison.

3. Sign. Positive field_map ⇔ the source k-space (sp.ifft of the image)
   carries exp(−i·2π·f·t(k)). This is the NEGATIVE of ring_b0_corr's
   field_map sign (ts3d scripts used fmap_scale=−1 with the old map).

4. Unwrap. Brian's ±π/2 sign flip is NOT off-resonance unwrapping — it
   rectifies the 0/π sign ambiguity of a real-valued band-pass image (50–70%
   of voxels flip on every shell). Rectifying around zero caps a uniform
   off-resonance at about 1/(4·Δt_max) (N=8: clean to ~110 Hz, fails by
   160 Hz; N=16: ~200 Hz). `unwrap="predict"` rectifies around a predicted
   increment instead — the coherence-weighted running phase slope of the
   previous shells, ridge-shrunk to zero where the shells are incoherent, so
   it degrades to Brian's flip in vivo where the evidence is weak. Default
   stays "flip" (= Brian's correction, bit-for-bit up to float32) so the
   corrected IMAGE is unchanged unless you ask for it.

Everything is 3D: 3D shells (per-axis-normalised k-radius), 3D Gaussian phase
smoothing, 3D field map.

Self-test:  .venv/bin/python ring_fieldmap.py --selftest
"""
from __future__ import annotations

import numpy as np
from scipy import fft as sfft
from scipy import ndimage


# ---------------------------------------------------------------------------
# FFT helpers — same convention as sigpy.fft/ifft (centered, ortho), but in
# single precision via scipy (numpy < 2 upcasts complex64 → complex128).
# ---------------------------------------------------------------------------
def _fft3(x):
    return sfft.fftshift(sfft.fftn(sfft.ifftshift(x), norm="ortho", workers=-1))


def _ifft3(x):
    return sfft.fftshift(sfft.ifftn(sfft.ifftshift(x), norm="ortho", workers=-1))


def k_radius_norm(shape_xyz: tuple[int, int, int]) -> np.ndarray:
    """ring_b0_corr's normalised k-radius on an [x, y, z] grid: each axis'
    Nyquist maps to 0.5 (inscribed sphere = 0.5, cube corners ≈ 0.87)."""
    nx, ny, nz = shape_xyz
    kx = np.arange(-np.floor(nx / 2), np.floor((nx - 1) / 2) + 1, dtype=np.float32)
    ky = np.arange(-np.floor(ny / 2), np.floor((ny - 1) / 2) + 1, dtype=np.float32)
    kz = np.arange(-np.floor(nz / 2), np.floor((nz - 1) / 2) + 1, dtype=np.float32)
    aspect = (1.0, ny / nx, nz / nx)
    kr2 = ((kx / aspect[0]) ** 2)[:, None, None] + ((ky / aspect[1]) ** 2)[None, :, None] \
        + ((kz / aspect[2]) ** 2)[None, None, :]
    kr = np.sqrt(kr2, dtype=np.float32)
    return 0.5 * kr / float(np.max(ky / aspect[1]))


def ring_bounds(num_rings: int, ring_overlap: float = 1.0):
    inc = 0.5 / (num_rings - 1) if num_rings > 1 else 0.5
    width = inc * ring_overlap
    start = inc * np.arange(num_rings) - width
    return start, start + 2 * width, width


def ring_times_s(sample_kradius, sample_times, num_rings: int, kr_edge: float = 1.0,
                 weighting: str = "hamming") -> np.ndarray:
    """Effective acquisition time of each shell from the cones (kr, t) samples.
    `sample_kradius` is converted so that kr_edge → 0.5 (the code's sphere
    edge). weighting: 'hamming' (matches the shell filter) or 'mean' (Brian)."""
    skr = np.asarray(sample_kradius, dtype=np.float64) * (0.5 / kr_edge)
    t = np.asarray(sample_times, dtype=np.float64)
    start, end, width = ring_bounds(num_rings)
    out = np.full(num_rings, np.nan)
    for i in range(num_rings):
        m = (skr >= start[i]) & (skr < end[i])
        if not m.any():
            continue
        if weighting == "hamming":
            w = 0.54 - 0.46 * np.cos(np.pi * (skr[m] - start[i]) / width)
            out[i] = float(np.sum(w * t[m]) / np.sum(w))
        else:
            out[i] = float(np.mean(t[m]))
    return out


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def matlab_fspecial3_gaussian(kernelsize: int) -> np.ndarray:
    """MATLAB fspecial3('gaussian', kernelsize*[1 1 1], kernelsize/2) exactly:
    a Gaussian with σ = kernelsize/2 sampled on the (kernelsize)³ integer grid
    and TRUNCATED to that box, normalised to sum 1. This is what Brian's
    ring_b0_corr.m (kernelsize 7) and ts_b0_corr.m (kernelsize 9) use — NOT a
    full Gaussian of that σ: the box cuts it at ±(kernelsize−1)/2, so the
    effective smoothing std is ~2.0 (size 7) / ~2.4 (size 9) voxels, versus
    4.5 for scipy.ndimage.gaussian_filter(σ=4.5), which extends to ±18."""
    h = (kernelsize - 1) // 2
    x = np.arange(-h, h + 1, dtype=np.float64)
    s = kernelsize / 2.0
    g1 = np.exp(-(x ** 2) / (2 * s ** 2))
    g = g1[:, None, None] * g1[None, :, None] * g1[None, None, :]
    return (g / g.sum()).astype(np.float32)


def _smooth(x: np.ndarray, sigma: float, kernel) -> np.ndarray:
    """Per-shell phase smoothing. kernel=None → scipy Gaussian(σ) (historical
    Python behaviour); kernel=ndarray (separable, from matlab_fspecial3_gaussian)
    → exact MATLAB imfilter(...,'replicate'), applied as three 1D passes
    (identical result, ~27× faster than the full 3D convolution)."""
    if kernel is None:
        return ndimage.gaussian_filter(x, sigma=sigma, mode="nearest")
    k1 = kernel.sum(axis=(1, 2)).astype(np.float64)          # 1D factor; product = kernel (sum 1)
    k1 = k1 / k1.sum()

    def sep(a):
        out = a.astype(np.float32, copy=True)
        for ax in range(a.ndim):
            out = ndimage.convolve1d(out, k1, axis=ax, mode="nearest")
        return out

    if np.iscomplexobj(x):
        return (sep(x.real) + 1j * sep(x.imag)).astype(np.complex64)
    return sep(x)


# ---------------------------------------------------------------------------
# The cascade
# ---------------------------------------------------------------------------
def ring_b0_corr_v2(source_image_zxy: np.ndarray, num_rings: int = 8,
                    sample_kradius=None, sample_times=None, kr_edge: float = 1.0,
                    smooth_sigma: float = 4.5, unwrap: str = "flip",
                    fmap_method: str = "ls_increment", ring_time_weighting: str = "hamming",
                    verbose: bool = False, debug_mask_zxy: np.ndarray | None = None,
                    predict_ridge_coh: float = 0.25, matlab_kernelsize: int | None = None):
    """Hierarchical per-shell phase correction (Brian's ring cascade) with a
    calibrated 3D field map.

    Parameters
    ----------
    source_image_zxy : complex volume [z, x, y] (same order as ring_b0_corr).
    num_rings : number of overlapping Hamming shells (Brian's baseline: 8).
    sample_kradius, sample_times : cones (|k|, t) samples for the shell times.
        `kr_edge` = the sample_kradius value at the sampled sphere's edge
        (1.0 for data/tkr_profile_brian.npz — pass the raw `kr`).
    smooth_sigma : 3D Gaussian σ (voxels) of the per-shell phase smoothing
        (scipy full Gaussian — the historical Python behaviour).
    matlab_kernelsize : if set (7 = ring_b0_corr.m, 9 = ts_b0_corr.m), use
        MATLAB's fspecial3 truncated kernel of that size instead — this is what
        makes the correction 1:1 with Brian's MATLAB; smooth_sigma is ignored.
    unwrap : 'flip' (Brian's ±π/2 sign flip) or 'predict' (temporal unwrap
        around the coherence-weighted running phase slope; needs shell times).
    predict_ridge_coh : ridge for the prediction — equivalent to one prior
        shell at this coherence; below that the prediction shrinks to zero and
        the step degrades gracefully to Brian's flip.
    fmap_method : 'ls_increment' | 'ls_cumulative' | 'brian' — which map is
        returned as `field_map`; all three are in info['field_maps'].

    Returns
    -------
    corr_image [z,x,y] complex64, field_map [z,x,y] float32 (Hz, sign per
    module docstring), info dict (ring_times_s, ring_bounds, field_maps, ...).
    """
    if unwrap not in ("flip", "predict"):
        raise ValueError("unwrap must be 'flip' or 'predict'")
    if fmap_method not in ("ls_increment", "ls_cumulative", "ls_weighted", "brian", "matlab"):
        raise ValueError("fmap_method must be ls_increment | ls_cumulative | ls_weighted | brian | matlab")

    kern = matlab_fspecial3_gaussian(matlab_kernelsize) if matlab_kernelsize else None
    img = np.ascontiguousarray(np.transpose(source_image_zxy, (1, 2, 0))).astype(np.complex64)
    nx, ny, nz = img.shape
    ks = _ifft3(img).astype(np.complex64)       # ring_b0_corr: source_kspace = ift3(image)
    del img
    kr = k_radius_norm((nx, ny, nz))
    start, end, width = ring_bounds(num_rings)

    has_t = sample_kradius is not None and sample_times is not None
    t_ring = (ring_times_s(sample_kradius, sample_times, num_rings, kr_edge,
                           ring_time_weighting) if has_t else np.full(num_rings, np.nan))
    if unwrap == "predict" and not has_t:
        raise ValueError("unwrap='predict' needs sample_kradius/sample_times")

    corr_image = np.zeros((nx, ny, nz), np.complex64)
    corr_phase = np.ones((nx, ny, nz), np.complex64)
    # Field-map accumulators (all float32, all 3D)
    cum_phase = np.zeros((nx, ny, nz), np.float32)          # Σ increments (unwrapped by construction)
    acc_inc_num = np.zeros((nx, ny, nz), np.float32)        # Σ Δφ_r · Δt_r
    acc_cum_num = np.zeros((nx, ny, nz), np.float32)        # Σ Φ_r · T_r
    acc_brian = np.zeros((nx, ny, nz), np.float32)          # Σ Δφ_r / T_r
    # Coherence-weighted LS: w_r = coh_r², coh_r = |smoothed phasor| / smoothed |phasor|
    # (1 where the shell's phase is spatially coherent, →0 where the flip/noise/
    # dephasing randomised it). In vivo the late shells lose coherence exactly
    # where |B0| is large, and an unweighted dt²-LS is pulled to zero there.
    acc_w_num = np.zeros((nx, ny, nz), np.float32)          # Σ coh² Δφ_r Δt_r
    acc_w_den = np.zeros((nx, ny, nz), np.float32)          # Σ coh² Δt_r²
    # EXACT ts_b0_corr.m estimator (reproduced literally, including its use of
    # the accumulated corr_phase a second time):
    #   ex_field_map += angle(image_copy_smoothed .* corr_phase) / (ring_time(r) − ring_time(r−1))
    #   ex_field_map  = ex_field_map / (num_rings − 1) / (2π)
    acc_matlab = np.zeros((nx, ny, nz), np.float32)
    n_matlab = 0
    coh_median = []
    den_inc = 0.0; den_cum = 0.0; n_brian = 0
    prev_inc = None; prev_dt = None
    n_flipped = []
    # debug_mask_zxy: a bool volume, or a dict {region_name: bool volume};
    # per-ring medians of the raw residual phase, prediction, residual,
    # increment and coherence are recorded per region in info['debug_rows'].
    if debug_mask_zxy is None:
        dbg_regions = None
    elif isinstance(debug_mask_zxy, dict):
        dbg_regions = {k: np.ascontiguousarray(np.transpose(v, (1, 2, 0))) for k, v in debug_mask_zxy.items()}
    else:
        dbg_regions = {"mask": np.ascontiguousarray(np.transpose(debug_mask_zxy, (1, 2, 0)))}
    dbg = dbg_regions["mask"] if (dbg_regions is not None and "mask" in dbg_regions) else None
    debug_rows = []
    _dbg_pending: dict = {}

    for r in range(num_rings):
        if verbose:
            print(f"ring {r + 1}/{num_rings}", flush=True)
        m = (kr >= start[r]) & (kr < end[r])
        filt = np.zeros((nx, ny, nz), np.float32)
        filt[m] = 0.54 - 0.46 * np.cos(np.pi * (kr[m] - start[r]) / width)
        ring_image = _fft3(ks * filt).astype(np.complex64)   # ring_b0_corr: ft3(masked k-space)
        del filt, m

        if r == 0:
            corr_phase = np.exp(-1j * np.angle(ring_image)).astype(np.complex64)
            corr_image = (ring_image * corr_phase).astype(np.complex64)
            continue

        image_copy = (ring_image * corr_phase).astype(np.complex64)   # residual after prior shells
        dt = (t_ring[r] - t_ring[r - 1]) if has_t else np.nan
        raw_med = float(np.median(np.angle(image_copy)[dbg])) if dbg is not None else np.nan
        if dbg_regions is not None:
            _ang = np.angle(image_copy)
            _dbg_pending = {name: {"ring": r, "dt_ms": dt * 1e3, "raw_med": float(np.median(_ang[m_]))}
                            for name, m_ in dbg_regions.items()}
            del _ang
        if unwrap == "predict" and r >= 2 and np.isfinite(dt) and dt > 0:
            # Temporal unwrap: rotate by the increment predicted for this shell,
            # THEN rectify the band-pass sign ambiguity (a real-valued band-pass
            # image has phase 0 or π, so ~50% of voxels always sit at ±π; Brian's
            # flip is what makes the phasor average coherent) around the
            # prediction instead of around zero. This turns the ±π/2 cap on the
            # absolute increment into a ±π/2 cap on the prediction ERROR.
            #
            # The prediction is the running COHERENCE-WEIGHTED phase slope from
            # all previous shells (Σ coh²·Δφ·Δt / Σ coh²·Δt²) times this Δt, with
            # a ridge that shrinks it to zero where the accumulated coherent
            # evidence is weak — so in incoherent regions (in vivo, late shells
            # at tissue/air boundaries) it reduces exactly to Brian's flip.
            # Predicting from the previous shell alone is NOT safe in vivo: its
            # increment is mostly structural noise, and rectifying around noise
            # is worse than rectifying around zero (ghosted edges).
            lam = np.float32((predict_ridge_coh * dt) ** 2)
            pred = (acc_w_num * np.float32(dt) / (acc_w_den + lam)).astype(np.float32)
            image_copy *= np.exp(-1j * pred).astype(np.complex64)
            flip = np.abs(np.angle(image_copy)) > (np.pi / 2)
            n_flipped.append(float(flip.mean()))
            if dbg_regions is not None:
                for name, m_ in dbg_regions.items():
                    _dbg_pending[name]["flip_frac"] = float(flip[m_].mean())
            image_copy[flip] = -image_copy[flip]
            del flip
            sm = _smooth(image_copy, smooth_sigma, kern)
            mag_s = _smooth(np.abs(image_copy), smooth_sigma, kern)
            resid = np.angle(sm).astype(np.float32)
            inc = (pred + resid).astype(np.float32)
            if dbg_regions is not None:
                for name, m_ in dbg_regions.items():
                    _dbg_pending[name].update({"pred_med": float(np.median(pred[m_])),
                                               "resid_med": float(np.median(resid[m_])),
                                               "inc_med": float(np.median(inc[m_]))})
            del pred, resid
        else:
            flip = np.abs(np.angle(image_copy)) > (np.pi / 2)
            n_flipped.append(float(flip.mean()))
            if dbg_regions is not None:
                for name, m_ in dbg_regions.items():
                    _dbg_pending[name]["flip_frac"] = float(flip[m_].mean())
            image_copy[flip] = -image_copy[flip]
            del flip
            sm = _smooth(image_copy, smooth_sigma, kern)
            mag_s = _smooth(np.abs(image_copy), smooth_sigma, kern)
            inc = np.angle(sm).astype(np.float32)
            if dbg_regions is not None:
                for name, m_ in dbg_regions.items():
                    _dbg_pending[name].update({"inc_med": float(np.median(inc[m_]))})
        if has_t and np.isfinite(dt) and dt > 0:
            acc_matlab += (np.angle(sm * corr_phase) / np.float32(dt)).astype(np.float32)
            n_matlab += 1
        coh = (np.abs(sm) / (mag_s + np.float32(1e-12))).astype(np.float32)
        np.clip(coh, 0.0, 1.0, out=coh)
        coh_median.append(float(np.median(coh)))
        if dbg_regions is not None:
            for name, m_ in dbg_regions.items():
                _dbg_pending[name].update({"region": name, "coh_med": float(np.median(coh[m_]))})
                debug_rows.append(_dbg_pending[name])
        del image_copy, sm, mag_s

        corr_phase = (corr_phase * np.exp(-1j * inc)).astype(np.complex64)
        corr_image += (ring_image * corr_phase).astype(np.complex64)
        del ring_image

        # --- field-map accumulation (3D, on the fly) ---
        cum_phase += inc
        if has_t and np.isfinite(dt) and dt > 0:
            T = t_ring[r] - t_ring[0]
            acc_inc_num += inc * np.float32(dt);     den_inc += dt * dt
            acc_cum_num += cum_phase * np.float32(T); den_cum += T * T
            w = coh * coh
            acc_w_num += w * inc * np.float32(dt); acc_w_den += w * np.float32(dt * dt)
            del w
            if T > 0:
                acc_brian += inc / np.float32(T); n_brian += 1
        del coh
        prev_inc, prev_dt = inc, dt

    # Sign: increments are ≈ −2π f Δt for k-space phase exp(−i2πf t) → negate.
    field_maps = {}
    if has_t and den_inc > 0:
        field_maps["ls_increment"] = (-acc_inc_num / np.float32(2 * np.pi * den_inc)).astype(np.float32)
        field_maps["ls_cumulative"] = (-acc_cum_num / np.float32(2 * np.pi * den_cum)).astype(np.float32)
        field_maps["brian"] = (acc_brian / np.float32(2 * np.pi * max(n_brian, 1))).astype(np.float32)
        field_maps["ls_weighted"] = (-acc_w_num / (np.float32(2 * np.pi) * (acc_w_den + np.float32(1e-12)))).astype(np.float32)
        # MATLAB: ex_field_map / (num_rings-1) / (2*pi)  — MATLAB's sign, no negation
        field_maps["matlab"] = (acc_matlab / np.float32((num_rings - 1) * 2 * np.pi)).astype(np.float32)
    else:
        for k in ("ls_increment", "ls_cumulative", "ls_weighted", "brian", "matlab"):
            field_maps[k] = np.zeros((nx, ny, nz), np.float32)
    field_map = field_maps[fmap_method]

    to_zxy = lambda a: np.ascontiguousarray(np.transpose(a, (2, 0, 1)))
    info = {
        "ring_times_s": t_ring, "ring_start": start, "ring_end": end, "ring_width": width,
        "kr_edge": kr_edge, "smooth_sigma": smooth_sigma, "unwrap": unwrap,
        "matlab_kernelsize": matlab_kernelsize,
        "fmap_method": fmap_method, "flip_fraction_per_ring": n_flipped,
        "coherence_median_per_ring": coh_median,
        "sign_note": "positive = k-space phase exp(-i2πft) in the sp.ifft(image) "
                     "convention; NEGATIVE of ring_b0_corr's field_map sign",
        "field_maps": {k: to_zxy(v) for k, v in field_maps.items()},
        "debug_rows": debug_rows,
    }
    return to_zxy(corr_image), to_zxy(field_map), info


def ring_b0_corr_matlab(source_image_zxy: np.ndarray, num_rings: int = 8,
                        sample_kradius=None, sample_times=None, kernelsize: int = 9,
                        kr_edge: float = 1.0, verbose: bool = False):
    """EXACT Python equivalent of Brian's MATLAB:
         kernelsize=9 → ts_b0_corr.m   [corr_image, ex_field_map] = ring_b0_corr(src, N, kr, t)
         kernelsize=7 → ring_b0_corr.m corr_image = ring_b0_corr(src, N)   (no field map)
    Same shells, Hamming windows, cascade, ±π/2 sign rule, fspecial3 truncated
    kernel with imfilter 'replicate', plain-mean shell times, and the literal
    ts_b0_corr.m field-map accumulation and sign. Input/output [z, x, y].
    `sample_kradius` may be given in the profile's units (sphere edge = kr_edge,
    1.0 for data/tkr_profile_brian.npz); MATLAB expects it in the shell units
    (edge = 0.5) — pass kr_edge=0.5 if you already converted.
    Returns (corr_image, ex_field_map, info)."""
    return ring_b0_corr_v2(source_image_zxy, num_rings, sample_kradius, sample_times,
                           kr_edge=kr_edge, unwrap="flip", fmap_method="matlab",
                           ring_time_weighting="mean", matlab_kernelsize=kernelsize,
                           verbose=verbose)


# ---------------------------------------------------------------------------
# Self-test: regression vs cones_recon_fcns + calibration + aliasing sweep
# ---------------------------------------------------------------------------
def _selftest():
    import contextlib, io, os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import cones_recon_fcns as crf
    import sigpy as sp
    from scipy.interpolate import interp1d
    from scipy.ndimage import gaussian_filter

    prof = np.load("data/tkr_profile_brian.npz"); raw = prof["kr"]; t_p = prof["t_s"]
    o = np.argsort(raw)
    tfun = interp1d(raw[o], t_p[o], bounds_error=False, fill_value=(t_p[o][0], t_p[o][-1]))

    nx, ny, nz = 64, 64, 32
    x, y, z = np.mgrid[:nx, :ny, :nz]; c = np.array([nx / 2, ny / 2, nz / 2])
    obj = 1.0 * (((x - c[0]) / 22) ** 2 + ((y - c[1]) / 26) ** 2 + ((z - c[2]) / 11) ** 2 < 1)
    obj += 0.8 * (((x - c[0] - 8) / 7) ** 2 + ((y - c[1] + 5) / 9) ** 2 + ((z - c[2]) / 5) ** 2 < 1)
    obj -= 0.5 * (((x - c[0] + 10) / 6) ** 2 + ((y - c[1] - 6) / 6) ** 2 + ((z - c[2]) / 4) ** 2 < 1)
    img = gaussian_filter(obj, 0.8).astype(np.complex128)           # [x,y,z]
    kr = k_radius_norm((nx, ny, nz))
    t_true = tfun(np.clip(2 * kr, 0, raw.max()))                      # code 0.5 = raw 1.0
    truth = np.abs(np.transpose(img, (2, 0, 1))); mask = truth > 0.5

    def blur(f_map_xyz):
        ks = sp.ifft(img, axes=(-3, -2, -1)) * np.exp(-2j * np.pi * f_map_xyz * t_true)
        return np.transpose(sp.fft(ks, axes=(-3, -2, -1)), (2, 0, 1))

    # 1. Regression: unwrap='flip' must reproduce crf.ring_b0_corr's image
    b = blur(np.full((nx, ny, nz), 60.0))
    with contextlib.redirect_stdout(io.StringIO()):
        c_ref, fm_ref = crf.ring_b0_corr(b, 8, sample_kradius=raw / 2, sample_times=t_p)
    c_new, fm_new, info = ring_b0_corr_v2(b, 8, raw, t_p, kr_edge=1.0)
    rel = np.linalg.norm(c_new - c_ref) / np.linalg.norm(c_ref)
    print(f"[regression] corr_image v2 vs ring_b0_corr (flip): rel diff {rel:.2e}  "
          f"{'OK' if rel < 1e-4 else 'MISMATCH'}")
    print(f"[timing] ring times (ms): {np.round(info['ring_times_s'] * 1e3, 2).tolist()}")

    # 2. Calibration on a uniform 60 Hz field
    print("[calibration] uniform +60 Hz:")
    for k, v in info["field_maps"].items():
        print(f"    {k:14s} median in object {np.median(v[mask]):+7.2f} Hz  (ratio {np.median(v[mask]) / 60:+.3f})")
    print(f"    ring_b0_corr (raw/2 times)    {np.median(fm_ref[mask]):+7.2f} Hz  (ratio {np.median(fm_ref[mask]) / 60:+.3f})")
    # exact-MATLAB wrapper: image must equal the kernel-9 cascade; report its literal field map
    c_m, fm_m, info_m = ring_b0_corr_matlab(b, 8, raw, t_p, kernelsize=9)
    c_k9, _, _ = ring_b0_corr_v2(b, 8, raw, t_p, matlab_kernelsize=9)
    print(f"[matlab wrapper] image == kernel-9 cascade: rel diff {np.linalg.norm(c_m - c_k9) / np.linalg.norm(c_k9):.1e}; "
          f"ts_b0_corr.m literal field map median {np.median(fm_m[mask]):+.2f} Hz (ratio {np.median(fm_m[mask]) / 60:+.3f}); "
          f"mean shell times ms {np.round(info_m['ring_times_s'] * 1e3, 2).tolist()}")

    # 3. Calibration on a smooth spatially varying field (linear + bump), RMSE in object
    fx = 40.0 * (x - c[0]) / nx + 30.0 * np.exp(-(((x - c[0] - 6) / 9) ** 2 + ((y - c[1]) / 9) ** 2 + ((z - c[2]) / 5) ** 2))
    fx = fx.astype(np.float64)
    b2 = blur(fx); f_zxy = np.transpose(fx, (2, 0, 1))
    _, fm2, info2 = ring_b0_corr_v2(b2, 8, raw, t_p)
    for k, v in info2["field_maps"].items():
        err = v[mask] - f_zxy[mask]
        print(f"[spatial field] {k:14s} RMSE {np.sqrt(np.mean(err ** 2)):6.2f} Hz, bias {np.mean(err):+6.2f} Hz "
              f"(field range {f_zxy[mask].min():+.0f}..{f_zxy[mask].max():+.0f} Hz)")

    # 4. Aliasing sweep: flip vs predict, N=8
    print("[aliasing sweep] NRMSE of |corr|/gain vs truth, N=8:")
    print(f"    {'f Hz':>6s} {'blurred':>8s} {'flip':>8s} {'predict':>8s} {'fmap_flip':>10s} {'fmap_pred':>10s}")
    for f in (60, 110, 130, 160, 200, 250):
        bf = blur(np.full((nx, ny, nz), float(f)))
        nb = np.linalg.norm(np.abs(bf) - truth) / np.linalg.norm(truth)
        row = []
        for uw in ("flip", "predict"):
            cc, fmm, _ = ring_b0_corr_v2(bf, 8, raw, t_p, unwrap=uw)
            g = np.median(np.abs(cc)[mask] / truth[mask])
            row.append((np.linalg.norm(np.abs(cc) / g - truth) / np.linalg.norm(truth), np.median(fmm[mask])))
        print(f"    {f:6d} {nb:8.3f} {row[0][0]:8.3f} {row[1][0]:8.3f} {row[0][1]:+10.1f} {row[1][1]:+10.1f}")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
