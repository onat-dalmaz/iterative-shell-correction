"""Pixel-level shell/phase diagnostics — the SHARED backend behind every phase panel.

Purpose (reviewer priority 2, Brian's question "linear phase, or a bit more?"):
extract, at a handful of named voxels, everything needed to decide whether the
per-shell phase of the ring cascade really is linear in the shell's acquisition
time — and to do it with the SAME extraction code on real data and on
simulations, so a panel from a phantom and a panel from the breast are
comparable line by line.

No plotting lives here (a renderer is not included yet); this module only
measures.

--------------------------------------------------------------------------
THE ONE DISTINCTION THAT MATTERS
--------------------------------------------------------------------------
Two different "shell phases" are floating around in this project and mixing
them manufactures apparent non-linearity in the display itself:

  FIXED reference   phi_fixed[r] = angle(S_r) - angle(S_0)
                    S_r = the Hamming-shell-r-filtered image at the voxel.
                    Every shell is measured against ONE reference (shell 0).
                    This is the quantity a linear-phase model predicts:
                    phi_fixed[r] ~= -2 pi f (t_r - t_0)  (mod pi, see below).

  EVOLVING reference  phi_casc[r] = angle(S_r * C_{r-1})
                    C_{r-1} = the correction phasor the cascade has ACCUMULATED
                    after shells 0..r-1.  This is what ring_b0_corr_v2 actually
                    looks at when it forms `image_copy`.  It is an INCREMENT
                    (shell r relative to the corrected state at shell r-1), not
                    a cumulative phase.  Plotting it on the same axis as
                    phi_fixed, or fitting a line through it, is a category
                    error: a perfectly linear voxel gives phi_fixed rising
                    steadily and phi_casc hovering near a small constant.

Both are returned, under those names, and the panel labels them.

--------------------------------------------------------------------------
THE SIGN / BRANCH AMBIGUITY
--------------------------------------------------------------------------
A band-pass-filtered image of a REAL object is real up to the off-resonance
phase, so at each shell the measured phase is

        phi[r] = -2 pi f (t_r - t_0) + s_r * pi   (mod 2 pi),  s_r in {0, 1}

i.e. about half the voxels sit near +-pi on any given shell.  Brian's +-pi/2
flip is exactly the rectification of s_r.  We therefore fit MODULO pi and
return the whole cost-vs-frequency curve, so the competing branches are
visible instead of silently resolved.  A plain (non-sign-aware) fit is
returned alongside, to show what the ambiguity does to a naive line fit.

--------------------------------------------------------------------------
WEAK RESPONSE
--------------------------------------------------------------------------
Near a zero of the shell's band-pass response |S_r| collapses and its phase is
noise.  Such a shell must not be read, or weighted, like a strong one.  Every
shell carries `rel_response_to_image` (|S_r| / local image level), `shell_snr`
(|S_r| / per-shell noise level from a noise mask), `notch_resid_log` (|S_r|
against the voxel's own smooth spectral trend) and a boolean `weak` with the
three reasons split out (`weak_noise`, `weak_tiny`, `weak_notch`).
Weak shells get zero weight in every fit and are meant to be drawn hollow.

--------------------------------------------------------------------------
ACQUISITION-TIME SUPPORT
--------------------------------------------------------------------------
`ring_times_s()` gives each shell ONE nominal time (a Hamming-weighted mean of
the radial (kr, t) profile).  The shell is not a single time: it is a band of
k-radii, and on a real 3D cones trajectory t(k) inside one shell spans a wide
range.  We return the nominal time AND the Hamming-weighted quantiles
(min / p10 / p50 / p90 / max) of the REAL 3D t(k) inside the shell.  The gap
between them is a first-order reason a shell phase need not be a clean sample
of a linear phase ramp.

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    from shell_diag import extract_shell_diagnostics, save_diag_npz

    diag = extract_shell_diagnostics(vol_zxy, tmap_zxy,
                                     locations=[(42, 325, 708), ...],
                                     names=["skin_trouble", ...],
                                     num_rings=8, nbhd=(1, 3, 3),
                                     profile_npz="data/tkr_profile_brian.npz",
                                     noise_mask_zxy=~breast_mask,
                                     extra_freq_maps={"fblend": path_or_array})
    save_diag_npz("cache.npz", diag)

Self-test (synthetic phantom with a KNOWN uniform off-resonance, so the fit has
a right answer):   .venv/bin/python cones_offres_lean/shell_diag.py --selftest
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy import ndimage

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from ring_b0_corr import (_fft3, _ifft3, _smooth, k_radius_norm,  # noqa: E402
                          matlab_fspecial3_gaussian, ring_bounds, ring_times_s)
from autofocus import kr_norm_zxy  # noqa: E402

__all__ = ["extract_shell_diagnostics", "save_diag_npz", "load_diag_npz",
           "wrap_pi", "wrap_half_pi", "fit_signed_linear", "fit_signed_quadratic",
           "fit_signed_support", "shell_phasor_model"]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def wrap_pi(a):
    """wrap into (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def wrap_half_pi(a):
    """wrap into (-pi/2, pi/2] — the modulo-pi residual of the 0/pi branch."""
    return (np.asarray(a) + np.pi / 2) % np.pi - np.pi / 2


def _mirror_centered(a: np.ndarray) -> np.ndarray:
    """Index negation on a centered (fftshifted) grid.

    With this module's centered/ortho convention, _ifft3(x)[m] == _fft3(x)[-m].
    So a map indexed against `_fft3(image)` (the convention `autofocus.py` and
    `data/kxyz_time_768x768x112.npy` use) must be mirrored before it is indexed
    against `_ifft3(image)` (the convention `ring_b0_corr.py` uses)."""
    return np.ascontiguousarray(np.roll(a[::-1, ::-1, ::-1], 1, axis=(0, 1, 2)))


def _circ_std(ang, axis=None):
    """Circular standard deviation (radians) of a set of angles."""
    z = np.mean(np.exp(1j * np.asarray(ang)), axis=axis)
    r = np.clip(np.abs(z), 1e-12, 1.0)
    return float(np.sqrt(-2.0 * np.log(r))) if np.isscalar(r) or r.ndim == 0 else np.sqrt(-2.0 * np.log(r))


def _block_slices(shape_zxy, loc, nbhd):
    """Slices of a (dz, dx, dy) block centered on `loc`, clipped to the volume."""
    sl = []
    for n, c, d in zip(shape_zxy, loc, nbhd):
        h = d // 2
        a = int(np.clip(c - h, 0, n - 1))
        b = int(np.clip(c + h + 1, 1, n))
        sl.append(slice(a, b))
    return tuple(sl)


def _weighted_quantiles(vals, wts, qs=(0.0, 0.1, 0.5, 0.9, 1.0), max_n=4_000_000, seed=0):
    """Weighted quantiles of `vals`; subsamples uniformly above `max_n` (the
    subsample is uniform so it does not bias the weighted distribution)."""
    vals = np.asarray(vals, dtype=np.float64).ravel()
    wts = np.asarray(wts, dtype=np.float64).ravel()
    n_all = vals.size
    sub = False
    if n_all > max_n:
        rng = np.random.default_rng(seed)
        sel = rng.choice(n_all, size=max_n, replace=False)
        vals = vals[sel]
        wts = wts[sel]
        sub = True
    o = np.argsort(vals, kind="stable")
    v = vals[o]
    cw = np.cumsum(wts[o])
    if cw[-1] <= 0:
        return np.full(len(qs), np.nan), n_all, sub
    cw = cw / cw[-1]
    out = np.interp(qs, cw, v)
    out[0] = v[0]
    out[-1] = v[-1]
    return out.astype(np.float64), n_all, sub


# ---------------------------------------------------------------------------
# sign-aware phase fits
# ---------------------------------------------------------------------------
def fit_signed_linear(phi, dt, w, f_grid=None):
    """Sign-aware LS of phi[r] ~ -2 pi f dt[r]  MODULO pi.

    `phi` are the FIXED-reference phases (rad), `dt` the shell times relative to
    shell 0 (s), `w` per-shell weights (0 for weak shells).  Returns the whole
    cost curve so the competing branches stay visible.

    cost(f) = sum_r w_r * wrap_half_pi(phi_r + 2 pi f dt_r)^2 / sum_r w_r
    """
    phi = np.asarray(phi, float); dt = np.asarray(dt, float); w = np.asarray(w, float)
    if f_grid is None:
        f_grid = np.arange(-400.0, 400.01, 0.5)
    f_grid = np.asarray(f_grid, float)
    ok = np.isfinite(phi) & np.isfinite(dt) & (w > 0)
    wsum = float(w[ok].sum())
    if not ok.any() or wsum <= 0:
        nan = np.full(f_grid.shape, np.nan)
        return {"f_grid_hz": f_grid, "cost": nan, "cost_naive": nan, "best_f_hz": np.nan,
                "best_cost": np.nan, "branch": np.full(phi.shape, -1, np.int8),
                "resid_rad": np.full(phi.shape, np.nan), "model_rad": np.full(phi.shape, np.nan),
                "branch_resolved_rad": np.full(phi.shape, np.nan),
                "local_minima_hz": np.zeros(0), "local_minima_cost": np.zeros(0),
                "second_best_f_hz": np.nan, "second_best_cost": np.nan, "n_used": 0}
    res = phi[None, ok] + 2 * np.pi * f_grid[:, None] * dt[None, ok]
    cost = (w[ok][None, :] * wrap_half_pi(res) ** 2).sum(1) / wsum          # modulo-pi (sign-aware)
    cost_naive = (w[ok][None, :] * wrap_pi(res) ** 2).sum(1) / wsum          # plain, for contrast
    i = int(np.argmin(cost))
    f_best = float(f_grid[i])

    # all local minima of the sign-aware cost, with their depths
    c = cost
    lm = np.flatnonzero((c[1:-1] <= c[:-2]) & (c[1:-1] < c[2:])) + 1
    order = np.argsort(c[lm]) if lm.size else np.zeros(0, int)
    lm = lm[order]
    # second-best DISTINCT branch (>= 10 Hz from the best)
    f2, c2 = np.nan, np.nan
    for j in lm:
        if abs(f_grid[j] - f_best) >= 10.0:
            f2, c2 = float(f_grid[j]), float(c[j]); break

    model = -2 * np.pi * f_best * dt                                        # the fitted line
    resid = wrap_half_pi(phi - model)                                       # modulo-pi residual
    resid = np.where(np.isfinite(phi), resid, np.nan)
    branch = np.where(np.isfinite(phi), np.rint((wrap_pi(phi - model) - resid) / np.pi).astype(float) % 2, -1)
    return {"f_grid_hz": f_grid, "cost": cost, "cost_naive": cost_naive,
            "best_f_hz": f_best, "best_cost": float(c[i]),
            "branch": branch.astype(np.int8), "resid_rad": resid, "model_rad": model,
            "branch_resolved_rad": model + resid,
            "local_minima_hz": f_grid[lm].astype(float), "local_minima_cost": c[lm].astype(float),
            "second_best_f_hz": f2, "second_best_cost": c2, "n_used": int(ok.sum())}


def shell_phasor_model(f, t_bins, w_bins):
    """Expected complex shell response at off-resonance f, USING the shell's
    real acquisition-time distribution:  E_r(f) = sum_b w_b exp(-i 2 pi f t_b).

    A shell is a band of k-radii, and on a real 3D cones trajectory t(k) inside
    one shell spans ~1-2 ms.  The shell therefore does NOT carry the single
    phase -2 pi f t_bar_r: it carries the phase of a sum of phasors, which is
    both shifted away from -2 pi f t_bar_r and attenuated.  |E_r| is that
    attenuation, and it is the honest first explanation for a shell phase that
    looks 'a bit more than linear'."""
    t_bins = np.asarray(t_bins, float); w_bins = np.asarray(w_bins, float)
    ph = np.exp(-2j * np.pi * np.asarray(f, float)[..., None, None] * t_bins)
    return (w_bins * ph).sum(-1) / np.maximum(w_bins.sum(-1), 1e-30)


def fit_signed_support(phi, t_bins, w_bins, w, f_grid=None):
    """Sign-aware fit of phi[r] against the TIME-SUPPORT model (above), modulo pi.

    Differs from `fit_signed_linear` only in the model: the predicted phase is
    angle(E_r(f)) - angle(E_0(f)) instead of -2 pi f (t_r - t_0)."""
    phi = np.asarray(phi, float); w = np.asarray(w, float)
    if f_grid is None:
        f_grid = np.arange(-400.0, 400.01, 0.5)
    f_grid = np.asarray(f_grid, float)
    ok = np.isfinite(phi) & (w > 0)
    wsum = float(w[ok].sum())
    if not ok.any() or wsum <= 0 or not np.isfinite(t_bins).all():
        nan = np.full(f_grid.shape, np.nan)
        return {"f_grid_hz": f_grid, "cost": nan, "best_f_hz": np.nan, "best_cost": np.nan,
                "model_rad": np.full(phi.shape, np.nan), "resid_rad": np.full(phi.shape, np.nan),
                "attenuation": np.full(phi.shape, np.nan), "second_best_f_hz": np.nan, "n_used": 0}
    E = shell_phasor_model(f_grid, t_bins, w_bins)                     # (F, R)
    pred = np.angle(E) - np.angle(E[:, :1])
    cost = (w[None, ok] * wrap_half_pi(phi[None, ok] - pred[:, ok]) ** 2).sum(1) / wsum
    i = int(np.argmin(cost))
    f_best = float(f_grid[i])
    c = cost
    lm = np.flatnonzero((c[1:-1] <= c[:-2]) & (c[1:-1] < c[2:])) + 1
    lm = lm[np.argsort(c[lm])] if lm.size else np.zeros(0, int)
    f2 = np.nan
    for j in lm:
        if abs(f_grid[j] - f_best) >= 10.0:
            f2 = float(f_grid[j]); break
    Eb = shell_phasor_model(np.array([f_best]), t_bins, w_bins)[0]
    model = np.angle(Eb) - np.angle(Eb[0])
    return {"f_grid_hz": f_grid, "cost": cost, "best_f_hz": f_best, "best_cost": float(c[i]),
            "model_rad": model, "resid_rad": wrap_half_pi(phi - model),
            "attenuation": np.abs(Eb) / max(abs(Eb[0]), 1e-30),
            "local_minima_hz": f_grid[lm].astype(float), "second_best_f_hz": f2,
            "n_used": int(ok.sum())}


def fit_signed_quadratic(phi, dt, w, f_grid=None, c_grid=None):
    """Same, with an extra quadratic term: phi ~ -2 pi (f dt + c dt^2), modulo pi.

    `c` is in Hz/s; the extra phase it contributes at the last shell is
    -2 pi c dt_max^2 rad, which is what `extra_phase_last_shell_rad` reports.
    Answers "linear, or a bit more?" quantitatively: if the best |c| buys only a
    small cost reduction the data do not ask for more than a line."""
    phi = np.asarray(phi, float); dt = np.asarray(dt, float); w = np.asarray(w, float)
    if f_grid is None:
        f_grid = np.arange(-400.0, 400.01, 2.0)
    if c_grid is None:
        c_grid = np.linspace(-20000.0, 20000.0, 161)
    f_grid = np.asarray(f_grid, float); c_grid = np.asarray(c_grid, float)
    ok = np.isfinite(phi) & np.isfinite(dt) & (w > 0)
    if ok.sum() < 3:
        return {"best_f_hz": np.nan, "best_c_hz_per_s": np.nan, "best_cost": np.nan,
                "extra_phase_last_shell_rad": np.nan, "n_used": int(ok.sum())}
    wsum = float(w[ok].sum())
    base = phi[None, None, ok] + 2 * np.pi * (f_grid[:, None, None] * dt[None, None, ok]
                                              + c_grid[None, :, None] * dt[None, None, ok] ** 2)
    cost = (w[ok][None, None, :] * wrap_half_pi(base) ** 2).sum(-1) / wsum
    i, j = np.unravel_index(int(np.argmin(cost)), cost.shape)
    dtmax = float(np.nanmax(np.abs(dt[ok])))
    return {"best_f_hz": float(f_grid[i]), "best_c_hz_per_s": float(c_grid[j]),
            "best_cost": float(cost[i, j]),
            "extra_phase_last_shell_rad": float(-2 * np.pi * c_grid[j] * dtmax ** 2),
            "n_used": int(ok.sum())}


# ---------------------------------------------------------------------------
# the extractor
# ---------------------------------------------------------------------------
def extract_shell_diagnostics(
    vol_zxy,
    tmap_zxy,
    locations,
    names=None,
    num_rings: int = 8,
    nbhd=(1, 3, 3),
    smooth_sigma: float = 4.5,
    matlab_kernelsize: int | None = None,
    profile_npz: str | None = None,
    kr_edge: float = 1.0,
    noise_mask_zxy=None,
    weak_rel_image: float = 0.01,
    notch_ratio: float = 0.4,
    weak_snr: float = 3.0,
    run_autofocus: bool = True,
    af_freqs_hz=None,
    af_window=(3, 5, 5),
    af_idx_smooth_sigma: float = 5.0,
    af_phase_ref_kr: float = 0.1,
    extra_freq_maps: dict | None = None,
    f_grid_hz=None,
    verbose: bool = False,
):
    """Extract shell + autofocus diagnostics at a few voxels.

    Parameters
    ----------
    vol_zxy   : complex volume [z, x, y] (the UNCORRECTED image).
    tmap_zxy  : real 3D acquisition-time map t(k) in SECONDS, [z, x, y], indexed
                against `_fft3(image)` (the convention of
                data/kxyz_time_768x768x112.npy and autofocus.py).  Mirrored
                internally where the ring cascade's `_ifft3` k-space needs it.
    locations : list of (z, x, y) integer voxel indices.
    names     : optional list of labels, same length.
    num_rings : Hamming shells, Brian's baseline 8.
    nbhd      : (dz, dx, dy) odd block extracted around each location.
    smooth_sigma / matlab_kernelsize : the cascade's per-shell phase smoothing;
                defaults reproduce the deployed Python ring (scipy sigma 4.5).
    profile_npz : radial (kr, t) profile for the NOMINAL `ring_times_s`.
    noise_mask_zxy : bool volume marking signal-free voxels, for the per-shell
                noise level.  If None, the 2nd percentile of |vol| is used and
                `meta['noise_mask_source']` says so.
    weak_snr / weak_rel_image / notch_ratio : a shell is flagged weak when its
                SNR against the noise floor is below weak_snr, or when |S_r| is
                below weak_rel_image times the LOCAL image level, or when |S_r|
                dips below notch_ratio times the voxel's own smooth spectral
                trend across shells (a band-pass response ZERO).  Reported
                separately as weak_noise / weak_tiny / weak_notch.
    run_autofocus : also sweep the autofocus criterion (the expensive part).
    extra_freq_maps : {label: array_or_path} of [z, x, y] Hz maps (the DEPLOYED
                smoothed / coarse-to-fine / field-blend selections) to read at
                each location.

    Returns
    -------
    dict with keys 'meta', 'shells', 'locations' (list of per-location dicts).
    """
    vol = np.asarray(vol_zxy)
    if vol.ndim != 3:
        raise ValueError("vol_zxy must be 3D [z, x, y]")
    nz, nx, ny = vol.shape
    tmap = np.asarray(tmap_zxy, dtype=np.float32)
    if tmap.shape != vol.shape:
        raise ValueError(f"tmap {tmap.shape} != vol {vol.shape}")
    locations = [tuple(int(v) for v in L) for L in locations]
    if names is None:
        names = [f"loc{i}" for i in range(len(locations))]
    if len(names) != len(locations):
        raise ValueError("names and locations must have the same length")
    nbhd = tuple(int(v) for v in nbhd)
    blocks = [_block_slices(vol.shape, L, nbhd) for L in locations]
    # index of the centre voxel inside its block
    centres = [tuple(int(L[a] - blocks[i][a].start) for a in range(3)) for i, L in enumerate(locations)]
    nloc = len(locations)

    mag_vol = np.abs(vol).astype(np.float32)
    # --- local signal level and noise level -------------------------------
    # local signal level = mean |vol| over a (3, 7, 7) box centred on the voxel
    local_level = np.array([float(np.mean(
        mag_vol[max(L[0] - 1, 0):L[0] + 2, max(L[1] - 3, 0):L[1] + 4, max(L[2] - 3, 0):L[2] + 4]))
        for L in locations], np.float64)
    mag_centre = np.array([float(mag_vol[L]) for L in locations], np.float64)
    if noise_mask_zxy is None:
        thr = float(np.percentile(mag_vol, 2.0))
        noise_mask = mag_vol <= thr
        noise_src = f"auto: |vol| <= 2nd percentile ({thr:.4g})"
    else:
        noise_mask = np.asarray(noise_mask_zxy, bool)
        noise_src = "caller-supplied noise mask"
    nidx = np.flatnonzero(noise_mask.ravel())
    if nidx.size > 400_000:
        nidx = np.random.default_rng(0).choice(nidx, 400_000, replace=False)
    del noise_mask

    # ======================================================================
    # 1. the ring cascade, faithful to ring_b0_corr_v2(unwrap='flip')
    # ======================================================================
    kern = matlab_fspecial3_gaussian(matlab_kernelsize) if matlab_kernelsize else None
    img_xyz = np.ascontiguousarray(np.transpose(vol, (1, 2, 0))).astype(np.complex64)
    ks = _ifft3(img_xyz).astype(np.complex64)
    del img_xyz
    kr = k_radius_norm((nx, ny, nz))                       # ring units: inscribed sphere = 0.5
    start, end, width = ring_bounds(num_rings)
    # time map in the ring's k-index convention (mirror; see _mirror_centered)
    t_ring_k = np.ascontiguousarray(np.transpose(_mirror_centered(tmap), (1, 2, 0)))
    t_phys_k = np.ascontiguousarray(np.transpose(tmap, (1, 2, 0)))

    nominal_t = np.full(num_rings, np.nan)
    if profile_npz is not None:
        p = np.load(profile_npz)
        nominal_t = ring_times_s(p["kr"], p["t_s"], num_rings, kr_edge=kr_edge, weighting="hamming")

    R = num_rings
    NB_T = 48                                       # bins of the per-shell t(k) histogram
    t_quant = np.full((R, 5), np.nan)               # min p10 p50 p90 max of REAL t(k) in the shell
    t_quant_phys = np.full((R, 5), np.nan)          # same, unmirrored (mirroring sanity check)
    t_hmean = np.full(R, np.nan)                    # Hamming-weighted mean of the real t(k)
    t_bins = np.full((R, NB_T), np.nan)             # bin centres of the Hamming-weighted t(k) hist
    t_bw = np.zeros((R, NB_T))                      # bin weights
    shell_nk = np.zeros(R, np.int64)
    shell_kr = np.zeros((R, 2))
    shell_noise = np.full(R, np.nan)

    _bshape = [tuple(sl.stop - sl.start for sl in b) for b in blocks]
    sm0_b = [np.full((R,) + bs, np.nan, np.float32) for bs in _bshape]      # angle(smooth(S_r C_0))
    sm0f_b = [np.full((R,) + bs, np.nan, np.float32) for bs in _bshape]     # same, after the +-pi/2 flip
    coh0_b = [np.full((R,) + bs, np.nan, np.float32) for bs in _bshape]     # coherence of the unflipped smooth
    coh0f_b = [np.full((R,) + bs, np.nan, np.float32) for bs in _bshape]    # coherence of the flipped smooth
    S = [np.zeros((R,) + tuple(s.stop - s.start for s in b), np.complex64) for b in blocks]   # raw shell values
    Ccum = [np.zeros((R,) + tuple(s.stop - s.start for s in b), np.complex64) for b in blocks]  # corr_phase AFTER shell r
    resid_c = [np.full((R,) + tuple(s.stop - s.start for s in b), np.nan, np.float32) for b in blocks]  # angle(S_r*C_{r-1})
    inc_b = [np.full((R,) + tuple(s.stop - s.start for s in b), np.nan, np.float32) for b in blocks]
    flip_b = [np.zeros((R,) + tuple(s.stop - s.start for s in b), np.int8) for b in blocks]
    coh_b = [np.full((R,) + tuple(s.stop - s.start for s in b), np.nan, np.float32) for b in blocks]
    corr_img_b = [np.zeros(tuple(s.stop - s.start for s in b), np.complex64) for b in blocks]

    corr_phase = np.ones((nx, ny, nz), np.complex64)
    flip_frac = np.zeros(R)

    def _xyz_block(i):
        bz, bx, by = blocks[i]
        return (bx, by, bz)

    def _grab(arr, i):
        """arr is [x, y, z]; return the block transposed back to [z, x, y]."""
        bx, by, bz = _xyz_block(i)
        return np.ascontiguousarray(np.transpose(arr[bx, by, bz], (2, 0, 1)))

    for r in range(R):
        if verbose:
            print(f"  shell {r + 1}/{R}", flush=True)
        m = (kr >= start[r]) & (kr < end[r])
        filt = np.zeros((nx, ny, nz), np.float32)
        filt[m] = 0.54 - 0.46 * np.cos(np.pi * (kr[m] - start[r]) / width)
        shell_nk[r] = int(m.sum())
        shell_kr[r] = (float(start[r]), float(end[r]))
        if shell_nk[r] > 0:
            t_quant[r], _, _ = _weighted_quantiles(t_ring_k[m], filt[m], seed=r)
            t_quant_phys[r], _, _ = _weighted_quantiles(t_phys_k[m], filt[m], seed=r)
            wm = filt[m].astype(np.float64)
            tm = t_ring_k[m].astype(np.float64)
            t_hmean[r] = float(np.sum(wm * tm) / np.sum(wm))
            lo, hi = float(tm.min()), float(tm.max())
            if hi <= lo:
                hi = lo + 1e-9
            h, e = np.histogram(tm, bins=NB_T, range=(lo, hi), weights=wm)
            t_bins[r] = 0.5 * (e[:-1] + e[1:])
            t_bw[r] = h
            del wm, tm
        del m

        ring_image = _fft3(ks * filt).astype(np.complex64)
        del filt
        # per-shell noise level from the noise mask (on the [z,x,y] view)
        ri_zxy_flat = np.transpose(ring_image, (2, 0, 1)).ravel()
        shell_noise[r] = float(np.median(np.abs(ri_zxy_flat[nidx]))) if nidx.size else np.nan
        del ri_zxy_flat
        for i in range(nloc):
            S[i][r] = _grab(ring_image, i)

        if r == 0:
            corr_phase = np.exp(-1j * np.angle(ring_image)).astype(np.complex64)
            C0 = corr_phase.copy()
            for i in range(nloc):
                Ccum[i][0] = _grab(corr_phase, i)
                corr_img_b[i] += S[i][0] * Ccum[i][0]
                resid_c[i][0] = np.angle(S[i][0] * Ccum[i][0])     # == 0 by construction
                inc_b[i][0] = np.nan
                sm0_b[i][0] = 0.0; sm0f_b[i][0] = 0.0
                coh0_b[i][0] = 1.0; coh0f_b[i][0] = 1.0
            del ring_image
            continue

        # ---- FIXED-reference smoothed phase: the same sigma-4.5 spatial
        # average the cascade uses, but always referenced to shell 0 instead of
        # to the evolving cascade.  This is the quantity that separates the
        # per-voxel structural scatter of a shell phase from the underlying
        # field: raw phases scatter, their local average need not.
        fx = (ring_image * C0).astype(np.complex64)
        magf = _smooth(np.abs(fx), smooth_sigma, kern)
        smf = _smooth(fx, smooth_sigma, kern)
        ang_ = np.angle(smf).astype(np.float32)
        coh_ = (np.abs(smf) / (magf + np.float32(1e-12))).astype(np.float32)
        for i in range(nloc):
            sm0_b[i][r] = _grab(ang_, i); coh0_b[i][r] = _grab(coh_, i)
        del smf, ang_, coh_
        fl = np.abs(np.angle(fx)) > (np.pi / 2)
        fx[fl] = -fx[fl]
        del fl
        smf = _smooth(fx, smooth_sigma, kern)
        ang_ = np.angle(smf).astype(np.float32)
        coh_ = (np.abs(smf) / (magf + np.float32(1e-12))).astype(np.float32)
        for i in range(nloc):
            sm0f_b[i][r] = _grab(ang_, i); coh0f_b[i][r] = _grab(coh_, i)
        del smf, ang_, coh_, magf, fx

        image_copy = (ring_image * corr_phase).astype(np.complex64)
        for i in range(nloc):
            resid_c[i][r] = _grab(np.angle(image_copy).astype(np.float32), i)
        flip = np.abs(np.angle(image_copy)) > (np.pi / 2)
        flip_frac[r] = float(flip.mean())
        for i in range(nloc):
            flip_b[i][r] = _grab(flip.astype(np.int8), i)
        image_copy[flip] = -image_copy[flip]
        del flip
        sm = _smooth(image_copy, smooth_sigma, kern)
        mag_s = _smooth(np.abs(image_copy), smooth_sigma, kern)
        del image_copy
        inc = np.angle(sm).astype(np.float32)
        coh = (np.abs(sm) / (mag_s + np.float32(1e-12))).astype(np.float32)
        np.clip(coh, 0.0, 1.0, out=coh)
        del sm, mag_s
        for i in range(nloc):
            inc_b[i][r] = _grab(inc, i)
            coh_b[i][r] = _grab(coh, i)
        del coh
        corr_phase = (corr_phase * np.exp(-1j * inc)).astype(np.complex64)
        del inc
        for i in range(nloc):
            Ccum[i][r] = _grab(corr_phase, i)
            corr_img_b[i] += S[i][r] * Ccum[i][r]
        del ring_image
    del ks, corr_phase, C0, kr, t_ring_k, t_phys_k

    # ======================================================================
    # 2. autofocus criterion sweep (same code path as autofocus_3d)
    # ======================================================================
    af = {"ran": bool(run_autofocus)}
    if run_autofocus:
        if af_freqs_hz is None:
            af_freqs_hz = np.arange(-300.0, 300.01, 20.0)
        af_freqs_hz = np.asarray(af_freqs_hz, float)
        K = _fft3(np.asarray(vol).astype(np.complex64)).astype(np.complex64)
        krz = kr_norm_zxy(K.shape)
        low = krz <= af_phase_ref_kr
        del krz
        w_low = np.abs(_ifft3(np.where(low, K, 0))).astype(np.float32)
        two_pi_t = (2 * np.pi * tmap).astype(np.float32)
        nF = len(af_freqs_hz)
        crit = [np.zeros((nF,) + tuple(s.stop - s.start for s in b), np.float32) for b in blocks]
        best_c = None; worst_c = None; best_idx = np.zeros(K.shape, np.int16)
        for i, f in enumerate(af_freqs_hz):
            if verbose:
                print(f"  autofocus {i + 1}/{nF} ({f:+.0f} Hz)", flush=True)
            Kf = (K * np.exp(1j * (two_pi_t * np.float32(f))).astype(np.complex64)).astype(np.complex64)
            mm = _ifft3(Kf).astype(np.complex64)
            mm *= np.exp(-1j * np.angle(_ifft3(np.where(low, Kf, 0)))).astype(np.complex64)
            del Kf
            c = ndimage.uniform_filter(np.abs(mm.imag), size=af_window, mode="nearest").astype(np.float32)
            del mm
            for j, b in enumerate(blocks):
                crit[j][i] = c[b]
            if best_c is None:
                best_c = c.copy(); worst_c = c
            else:
                better = c < best_c
                best_c[better] = c[better]; best_idx[better] = i
                np.maximum(worst_c, c, out=worst_c)
                del better, c
        del K, two_pi_t, low
        focus_w = ((worst_c - best_c) * w_low).astype(np.float32)
        del best_c, worst_c, w_low
        idx_s = best_idx.astype(np.float32)
        if af_idx_smooth_sigma and af_idx_smooth_sigma > 0:
            num = ndimage.gaussian_filter(idx_s * focus_w, sigma=af_idx_smooth_sigma, mode="nearest")
            den = ndimage.gaussian_filter(focus_w, sigma=af_idx_smooth_sigma, mode="nearest")
            idx_s = num / (den + np.float32(1e-6) * float(focus_w.max()))
            del num, den
        idx_s = np.clip(np.round(idx_s), 0, nF - 1).astype(np.int16)
        af_raw = af_freqs_hz[best_idx.astype(np.intp)].astype(np.float32)
        af_sm = af_freqs_hz[idx_s.astype(np.intp)].astype(np.float32)
        af["freqs_hz"] = af_freqs_hz
        af["raw_blocks"] = [af_raw[b] for b in blocks]
        af["smoothed_blocks"] = [af_sm[b] for b in blocks]
        af["focus_w_blocks"] = [focus_w[b] for b in blocks]
        af["criterion_blocks"] = crit
        del af_raw, af_sm, focus_w, idx_s, best_idx

    # ======================================================================
    # 3. externally supplied frequency maps (deployed estimators)
    # ======================================================================
    ext = {}
    if extra_freq_maps:
        for lab, src in extra_freq_maps.items():
            arr = np.load(src, mmap_mode="r") if isinstance(src, str) else np.asarray(src)
            ext[lab] = [np.asarray(arr[b], np.float32) for b in blocks]
            del arr

    # ======================================================================
    # 4. assemble per-location records + fits
    # ======================================================================
    dt_shell = t_hmean - t_hmean[0]                      # REAL Hamming-weighted shell times
    dt_nom = nominal_t - nominal_t[0]
    out_locs = []
    for i in range(nloc):
        ci = centres[i]
        sc = S[i][(slice(None),) + ci]                                   # (R,) complex, centre voxel
        mag_r = np.abs(sc).astype(float)
        phase_abs = np.angle(sc).astype(float)
        phi_fixed = wrap_pi(phase_abs - phase_abs[0])                    # FIXED reference = shell 0
        phi_sm = wrap_pi(sm0_b[i][(slice(None),) + ci].astype(float))    # FIXED ref, sigma-smoothed
        phi_smf = wrap_pi(sm0f_b[i][(slice(None),) + ci].astype(float))  # FIXED ref, flip + smoothed
        coh_sm = coh0_b[i][(slice(None),) + ci].astype(float)
        coh_smf = coh0f_b[i][(slice(None),) + ci].astype(float)
        phi_casc = resid_c[i][(slice(None),) + ci].astype(float)         # EVOLVING reference
        inc_c = inc_b[i][(slice(None),) + ci].astype(float)
        acc_c = np.angle(Ccum[i][(slice(None),) + ci]).astype(float)
        coh_c = coh_b[i][(slice(None),) + ci].astype(float)
        flip_c = flip_b[i][(slice(None),) + ci].astype(int)

        # --- weak-response bookkeeping -----------------------------------
        # `rel` is the shell response relative to the STRONGEST shell at this
        # voxel: it says "this shell sits near a zero of the band-pass
        # response", independently of absolute scale.  `snr` is the decisive
        # one: phase noise is ~1/snr rad, so a low-snr shell's phase is not a
        # measurement.  Weak shells get zero weight and are drawn hollow.
        rel = mag_r / max(float(np.nanmax(mag_r)), 1e-30)
        rel_to_image = mag_r / max(local_level[i], 1e-30)
        snr = mag_r / np.where(np.isfinite(shell_noise) & (shell_noise > 0), shell_noise, np.nan)
        # Three independent reasons a shell's phase must not be read like a
        # strong one; reported separately so it is visible WHICH one fired.
        #  (a) NOISE   : |S_r| within a few sigma of the noise floor.
        #  (b) NOTCH   : |S_r| dips below the voxel's OWN smooth spectral trend
        #                (quadratic in shell index fitted to log|S_r|) — the
        #                band-pass response of the local structure passing
        #                through zero.  The trend subtraction matters: every
        #                natural image falls off steeply with |k|, and that
        #                fall-off alone is not a response zero.
        #  (c) TINY    : |S_r| is a negligible fraction of the local image
        #                level, so whatever phase it carries is not this
        #                voxel's structure.
        lg = np.log(np.maximum(mag_r, 1e-30))
        if R >= 4:
            notch_resid = lg - np.polyval(np.polyfit(np.arange(R), lg, 2), np.arange(R))
        else:
            notch_resid = np.zeros(R)
        weak_notch = notch_resid < np.log(notch_ratio)
        weak_noise = np.asarray(snr < weak_snr, bool) if np.all(np.isfinite(snr)) else np.zeros(R, bool)
        weak_tiny = rel_to_image < weak_rel_image
        weak = weak_noise | weak_notch | weak_tiny
        # inverse-variance weights: var(phase) ~ 1/snr^2 (clipped so one shell
        # cannot own the fit); falls back to |S_r|^2 with no noise model.
        base_w = np.clip(snr, 0, 50.0) ** 2 if np.all(np.isfinite(snr)) else rel ** 2
        w_fit = np.where(weak, 0.0, base_w)

        # phase spread over the neighbourhood (stability of the reading)
        nb_phi = wrap_pi(np.angle(S[i]) - np.angle(S[i][(slice(None),) + ci])[:, None, None, None])
        nb_spread = _circ_std(nb_phi.reshape(R, -1), axis=1)

        lin = fit_signed_linear(phi_fixed, dt_shell, w_fit, f_grid=f_grid_hz)
        quad = fit_signed_quadratic(phi_fixed, dt_shell, w_fit)
        lin_nom = fit_signed_linear(phi_fixed, dt_nom, w_fit, f_grid=f_grid_hz)
        sup = fit_signed_support(phi_fixed, t_bins, t_bw, w_fit, f_grid=f_grid_hz)
        COH_FLOOR = 0.10          # below this the smoothed phasor has cancelled; no information
        w_sm = np.where(np.isfinite(phi_sm) & (coh_sm > COH_FLOOR), np.clip(coh_sm, 0, 1) ** 2, 0.0)
        lin_sm = fit_signed_linear(phi_sm, dt_shell, w_sm, f_grid=f_grid_hz)
        w_smf = np.where(np.isfinite(phi_smf) & (coh_smf > COH_FLOOR), np.clip(coh_smf, 0, 1) ** 2, 0.0)
        lin_smf = fit_signed_linear(phi_smf, dt_shell, w_smf, f_grid=f_grid_hz)
        # The cascade's own cumulative phase estimate, directly comparable with
        # phi_fixed: sum of the increments it applied, unwrapped by construction.
        cum_rel = np.nancumsum(np.where(np.isfinite(inc_c), inc_c, 0.0))
        gd = np.isfinite(dt_shell) & np.isfinite(cum_rel)
        gd[0] = False
        f_casc_ls = (float(-np.sum(cum_rel[gd] * dt_shell[gd]) / (2 * np.pi * np.sum(dt_shell[gd] ** 2)))
                     if gd.any() else np.nan)
        fit_casc = {"best_f_hz": f_casc_ls,
                    "model_rad": -2 * np.pi * f_casc_ls * dt_shell,
                    "resid_rad": cum_rel + 2 * np.pi * f_casc_ls * dt_shell,
                    "n_used": int(gd.sum())}
        # the cascade's own implied field (ls_increment, at this voxel)
        dts = np.diff(t_hmean)
        good = np.isfinite(dts) & (dts > 0) & np.isfinite(inc_c[1:])
        f_casc = (float(-np.sum(inc_c[1:][good] * dts[good]) / (2 * np.pi * np.sum(dts[good] ** 2)))
                  if good.any() else np.nan)

        rec = {
            "name": names[i], "index_zxy": np.array(locations[i], int),
            "block_start_zxy": np.array([b.start for b in blocks[i]], int),
            "centre_in_block": np.array(ci, int),
            "nbhd": np.array(nbhd, int),
            "mag_centre": float(mag_centre[i]), "local_level": float(local_level[i]),
            # ---- RAW shell values, FIXED reference -----------------------
            "shell_complex": S[i],
            "shell_mag": mag_r,
            "phase_abs_fixed_rad": phase_abs,
            "phase_fixed_rel_rad": phi_fixed,
            "phase_fixed_rel_smoothed_rad": phi_sm,
            "phase_fixed_rel_smoothed_flip_rad": phi_smf,
            "coh_fixed_smoothed": coh_sm, "coh_fixed_smoothed_flip": coh_smf,
            "nbhd_phase_circstd_rad": nb_spread,
            # ---- EVOLVING (cascade) reference ----------------------------
            "phase_cascade_resid_rad": phi_casc,
            "cascade_inc_rad": inc_c,
            "cascade_acc_rad": acc_c,
            "cascade_cum_rel_rad": cum_rel,
            "cascade_flip": flip_c,
            "cascade_coherence": coh_c,
            "cascade_corr_value": corr_img_b[i][ci],
            "cascade_field_hz_ls_increment": f_casc,
            # ---- weak-response bookkeeping -------------------------------
            "rel_response": rel, "rel_response_to_image": rel_to_image,
            "shell_snr": snr, "weak": weak.astype(np.int8),
            "weak_noise": weak_noise.astype(np.int8), "weak_notch": weak_notch.astype(np.int8),
            "weak_tiny": weak_tiny.astype(np.int8), "notch_resid_log": notch_resid,
            "reference_shell_weak": int(weak[0]), "fit_weight": w_fit,
            # ---- fits ----------------------------------------------------
            "fit": lin, "fit_quadratic": quad, "fit_nominal_times": lin_nom,
            "fit_support": sup, "fit_smoothed": lin_sm, "fit_smoothed_flip": lin_smf,
            "fit_cascade": fit_casc,
        }
        if run_autofocus:
            cb = af["criterion_blocks"][i][(slice(None),) + ci].astype(float)
            rec["af_freqs_hz"] = af["freqs_hz"]
            rec["af_criterion"] = cb
            rec["af_criterion_nbhd_mean"] = af["criterion_blocks"][i].reshape(len(af["freqs_hz"]), -1).mean(1)
            rec["af_raw_argmin_hz"] = float(af["freqs_hz"][int(np.argmin(cb))])
            rec["af_raw_map_hz"] = float(af["raw_blocks"][i][ci])
            rec["af_smoothed_hz"] = float(af["smoothed_blocks"][i][ci])
            rec["af_focus_weight"] = float(af["focus_w_blocks"][i][ci])
        for lab, blks in ext.items():
            rec[f"ext_{lab}_hz"] = float(blks[i][ci])
        out_locs.append(rec)

    meta = {
        "num_rings": R, "nbhd": np.array(nbhd, int), "smooth_sigma": float(smooth_sigma),
        "matlab_kernelsize": -1 if matlab_kernelsize is None else int(matlab_kernelsize),
        "weak_rel_image": float(weak_rel_image), "weak_snr": float(weak_snr),
        "notch_ratio": float(notch_ratio),
        "noise_mask_source": noise_src, "n_noise_voxels": int(nidx.size),
        "profile_npz": profile_npz or "", "kr_edge": float(kr_edge),
        "vol_shape_zxy": np.array(vol.shape, int),
        "af_window": np.array(af_window, int), "af_idx_smooth_sigma": float(af_idx_smooth_sigma),
        "af_phase_ref_kr": float(af_phase_ref_kr),
        "convention": ("phase e^{-i2pi f t(k)}; k-index of the ring cascade is the MIRROR of the "
                       "k-index of tmap/autofocus, handled internally"),
    }
    shells = {
        "nominal_time_s": nominal_t, "nominal_dt_s": dt_nom,
        "real_time_hamming_mean_s": t_hmean, "real_dt_s": dt_shell,
        "real_time_quantiles_s": t_quant,             # min p10 p50 p90 max, ring k-index convention
        "real_time_quantiles_phys_s": t_quant_phys,   # same, unmirrored — mirroring sanity check
        "time_hist_centres_s": t_bins, "time_hist_weights": t_bw,
        "kr_bounds": shell_kr, "n_k_samples": shell_nk,
        "noise_level": shell_noise, "flip_fraction": flip_frac,
    }
    return {"meta": meta, "shells": shells, "locations": out_locs}


# ---------------------------------------------------------------------------
# npz caching (so re-rendering never re-runs a reconstruction)
# ---------------------------------------------------------------------------
def _flatten(prefix, obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}/{k}" if prefix else str(k), v, out)
    elif isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], dict):
        out[f"{prefix}/__len__"] = np.array(len(obj))
        for i, v in enumerate(obj):
            _flatten(f"{prefix}/{i}", v, out)
    else:
        out[prefix] = np.asarray(obj) if not isinstance(obj, str) else np.array(obj, dtype=object)


def save_diag_npz(path: str, diag: dict):
    flat = {}
    _flatten("", diag, flat)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **{k.replace("/", "⁄"): v for k, v in flat.items()})
    return path


def load_diag_npz(path: str) -> dict:
    z = np.load(path, allow_pickle=True)
    root: dict = {}
    for key in z.files:
        parts = key.replace("⁄", "/").split("/")
        node = root
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = z[key]

    def relist(node):
        if not isinstance(node, dict):
            return node
        if "__len__" in node:
            n = int(node["__len__"])
            return [relist(node[str(i)]) for i in range(n)]
        return {k: relist(v) for k, v in node.items()}

    out = relist(root)
    for rec in out.get("locations", []):
        for k in ("name",):
            if k in rec and rec[k].dtype == object:
                rec[k] = str(rec[k])
    if "meta" in out:
        for k in ("noise_mask_source", "profile_npz", "convention"):
            if k in out["meta"] and getattr(out["meta"][k], "dtype", None) == object:
                out["meta"][k] = str(out["meta"][k])
    return out


# ---------------------------------------------------------------------------
# self-test: a synthetic phantom with a KNOWN uniform off-resonance
# ---------------------------------------------------------------------------
def _selftest():
    """Two separate checks, deliberately not mixed.

    (1) UNIT test of the fitter: synthetic phase vectors with a known slope and
        random +-pi branch flips.  The fitter must recover the slope exactly;
        this tests code, not physics.
    (2) PHYSICS smoke test on a phantom blurred with the REAL cones t(k) and a
        known uniform off-resonance.  Here the per-voxel raw shell phase is NOT
        expected to sit on the line (see the module docstring); what is expected
        is that the sigma-smoothed fixed-reference phase does.
    """
    from scipy.interpolate import interp1d
    from scipy.ndimage import binary_dilation, gaussian_filter

    root = os.path.dirname(_HERE)
    prof = os.path.join(root, "data", "tkr_profile_brian.npz")
    p = np.load(prof); raw = p["kr"]; tp = p["t_s"]
    o = np.argsort(raw)
    tfun = interp1d(raw[o], tp[o], bounds_error=False, fill_value=(tp[o][0], tp[o][-1]))

    # ---- (1) fitter unit test -------------------------------------------
    rng = np.random.default_rng(1)
    dt = np.array([0.0, 0.57, 1.62, 2.99, 4.40, 5.63, 6.82, 7.87]) * 1e-3
    ok = True
    for f_true in (0.0, 37.0, -120.0, 210.0):
        phi = -2 * np.pi * f_true * dt + np.pi * rng.integers(0, 2, dt.size)
        phi = wrap_pi(phi + rng.normal(0, 0.02, dt.size))
        r = fit_signed_linear(phi, dt, np.ones_like(dt), f_grid=np.arange(-400, 400.01, 0.5))
        hit = abs(r["best_f_hz"] - f_true) <= 1.0
        ok &= hit
        print(f"[selftest 1] branch-flipped slope {f_true:+7.1f} Hz -> fit {r['best_f_hz']:+7.1f} Hz "
              f"({'OK' if hit else 'MISS'}); next distinct branch {r['second_best_f_hz']:+.1f} Hz, "
              f"depth ratio {r['second_best_cost'] / max(r['best_cost'], 1e-12):.0f}x")
    print(f"[selftest 1] fitter {'PASS' if ok else 'FAIL'}")

    # ---- (2) phantom -----------------------------------------------------
    nz, nx, ny = 32, 64, 64
    z, x, y = np.mgrid[:nz, :nx, :ny]
    c = (nz / 2, nx / 2, ny / 2)
    obj = 1.0 * (((x - c[1]) / 24) ** 2 + ((y - c[2]) / 26) ** 2 + ((z - c[0]) / 12) ** 2 < 1)
    obj += 0.8 * (((x - c[1] - 8) / 7) ** 2 + ((y - c[2] + 5) / 9) ** 2 + ((z - c[0]) / 5) ** 2 < 1)
    obj = obj * (1 + 0.35 * np.sin(x / 2.0) * np.sin(y / 2.3))
    img = gaussian_filter(obj, 0.5).astype(np.complex64)
    tmap = tfun(np.clip(kr_norm_zxy((nz, nx, ny)), 0, raw.max())).astype(np.float32)
    far = ~binary_dilation(np.abs(img) > 1e-3, iterations=6)
    rng = np.random.default_rng(0)
    noise = (rng.normal(0, 1, img.shape) + 1j * rng.normal(0, 1, img.shape)).astype(np.complex64)
    locs = [(16, 32, 32), (16, 32, 44), (16, 24, 37)]
    names_ = ["centre", "edge", "lesion"]
    for f_true in (0.0, 60.0, 120.0):
        K = _fft3(img) * np.exp(-2j * np.pi * f_true * tmap)                # e^{-i2pi f t}
        blurred = (_ifft3(K) + 0.002 * noise).astype(np.complex64)
        d = extract_shell_diagnostics(blurred, tmap, locs, names=names_, num_rings=8,
                                      profile_npz=prof, run_autofocus=False, noise_mask_zxy=far)
        for r in d["locations"]:
            print(f"[selftest 2] true {f_true:+6.1f} Hz  {r['name']:7s}  raw-fit {r['fit']['best_f_hz']:+7.1f}  "
                  f"sm(noflip) {r['fit_smoothed']['best_f_hz']:+7.1f} (n={r['fit_smoothed']['n_used']})  "
                  f"sm(flip) {r['fit_smoothed_flip']['best_f_hz']:+7.1f}  "
                  f"cascade {r['fit_cascade']['best_f_hz']:+7.1f}  "
                  f"weak {int(r['weak'].sum())}/{len(r['weak'])}  "
                  f"raw scatter {np.nanstd(r['fit']['resid_rad']):.2f} rad")
    s_ = d["shells"]
    print("[selftest 2] shell times ms: nominal " + np.array2string(s_["nominal_time_s"] * 1e3, precision=2))
    print("[selftest 2] real Hamming mean " + np.array2string(s_["real_time_hamming_mean_s"] * 1e3, precision=2))
    print("[selftest 2] real p10..p90     "
          + str([f"{a * 1e3:.2f}-{b * 1e3:.2f}" for a, b in zip(s_["real_time_quantiles_s"][:, 1],
                                                                s_["real_time_quantiles_s"][:, 3])]))
    print("[selftest 2] within-shell dephasing attenuation at the fitted f: "
          + np.array2string(d["locations"][0]["fit_support"]["attenuation"], precision=2))
    mm = np.abs(s_["real_time_quantiles_s"] - s_["real_time_quantiles_phys_s"]).max()
    print(f"[selftest 2] k-index mirroring changes the shell time support by at most {mm * 1e3:.3f} ms")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        pth = save_diag_npz(os.path.join(td, "t.npz"), d)
        d2 = load_diag_npz(pth)
        same = np.allclose(d["locations"][0]["phase_fixed_rel_rad"], d2["locations"][0]["phase_fixed_rel_rad"])
        print(f"[selftest] npz round-trip: {'OK' if same else 'MISMATCH'}; name={d2['locations'][0]['name']}")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
