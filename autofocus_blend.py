"""Refinements of the 3D autofocus: coarse-to-fine search, margin reliability, field-map blend, gated blend.

These are the methods behind `autofocus_c2f`, `autofocus_fblend` and `autofocus_fblend_gated60`, collected
from the research scripts into one module with the same maths, parameters and defaults.

Pipeline, all on the same candidate grid (31 frequencies, -300..+300 Hz in 20 Hz steps):

  1. smoothed autofocus     autofocus_3d(vol, tmap, freqs, idx_smooth_sigma=5)          -> f_smooth
     Per voxel, the frequency minimising |Im| against that candidate's own low-pass phase, then the
     selected index is smoothed. Conservative: reliable, but biased toward small offsets at the skin.

  2. coarse-to-fine         coarse_to_fine(vol, tmap, freqs, f_smooth, halfwin=60, radius=5) -> f_c2f
     Re-search only f in [f_smooth - 60, f_smooth + 60] Hz, box-filter the cost inside that window,
     parabolic sub-bin. Aggressive: sharper edges, but it can lock onto a different minimum than the
     smoothed estimate, which is what produced the held-out artifacts.

  3. margin                 margin_map(vol, tmap, freqs, sep=3)                          -> margin
     How decisive the autofocus minimum is: depth of the best competing minimum at least `sep` bins
     away, divided by the cost range. 1 = one clear minimum, 0 = two equally good branches.

  4. field blend            w = blend_weight(margin, sigma=5)
                            f = w * f_c2f + (1 - w) * f_smooth                           -> f_blend
     Blend the FIELD MAPS, not the images, then reconstruct once (blending images softens wherever
     the two disagree).

  5. gated blend            w_eff = w * exp(-(|f_c2f - f_smooth| / gate_hz)^2), gate_hz = 60  -> f_gated
     Where the two parents disagree by more than about gate_hz they have picked different minima, and
     the blend falls back to the smoothed map. Removes 92 % of the held-out volume's artifact voxels and
     is indistinguishable from the plain blend on the development volume.

  6. assemble(vol, tmap, freqs, fmap)  soft assembly: each voxel interpolates linearly between the two
     candidate reconstructions that bracket its frequency.

Sign convention (same as autofocus.py): candidate f demodulates k-space by exp(+i 2 pi f t(k)), so the
selected map is the off-resonance of the acquisition exp(-i 2 pi f t(k)).

  python autofocus_blend.py --orig volume.npy --tmap kxyz_time.npy [--out results]
  python autofocus_blend.py --selftest
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from autofocus import _fft3, _ifft3, kr_norm_zxy, autofocus_3d, time_map_from_profile   # noqa: E402

FREQS = np.arange(-300.0, 300.1, 20.0)


def _criterion_fn(K, two_pi_t, window=(3, 5, 5), phase_ref_kr=0.1):
    """|Im| of the candidate image after removing that candidate's own low-pass phase, box-averaged."""
    low = kr_norm_zxy(K.shape) <= phase_ref_kr

    def criterion(f):
        Kf = (K * np.exp(1j * (two_pi_t * np.float32(f))).astype(np.complex64)).astype(np.complex64)
        m = _ifft3(Kf).astype(np.complex64)
        m *= np.exp(-1j * np.angle(_ifft3(np.where(low, Kf, 0)))).astype(np.complex64); del Kf
        c = ndimage.uniform_filter(np.abs(m.imag), size=window, mode="nearest").astype(np.float32); del m
        return c
    return criterion


def coarse_to_fine(vol_zxy, tmap_zxy, freqs, f_coarse, halfwin=60.0, radius=5, window=(3, 5, 5),
                   phase_ref_kr=0.1, verbose=True):
    """Fine search restricted to +-halfwin around f_coarse. Returns (complex image, selected Hz map)."""
    freqs = np.asarray(freqs, np.float64); df = float(freqs[1] - freqs[0])
    K = _fft3(np.asarray(vol_zxy).astype(np.complex64)).astype(np.complex64)
    two_pi_t = (2 * np.pi * np.asarray(tmap_zxy, np.float32)).astype(np.float32)
    criterion = _criterion_fn(K, two_pi_t, window, phase_ref_kr)
    fc = np.asarray(f_coarse, np.float32)

    def box(a):
        return ndimage.uniform_filter(a, size=(3, 2 * radius + 1, 2 * radius + 1), mode="nearest")

    BIG = np.float32(1e30)
    best = np.full(K.shape, BIG, np.float32)
    cm = np.full(K.shape, BIG, np.float32); cp = np.full(K.shape, BIG, np.float32)
    idx = np.full(K.shape, -1, np.int16); prev = None
    for i, f in enumerate(freqs):
        inwin = np.abs(fc - np.float32(f)) <= np.float32(halfwin)
        if not inwin.any():
            prev = None                      # window gap: no left neighbour for the next frequency
            continue
        if verbose:
            print(f"  coarse-to-fine {i + 1}/{len(freqs)} ({f:+.0f} Hz, {100 * inwin.mean():.0f}% of voxels in window)", flush=True)
        cf = box(criterion(f).astype(np.float32))
        cf = np.where(inwin, cf, BIG).astype(np.float32)
        if prev is not None:
            js = idx == (i - 1); cp[js] = cf[js]; del js
        nb = cf < best
        cm[nb] = BIG if prev is None else prev[nb]
        best[nb] = cf[nb]; idx[nb] = i; cp[nb] = BIG
        del nb, inwin
        prev = cf
    del prev

    den = (cm - 2.0 * best + cp).astype(np.float32)
    ok = (cm < BIG) & (cp < BIG) & (best < BIG) & (den > 1e-20)
    delta = np.zeros(best.shape, np.float32)
    delta[ok] = 0.5 * (cm[ok] - cp[ok]) / den[ok]                    # parabolic sub-bin
    np.clip(delta, -0.5, 0.5, out=delta)
    del den, ok, cm, cp, best
    idx_safe = np.where(idx >= 0, idx, 0).astype(np.intp)
    fsel = (freqs[idx_safe] + delta * df).astype(np.float32)
    fsel = np.where(idx >= 0, fsel, fc).astype(np.float32)          # fall back to the coarse value
    del delta, idx, idx_safe
    np.clip(fsel, freqs[0], freqs[-1], out=fsel)
    return _assemble_K(K, two_pi_t, freqs, fsel, verbose), fsel


def margin_map(vol_zxy, tmap_zxy, freqs, sep=3, window=(3, 5, 5), phase_ref_kr=0.1, verbose=True):
    """Normalised depth of the best competing minimum >= `sep` bins from the argmin (two passes)."""
    freqs = np.asarray(freqs, np.float64)
    K = _fft3(np.asarray(vol_zxy).astype(np.complex64)).astype(np.complex64)
    two_pi_t = (2 * np.pi * np.asarray(tmap_zxy, np.float32)).astype(np.float32)
    criterion = _criterion_fn(K, two_pi_t, window, phase_ref_kr)
    best = None; worst = None; idx = np.zeros(K.shape, np.int16)
    for i, f in enumerate(freqs):
        if verbose:
            print(f"  margin pass 1 {i + 1}/{len(freqs)}", flush=True)
        c = criterion(f)
        if best is None:
            best = c.copy(); worst = c
        else:
            b = c < best; best[b] = c[b]; idx[b] = i; np.maximum(worst, c, out=worst); del b
        del c
    second = np.full(K.shape, np.inf, np.float32)
    for i, f in enumerate(freqs):
        if verbose:
            print(f"  margin pass 2 {i + 1}/{len(freqs)}", flush=True)
        c = criterion(f)
        far = np.abs(idx.astype(np.int32) - i) >= sep
        np.minimum(second, np.where(far, c, np.inf).astype(np.float32), out=second)
        del c, far
    del K, two_pi_t
    rng = np.maximum(worst - best, 1e-12); del worst
    return np.clip((second - best) / rng, 0.0, 1.0).astype(np.float32)


def blend_weight(margin, sigma=5.0):
    """Smooth the margin, normalise by its 99th percentile, clip to [0, 1]."""
    w = ndimage.gaussian_filter(np.asarray(margin, np.float32), sigma, mode="nearest")
    w = (w / max(float(np.percentile(w, 99.0)), 1e-9)).astype(np.float32)
    np.clip(w, 0.0, 1.0, out=w)
    return w


def blend_fieldmaps(f_aggressive, f_conservative, w, gate_hz=None):
    """Field-map blend; with gate_hz, fall back to the conservative map where the two disagree."""
    fa = np.asarray(f_aggressive, np.float32); fc = np.asarray(f_conservative, np.float32)
    w = np.asarray(w, np.float32)
    if gate_hz:
        w = (w * np.exp(-(np.abs(fa - fc) / np.float32(gate_hz)) ** 2)).astype(np.float32)
    return (w * fa + (1.0 - w) * fc).astype(np.float32)


def _assemble_K(K, two_pi_t, freqs, fmap, verbose=True):
    df = float(freqs[1] - freqs[0])
    out = np.zeros(K.shape, np.complex64); wsum = np.zeros(K.shape, np.float32)
    for i, f in enumerate(freqs):
        w = (1.0 - np.abs(fmap - np.float32(f)) / np.float32(df)).astype(np.float32)
        np.clip(w, 0.0, None, out=w)
        if not w.any():
            continue
        if verbose:
            print(f"  assemble {i + 1}/{len(freqs)} ({f:+.0f} Hz)", flush=True)
        m = _ifft3(K * np.exp(1j * (two_pi_t * np.float32(f))).astype(np.complex64)).astype(np.complex64)
        out += m * w; wsum += w; del m, w
    out /= np.maximum(wsum, 1e-6).astype(np.float32)
    return out


def assemble(vol_zxy, tmap_zxy, freqs, fmap, verbose=True):
    """Reconstruct once from a field map: linear interpolation between the bracketing candidates."""
    K = _fft3(np.asarray(vol_zxy).astype(np.complex64)).astype(np.complex64)
    two_pi_t = (2 * np.pi * np.asarray(tmap_zxy, np.float32)).astype(np.float32)
    return _assemble_K(K, two_pi_t, np.asarray(freqs, np.float64), np.asarray(fmap, np.float32), verbose)


def run_pipeline(vol_zxy, tmap_zxy, freqs=FREQS, halfwin=60.0, radius=5, sep=3, sigma=5.0, gate_hz=60.0,
                 verbose=True):
    """Everything above, in order. Returns {name: (complex image, Hz map)} plus the margin."""
    out = {}
    img, f_smooth, _ = autofocus_3d(vol_zxy, tmap_zxy, freqs_hz=freqs, idx_smooth_sigma=5.0, verbose=verbose)
    out["autofocus_r300"] = (img, np.asarray(f_smooth, np.float32)); del img
    img, f_c2f = coarse_to_fine(vol_zxy, tmap_zxy, freqs, f_smooth, halfwin=halfwin, radius=radius, verbose=verbose)
    out["autofocus_c2f"] = (img, f_c2f); del img
    margin = margin_map(vol_zxy, tmap_zxy, freqs, sep=sep, verbose=verbose)
    w = blend_weight(margin, sigma)
    f_b = blend_fieldmaps(f_c2f, f_smooth, w)
    out["autofocus_fblend"] = (assemble(vol_zxy, tmap_zxy, freqs, f_b, verbose), f_b)
    if gate_hz:
        f_g = blend_fieldmaps(f_c2f, f_smooth, w, gate_hz=gate_hz)
        out[f"autofocus_fblend_gated{gate_hz:g}"] = (assemble(vol_zxy, tmap_zxy, freqs, f_g, verbose), f_g)
    out["_margin"] = margin
    return out


LABELS = {"autofocus_r300": "autofocus 3D — |Imag| criterion, 31 freqs ±300 Hz (20 Hz), σ_idx 5",
          "autofocus_c2f": "autofocus, coarse-to-fine (±60 Hz window, cost-filtered r=5, sub-bin)",
          "autofocus_fblend": "autofocus, field-map blend (coarse-to-fine where the minimum is unambiguous, else smoothed)",
          "autofocus_fblend_gated60": "autofocus, field-map blend gated at 60 Hz (smoothed map where the parents disagree)"}


def _selftest():
    """Uniform +60 Hz offset on a small phantom through the real cones t(k): every method must recover it."""
    here = os.path.dirname(os.path.abspath(__file__))
    shape = (12, 48, 48)
    z, x, y = np.mgrid[-6:6, -24:24, -24:24]
    r = np.sqrt((z * 4.0) ** 2 + x ** 2 + y ** 2)
    m = np.where(r < 18, 1.0, 0.0) + np.where((r > 15) & (r < 18), 0.8, 0.0)
    m = ndimage.gaussian_filter(m, 0.6).astype(np.complex64)
    t = time_map_from_profile(shape, os.path.join(here, "data", "tkr_profile_brian.npz")).astype(np.float32)
    f_true = 60.0
    y_img = _ifft3(_fft3(m) * np.exp(-1j * 2 * np.pi * f_true * t).astype(np.complex64))
    res = run_pipeline(y_img, t, verbose=False)
    mask = np.abs(m) > 0.3
    err0 = np.linalg.norm((np.abs(y_img) - np.abs(m))[mask]) / np.linalg.norm(np.abs(m)[mask])
    ok = True
    print(f"selftest: uniform {f_true:+.0f} Hz, phantom {shape}; uncorrected magnitude error {err0:.3f}")
    for k, v in res.items():
        if k.startswith("_"):
            continue
        img, fm = v
        e = np.linalg.norm((np.abs(img) - np.abs(m))[mask]) / np.linalg.norm(np.abs(m)[mask])
        med = float(np.median(fm[mask]))
        good = abs(med - f_true) <= 10.0 and e < err0
        ok &= good
        print(f"  {k:26s} median field {med:+6.1f} Hz   magnitude error {e:.3f}   {'OK' if good else 'FAIL'}")
    print("selftest", "PASSED" if ok else "FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--orig"); ap.add_argument("--tmap")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    ap.add_argument("--gate_hz", type=float, default=60.0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    if not (a.orig and a.tmap):
        ap.error("--orig and --tmap are required (or --selftest)")
    t0 = time.time()
    vol = np.load(a.orig).astype(np.complex64); tmap = np.load(a.tmap).astype(np.float32)
    res = run_pipeline(vol, tmap, gate_hz=a.gate_hz)
    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, "autofocus_margin.npy"), res.pop("_margin"))
    lab_p = os.path.join(a.out, "labels.json"); labels = json.load(open(lab_p)) if os.path.exists(lab_p) else {}
    for name, (img, fm) in res.items():
        np.save(os.path.join(a.out, f"{name}.npy"), np.abs(img).astype(np.float32))    # abs BEFORE the cast
        np.save(os.path.join(a.out, f"{name}_selected_freq_hz.npy"), fm.astype(np.float32))
        labels[name] = LABELS.get(name, name)
        print(f"wrote {name}", flush=True)
    json.dump(labels, open(lab_p, "w"), indent=1, ensure_ascii=False)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
