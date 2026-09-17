"""1:1 Python port of Brian's multi-frequency reconstruction:
  freq_corr.m      — reconstruct at ONE offset frequency:  ift3( ft3(image) · exp(+2πi·t(k)·f) )
  freq_seg_corr.m  — frequency-segmented autofocus (Mar 2026): demodulate the
                     source by its low-res phase ONCE, reconstruct at each
                     frequency, low-pass (fspecial3 kernel 5) the SIGNED
                     imaginary part on a saved patch, pick per voxel the
                     frequency with the MINIMUM smoothed imaginary value
                     (literally `min(patch_imag,[],4)` — the most negative,
                     not the smallest |Imag|), output = that patch.

Reproduced as committed, including two properties worth knowing:
  * the phase reference is taken once from the uncorrected low-res image
    (time_segment(source, 1, 16, 2)); after demodulating at f ≠ 0 the flat
    interior carries phase ≈ 2π f t_low again, so flat tissue votes for 0 Hz
    (our 3D autofocus references each candidate to its own low-pass phase
    for that reason);
  * no smoothing of the selected-frequency map → per-voxel checkerboarding.
Arrays [z, x, y] at the public interface; MATLAB [x, y, z] internally.
"""
from __future__ import annotations

import numpy as np
from scipy import fft as sfft
from time_seg_corr import ft3, ift3, fspecial3_gaussian, imfilter_replicate, time_segment


def freq_corr_xyz(uncorr, kxyz_time, freq_offset: float, skip_fwd_fft: bool = False):
    ks = uncorr if skip_fwd_fft else ft3(uncorr)
    return ift3(ks * np.exp(2j * np.pi * kxyz_time * freq_offset).astype(np.complex64)).astype(np.complex64)


def freq_seg_corr_xyz(source_xyz, kxyz_time_xyz, corr_freqs, patch, kernelsize: int = 5, verbose: bool = False):
    """patch = (x0, x1, y0, y1, z0, z1) as Python half-open slices on the [x,y,z] arrays.
    Returns (corr_patch, field_map_patch, saved_patch[x,y,z,f], patch_imag[x,y,z,f])."""
    x0, x1, y0, y1, z0, z1 = patch
    kern = fspecial3_gaussian(kernelsize)
    lowres, _, _ = time_segment(source_xyz, kxyz_time_xyz, 1, 16, 2.0)
    source_kspace = ft3(source_xyz * np.exp(-1j * np.angle(lowres)).astype(np.complex64)).astype(np.complex64)
    del lowres
    nf = len(corr_freqs)
    saved = np.zeros((x1 - x0, y1 - y0, z1 - z0, nf), np.complex64)
    pimag = np.zeros(saved.shape, np.float32)
    for i, f in enumerate(corr_freqs):
        if verbose:
            print(f"Frequency {i + 1} of {nf} ({f:g} Hz)", flush=True)
        cur = freq_corr_xyz(source_kspace, kxyz_time_xyz, float(f), skip_fwd_fft=True)
        saved[..., i] = cur[x0:x1, y0:y1, z0:z1]
        pimag[..., i] = imfilter_replicate(saved[..., i].imag.astype(np.float32), kern)
        del cur
    min_index = np.argmin(pimag, axis=3)                      # MATLAB: [~, min_index] = min(patch_imag, [], 4)
    corr = np.take_along_axis(saved, min_index[..., None], axis=3)[..., 0]
    fmap = np.asarray(corr_freqs, dtype=np.float32)[min_index]
    return corr, fmap, saved, pimag


def freq_seg_corr(source_zxy, kxyz_time_zxy, corr_freqs, zs: slice, rows: slice, cols: slice,
                  kernelsize: int = 5, verbose: bool = False, embed: bool = True):
    """Repo-order wrapper: the volume is [z, rows, cols] (axis 1 = image rows,
    axis 2 = image columns — the same indexing the viewer's zoom box uses:
    zoom=y0,y1,x0,x1 ⇔ rows y0:y1, cols x0:x1). MATLAB's [x, y, z] internal
    order maps axis 1 → x, axis 2 → y. With embed=True returns a FULL-size
    volume equal to the source outside the patch and to Brian's MF selection
    inside it (so the viewer can show it), plus the patch field map."""
    src = np.ascontiguousarray(np.transpose(source_zxy, (1, 2, 0))).astype(np.complex64)
    tm = np.ascontiguousarray(np.transpose(kxyz_time_zxy, (1, 2, 0))).astype(np.float64)
    patch = (rows.start, rows.stop, cols.start, cols.stop, zs.start, zs.stop)
    corr_p, fmap_p, _, _ = freq_seg_corr_xyz(src, tm, corr_freqs, patch, kernelsize, verbose)
    corr_p = np.transpose(corr_p, (2, 0, 1)); fmap_p = np.transpose(fmap_p, (2, 0, 1))
    if not embed:
        return corr_p, fmap_p
    out = np.array(source_zxy, dtype=np.complex64, copy=True)
    out[zs, rows, cols] = corr_p
    return out, fmap_p


class MFSliceRecon:
    """Brian's freq_corr.m — ift3( ft3(image) .* exp(2πi·kxyz_time·f) ) — evaluated
    EXACTLY for one slice, fast enough for a viewer slider.

    ift3 is a centred inverse DFT along each axis. Restricting the output to one
    z-index turns the z-transform into a weighted sum over k_z planes:
        S(kx,ky) = (1/nz) Σ_kz K(kx,ky,kz)·exp(2πi·t(k)·f)·exp(2πi (kz−nz/2)(z−nz/2)/nz)
    followed by the centred 2D inverse DFT of S. Same normalisation and shift
    conventions as ft3/ift3 (verified in --selftest to 1e-6). Cost: one pass
    over k-space (~0.5 s at 768×768×112) + one 2D IFFT, instead of a 3D IFFT.
    Public interface in repo order: volume [z, rows, cols] → slice [rows, cols].
    """

    def __init__(self, vol_zxy, kxyz_time_zxy):
        self.K = ft3(np.ascontiguousarray(np.transpose(vol_zxy, (1, 2, 0))).astype(np.complex64)).astype(np.complex64)   # [x,y,z]
        self.t = np.ascontiguousarray(np.transpose(kxyz_time_zxy, (1, 2, 0))).astype(np.float32)
        self.nz = self.K.shape[2]

    def slice_at(self, f_hz: float, z: int) -> np.ndarray:
        nz = self.nz
        kz = np.arange(nz)
        w = (np.exp(2j * np.pi * (kz - nz // 2) * (z - nz // 2) / nz) / nz).astype(np.complex64)
        S = np.zeros(self.K.shape[:2], np.complex64)
        two_pi_f = np.float32(2 * np.pi * f_hz)
        for k in range(nz):
            S += self.K[:, :, k] * np.exp(1j * (two_pi_f * self.t[:, :, k])).astype(np.complex64) * w[k]
        return sfft.ifftshift(sfft.ifft2(sfft.ifftshift(S))).astype(np.complex64)      # [rows, cols]


def _selftest():
    rng = np.random.default_rng(0)
    vol = (rng.standard_normal((12, 16, 20)) + 1j * rng.standard_normal((12, 16, 20))).astype(np.complex64)   # [z,rows,cols]
    t = (rng.random((12, 16, 20)) * 8e-3).astype(np.float32)
    mf = MFSliceRecon(vol, t)
    for f in (-120.0, 0.0, 75.0):
        full = ift3(mf.K * np.exp(2j * np.pi * mf.t * f).astype(np.complex64))          # [x,y,z] = freq_corr_xyz
        for z in (0, 5, 11):
            a = full[:, :, z]; b = mf.slice_at(f, z)
            print(f"f={f:+6.1f} Hz z={z:2d}: max|Δ|/max|full| = {np.abs(a - b).max() / np.abs(a).max():.2e}")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
