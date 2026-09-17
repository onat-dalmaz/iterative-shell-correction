"""1:1 Python port of Brian Hargreaves' time_seg_corr.m + time_segment.m
(mribri999/timesegmentedphase, Feb–Mar 2026) — the CURRENT method in his
Off-Res Log ("hierarchical time-segmented correction", "16 segments better").

Differences from ring_b0_corr.m / ts_b0_corr.m (which cones_recon_fcns.py
translates) — these are why our N=8-vs-N=16 conclusions did not transfer:
  * segmentation by the ACTUAL 3D sample-time map kxyz_time (equally spaced
    in TIME), not by radial k-space shells;
  * Hamming windows with overlap = 8 (each window spans ±4 segment spacings:
    heavily overlapping sliding windows), not 50 % overlap;
  * kernelsize = 7 → MATLAB fspecial3 truncated Gaussian (σ = 3.5 on a 7³
    box, effective std ≈ 2.0 voxels), not a full Gaussian;
  * FFT direction: source_kspace = ft3(image), segment = ift3(W·kspace)
    (ring_b0_corr uses ift3 to k-space and ft3 back — equivalent for a
    symmetric window, mirrored here anyway);
  * field map = first-interval increment only (segments 1→2).

MATLAB primitives reproduced exactly:
  ft3  = fftshift(fft(fftshift(a,k),[],k),k) over k=1..3   (unnormalised)
  ift3 = ifftshift(ifft(ifftshift(a,k),[],k),k)           (1/N)
  fspecial3('gaussian', s*[1 1 1], s/2), imfilter(...,'replicate')
Arrays are handled in Brian's [x, y, z] order internally; the public function
takes/returns [z, x, y] like the rest of this repo.
"""
from __future__ import annotations

import numpy as np
from scipy import fft as sfft
from scipy import ndimage


def ft3(a):
    return sfft.fftshift(sfft.fftn(sfft.fftshift(a), workers=-1))


def ift3(a):
    return sfft.ifftshift(sfft.ifftn(sfft.ifftshift(a), workers=-1))


def fspecial3_gaussian(kernelsize: int) -> np.ndarray:
    h = (kernelsize - 1) // 2
    x = np.arange(-h, h + 1, dtype=np.float64)
    s = kernelsize / 2.0
    g1 = np.exp(-(x ** 2) / (2 * s ** 2))
    g = g1[:, None, None] * g1[None, :, None] * g1[None, None, :]
    return (g / g.sum()).astype(np.float32)


def imfilter_replicate(x, kern):
    """imfilter(x, kern, 'replicate') for the separable fspecial3 kernel:
    three 1D passes (identical to the full 3D convolution, ~27× faster)."""
    k1 = kern.sum(axis=(1, 2)).astype(np.float64); k1 = k1 / k1.sum()

    def sep(a):
        out = a.astype(np.float32, copy=True)
        for ax in range(a.ndim):
            out = ndimage.convolve1d(out, k1, axis=ax, mode="nearest")
        return out

    if np.iscomplexobj(x):
        return (sep(x.real) + 1j * sep(x.imag)).astype(np.complex64)
    return sep(x)


def time_segment(image_or_kspace, sample_times, segment: int, num_segments: int = 8,
                 overlap: float = 2.0, skip_fft: bool = False, modulation: int = 0):
    """time_segment.m. `segment` is 1-based like MATLAB. Returns
    (im_component, time_window, seg_mid_time)."""
    t = sample_times
    start_time = 0.0
    end_time = float(t.max())
    seg_mid_time = (segment - 1) / (num_segments - 1) * end_time
    seg_spacing = (end_time - start_time) / (num_segments - 1)
    seg_half_width = seg_spacing * overlap / 2.0
    seg_start = seg_mid_time - seg_half_width
    seg_end = seg_mid_time + seg_half_width
    pts = (t > seg_start) & (t < seg_end)
    W = np.zeros(t.shape, np.float32)
    W[pts] = 0.54 + 0.46 * np.cos(np.pi * (t[pts] - seg_mid_time) / seg_half_width)
    if modulation == 1:
        W[pts] *= (t[pts] - seg_mid_time) / seg_half_width
    ks = image_or_kspace if skip_fft else ft3(image_or_kspace)
    comp = ift3(W * ks).astype(np.complex64)
    return comp, W, seg_mid_time


def time_seg_corr_xyz(source_image_xyz, kxyz_time_xyz, num_segments: int,
                      kernelsize: int = 7, overlap: float = 8.0, verbose: bool = False):
    """time_seg_corr.m on [x, y, z] arrays. Returns (corr_image, field_map_hz, info)."""
    kern = fspecial3_gaussian(kernelsize)
    source_kspace = ft3(source_image_xyz.astype(np.complex64)).astype(np.complex64)
    corr_phase = None; corr_image = None; field_map = None
    last_seg_time = None; seg_times = []; flip_frac = []
    for s in range(1, num_segments + 1):
        if verbose:
            print(f"Correcting segment {s} of {num_segments}", flush=True)
        seg_image, _, seg_time = time_segment(source_kspace, kxyz_time_xyz, s, num_segments,
                                              overlap, skip_fft=True)
        seg_times.append(seg_time)
        if s == 1:
            corr_phase = np.exp(-1j * np.angle(seg_image)).astype(np.complex64)
            corr_image = (seg_image * corr_phase).astype(np.complex64)
        else:
            image_copy = (seg_image * corr_phase).astype(np.complex64)
            flip = np.abs(np.angle(image_copy)) > (np.pi / 2)
            flip_frac.append(float(flip.mean()))
            image_copy[flip] = -image_copy[flip]
            del flip
            image_copy = imfilter_replicate(image_copy, kern)
            corr_phase = (corr_phase * np.exp(-1j * np.angle(image_copy))).astype(np.complex64)
            corr_image += (seg_image * corr_phase).astype(np.complex64)
            if s == 2:
                field_map = (np.angle(image_copy) / (2 * np.pi) / (seg_time - last_seg_time)).astype(np.float32)
            del image_copy
        del seg_image
        last_seg_time = seg_time
    info = {"seg_times_s": np.array(seg_times), "flip_fraction_per_segment": flip_frac,
            "kernelsize": kernelsize, "overlap": overlap, "num_segments": num_segments,
            "sign_note": "field_map sign = MATLAB's: angle(image_copy)/(2π Δt); this is the "
                         "NEGATIVE of ring_fieldmap's convention"}
    return corr_image, field_map, info


def time_seg_corr(source_image_zxy, kxyz_time_zxy, num_segments: int, kernelsize: int = 7,
                  overlap: float = 8.0, verbose: bool = False):
    """Repo-order wrapper: [z, x, y] in, [z, x, y] out."""
    img = np.ascontiguousarray(np.transpose(source_image_zxy, (1, 2, 0)))
    tmap = np.ascontiguousarray(np.transpose(kxyz_time_zxy, (1, 2, 0))).astype(np.float64)
    corr, fmap, info = time_seg_corr_xyz(img, tmap, num_segments, kernelsize, overlap, verbose)
    back = lambda a: np.ascontiguousarray(np.transpose(a, (2, 0, 1)))
    return back(corr), (back(fmap) if fmap is not None else None), info


def _selftest():
    """Structural checks on Brian's own sample data (sampledata.mat + time_kxyz.mat)."""
    import scipy.io as sio
    t = sio.loadmat("brian_ref/tsphase/time_kxyz.mat")["samp_times"]
    img = sio.loadmat("brian_ref/tsphase/sampledata.mat")["source_image"]
    print("sample:", img.shape, img.dtype, "| times ms", t.min() * 1e3, t.max() * 1e3)
    # window sums (README: adjacent windows add up) for overlap 2 and 8
    for ov in (2.0, 8.0):
        S = np.zeros(t.shape, np.float32)
        for s in range(1, 9):
            _, W, _ = time_segment(np.zeros((2, 2, 2)), t, s, 8, ov, skip_fft=True) if False else (None, None, None)
        # cheaper: compute windows directly
        end = float(t.max()); sp = end / 7; hw = sp * ov / 2
        S[:] = 0
        for s in range(1, 9):
            mid = (s - 1) / 7 * end
            p = (t > mid - hw) & (t < mid + hw)
            S[p] += 0.54 + 0.46 * np.cos(np.pi * (t[p] - mid) / hw)
        inner = t > 0.5e-3
        print(f"overlap {ov:.0f}: window sum over sampled k-space p5/p50/p95 = "
              f"{np.percentile(S[inner], 5):.3f}/{np.median(S[inner]):.3f}/{np.percentile(S[inner], 95):.3f}")
    # run the correction on a 128^3 central crop of the sample (fast), N=8 and 16
    c = slice(64, 192)
    sub = np.ascontiguousarray(img[c, c, c]); tsub = np.ascontiguousarray(t[c, c, c])  # NOTE: cropping k-space
    # (a crop of the image is fine; the time map must stay on the k-grid of the crop —
    #  here we approximate by cropping the time map's central k-region: same normalised extent)
    for N in (8, 16):
        corr, fmap, info = time_seg_corr_xyz(sub, tsub, N, verbose=False)
        g = np.median(np.abs(corr) / (np.abs(sub) + 1e-9))
        print(f"N={N}: seg times ms {np.round(info['seg_times_s'] * 1e3, 2).tolist()}; "
              f"flip frac {np.round(info['flip_fraction_per_segment'], 2).tolist()}; gain median {g:.3f}; "
              f"fmap Hz p5/p50/p95 {np.percentile(fmap, 5):+.0f}/{np.median(fmap):+.0f}/{np.percentile(fmap, 95):+.0f}")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
