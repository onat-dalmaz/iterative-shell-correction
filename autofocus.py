"""Frequency-selecting autofocus used in the abstract.

Noll et al., Magn Reson Med 25:319–333 (1992), Eq. 14,
doi:10.1002/mrm.1910250210. The 3D grid and local window are the
reported cones adaptation; selected fields are not smoothed.
"""
from __future__ import annotations

import numpy as np
from scipy import fft, ndimage


def _fft3(x):
    return fft.fftshift(fft.fftn(fft.ifftshift(x), norm="ortho", workers=-1))


def _ifft3(x):
    return fft.fftshift(fft.ifftn(fft.ifftshift(x), norm="ortho", workers=-1))


def kr_norm_zxy(shape_zxy):
    """Per-axis-normalized radius; the sampled sphere ends at one."""
    out = None
    for axis, n in enumerate(shape_zxy):
        f = ((np.arange(n) - n // 2) / (n / 2)).astype(np.float32) ** 2
        view = [None] * 3
        view[axis] = slice(None)
        out = f[tuple(view)] if out is None else out + f[tuple(view)]
    return np.sqrt(out, dtype=np.float32)


def _eq14_cost(candidate, lowpass_candidate):
    phase = np.angle(lowpass_candidate)
    referenced = candidate * np.exp(-1j * phase).astype(np.complex64)
    point_cost = np.abs(referenced.imag).astype(np.float32)
    point_cost = np.power(point_cost, np.float32(0.5)).astype(np.float32)
    cost = ndimage.uniform_filter(point_cost, size=(1, 41, 41), mode="nearest")
    return (cost * np.float32(41 * 41)).astype(np.float32)


def autofocus(volume, timing, *, workers=1):
    """Return corrected image, selected Hz map and settings for [z,x,y] arrays.

    Timing is in seconds on the centered FFT grid. Each candidate uses its
    own low-pass phase; a positive field cancels exp(-i*2*pi*f*t(k)).
    """
    vol = np.asarray(volume)
    tmap = np.asarray(timing)
    if vol.ndim != 3 or vol.shape != tmap.shape or any(n < 1 for n in vol.shape):
        raise ValueError("volume and timing must have the same nonempty [z,x,y] shape")
    if not np.iscomplexobj(vol) or not np.all(np.isfinite(vol)):
        raise ValueError("volume must contain finite complex image values")
    if np.iscomplexobj(tmap) or not np.all(np.isfinite(tmap)) or np.any(tmap < 0):
        raise ValueError("timing must contain finite nonnegative seconds")
    if isinstance(workers, bool) or not isinstance(workers, (int, np.integer)) or workers < 1:
        raise ValueError("workers must be a positive integer")

    def ft(a):
        return fft.fftshift(fft.fftn(fft.ifftshift(a), norm="ortho", workers=workers))

    def ift(a):
        return fft.fftshift(fft.ifftn(fft.ifftshift(a), norm="ortho", workers=workers))

    freqs = np.arange(-500.0, 501.0, 10.0, dtype=np.float64)
    kspace = ft(vol.astype(np.complex64)).astype(np.complex64)
    radius2 = np.zeros(vol.shape, dtype=np.float32)
    for axis, n in enumerate(vol.shape):
        coordinate = ((np.arange(n) - n // 2) / (n / 2)).astype(np.float32)
        view = [1, 1, 1]
        view[axis] = n
        radius2 += (coordinate * coordinate).reshape(view)
    lowmask = np.sqrt(radius2) <= np.float32(0.1)
    phase_per_hz = (2 * np.pi * tmap.astype(np.float32)).astype(np.float32)
    image = np.zeros(vol.shape, dtype=np.complex64)
    field = np.full(vol.shape, freqs[0], dtype=np.float32)
    best = np.full(vol.shape, np.inf, dtype=np.float32)
    for frequency in freqs:
        ramp = np.exp(1j * phase_per_hz * np.float32(frequency)).astype(np.complex64)
        candidate_kspace = (kspace * ramp).astype(np.complex64)
        candidate = ift(candidate_kspace).astype(np.complex64)
        lowpass = ift(np.where(lowmask, candidate_kspace, 0)).astype(np.complex64)
        cost = _eq14_cost(candidate, lowpass)
        better = cost < best  # Ascending frequencies: the first candidate wins ties.
        best[better] = cost[better]
        field[better] = np.float32(frequency)
        image[better] = candidate[better]
    info = {
        "method": "Noll1992 Eq14 (cones adaptation)",
        "paper_doi": "10.1002/mrm.1910250210",
        "alpha": 0.5,
        "freqs_hz": freqs.tolist(),
        "integration_window_zxy": [1, 41, 41],
        "boundary_mode": "nearest",
        "phase_ref_kr": 0.1,
        "phase_reference": "candidate-specific hard low-pass phase",
        "selection": "pointwise argmin; first frequency wins exact ties",
        "assembly": "original complex candidate; no smoothing or interpolation",
        "workers": int(workers),
    }
    return image, field, info
