"""Iterative shell correction for coil-combined center-out MRI.

Volumes, fields and centered-FFT timing maps use [z,x,y]. Timing is in
seconds and fields are in Hz: positive f cancels exp(-i*2*pi*f*t(k)).
Shell demodulation follows Brian Hargreaves' hierarchical correction.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from autofocus import _fft3, _ifft3
from _shells import (
    C64, TWO_PI, AppliedField, Kern, LockHorizon, TimeBank,
    _sample_idx, infer_voxel_mm, kernel_from_mm, kernel_schedule,
    noise_floor_table, noise_region, step_k, to_zxy,
)

SETTINGS = {
    "shells": 16,
    "spacing_ms": 0.6,
    "field_updates": 3,
    "update_clip_hz": 60.0,
    "field_smoothing_sigma_zxy": [1.5, 4.0, 4.0],
    "minimum_fit_time_ms": 1.0,
    "stop_hz": 3.0,
    "cp_step_hz": 2.0,
    "cp_clip_hz": 400.0,
}


def _validate(volume, timing):
    vol = np.asarray(volume)
    tm = np.asarray(timing)
    if vol.ndim != 3 or vol.shape != tm.shape or any(n < 1 for n in vol.shape):
        raise ValueError("volume and timing must have the same nonempty [z,x,y] shape")
    if not np.iscomplexobj(vol) or not np.all(np.isfinite(vol)):
        raise ValueError("volume must contain finite complex image values")
    if np.iscomplexobj(tm) or not np.all(np.isfinite(tm)) or np.any(tm < 0):
        raise ValueError("timing must contain finite nonnegative seconds")
    return vol.astype(C64), np.asarray(tm, np.float32)


def _cp(vol, tmap, field, df=2.0, clip=400.0):
    # Occupied frequency groups are reconstructed from the original k-space.
    kspace = _fft3(np.asarray(vol).astype(C64)).astype(C64)
    q = np.rint(np.clip(np.asarray(field, np.float32), -clip, clip) / df).astype(np.int32).ravel()
    order = np.argsort(q, kind="stable")
    qs = q[order]
    occupied, starts = np.unique(qs, return_index=True)
    ends = np.r_[starts[1:], qs.size]
    phi = (2.0 * np.pi * df * np.asarray(tmap, np.float32)).astype(np.float32)
    step = np.exp(1j * phi).astype(C64)
    out = np.empty(kspace.size, C64)
    qmin, qmax = int(occupied[0]), int(occupied[-1])
    ramp = np.exp(1j * (phi * np.float32(qmin))).astype(C64)
    j = 0
    for qv in range(qmin, qmax + 1):
        if qv != qmin and (qv - qmin) % 16 == 0:
            ramp = np.exp(1j * (phi * np.float32(qv))).astype(C64)
        if j < len(occupied) and int(occupied[j]) == qv:
            image = _ifft3(kspace * ramp).astype(C64).ravel()
            indices = order[starts[j]:ends[j]]
            out[indices] = image[indices]
            j += 1
        if qv < qmax:
            ramp *= step
    return out.reshape(kspace.shape)


def conjugate_phase(volume, field_hz, timing, *, df_hz=2.0, clip_hz=400.0):
    """CP correction with a supplied field; measured-reference runs use clip_hz=600."""
    vol, tm = _validate(volume, timing)
    field = np.asarray(field_hz)
    if field.shape != vol.shape or np.iscomplexobj(field) or not np.all(np.isfinite(field)):
        raise ValueError("field_hz must contain finite real Hz values on the volume grid")
    if not np.isfinite(df_hz) or df_hz <= 0 or not np.isfinite(clip_hz) or clip_hz <= 0:
        raise ValueError("df_hz and clip_hz must be positive")
    return _cp(vol, tm, field, df=float(df_hz), clip=float(clip_hz))


class _ResidualFit:
    def __init__(self, times):
        self.times = np.asarray(times, float)
        self.num = self.den = self.count = None
        self.used = []

    def add(self, r, increment, horizon):
        if self.num is None:
            self.num = np.zeros(increment.shape, np.float32)
            self.den = np.zeros(increment.shape, np.float32)
            self.count = np.zeros(increment.shape, np.float32)
        if self.times[r] < 1e-3:
            return
        alive = (horizon == r).astype(np.float32)
        dt = np.float32((self.times[r] - self.times[r - 1]) * 1e3)
        self.num += alive * increment * dt
        self.den += alive * (dt * dt)
        self.count += alive
        self.used.append(int(r))

    def result(self):
        field = -self.num / (np.float32(TWO_PI) * (self.den + np.float32(1e-30))) * np.float32(1e3)
        return to_zxy(field.astype(np.float32)), to_zxy(self.count)


def _pass(volume, timing, voxel_mm, cache_dir):
    bank = TimeBank(volume, timing, dt_ms=0.6)
    if bank.N != 16:
        raise ValueError(f"the reported workflow requires 16 shells; this timing map gives {bank.N}")
    table = noise_floor_table(timing, volume.shape, voxel_mm, bank.N, bank.t_max, cache_dir)
    sizes, _ = kernel_schedule(table)
    kernels = [None] + [Kern(size) for size in sizes[1:]]
    energy_kernel = Kern(kernel_from_mm((13.5, 4.9, 4.9), voxel_mm, 1.0, 19.5))
    noise_indices = _sample_idx(noise_region(volume))
    z0 = bank(0)
    reference = np.exp(-1j * np.angle(z0)).astype(C64)
    image = (z0 * reference).astype(C64)
    correction = reference.copy()
    phase = np.zeros(image.shape, np.float32)
    applied = AppliedField(image.shape)
    horizon = LockHorizon(image.shape)
    fit = _ResidualFit(bank.t_eff)
    thresholds = []
    for r in range(1, bank.N):
        shell = bank(r)
        residual = (shell * correction).astype(C64)
        increment, coherence = step_k(residual, kernels[r])
        correction = (correction * np.exp(-1j * increment)).astype(C64)
        image += shell * correction  # Sum the unrectified complex shell.
        phase += increment
        energy = energy_kernel((np.abs(shell) ** 2).astype(np.float32))
        applied.add(phase, energy, bank.t_eff[r] - bank.t_eff[0])
        threshold = float(np.quantile(coherence.ravel()[noise_indices], 0.99))
        horizon.add(r, coherence, threshold)
        fit.add(r, increment, horizon.L)
        thresholds.append(threshold)
    residual_field, count = fit.result()
    info = {"t_eff_ms": [float(t) * 1e3 for t in bank.t_eff],
            "kernel_schedule_zxy": [None] + [list(size) for size in sizes[1:]],
            "c2_noise_q99": thresholds, "rings_used": fit.used,
            "noise_floor_q99": table["q99"]}
    return to_zxy(image), to_zxy(applied.result()), residual_field, count, info


def _smooth_update(field, weight):
    field = np.clip(field, -60.0, 60.0).astype(np.float32)
    weight = np.asarray(weight, np.float32)
    weighted_field = field * weight
    numerator = np.zeros_like(field)
    denominator = np.zeros_like(field)
    for scale, alpha in zip((1, 3, 9), (1.0, 0.03, 0.001)):
        sigma = tuple(scale * x for x in (1.5, 4.0, 4.0))
        numerator += np.float32(alpha) * ndimage.gaussian_filter(weighted_field, sigma, mode="nearest")
        smoothed_weight = ndimage.gaussian_filter(weight, sigma, mode="nearest")
        if scale == 1:
            support = smoothed_weight >= np.float32(1e-3) * smoothed_weight.max()
        denominator += np.float32(alpha) * smoothed_weight
    epsilon = np.float32(1e-6) * denominator.max()
    return (numerator / (denominator + epsilon)).astype(np.float32), support


def correct(volume, timing, *, voxel_mm=None, cache_dir=None):
    """Return image, accumulated CP field, effective field and iteration info.

    The effective field adds the final applied shell-phase slope to the
    accumulated CP field. The latter is the map shown in Figure 5.
    """
    vol, tm = _validate(volume, timing)
    voxels = infer_voxel_mm(vol.shape, voxel_mm)
    field = np.zeros(vol.shape, np.float32)
    history = []
    previous_median, growing, previous_field = None, 0, None
    converged = diverged = False
    for k in range(3):
        corrected = vol if k == 0 else _cp(vol, tm, field)
        image, _, residual, count, info = _pass(corrected, tm, voxels, cache_dir)
        weight = np.abs(image).astype(np.float32) * count
        update, support = _smooth_update(residual, weight)
        step = (np.float32(1.0) * update).astype(np.float32)
        magnitude = np.abs(step[support])
        median = float(np.median(magnitude)) if magnitude.size else 0.0
        row = {"k": k, "step_abs_median_hz": median,
               "support_frac": float(support.mean()), **info}
        growing = growing + 1 if previous_median is not None and median > previous_median else 0
        if growing >= 2:
            diverged = True
            field = previous_field
            row["action"] = "revert previous update and stop"
            history.append(row)
            break
        previous_field = field
        field = (field + step).astype(np.float32)
        previous_median = median
        history.append(row)
        if median < 3.0:
            converged = True
            break
    # Every CP pass starts from vol, including the fresh final shell pass.
    corrected = _cp(vol, tm, field)
    image, shell_field, _, _, final_info = _pass(corrected, tm, voxels, cache_dir)
    effective = (field + shell_field).astype(np.float32)
    info = {**SETTINGS, "voxel_mm_zxy": voxels, "iterations": history,
            "converged": converged, "diverged": diverged, "final_pass": final_info,
            "effective_field": "accumulated CP field + final applied shell-phase slope"}
    return image, field, effective, info
