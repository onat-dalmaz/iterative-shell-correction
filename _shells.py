"""Time-segmented shell filtering used by iterative shell correction.

Input volumes and timing maps are [z,x,y]; shell phase tracking uses [x,y,z].
The truncated Gaussian and complex sign rule follow Brian Hargreaves' MATLAB.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import numpy as np
from scipy import ndimage
from autofocus import _fft3 as af_fft3, _ifft3 as af_ifft3, kr_norm_zxy
C64 = np.complex64
TWO_PI = 2.0 * np.pi
_POOL = None
_POOL_LOCK = threading.Lock()

def _gaussian_kernel(kernelsize: int) -> np.ndarray:
    # MATLAB fspecial3: sigma=n/2, truncated to the n-voxel support.
    h = (kernelsize - 1) // 2
    x = np.arange(-h, h + 1, dtype=np.float64)
    s = kernelsize / 2.0
    g1 = np.exp(-x ** 2 / (2 * s ** 2))
    g = g1[:, None, None] * g1[None, :, None] * g1[None, None, :]
    return (g / g.sum()).astype(np.float32)

def to_xyz(a):
    return np.ascontiguousarray(np.transpose(a, (1, 2, 0)))

def to_zxy(a):
    return np.ascontiguousarray(np.transpose(a, (2, 0, 1)))

def rect(ic):
    return np.where(ic.real < 0, -ic, ic).astype(C64)

def noise_region(vol_zxy):
    # Scan-specific low-signal proxy: bottom tenth after local averaging.
    mag = ndimage.uniform_filter(np.abs(np.asarray(vol_zxy)).astype(np.float32), size=(1, 5, 5))
    nz = mag > 0
    thr = np.percentile(mag[nz][::max(1, int(nz.sum() // 2000000))], 10)
    return to_xyz(nz & (mag <= thr))

def n_threads():
    for k in ('RING_PLUS_THREADS', 'SLURM_CPUS_PER_TASK', 'OMP_NUM_THREADS'):
        v = os.environ.get(k)
        if v and v.isdigit() and (int(v) > 0):
            return int(v)
    return os.cpu_count() or 1

def _pool():
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = ThreadPoolExecutor(max(1, n_threads()))
    return _POOL

def odd_round(v):
    return max(1, 2 * int(math.floor((float(v) - 1.0) / 2.0 + 0.5)) + 1)

def odd_floor(v):
    return max(1, 2 * int(math.floor((float(v) - 1.0) / 2.0)) + 1)

def kernel_from_mm(kern_mm_zxy, voxel_mm_zxy, scale=1.0, z_cap_mm=None):
    n = [odd_round(scale * k / v) for k, v in zip(kern_mm_zxy, voxel_mm_zxy)]
    if z_cap_mm is not None:
        n[0] = min(n[0], odd_floor(z_cap_mm / voxel_mm_zxy[0]))
    return tuple((int(x) for x in n))

@lru_cache(maxsize=None)
def _k1(n):
    k = _gaussian_kernel(int(n)).sum(axis=(1, 2)).astype(np.float64)
    return k / k.sum()

def _conv1d(a, k1, ax):
    T = n_threads()
    other = max((d for d in range(a.ndim) if d != ax), key=lambda d: a.shape[d])
    n = a.shape[other]
    if T <= 1 or n < 2 or a.size < 200000:
        return ndimage.convolve1d(a, k1, axis=ax, mode='nearest')
    nb = min(T, n)
    b = np.linspace(0, n, nb + 1).astype(int)
    out = np.empty_like(a)

    def work(i):
        sl = [slice(None)] * a.ndim
        sl[other] = slice(b[i], b[i + 1])
        sl = tuple(sl)
        out[sl] = ndimage.convolve1d(a[sl], k1, axis=ax, mode='nearest')
    list(_pool().map(work, range(nb)))
    return out

def smooth_aniso(x, sizes_xyz):
    sizes = tuple((int(s) for s in sizes_xyz))
    if len(sizes) != np.ndim(x):
        raise ValueError(f'sizes {sizes} vs ndim {np.ndim(x)}')

    def sep(a):
        out = a.astype(np.float32, copy=True)
        for ax in range(a.ndim):
            if sizes[ax] > 1:
                out = _conv1d(out, _k1(sizes[ax]), ax)
        return out
    if np.iscomplexobj(x):
        return (sep(x.real) + 1j * sep(x.imag)).astype(C64)
    return sep(x)

class Kern:

    def __init__(self, sizes_zxy):
        self.zxy = tuple((int(s) for s in sizes_zxy))
        self.xyz = (self.zxy[1], self.zxy[2], self.zxy[0])

    def __call__(self, a):
        return smooth_aniso(a, self.xyz)

def step_k(residual, kernel):
    magnitude = np.abs(residual).astype(np.float32)
    smooth_magnitude = kernel(magnitude).astype(np.float32)
    doubled = (residual * residual / (magnitude + np.float32(1e-30))).astype(C64)
    coherence = (np.abs(kernel(doubled)) / (smooth_magnitude + np.float32(1e-30))).astype(np.float32)
    np.clip(coherence, 0, 1, out=coherence)
    increment = np.angle(kernel(rect(residual))).astype(np.float32)
    return increment, coherence


class TimeBank:
    def __init__(self, volume, timing, dt_ms=0.6, n=None, t_max=None):
        vol = np.asarray(volume)
        tm = np.asarray(timing, np.float32)
        if vol.shape != tm.shape:
            raise ValueError("volume and timing must share the same [z,x,y] grid")
        self.K = af_fft3(vol.astype(C64)).astype(C64)
        inside = kr_norm_zxy(self.K.shape) <= 1.0
        self.t_max = float(tm[inside].max()) if t_max is None else float(t_max)
        self.t = np.where(inside, tm, np.float32(self.t_max)).astype(np.float32)
        self.dt = float(dt_ms) * 1e-3
        self.N = int(n) if n is not None else int(math.ceil(self.t_max / self.dt)) + 1
        nz, nx, ny = self.K.shape
        self.shape = (nx, ny, nz)
        total = np.zeros(self.K.shape, np.float32)
        for r in range(self.N):
            total += self._raw(r)
        self.inv = np.where(total > 0, 1.0 / np.maximum(total, np.float32(1e-12)), 0).astype(np.float32)
        # Effective times use each shell's energy, not just its window center.
        energy = (np.abs(self.K) ** 2).astype(np.float32) * inside
        self.t_eff = np.zeros(self.N)
        for r in range(self.N):
            weighted_energy = self.window(r) * energy
            denominator = float(weighted_energy.sum(dtype=np.float64))
            self.t_eff[r] = float((weighted_energy * self.t).sum(dtype=np.float64) / max(denominator, 1e-300))

    def _raw(self, r):
        offset = (self.t - np.float32(r * self.dt)) / np.float32(self.dt)
        window = np.zeros(self.t.shape, np.float32)
        inside = np.abs(offset) < 1
        window[inside] = 0.54 + 0.46 * np.cos(np.pi * offset[inside])
        return window

    def window(self, r):
        return (self._raw(r) * self.inv).astype(np.float32)

    def __call__(self, r):
        return to_xyz(af_ifft3(self.K * self.window(r)).astype(C64))


def _md5(a):
    return hashlib.md5(np.ascontiguousarray(a).data).hexdigest()

def reduced_stride(shape_zxy):
    _, nx, ny = shape_zxy
    if nx > 384 and nx == ny:
        s = nx // 256
        if s > 1 and nx % s == 0:
            return s
    return 1

class AppliedField:

    def __init__(self, shape):
        self.num = np.zeros(shape, np.float32)
        self.den = np.zeros(shape, np.float32)

    def add(self, C, w, tau_s):
        tau = np.float32(tau_s * 1000.0)
        self.num += w * C * tau
        self.den += w * (tau * tau)

    def result(self):
        tiny = np.float32(1e-30)
        return (-self.num / (np.float32(TWO_PI) * (self.den + tiny)) * np.float32(1000.0)).astype(np.float32)

class LockHorizon:

    def __init__(self, shape):
        self.alive = np.ones(shape, bool)
        self.L = np.zeros(shape, np.uint8)

    def add(self, r, c2, thr):
        # A voxel stays eligible only while every preceding shell passed.
        self.alive &= c2 > np.float32(thr)
        self.L[self.alive] = np.uint8(min(r, 255))

def _sample_idx(m, cap=2000000):
    idx = np.flatnonzero(m.ravel())
    return idx[::max(1, idx.size // cap)] if idx.size else idx

def infer_voxel_mm(shape_zxy, voxel_mm=None):
    if voxel_mm is not None:
        v = tuple((float(x) for x in voxel_mm))
        if len(v) != 3 or any((not np.isfinite(x) or x <= 0 for x in v)):
            raise ValueError('voxel_mm must contain three positive [z,x,y] spacings in mm')
        return list(v)
    _, nx, ny = shape_zxy
    if nx == ny and nx in (384, 768):
        xy = {768: 0.547, 384: 1.094}[nx]
        return [1.5, xy, xy]
    raise ValueError('provide voxel_mm=(z,x,y) for this input grid')

def kernel_schedule(table):
    kernels = [tuple(k) for k in table['kernels_zxy']]
    schedule, indices = ([None], [None])
    for r in range(1, table['n_rings']):
        ok = [i for i, value in enumerate(table['q99'][r]) if value <= 0.5]
        index = ok[0] if ok else len(kernels) - 1
        schedule.append(kernels[index])
        indices.append(index)
    return (schedule, indices)

def noise_floor_table(tmap_zxy, shape_zxy, voxel_mm_zxy, n, t_max, cache_dir=None):
    # Two fixed complex-noise draws select one smoothing kernel per shell.
    dt_ms = 0.6
    kern_mm = (13.5, 4.9, 4.9)
    kscale = (1.0, 1.44, 1.89, 2.33)
    z_cap_mm = 19.5
    seeds = (0, 1)
    q = 0.99
    tmap_md5 = _md5(np.asarray(tmap_zxy, np.float32))
    stride = reduced_stride(shape_zxy)
    kernels = [kernel_from_mm(kern_mm, voxel_mm_zxy, s, z_cap_mm) for s in kscale]
    key = {'v': 1, 'tmap_md5': tmap_md5, 'shells': 'time', 'dt_ms': dt_ms, 'N': None, 'n': int(n), 't_max': round(float(t_max), 9), 'grid_zxy': [int(x) for x in shape_zxy], 'stride': stride, 'kernels_zxy': [list(k) for k in kernels], 'seeds': list(seeds), 'q': q, 'support': 'kr_norm<=1', 'margin': 'half kernel'}
    h = hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    path = os.path.join(cache_dir, f'nf_time_{h}.json') if cache_dir else None
    if path and os.path.exists(path):
        with open(path) as handle:
            table = json.load(handle)
        if table.get('key') == key:
            return table
    tm = np.ascontiguousarray(np.asarray(tmap_zxy, np.float32)[:, ::stride, ::stride])
    shape = tm.shape
    inside = kr_norm_zxy(shape) <= 1.0
    banks = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        noise_k = (rng.standard_normal(shape, dtype=np.float32) + 1j * rng.standard_normal(shape, dtype=np.float32)).astype(C64)
        noise_k *= inside
        noise = af_ifft3(noise_k).astype(C64)
        banks.append(TimeBank(noise, tm, dt_ms=dt_ms, n=n, t_max=t_max))
    q99 = [[None] * len(kernels)]
    for r in range(1, n):
        shells = [bank(r) for bank in banks]
        magnitudes = [np.abs(z).astype(np.float32) for z in shells]
        doubled = [(z * z / (m + np.float32(1e-30))).astype(C64) for z, m in zip(shells, magnitudes)]
        row = []
        for sizes in kernels:
            kernel = Kern(sizes)
            hx, hy, hz = (s // 2 for s in kernel.xyz)
            interior = (slice(hx, magnitudes[0].shape[0] - hx), slice(hy, magnitudes[0].shape[1] - hy), slice(hz, magnitudes[0].shape[2] - hz))
            values = []
            for magnitude, phase in zip(magnitudes, doubled):
                coherence = (np.abs(kernel(phase)) / (kernel(magnitude) + np.float32(1e-30))).astype(np.float32)
                values.append(coherence[interior].ravel())
            if not any((v.size for v in values)):
                raise ValueError('grid is smaller than the noise-calibration kernel footprint')
            row.append(float(np.quantile(np.concatenate(values), q)))
        q99.append(row)
    table = {'key': key, 'q99': q99, 'kernels_zxy': [list(k) for k in kernels], 'kscale': list(kscale), 'reduced_grid_zxy': list(shape), 'n_rings': n}
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        temporary = f'{path}.{os.getpid()}.tmp'
        with open(temporary, 'w') as handle:
            json.dump(table, handle, indent=1)
        os.replace(temporary, path)
    return table
