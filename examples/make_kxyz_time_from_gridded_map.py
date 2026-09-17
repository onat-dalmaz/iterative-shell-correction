"""Build a 3D sample-time map for the real 768×768×112 grid by resampling
Brian's gridded samp_times (brian_ref/tsphase/time_kxyz.mat, 256³, per-axis
normalised k to ±0.5 inverse pixels) in per-axis-normalised k-coordinates.

APPROXIMATION, flagged: this assumes the 11Nov25 breast acquisition uses the
same cones trajectory design as Brian's sample (Feb-3 dataset, 0.5×0.5×1.5 mm).
The exact map for our volume would come from its own kacq_cone trajectory
files on the cluster. Sanity: the radial median of the resampled map must
match data/tkr_profile_brian.npz (printed).

Output: data/kxyz_time_768x768x112.npy  (float32 seconds, [z, x, y] order)
"""
from __future__ import annotations

import numpy as np
import scipy.io as sio
from scipy.ndimage import map_coordinates
from scipy.interpolate import interp1d

t256 = sio.loadmat("brian_ref/tsphase/time_kxyz.mat")["samp_times"].astype(np.float32)  # [x,y,z] 256³
n = t256.shape[0]
NZ, NY, NX = 112, 768, 768                     # target [z, x, y] as stored in im_grid_ezxy.npy


def norm_axis(N):
    # per-axis normalised k: index i → (i − N/2)/(N/2) ∈ [−1, 1)
    return (np.arange(N) - N // 2) / (N / 2)


# target normalised coordinates → source fractional indices: src = k_norm*(n/2) + n/2
kx = norm_axis(NX) * (n / 2) + n / 2
ky = norm_axis(NY) * (n / 2) + n / 2
kz = norm_axis(NZ) * (n / 2) + n / 2
out = np.empty((NZ, NX, NY), np.float32)
X, Y = np.meshgrid(kx, ky, indexing="ij")     # [x, y]
for iz, z in enumerate(kz):
    coords = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)])
    out[iz] = map_coordinates(t256, coords, order=1, mode="nearest").reshape(NX, NY)
np.save("data/kxyz_time_768x768x112.npy", out)

# sanity vs the radial profile
prof = np.load("data/tkr_profile_brian.npz"); raw = prof["kr"]; tp = prof["t_s"]
o = np.argsort(raw); tf = interp1d(raw[o], tp[o], bounds_error=False, fill_value=(tp[o][0], tp[o][-1]))
kxn, kyn, kzn = norm_axis(NX), norm_axis(NY), norm_axis(NZ)
kr = np.sqrt(kzn[:, None, None] ** 2 + kxn[None, :, None] ** 2 + kyn[None, None, :] ** 2)
print(f"resampled map: min {out.min()*1e3:.3f} max {out.max()*1e3:.3f} ms; shape {out.shape}")
for a, b in ((0.05, 0.1), (0.2, 0.25), (0.45, 0.5), (0.7, 0.75), (0.95, 1.0)):
    m = (kr >= a) & (kr < b)
    print(f"  kr {a:.2f}-{b:.2f}: resampled median {np.median(out[m])*1e3:.3f} ms "
          f"(p5 {np.percentile(out[m],5)*1e3:.2f}, p95 {np.percentile(out[m],95)*1e3:.2f})  "
          f"profile {float(tf((a+b)/2))*1e3:.3f} ms")
