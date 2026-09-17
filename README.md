# ring-b0-correction — off-resonance correction for 3D cones MRI (Python ports of Brian Hargreaves' MATLAB)

Lean, dependency-light (numpy / scipy / scikit-image; dash + plotly for the
viewer) implementations of the hierarchical phase-correction methods in
`mribri999/timesegmentedphase`, one field-map-free autofocus baseline, two
metrics, and a slice viewer. Everything is 3D.

```bash
pip install -r requirements.txt
python run_methods.py --orig volume.npy [--tmap kxyz_time.npy]   # results/*.npy + labels.json
python run_mf_stack.py --orig volume.npy --tmap kxyz_time.npy    # results/mf_stack/ (Brian's MF recon at 31 frequencies)
python compute_metrics.py --orig volume.npy                      # results/metrics.json
python viewer.py --orig volume.npy --port 8053                   # http://127.0.0.1:8053/?z=42&a=ring_k9_N8&b=mf_stack&f=-60&zoom=217,327,644,754
python ring_b0_corr.py --selftest ; python time_seg_corr.py --selftest
```

`volume.npy` is a complex `[z, rows, cols]` image volume (coil-combined, gridded).
`kxyz_time.npy` is the 3D sample-time map on the same grid (seconds); without
it the shell times come from the radial profile `data/tkr_profile_brian.npz`.

| file | what |
|---|---|
| `ring_b0_corr.py` | radial-shell cascade (`ring_b0_corr.m` / `ts_b0_corr.m`) — `ring_b0_corr_matlab()` is the exact MATLAB equivalent; `ring_b0_corr_v2()` adds calibrated field maps and the predict-unwrap variant |
| `time_seg_corr.py` | `time_seg_corr.m` + `time_segment.m` (Brian's current method: windows in acquisition time) |
| `freq_seg_corr.py` | `freq_corr.m` (single-frequency recon) + `freq_seg_corr.m` (frequency-segmented per-voxel selection) |
| `autofocus.py` | 3D field-map-free autofocus, streamed |
| `metrics.py` | gradient entropy (R12) and normalised gradient energy (R13), 2D/3D |
| `run_methods.py`, `run_mf_stack.py`, `compute_metrics.py`, `viewer.py`, `examples/` | workflow, Brian's MF stack for the slider, metrics cache, Dash viewer (synced zoom, frequency slider, per-box metrics), view rendering, time-map resampling |

---

## 1. Conventions

* Volumes are `[z, rows, cols]`; the viewer's zoom box `y0,y1,x0,x1` means rows `y0:y1`, cols `x0:x1`. MATLAB's `[x, y, z]` is handled internally (axis 1 → x, axis 2 → y).
* k-space is the centred FFT of the image: $K(\mathbf{k}) = \mathcal{F}\{m\}$. The normalised k-radius is per axis, $|\mathbf{k}|^2 = (k_x/k_x^{\rm Nyq})^2 + (k_y/k_y^{\rm Nyq})^2 + (k_z/k_z^{\rm Nyq})^2$, so the sampled cones sphere is inscribed with its edge at $|\mathbf{k}|=1$ (the ring code puts that edge at 0.5; the conversion is internal via `kr_edge`).
* $t(\mathbf{k})$ is the acquisition time of the sample that landed at grid point $\mathbf{k}$ (0 → 8.5 ms for the breast protocol). It is only approximately radial (±16 % within a shell), which is why time windows ≠ radial shells.
* Gains: the ring cascade returns $1.08\times$ the input (overlapping Hamming shells sum to 1.080); `time_seg_corr` with overlap 8 returns ≈ 2.9×. Divide by the gain before comparing intensities.
* Masks: metrics use an Otsu tissue mask by default; `compute_metrics.py --mask` and `viewer.py --mask` take a boolean `[z, rows, cols]` volume instead — for the breast case a crude anatomical mask (Otsu tissue anterior to a slice-dependent chest-wall row, columns 40–730) restricts everything to breast tissue and excludes heart, chest wall and axilla.

## 2. The problem

With off-resonance $\Delta f(\mathbf{r})$ the acquired signal is

$$
K(\mathbf{k}) = \int m(\mathbf{r})\, e^{-i 2\pi \Delta f(\mathbf{r})\, t(\mathbf{k})}\, e^{-i 2\pi \mathbf{k}\cdot\mathbf{r}}\, d\mathbf{r}.
$$

Because $t$ grows with $|\mathbf{k}|$ along a centre-out cones readout, a voxel at frequency $\Delta f$ acquires the phase $2\pi\Delta f\,t(\mathbf{k})$ across k-space — a radially varying phase ramp, i.e. a blur whose width scales with $\Delta f \cdot T$ ($T$ = readout duration). Every method below tries to remove $e^{-i2\pi\Delta f(\mathbf{r}) t(\mathbf{k})}$ without a separately acquired field map.

## 3. Ring cascade — `ring_b0_corr.m` / `ts_b0_corr.m` (method A)

**Shells.** $N$ spherical shells in normalised k-radius with centre-to-centre increment $\Delta = 0.5/(N-1)$; shell $n$ spans $[(n-1)\Delta, (n+1)\Delta)$ with a Hamming window

$$
W_n(\mathbf{k}) = 0.54 - 0.46\cos\!\Big(\pi\,\frac{|\mathbf{k}| - (n-1)\Delta}{\Delta}\Big),
$$

1.0 at the shell centre, 0.08 at its edges, 50 % overlap with each neighbour, and $\sum_n W_n = 1.080$ everywhere inside the sphere. Shell images $m_n = \mathcal{F}^{-1}\{W_n K\}$ are band-pass views of the anatomy acquired around the shell time $t_n$ (mean $t(\mathbf{k})$ over the shell; for $N=8$: 0.3, 0.8, 1.9, 3.4, 4.8, 6.1, 7.4, 8.2 ms).

**Why the phase of a shell image is usable.** For a real-valued object a band-pass image is real, so its structural phase is 0 or π. Off-resonance adds $2\pi\Delta f\, t_n$. Hence, relative to the previous shell, the phase of $m_n$ is
$\{0,\pi\} + 2\pi\Delta f\,(t_n - t_{n-1})$ — a sign ambiguity plus a small coherent increment.

**The cascade** (accumulated phase $\Phi_n$, kernel $G$):

$$
\begin{aligned}
\Phi_0 &= \angle m_0 \quad (\text{per voxel, unsmoothed}) \\
r_n &= m_n\, e^{-i\Phi_{n-1}} && \text{(residual after removing earlier shells' phase)}\\
s_n &= \mathrm{sgn}\!\big(\mathrm{Re} r_n\big)\, r_n && \text{(the +-pi/2 rule: rectify the 0/pi ambiguity)}\\
\Delta\phi_n &= \angle\,(G * s_n) && \text{(magnitude-weighted phasor average over a neighbourhood)}\\
\Phi_n &= \Phi_{n-1} + \Delta\phi_n,\qquad \hat m = \sum_n m_n\, e^{-i\Phi_n}.
\end{aligned}
$$

So the cascade is a conjugate-phase (time-segmented) correction whose per-segment phase is *estimated from the data itself*: the shells play the role of echoes at times $t_n$, the rectification removes the structural sign, and the smoothing kernel is the smooth-field prior. $G$ in MATLAB is `fspecial3('gaussian', s³, s/2)` — a Gaussian **truncated to an $s^3$ box**, effective std ≈ 1.9 voxels ($s=7$, `ring_b0_corr.m`) or 2.4 ($s=9$, `ts_b0_corr.m`). (`scipy.ndimage.gaussian_filter(σ=4.5)`, used in earlier Python work, is a full Gaussian twice as wide; on the breast volume the corrected images differ by 0.000 px in edge width.)

**Limits that follow from the construction.**
* *Frequency cap.* Rectification around zero assumes $|2\pi\Delta f\,\Delta t_n| < \pi/2$, i.e. $|\Delta f| < 1/(4\,\Delta t_{\max})$. For $N=8$ the largest consecutive step is 1.5 ms → nominal 166 Hz; on synthetic uniform fields the correction is clean to ~110 Hz and fails by 160 Hz. $N=16$ halves $\Delta t_{\max}$ (cap ~330 Hz) at the price of thinner, noisier shells.
* *Coherence floor.* The phasor average $|G*s_n| / (G*|s_n|)$ equals $2/\pi = 0.637$ when the residual phase is uniformly random (folded into ±π/2): a shell at that floor carries no readable phase. In vivo, shells ≥ 3 of $N=8$ sit at the floor in most tissue — the cascade effectively corrects with shells 0–2 and applies ~0 to the rest, which is the safe behaviour.
* *Footprint rule.* A single per-voxel frequency cannot flatten a field that varies by more than half a cycle across the PSF footprint: correction is complete where $|\nabla B_0|\cdot w_{\rm PSF} \lesssim 1/(2T)$ (59 Hz for $T$ = 8.5 ms) and degrades beyond.

**Field maps from the cascade.** With shell times $t_n$ the increments are $\Delta\phi_n \approx -2\pi f\,\Delta t_n$ (k-space phase $e^{-i2\pi f t}$ convention). `ring_b0_corr_v2` returns

$$
\hat f_{\rm ls} = -\frac{\sum_n w_n\,\Delta\phi_n\,\Delta t_n}{2\pi\sum_n w_n\,\Delta t_n^2},
$$

with $w_n = 1$ (`ls_increment`) or $w_n = c_n^2$, $c_n$ the shell's phasor coherence (`ls_weighted`); both read 1.01× a synthetic uniform field. `ring_b0_corr_matlab` returns the field map exactly as `ts_b0_corr.m` accumulates it — per shell `ex_field_map += angle(image_copy .* corr_phase) / (ring_time(n) − ring_time(n−1))`, i.e. $\angle\big((G*s_n)\,e^{-i\Phi_{n-1}}\big)/\Delta t_n$, then divided by $(N-1)\,2\pi$ — reproduced literally; note it reads 1.97× the synthetic field with MATLAB's sign (the second multiplication by the accumulated phase; marked "still being tested" in the source). Shell-derived maps are blind above the cap: a true 160 Hz reads as ~30 Hz.

## 4. Predict-unwrap variant (method B, `unwrap="predict"`)

Identical shells, kernel and combination; only the rectification reference changes. From the previous shells' increments a running, coherence-weighted phase slope $\hat s = \sum w\,\Delta\phi\,\Delta t / (\sum w\,\Delta t^2 + \lambda)$ predicts this shell's increment $p_n = \hat s\,\Delta t_n$, and the sign is rectified *around the prediction*:

$$
s_n = \mathrm{sgn}\!\big(\mathrm{Re}(r_n e^{-ip_n})\big)\, r_n e^{-ip_n},\qquad \Delta\phi_n = p_n + \angle(G * s_n).
$$

This is temporal phase unwrapping as in multi-echo field mapping: the ±π/2 limit now applies to the *prediction error* rather than to the increment itself, so a uniform off-resonance is no longer capped (synthetic: 160 Hz → NRMSE 0.050 vs 0.179 for A; 200 Hz → 0.079 vs 0.256). The ridge $\lambda$ shrinks the prediction to zero where the accumulated coherent evidence is weak, so in incoherent regions B reduces to A.

**Why B is not shipped as the default.** In vivo the late shells sit at the coherence floor, so the prediction extrapolates from shells 1–2 into shells that cannot confirm or refute it. Where the field varies steeply (skin, chest wall) the smoothed prediction is wrong by more than π/2 over part of the boundary, the rectification locks onto the wrong branch, that shell is added with a π error, and the edge acquires a sidelobe — the doubled skin contour at high $|B_0|$. Fixed-edge widths still favour B (tie below 60 Hz, −0.08 px at 60–90 Hz) because a width measure cannot see a ghost; the images can. B is useful as an uncapped field-map estimator (its map reads a physically plausible p95 ≈ 120 Hz where A's is capped at ~30 Hz) and as a diagnostic, not as the correction.

## 5. Time-window cascade — `time_seg_corr.m` (Brian's current method)

Same cascade, but the segments are windows in acquisition time using the 3D map $t(\mathbf{k})$: $N$ centres $t_s = (s-1)\,t_{\max}/(N-1)$, half-width $h = \text{spacing}\cdot\text{overlap}/2$,

$$
W_s(\mathbf{k}) = 0.54 + 0.46\cos\!\Big(\pi\,\frac{t(\mathbf{k}) - t_s}{h}\Big)\ \ \text{for } |t(\mathbf{k}) - t_s| < h,
$$

segment images $\mathcal{F}^{-1}\{W_s K\}$, then steps identical to §3 with kernel size 7; the field map is the first interval only. With overlap 2 the windows sum to 1.080 and the method is the ring cascade in time units (on Brian's own sample: $N=8$ identical to the ring, $N=16$ a tie). With **overlap 8** — the value committed in `time_seg_corr.m` — each window spans ±4 spacings, the window sum runs 3.3 → 4.3 → 2.7 across the readout, and the output carries a mid-k boost that reads as sharpening but widens the median edge and lights up the high-frequency artifact guards. Which overlap produced the "16 segments best" images in the log is an open question for Brian.

## 6. Multi-frequency reconstruction — `freq_corr.m` / `freq_seg_corr.m`

`freq_corr`: reconstruct at one offset frequency exactly, $m_f = \mathcal{F}^{-1}\{K\,e^{+i2\pi f\,t(\mathbf{k})}\}$ — the building block of the multifrequency stack.
`freq_seg_corr` (as committed): remove the low-res phase of the *uncorrected* image once, reconstruct at each $f$ in a list, low-pass the **signed** imaginary part on a patch with a 5³ kernel, and pick per voxel `min(patch_imag, [], 4)`. Two consequences observed on the breast skin-edge patch: minimising the signed imaginary part selects the ends of the frequency range (±150 Hz for 61 % of tissue voxels), and the single reference makes flat tissue vote for 0 Hz. The port is literal so the behaviour can be discussed; §7 is the same idea with both points changed.

### 6a. The multifrequency stack in the viewer (`run_mf_stack.py`)

Brian's manual workflow — reconstruct at many frequencies and tune the frequency
while looking at a tissue or patch — is reproduced exactly: `run_mf_stack.py`
runs `freq_corr` in full 3D at every frequency on a grid (default −150…+150 Hz
in 10 Hz steps, 31 volumes) and stores each $|m_f|$; the viewer exposes the
stack as the method `freq_corr.m at slider frequency` with a **frequency
slider**, a zoom box that defines the patch, per-box gradient entropy / energy
for the current frequency, and a *scan* button that tabulates both ratios for
every frequency inside the box. Views are shareable:
`?z=42&a=ring_k9_N8&b=mf_stack&f=-60&zoom=217,327,644,754`. Tuned per patch by
eye this is the per-patch oracle; the scan table is only an aid (gradient
concentration also rises with noise and texture). Storage: 264 MB per
frequency (float32 magnitude; `--complex` keeps the phase at 2× disk).

## 7. Field-map-free autofocus, 3D (`autofocus.py`)

For each candidate frequency $f$ in a list: $m_f = \mathcal{F}^{-1}\{K e^{+i2\pi f t}\}$, referenced to **its own** low-pass phase $\psi_f = \angle\mathcal{F}^{-1}\{L\,K e^{+i2\pi f t}\}$ ($L$ = $|\mathbf{k}|\le 0.1$), and the focus criterion

$$
c_f(\mathbf{r}) = \big\langle\, |\mathrm{Im}(m_f e^{-i\psi_f})|\,\big\rangle_{\rm window}.
$$

A real object demodulated at its true frequency is real after removing the low-res phase; residual off-resonance leaves odd-symmetric imaginary sidelobes at edges, so $c_f$ is minimised at the true $f$ — but only where there are edges. The per-voxel argmin is therefore propagated into flat tissue by normalised-convolution smoothing with weight (criterion range over $f$) × (low-res magnitude), and a second pass assembles $\hat m(\mathbf{r}) = m_{f^*(\mathbf{r})}(\mathbf{r})$ so the candidate stack is never held in memory. Synthetic uniform +40/+90/+130 Hz → NRMSE 0.000–0.003, no frequency cap. On the breast volume it ties the ring cascade at every $|B_0|$ level and loses to it by +0.6 px on the extreme skin edges — consistent with the known ceiling of per-voxel *selection* (off-resonance blur is additive; a selected candidate cannot subtract its neighbours' out-of-focus contribution).

## 8. Metrics (`metrics.py`)

With $h(\mathbf{r}) = |\nabla I| / \sum_{\rm mask} |\nabla I|$ over an Otsu tissue mask:

$$
R_{12} = -\sum h \log_2 h \quad(\text{gradient entropy, max } \log_2 n),\qquad R_{13} = \sum h^2 \quad(\text{normalised gradient energy, min } 1/n),
$$

reported as ratios vs the original — $r_{12} = (R_{12}^{\max}-R_{12})/(R_{12}^{\max}-R_{12}^{\rm orig})$, $r_{13} = (R_{13}-R_{13}^{\min})/(R_{13}^{\rm orig}-R_{13}^{\min})$ — in 2D per slice and in 3D. They are gain-invariant and measure gradient *concentration*; a Gaussian blur of a noisy image raises both (it suppresses many small noise gradients more than the few coherent edge gradients), so they are an artifact-reading aid next to the image, not a sharpness score. Edge sharpness in our evaluation is the 10–90 % edge-spread width at edges fixed on a reference, with paired differences and bootstrap CIs (kept in the analysis repo, not here).

## 9. Results on the breast volume (5336 fixed edges, paired vs `ts_b0_corr.m` N=8)

| method | median edge width | paired vs A by \|B0\| bin (px; * CI excludes 0) | guards |
|---|---|---|---|
| uncorrected | 6.43 | | |
| **A: ts_b0_corr.m N=8, kernel 9** | 5.96 | — | clean |
| ring_b0_corr.m N=8, kernel 7 | 5.94 | 0.000 in every bin | clean |
| ts_b0_corr.m N=16 | 6.01 | +0.02…+0.05* below 60 Hz, −0.10 at 130–200 | clean |
| B: predict unwrap | — | 0.00 / +0.01 / +0.01 / −0.08* / −0.10 / −0.32 (n.s.) | ghosted skin edge |
| autofocus 3D | 5.76 | −0.01…−0.11, all CIs include 0 (tie); +0.64* on 82 extreme skin edges | clean, gain 1.0 |
| time_seg_corr N=16 (overlap 8, resampled time map) | 6.85 | +0.3…+1.1* above 60 Hz | HF boost, gain 2.96 |
| freq_seg_corr (patch) | 7.14 (patch) | +1.05* on the skin patch | range-end selections |

On Brian's own sample (`sampledata.mat`, exact time map, 995 edges): ring N=8 5.07 px; time_seg_corr N=8/N=16 with overlap 2: 5.04 / 5.14 (tie); overlap 8: 5.60 with gain 2.9 and r_corner 0.88.

## 10. 1:1 status against the MATLAB, and what would close it

Reproduced by exact primitives: shells, Hamming windows, cascade order, ±π/2 rule, `fspecial3` (separable, 1e-7), `imfilter 'replicate'`, `ft3`/`ift3` (identical for even sizes), plain-mean shell times, the literal `ts_b0_corr.m` field-map accumulation, the `time_segment` window and `freq_seg_corr` selection; regression vs the original Python translation 3.5e-7; complex64 here vs double in MATLAB (~1e-6 expected). **Not done:** a numerical diff against MATLAB output (no license here). Kit: `check_1to1.m` (runs the three MATLAB functions on `sampledata.mat`/`time_kxyz.mat`, ≈2 min) + `diff_1to1.py` against saved Python references for identical inputs. Two things only Brian can confirm: the `sample_kradius` units he passes to `ts_b0_corr`, and the `overlap` used for the "16 segments best" result.

The 3D sample-time map for the 768×768×112 breast grid was resampled from the 256³ `time_kxyz.mat` in per-axis-normalised k (radial medians agree with the profile to 0.05 ms); the exact map should come from the scan's own `kacq_cone` trajectory files.
