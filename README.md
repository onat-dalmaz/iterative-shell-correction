# Iterative shell correction

Code for iterative shell-based off-resonance correction in center-out MRI. The
method estimates small field updates from phase changes between k-space shells,
then removes the remaining shell phase to reduce blurring without a separate
field-map scan.

Runs use 16 normalized Hamming shells spaced by 0.6 ms and three field updates,
limited to ±60 Hz before smoothing. Each CP correction starts from the original
complex volume. The final step demodulates all 16 unrectified shells.

Use Python 3.10+ with NumPy, SciPy and resvg_py:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The manifests in `settings/` describe the three phantom conditions and five
volunteers. Supply coil-combined, gridded complex `.npy` volumes as `[z, x, y]`,
timing in seconds on the same centered k-space grid, and measured fields in Hz.
Voxel spacings are in millimetres. Scoring also needs the fixed private masks
and saved measured-field CP references listed in each manifest.

```bash
python experiments.py run phantom --inputs settings/phantom.json --data /path/to/private --out outputs/phantom
python experiments.py score phantom --inputs settings/phantom.json --data /path/to/private --results outputs/phantom --out outputs/phantom_scores
python experiments.py run volunteers --inputs settings/volunteers.json --data /path/to/private --out outputs/volunteers
python experiments.py score volunteers --inputs settings/volunteers.json --data /path/to/private --results outputs/volunteers --out outputs/scores
python plot_figures.py --figures 4 --out outputs/figures
python plot_figures.py --figures 2 3 5 --assets /path/to/private_plot_assets --out outputs/figures
```

Runs write complex corrected volumes, field maps and settings to the requested
directory. Figure 4 can be replotted from the included scalar results; the image
figures need private plotting assets. Scan data, image patches and masks are not
distributed here.

The phase-tracking step builds on Brian Hargreaves' [hierarchical shell
correction](https://github.com/mribri999/timesegmentedphase).
