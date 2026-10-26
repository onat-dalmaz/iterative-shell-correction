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

Inputs are coil-combined, gridded complex `.npy` volumes `[z, x, y]`, timing in
seconds on the same centered k-space grid, and measured fields in Hz. Voxel
spacings are in millimetres. Scoring uses fixed masks and saved measured-field CP
references; see `INPUTS.txt`.

[Example downloads](https://github.com/onat-dalmaz/iterative-shell-correction/releases/tag/ismrm-example-data-v1)
provide the standard-autoshim phantom and volunteer S03. Download both ZIPs into
`example-data/`, then extract them there. Use their bundled manifests:

```bash
python -m zipfile -e example-data/example-phantom-standard.zip example-data
python -m zipfile -e example-data/example-volunteer-S03.zip example-data
python experiments.py run phantom --inputs example-data/phantom.json --data example-data --case standard_autoshim --out outputs/phantom-example
python experiments.py run volunteers --inputs example-data/volunteers.json --data example-data --case S03 --out outputs/volunteer-example
```

The complete comparisons use three phantom conditions and five volunteers. Their
remaining scan arrays and prepared image tiles are not included:

```bash
python experiments.py run phantom --inputs settings/phantom.json --data /path/to/data --out outputs/phantom
python experiments.py score phantom --inputs settings/phantom.json --data /path/to/data --results outputs/phantom --out outputs/phantom_scores
python experiments.py run volunteers --inputs settings/volunteers.json --data /path/to/data --out outputs/volunteers
python experiments.py score volunteers --inputs settings/volunteers.json --data /path/to/data --results outputs/volunteers --out outputs/scores
python plot_figures.py --figures 4 --out outputs/figures
python plot_figures.py --figures 2 3 5 --assets /path/to/saved_tiles --out outputs/figures
```

Runs write corrected complex volumes, fields and settings under `--out`. Figure 4
plots the included scalar results without scan data. Full-cohort scoring requires
all cases; `--case` selects only a correction or input check.

The phase-tracking step builds on Brian Hargreaves' [hierarchical shell
correction](https://github.com/mribri999/timesegmentedphase).
