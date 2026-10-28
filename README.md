# Iterative shell correction

Off-resonance correction for center-out MRI without a separate field-map scan.

**Phantom**

![Phantom: uncorrected, iterative shell correction and measured-field CP reference](assets/phantom-example.png)

**Breast patches**

![Breast patches from central and inferior slices: uncorrected, iterative shell correction and measured-field CP reference](assets/breast-examples.png)

Requires Python 3.10–3.12. NumPy and SciPy are pinned to the experiment versions.

```bash
python -m pip install -r requirements.txt
```

[Example downloads](https://github.com/onat-dalmaz/iterative-shell-correction/releases/tag/ismrm-example-data-v1)
include one phantom and one volunteer. Save both ZIPs in `example-data/`, then run:

```bash
python -m zipfile -e example-data/example-phantom-standard.zip example-data
python -m zipfile -e example-data/example-volunteer-S03.zip example-data
python experiments.py run phantom --inputs example-data/phantom.json --data example-data --case standard_autoshim --out outputs/phantom-example
python experiments.py score phantom --inputs example-data/phantom.json --data example-data --case standard_autoshim --results outputs/phantom-example --out outputs/phantom-scores
python experiments.py run volunteers --inputs example-data/volunteers.json --data example-data --case S03 --out outputs/volunteer-example
python experiments.py score volunteers --inputs example-data/volunteers.json --data example-data --case S03 --results outputs/volunteer-example --out outputs/volunteer-scores
python plot_figures.py --figures 4 --out outputs/figures
```

Both examples use 16 time windows spaced by 0.6 ms, three residual-field updates
and final shell demodulation. Volunteer runs also include the autofocus baseline.

Outputs go to `outputs/`; individual scores are in `case_scores.json`.
Figure 4 plots the reported five-volunteer results without scan data.
The full comparisons use three phantom scans and five volunteers; the remaining
scan arrays and plotting tiles are not included. See [INPUTS.txt](INPUTS.txt) for
formats, settings and full-cohort commands.

The phase-tracking step builds on Brian Hargreaves' [hierarchical shell
correction](https://github.com/mribri999/timesegmentedphase).
