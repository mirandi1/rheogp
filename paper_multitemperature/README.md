# Multi-temperature master curves from chirp rheometry

Notebooks, helper modules and data behind the multi-temperature
(time–temperature superposition) paper. Results were produced with
RheoGP **v1.1.0**.

The workflow has two stages:

1. **Fitting (cluster).** The `01_*.ipynb` notebooks load the chirp
   records, run the joint multi-temperature fits and candidate-model
   comparisons, and write export bundles (`.json` + `.npz`) through
   `paper_utils.py`. These fits are the expensive step.
2. **Figures (workstation).** The plotting notebooks read those export
   bundles and draw every figure without refitting.

## Datasets and notebooks

| Dataset | Data | Fitting notebook | Export folder | Plotting notebook |
|---|---|---|---|---|
| Synthetic fractional Maxwell (FMM/FMG) | generated in the notebook | `01_synthetic_fmm_tts.ipynb` | `fmm_export/` | `fmm_plotting.ipynb` |
| Epoxy DGEBA-IPD (fractional Maxwell and Zener candidates) | `data/chirp_01_T*C.csv` | `01_epoxy_fzm.ipynb` | `dgeba_fzm_out/` | `fzm_dgeba.ipynb` |
| PT | `data/chirp_02_T*C.csv` | `01_pt_fzm.ipynb` | `pt_fzm_out/` | `fzm_pt_analysis.ipynb` |
| 3M acrylic, OWChirp (−40 to 150 °C) | `data_owchirp/chirp_*_T*C.csv` | `01_owchirp_tts.ipynb` | `owchirp_fzm_out/` | `fzm_acrylic_3M.ipynb` |

Each CSV holds one chirp at one temperature: 17 epoxy records, 13 PT
records and 20 OWChirp records. Columns are `Strain, Stress,
Temperature, Time` (epoxy, PT) or `t, strain, stress, temperature`
(OWChirp); the loader in the fitting notebooks matches either.

`01_synthetic_fmm_tts.ipynb` writes its bundles to the working
directory; move them into `fmm_export/` before running
`fmm_plotting.ipynb`.

## Helper modules

- `paper_utils.py` — export/load of master-curve bundles, model
  comparisons and noise sweeps; LaTeX macro and table writers.
- `mastercurve_compare.py` — comparison against a conventional
  FFT + `mastercurves` pipeline and against `pyRheo` SAOS fits (used by
  the plotting notebooks).
- `run_broadband_demo.py`, `run_broadband_demo_fmm.py` — the
  "one broadband chirp vs. the master curve" self-consistency check,
  built from an existing export. `run_broadband_demo.py` is set up for
  the DGEBA export; change `DATA_DIR`, `EXPORT_PATH` and `OUT_PATH` for
  the other datasets.

## Running

Run every notebook from this folder, so that the relative data and
export paths resolve and `paper_utils` / `mastercurve_compare` import.

```bash
pip install -e ..                 # RheoGP itself
pip install pandas pyRheo mastercurves SciencePlots   # plotting extras
```

Export bundles and generated figures are not tracked in git (see the
repository `.gitignore`); rerun the fitting notebooks to regenerate them.
