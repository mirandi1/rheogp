#!/usr/bin/env python3
"""
multichirp_master_curve.py
===========================
End-to-end example: build a time-temperature-superposition (TTS)
master curve from several chirps recorded at different temperatures,
using the `rheogp.mastercurve` module (a clean, tested replacement for
the original ad-hoc multichirp fitting script).

What changed from the original draft script
---------------------------------------------
* The joint fit / WLF regression / AIC-BIC model comparison now live
  in the PACKAGE (`rheogp.mastercurve.MasterCurve`,
  `compare_master_curve_models`), not copy-pasted per analysis --
  they're unit-tested in tests/test_mastercurve.py.
* The signal-cleaning helpers (`clean_chirp`, `rolling_rms`) moved to
  `rheogp.preprocessing`, importable and unit-tested
  (tests/test_mastercurve.py) instead of living only inside this
  script.
* The `patch_safe_report()` monkey-patch is gone -- the zero-division
  bugs it was working around (a degenerate record with sd_i = 0, and
  an unguarded truth-vs-zero relative error) are fixed directly in
  `MultiRecordSPGP.report()`.
* The master-curve credible band no longer needs a 200-repetition
  Monte-Carlo loop that resamples stress trajectories and re-runs the
  FFT/SNR-gating machinery per sample -- `MasterCurve.model_curve()`
  propagates the already-computed GP posterior prefactor samples
  through the closed-form G*(ω) formula directly (cheap, exact given
  the posterior, no FFT noise).
* Duplicated plotting blocks (the script had the master-curve figure
  built twice, once without and once with a temperature colour bar)
  are collapsed into one.
* You can now compare the Zener family (including the fractional
  Zener models added in this version) against the FMM-style models
  the draft was restricted to, by just adding names to CANDIDATE_MODELS.

Usage
-----
    python multichirp_master_curve.py                  # uses your CSVs
    python multichirp_master_curve.py --demo            # synthetic data

Real-data mode expects files matching DATA_GLOB, tab-delimited, with
"Time", "Strain"/"Gamma", "Stress"/"Sigma" columns (case-insensitive;
falls back to the first three columns), and a temperature encoded in
the filename as e.g. "..._T90C...csv" (see
`rheogp.preprocessing.parse_temperature` for the pattern).  If no file
matches DATA_GLOB, the script automatically falls back to a synthetic
demo dataset (`rheogp.utils.make_tts_dataset`) so it always runs.
"""

import argparse
import glob
import os

import numpy as np
import matplotlib.pyplot as plt

from rheogp.preprocessing import clean_chirp, parse_temperature
from rheogp.mastercurve import MasterCurve, compare_master_curve_models
from rheogp.utils import make_tts_dataset

# ------------------------------- configuration -------------------------------
DATA_GLOB = "chirp_*_T*C.csv"
T_WINDOW = (30.0, 100.0)          # keep records with T in this range, degC
CANDIDATE_MODELS = ["FractionalMaxwell", "FractionalZenerS",
                    "FractionalZener"]
CLEANKW = dict(discard_s=0.5, pad_s=0.5)
FITKW = dict(kernel="rbf+linear", inducing_points=200,
            n_epochs=4000, patience=500,
            n_epochs_linear=1500, patience_linear=300,
            learning_rate=0.01, learning_rate_physics=0.02,
            omega_min=1e-3, omega_max=1e3, verbose=True)
HEADER_MAP = dict(t=["t", "time", "sec", "s"],
                  strain=["gamma", "strain", "eps", "epsilon"],
                  stress=["sigma", "stress", "tau"])


# ------------------------------- data loading (experiment-specific) ----------

def load_csv_record(path):
    """(t, strain, stress) from a tab-delimited CSV; tries pandas for
    header matching, falls back to numpy / first-3-columns."""
    try:
        import pandas as pd
        df = pd.read_csv(path, sep=None, engine="python")
        cols = [c.strip().lower() for c in df.columns]

        def find_col(cands):
            for c in cands:
                if c in cols:
                    return df.iloc[:, cols.index(c)].values.astype(float)
            return None

        t = find_col(HEADER_MAP["t"])
        strain = find_col(HEADER_MAP["strain"])
        stress = find_col(HEADER_MAP["stress"])
        if t is None or strain is None or stress is None:
            arr = df.to_numpy(dtype=float)
            if arr.shape[1] < 3:
                raise ValueError("Need at least 3 columns (t, strain, stress).")
            t, strain, stress = arr[:, 0], arr[:, 1], arr[:, 2]
        return np.asarray(t, float), np.asarray(strain, float), np.asarray(stress, float)
    except Exception:
        arr = np.loadtxt(path, delimiter=",", dtype=float)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[1] < 3:
            raise ValueError(f"{path}: expected >=3 columns (t, strain, stress).")
        return arr[:, 0], arr[:, 1], arr[:, 2]


def load_real_dataset():
    files = sorted(glob.glob(DATA_GLOB))
    recs, temps = [], []
    for f in files:
        T = parse_temperature(f)
        if T is None:
            print(f"  (skipping {f}: no temperature found in filename)")
            continue
        if not (T_WINDOW[0] <= T <= T_WINDOW[1]):
            continue
        t, g, s = load_csv_record(f)
        tc, gc, sc = clean_chirp(t, g, s, **CLEANKW)
        recs.append((tc, gc, sc))
        temps.append(T)
    return recs, temps


def load_demo_dataset():
    print("  -> no files matched DATA_GLOB; using a synthetic demo "
         "dataset (rheogp.utils.make_tts_dataset).")
    truth, records = make_tts_dataset(
        model="fzs", temps=[40, 55, 70, 85, 100], T_ref=70.0,
        C1=6.0, C2=140.0, tau_c_ref=1.0,
        n=1200, dt=1e-3, omega1=0.05, omega2=20.0, noise=0.015)
    print(f"  ground truth: {truth}")
    return records, truth["temperatures"]


# ------------------------------------- main -----------------------------------

def main(demo=False):
    print(f"Looking for chirp files matching '{DATA_GLOB}' ...")
    records, temps = ([], []) if demo else load_real_dataset()
    if not records:
        records, temps = load_demo_dataset()
    print(f"Loaded {len(records)} records at temperatures: {temps}")

    # -- quick look at the cleaned records -------------------------------
    fig0, ax0 = plt.subplots(1, 2, figsize=(10, 3.2))
    cmap = plt.get_cmap("viridis")
    for i, (T, (t, g, s)) in enumerate(zip(temps, records)):
        col = cmap(i / max(1, len(temps) - 1))
        ax0[0].plot(t, g, lw=0.8, color=col, label=f"{T:g} C")
        ax0[1].plot(t, s, lw=0.8, color=col)
    ax0[0].set_xlabel("t [s]"); ax0[0].set_ylabel("strain")
    ax0[0].legend(frameon=False, fontsize=7, ncol=min(len(temps), 4))
    ax0[1].set_xlabel("t [s]"); ax0[1].set_ylabel("stress [Pa]")
    fig0.suptitle("cleaned records"); fig0.tight_layout()
    fig0.savefig("multichirp_cleaned_records.png", dpi=150)

    # -- rank candidate constitutive laws by BIC, jointly across records -
    print(f"\nComparing {CANDIDATE_MODELS} jointly across all records ...")
    best, table = compare_master_curve_models(
        records, temps, CANDIDATE_MODELS, criterion="bic", **FITKW)
    print(f"\nBest model by BIC: {best.model_name}")

    # -- WLF regression of the winning model's shift factors -------------
    best.fit_wlf()
    print(best.summary())

    # -- figures -----------------------------------------------------------
    fig1, ax1 = plt.subplots(1, 2, figsize=(9.5, 3.4))
    Tv = np.array(best.temperatures)
    ax1[0].semilogy(Tv, best.aT_, "o", ms=5, color="tab:orange",
                    label="fitted a_T")
    T_fine = np.linspace(Tv.min(), Tv.max(), 200)
    ax1[0].semilogy(T_fine, best.predict_aT(T_fine), "k-", lw=1.2,
                    label="WLF fit")
    ax1[0].axvline(best.T_ref, color="gray", lw=0.8, ls=":")
    ax1[0].set_xlabel("T [deg C]"); ax1[0].set_ylabel("a_T")
    ax1[0].set_title(f"{best.model_name} shift factors")
    ax1[0].legend(frameon=False, fontsize=8)

    for name, mc in table.items():
        ax1[1].bar(name, mc.metrics_["bic"])
    ax1[1].set_ylabel("BIC (lower is better)")
    ax1[1].set_title("model comparison")
    ax1[1].tick_params(axis="x", rotation=20)
    fig1.tight_layout()
    fig1.savefig("multichirp_shift_factors_and_comparison.png", dpi=150)

    # master curve: raw FFT points (colour = temperature) + model
    # median with a 95% credible band from the GP posterior
    fig2, ax2 = plt.subplots(1, 1, figsize=(7.2, 4.2))
    omega_ref, Gstar, rec_idx = best.reduced_data()
    norm = plt.Normalize(vmin=min(best.temperatures), vmax=max(best.temperatures))
    T_of_point = np.array(best.temperatures)[rec_idx]
    sc1 = ax2.scatter(omega_ref, np.real(Gstar), c=T_of_point, cmap=cmap,
                      norm=norm, s=8, alpha=0.6, marker="o")
    ax2.scatter(omega_ref, np.imag(Gstar), c=T_of_point, cmap=cmap,
               norm=norm, s=8, alpha=0.6, marker="s")
    ax2.set_xscale("log"); ax2.set_yscale("log")

    omega, Gp, Gdp, Gp_ci, Gdp_ci = best.model_curve()
    ax2.fill_between(omega, Gp_ci[0], Gp_ci[1], color="tab:blue", alpha=0.15)
    ax2.plot(omega, Gp, color="tab:blue", lw=2.0, label="G' (model median)")
    ax2.fill_between(omega, Gdp_ci[0], Gdp_ci[1], color="tab:red", alpha=0.15)
    ax2.plot(omega, Gdp, color="tab:red", lw=2.0, label="G'' (model median)")

    ax2.set_xlabel("reduced frequency ω_ref [rad/s]")
    ax2.set_ylabel("G', G'' [Pa]")
    ax2.set_title(f"Master curve  —  {best.model_name}  "
                 f"(T_ref = {best.T_ref:g} C)")
    ax2.legend(frameon=False, fontsize=8, loc="best")
    cb = fig2.colorbar(sc1, ax=ax2, pad=0.02)
    cb.set_label("Temperature [deg C]")
    fig2.tight_layout()
    fig2.savefig("multichirp_master_curve.png", dpi=150)

    print("\nSaved: multichirp_cleaned_records.png, "
         "multichirp_shift_factors_and_comparison.png, "
         "multichirp_master_curve.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true",
                        help="skip the CSV search and use the synthetic "
                             "demo dataset directly")
    args = parser.parse_args()
    main(demo=args.demo)
