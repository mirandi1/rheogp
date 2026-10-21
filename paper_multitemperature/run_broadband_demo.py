"""
Regenerate the "one huge chirp vs. 17-chirp master curve" self-consistency
demo (notebook 035_real_data_model_selection-Copy1.ipynb, cell 21), reusing
the ALREADY-FITTED FractionalZener physics recorded in
DGEBA-IPD_export_fzm.json/.npz instead of re-loading the raw CSVs and
re-running the full 17-record joint fit.

Only ONE genuinely new model fit happens here: a fresh, single-record,
no-TTS SPGP fit to a synthetic broadband chirp built from the winning
model's own analytic Gstar (the self-consistency check itself). Every
other ingredient (phys params, prefactors, master-curve coverage) is
read from the export.

Writes DGEBA-IPD_broadband_demo.npz + .json next to this script's CWD
(run from demos/), in the same load_export()-compatible format used by
fzm_dgeba.ipynb's other standalone plotting cells.
"""
import json
import time as _time

import numpy as np
import torch

from rheogp import SPGP
from rheogp.kernels import FractionalZenerKernel

# Both the export this reads and the bundle it writes live in the folder
# notebook 035 wrote its outputs to, so `fzm_dgeba.ipynb` finds every
# input in one place.
DATA_DIR = "dgeba_fzm_out"
EXPORT_PATH = f"{DATA_DIR}/DGEBA-IPD_export_fzm"
OUT_PATH = f"{DATA_DIR}/DGEBA-IPD_broadband_demo"

print("Loading export:", EXPORT_PATH)
meta = json.load(open(f"{EXPORT_PATH}.json"))
data = np.load(f"{EXPORT_PATH}.npz")

i_ref = int(meta["i_ref"])
bounds = meta["record_bounds"]
n_records = int(meta["n_records"])
phys_params = meta["phys_params"]
prefactors = {k: v["scalar_mean"] for k, v in meta["prefactors"].items()}
print("phys_params:", phys_params)
print("prefactors :", prefactors)

# ----------------------------------------------------------------
# Reference-record timing, straight from the export (no CSV reload needed)
# ----------------------------------------------------------------
t_ref = data[f"rec{i_ref}_t"]
dt_real = float(np.mean(np.diff(t_ref)))
N_ref = len(t_ref)
print(f"reference record i_ref={i_ref}: N={N_ref}, dt_real={dt_real:.4g} s")

# ----------------------------------------------------------------
# Full 17-chirp master-curve reduced-frequency coverage, from every
# record's own fft_omega_reduced (same quantity best.reduced_data() reads)
# ----------------------------------------------------------------
all_omega_r = []
for i in range(n_records):
    om = data[f"rec{i}_fft_omega_reduced"]
    all_omega_r.append(om[np.isfinite(om) & (om > 0)])
all_omega_r = np.concatenate(all_omega_r)
omega_lo, omega_hi = float(all_omega_r.min()), float(all_omega_r.max())
n_decades_full = np.log10(omega_hi / omega_lo)
print(f"Master curve reduced-frequency coverage: {omega_lo:.3g}-{omega_hi:.3g} rad/s "
      f"({n_decades_full:.1f} decades)")

periods_low, samples_per_period_high = 2, 6
t_needed = periods_low * 2 * np.pi / omega_lo
dt_needed = (2 * np.pi / omega_hi) / samples_per_period_high
n_needed = t_needed / dt_needed
print(f"A single chirp alone would need ~{t_needed:.3g} s (~{t_needed/86400:.2g} days), "
      f"sample rate > {omega_hi/(2*np.pi):.3g} Hz, ~{n_needed:.3g} samples.")

# ----------------------------------------------------------------
# Self-consistency demo: one chirp at the real instrument dt, run 6x
# longer than an actual record, at ONE temperature (T_ref)
# ----------------------------------------------------------------
demo_hi = np.pi / dt_real
N_DEMO = 6 * N_ref
demo_lo = periods_low * 2 * np.pi / (N_DEMO * dt_real)
n_decades_demo = np.log10(demo_hi / demo_lo)
print(f"One chirp at dt={dt_real:.3g}s, run {N_DEMO/N_ref:.0f}x longer "
      f"({N_DEMO*dt_real:.3g} s), reaches {demo_lo:.3g}-{demo_hi:.3g} rad/s "
      f"({n_decades_demo:.1f}/{n_decades_full:.1f} decades).")

t_demo = np.arange(N_DEMO) * dt_real
k_rate = np.log(demo_hi / demo_lo) / t_demo[-1]
omega_inst = demo_lo * np.exp(k_rate * t_demo)
phase = np.cumsum(omega_inst) * dt_real
strain_demo = 1e-3 * np.sin(phase)

m = max(int(0.05 * N_DEMO), 1)
ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(m) / m))
win = np.ones(N_DEMO); win[:m] = ramp; win[-m:] = ramp[::-1]
strain_demo *= win

# ----------------------------------------------------------------
# Ground truth stress from the WINNING model's own analytic Gstar
# (physics kernel constructed directly from the fitted phys_params --
# Gstar is a deterministic function of these, no re-training needed)
# ----------------------------------------------------------------
phys_truth = FractionalZenerKernel(
    alpha_init=phys_params["alpha"], beta_init=phys_params["beta"],
    kappa_init=phys_params["kappa"], tau_c_init=phys_params["tau_c"],
)

Gamma = np.fft.rfft(strain_demo)
omega_grid = 2 * np.pi * np.fft.rfftfreq(N_DEMO, dt_real)
omega_grid_safe = omega_grid.copy()
omega_grid_safe[0] = omega_grid_safe[1] * 1e-3
Gp_grid, Gdp_grid = phys_truth.Gstar(omega_grid_safe, prefactors)
Sigma = Gamma * (Gp_grid + 1j * Gdp_grid)
Sigma[0] = 0.0
stress_demo = np.fft.irfft(Sigma, n=N_DEMO)
rng = np.random.default_rng(0)
stress_demo = stress_demo + rng.normal(0, 0.01 * np.std(stress_demo), N_DEMO)

print(f"Synthetic broadband chirp: {N_DEMO:,} samples, duration={N_DEMO*dt_real:.3g} s, "
      f"sweeping {demo_lo:.3g}-{demo_hi:.3g} rad/s in ONE record.")

# ----------------------------------------------------------------
# THE genuinely new fit: fresh, single-record, no-TTS SPGP
# ----------------------------------------------------------------
device = "cpu"
print("torch:", torch.__version__, "device:", device)
t0 = _time.time()
demo_fit = SPGP(model="FractionalZener", kernel="linear", convolution="steady",
                quadrature="exact", inducing_points=250,
                learning_rate=0.05, learning_rate_physics=0.05,
                n_epochs=2000, patience=600,
                omega_min=demo_lo, omega_max=demo_hi, verbose=True,
                device=device)
demo_fit.fit(t_demo, strain_demo, stress_demo)
print(f"demo fit done in {(_time.time()-t0)/60:.1f} min")

omega_master = data["model_curve_omega"]
Gp_master = data["model_curve_Gp"]
Gdp_master = data["model_curve_Gdp"]
Gp_ci = data.get("model_curve_Gp_ci", None)
Gdp_ci = data.get("model_curve_Gdp_ci", None)

omega_demo = np.logspace(np.log10(demo_lo), np.log10(demo_hi), 400)
scalar_pf_demo = {k: v["scalar_mean"] for k, v in demo_fit.prefactors_.items()}
Gp_demo, Gdp_demo = demo_fit.gp_model_.phys.Gstar(omega_demo, scalar_pf_demo)

print(demo_fit.summary())

# ----------------------------------------------------------------
# Export -- pure numpy/json, standalone-plotting-friendly
# ----------------------------------------------------------------
arrays = dict(
    t_demo=t_demo, strain_demo=strain_demo, stress_demo=stress_demo,
    pred_demo=demo_fit.pred_, pred_std_demo=demo_fit.pred_std_,
    omega_demo=omega_demo, Gp_demo=Gp_demo, Gdp_demo=Gdp_demo,
    omega_master=omega_master, Gp_master=Gp_master, Gdp_master=Gdp_master,
)
if Gp_ci is not None:
    arrays["Gp_master_ci"] = Gp_ci
    arrays["Gdp_master_ci"] = Gdp_ci
if "hist_loss" in demo_fit.history_:
    arrays["demo_hist_loss"] = np.asarray(demo_fit.history_["loss"])
    for name, vals in demo_fit.history_.get("params", {}).items():
        arrays[f"demo_hist_param_{name}"] = np.asarray(vals)

np.savez_compressed(f"{OUT_PATH}.npz", **arrays)

meta_out = dict(
    source_export=EXPORT_PATH, i_ref=i_ref, T_ref=meta["T_ref"],
    dt_real=dt_real, N_ref=N_ref, N_DEMO=N_DEMO,
    omega_lo_full=omega_lo, omega_hi_full=omega_hi, n_decades_full=float(n_decades_full),
    demo_lo=demo_lo, demo_hi=demo_hi, n_decades_demo=float(n_decades_demo),
    t_needed_s=float(t_needed), dt_needed_s=float(dt_needed), n_needed=float(n_needed),
    demo_prefactors=scalar_pf_demo,
    demo_phys_params=demo_fit.gp_model_.phys.named_phys_params()
        if hasattr(demo_fit.gp_model_.phys, "named_phys_params") else None,
    demo_metrics=demo_fit.metrics_,
)


def _floatify(x):
    if isinstance(x, dict):
        return {k: _floatify(v) for k, v in x.items()}
    try:
        return float(x)
    except (TypeError, ValueError):
        return x


meta_out = _floatify(meta_out)
with open(f"{OUT_PATH}.json", "w") as fh:
    json.dump(meta_out, fh, indent=2)

print(f"wrote {OUT_PATH}.npz / .json")
