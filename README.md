# RheoGP

**Physics-Informed Sparse Gaussian Process for Fractional Rheology**

Fit fractional viscoelastic models directly to chirp (or any transient)
rheometry time series with a scikit-learn style API.  All constitutive
shape parameters (α, β, τ_c) are optimised *jointly* with the GP
hyperparameters through the variational ELBO, and the linear-kernel
sensitivities give time-resolved prefactors (𝕍(t), G(t), …) with
credible intervals.

New in **v0.3**:

* **Stress-controlled experiments** (`control="stress"`) — fit creep
  representations; moduli are reported automatically as derived
  parameters.
* **Two feature-construction modes** — causal convolution from t = 0
  (`convolution="causal"`, for from-rest experiments) or steady-state
  convolution from t = −∞ (`convolution="steady"`, exact frequency-space
  transfer functions for periodic/windowed protocols).
* **Exact quadrature** for the singular power-law memory kernels
  (`quadrature="exact"`) — removes the start-up transient and the
  systematic amplitude bias of naive rectangle rules.
* **`FitResult` container** — every quantity (signals, prediction ± σ,
  physical features, sensitivities, G′/G″, training history) as plain
  numpy, with `save()` / `load()`.
* **Complete information criteria** — AIC, AICc and BIC computed from
  the total variational NLL and counting *all* model parameters
  (shape parameters + prefactors + GP hyperparameters), with the full
  breakdown reported for transparency.

---

## Install

```bash
git clone https://github.com/<your-org>/rheogp
cd rheogp
pip install -e .            # or:  pip install -e ".[dev]"  for tests
```

**Dependencies:** `torch`, `gpytorch`, `scikit-learn`, `numpy`, `scipy`, `matplotlib`
(see `requirements.txt`; on CPU-only machines install torch from the CPU wheel index first).

---

## Quick start

```python
import rheogp

# one-liner: returns a fitted SPGP model
model = rheogp.fit(
    time, strain, stress,          # 1-D numpy arrays, uniform dt
    model="FractionalMaxwell",     # see table below
    control="strain",              # or "stress"  (creep)
    convolution="causal",          # or "steady"  (from -inf)
    quadrature="exact",            # "exact" | "midpoint" | "left"
)

print(model.summary())

res = model.results()              # FitResult: plain-numpy everything
res.t, res.prediction, res.prediction_std
res.features, res.feature_names    # physical features x_i(t)
res.sensitivities["V"]["mean"]     # time-resolved prefactor + CI
res.omega, res.Gp, res.Gdp         # model G', G''
res.save("my_fit")                 # -> my_fit/result.npz + result.json

# built-in plots
model.plot_fit()
model.plot_prefactors()
model.plot_Gstar(add_fft=True)
model.plot_convergence()
```

The classic API still works:

```python
from rheogp import SPGP
model = SPGP(model="Springpot", n_epochs=4000).fit(time, strain, stress)
```

---

## Supported models

| String (aliases) | Relaxation law (strain control) | Learnable shape params |
|---|---|---|
| `"Springpot"` | σ = 𝕍 D^α[γ] | α |
| `"Maxwell"` | single-mode Maxwell | τ_c |
| `"FractionalMaxwellGel"` / `"FMG"` | springpot + spring in series | α, τ_c |
| `"FractionalMaxwellLiquid"` / `"FML"` | springpot + dashpot in series | β, τ_c |
| `"FractionalMaxwell"` / `"FMM"` | two springpots in series | α, β, τ_c |
| `"FKVS"` | springpot ∥ spring | α |
| `"FKVD"` | springpot ∥ dashpot | β |
| `"FractionalKelvinVoigt"` / `"FKV"` | two springpots in parallel | α, β |
| `"KernelFree"` | non-parametric (γ, γ̇ inputs) | — |

Under `control="stress"` each model is re-expressed in its **creep
representation**: series elements' compliances add (features are
fractional integrals of σ, linear in 1/𝕍, 1/G, 1/η), while parallel
(FKV-family) models use a single Mittag-Leffler creep kernel
J(s) = s^α E_{α−β,1+α}(−(s/τ_c)^{α−β}) with τ_c learnable.  The
`summary()` and `results()` report the corresponding moduli as derived
parameters, so downstream analysis is identical in both modes.

---

## Causal vs steady features — which to use?

| | `convolution="causal"` | `convolution="steady"` |
|---|---|---|
| Assumes | material at rest for t < 0 | record = one period of a periodic protocol |
| Best for | from-rest experiments, step strain, start-up | optimally-windowed chirps (taper → 0 at both ends), oscillatory steady state |
| Implementation | product-integration convolution, `quadrature="exact"` integrates the kernel singularity analytically per bin | N-point FFT with analytic transfer functions, DC bin zeroed — **no padding** |

Notes:

* `quadrature="left"` reproduces the rheogp ≤ 0.2 discretisation
  (kept for back-compatibility).  It underestimates feature amplitude
  by ~15 % for α ≈ 0.6 on typical chirps; prefer `"exact"`.
* Zero-padding the FFT while using analytic kernel transforms
  implicitly prepends a block of zeros to the periodic history and
  systematically *underestimates* the response for long-memory
  kernels.  The steady mode therefore uses an unpadded N-point FFT.
* A model trained with `steady` features can still predict a
  non-periodic protocol: `model.predict(u, convolution="causal")`.

---

## Model selection: AIC and BIC

`model.metrics_` (also in `results().metrics`) reports

```python
{
  "rmse": ..., "r2": ...,
  "nll":  ...,            # N × (negative per-point variational ELBO)
  "aic":  ..., "aicc": ..., "bic": ...,
  "n_params": {
      "n_shape":       2,     # α, β, τ_c … of the chosen kernel
      "n_prefactors":  2,     # 𝕍, G, η (or 1/𝕍 … in stress control)
      "n_gp_hyper":    4,     # mean, kernel scales, likelihood noise
      "n_variational": 2550,  # reported, NOT counted (see below)
      "k_total":       8,     # what AIC/BIC use
  },
  "n_data": N,
}
```

with the standard definitions

```
AIC  = 2·NLL + 2k          AICc = AIC + 2k(k+1)/(N−k−1)
BIC  = 2·NLL + k·ln N
```

Two methodological choices, made explicit so they can be defended
(and changed) easily:

1. **Likelihood term.** The exact marginal likelihood of a sparse
   variational GP is intractable; the ELBO is its standard surrogate
   (a lower bound, so the criteria are conservative).  gpytorch returns
   the ELBO *per data point*; RheoGP multiplies by N before applying
   the formulas.
2. **Parameter count.** `k_total` includes the viscoelastic shape
   parameters, one scalar per prefactor (the sensitivities inferred
   through the GP posterior), and every GP hyperparameter (mean
   constant, kernel scales/lengthscales, observation noise).  The
   inducing-point variational parameters are *excluded*: they
   parametrise the approximate posterior, not the model — in an exact
   GP the posterior contributes no parameters either.  Their number is
   still reported in `n_params` so any alternative accounting can be
   recomputed from the stored quantities.

Both criteria are comparable **across constitutive models fitted to the
same data with the same settings** (same N, same inducing-point count).
BIC penalises complexity more strongly for large N; AICc converges to
AIC for N ≫ k.

---

## Repository layout

```
rheogp/            package (features, kernels, gp, model, results, plots, utils)
examples/          quickstart.py — full worked example
tests/             pytest suite (analytic references + parameter recovery)
demos/             research notebooks + experimental data
.github/workflows/ CI (pytest on push / PR)
```

---

## Testing

```bash
python -m pytest tests/
```

Runs analytic-reference checks of the feature engine plus end-to-end
parameter-recovery fits in both control modes.

See `examples/quickstart.py` for a complete worked example.

---

## Citing

If you use RheoGP in academic work, please cite the accompanying paper
(*Direct inference of viscoelastic memory from chirp rheometry via
physics-informed Gaussian processes*) — machine-readable metadata in
[`CITATION.cff`](CITATION.cff).

## License

MIT — see [`LICENSE`](LICENSE).
