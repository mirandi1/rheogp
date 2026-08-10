# Changelog

## 0.3.0 — 2026-07-22

### Added
- **Stress-controlled fitting** (`control="stress"`): creep representation
  for all models. Series models fit compliances (features are fractional
  integrals of stress); the FKV family uses a single Mittag-Leffler creep
  kernel with learnable τ_c. Moduli reported as derived parameters.
- **Steady-state features** (`convolution="steady"`): convolution from
  t = −∞ via unpadded N-point FFT with analytic transfer functions
  (DC bin zeroed). Exact for periodic / optimally-windowed chirps.
- **Exact quadrature** (`quadrature="exact"`, now the default) for the
  causal convolution: bin-integrated product-integration weights for the
  singular power-law kernels. Removes the start-up transient and the
  ~15 % amplitude bias of the previous left-rectangle rule
  (kept as `quadrature="left"` for reproducing rheogp ≤ 0.2).
- **`FitResult` container** (`model.results()`): every quantity as plain
  numpy — signals, prediction ± σ, physical features x_i(t),
  time-resolved sensitivities with credible intervals, G′/G″,
  FFT estimates, training history — with `save()` / `FitResult.load()`.
- **`rheogp.fit(...)` one-liner** and `predict(..., convolution=...)`
  override for applying a steady-trained model to non-periodic protocols.
- **Complete information criteria**: AIC, AICc and BIC now use the total
  variational NLL (N × per-point bound) and count *all* model
  parameters — viscoelastic shape parameters, prefactors, and GP
  hyperparameters — with the breakdown reported in
  `metrics_['n_params']`. Variational (inducing-point) parameters are
  reported but excluded, as they parametrise the approximate posterior.
- Tests (`tests/test_rheogp.py`), quickstart example, packaging metadata.

### Changed
- Default GP kernel is `"rbf+linear"` (two-phase training).
- Synthetic generators in `utils.py` use exact/midpoint quadrature so
  that ground truth is unbiased; `make_chirp` gained a `taper=` option
  (OWCh-style Tukey window).

### Fixed
- Frequency-space features previously zero-padded to 2N with analytic
  kernel transforms, which implicitly prepends a silent half-period to
  the periodic history and systematically underestimates the response
  amplitude of long-memory kernels. The steady mode uses an unpadded
  N-point FFT.

### Deprecated
- `kernels_legacy.py` / `model_legacy.py` retain the 0.2 implementation
  for reference and will be removed in 0.4.
