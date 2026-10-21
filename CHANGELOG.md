# Changelog

Versions 0.4.0 and 0.5.0 below were internal development versions; they
are released together as **1.1.0**. The 0.3.0 code was released on
GitHub as **v1.0.0 / v1.0.1** (the version cited by the single-chirp
paper).

## 1.1.0 — 2026-10-21

Multi-temperature (time–temperature superposition) analysis of chirp
rheometry. Single-chirp behavior is unchanged from v1.0.1: every new
option defaults to the previous behavior, and the original
`tests/test_rheogp.py` passes unmodified.

### Added
- Everything listed under 0.4.0, 0.4.1 and 0.5.0 below: joint
  multi-record fitting (`rheogp.rheogp_multi.MultiRecordSPGP`),
  `rheogp.mastercurve.MasterCurve` with WLF regression and
  `compare_master_curve_models`, the fractional Zener family and Prony
  series, `rheogp.preprocessing`, selectable noise models
  (`rheogp.likelihoods`), and prediction on new records at a given
  shift factor or temperature.
- `paper_multitemperature/`: the notebooks, helper modules and chirp
  data behind the multi-temperature master-curve paper.

### Changed
- Version numbers in `pyproject.toml`, `rheogp/__init__.py` and
  `CITATION.cff` now match the GitHub release tag.
- Author affiliation: Aalto University; Deltan Labs.
- `rheogp.preprocessing.clean_chirp` now discards the first
  `discard_s` seconds and removes a linear drift fitted on the first and
  last `pad_s` seconds (`detrend=False` skips the drift removal). It
  replaces the 0.4.0 onset-detection / taper interface (`rms_win_s`,
  `onset_thresh`, `prepad_s`, `tukey_taper`); the multi-temperature paper
  notebooks use the new interface. Test and example updated to match.
- Tests run on matplotlib's non-interactive backend (`tests/conftest.py`),
  so `compare_models()`'s `plt.show()` no longer blocks the suite on
  desktop backends.

## 0.5.0 — 2026-08-16

### Added
- `MultiRecordSPGP.predict_at_shift(records_with_aT)` and
  `MultiRecordSPGP.predict_at_temperature(records_with_T, shift_law=None)`,
  mirrored on `MasterCurve` (`MasterCurve.fit()` now also passes
  `temps=self.temperatures` through to the underlying
  `MultiRecordSPGP` so its own `predict_at_temperature` works
  out of the box): real prediction on NEW records for a trained
  joint/master-curve fit. The inherited `SPGP.predict()` silently
  mis-slices anything that isn't byte-identical in per-record
  length/dt to the training layout (it reuses `bounds`/`dts` baked in
  at construction, ignoring what you actually pass it) -- confirmed to
  corrupt/splice new input rather than raise. The new methods derive
  `bounds`/`dts` fresh from whatever new records are given (so a
  different record count/length than training works), reuse the
  trained GP posterior unmodified (same inducing points/variational
  distribution/hyperparameters -- no retraining), and return both mean
  and posterior std.
  `predict_at_shift()` takes shift factors directly (`a_T` or a
  `tau_c` you convert via the new `tau_ref` property) and makes no
  assumption about temperature dependence whatsoever.
  `predict_at_temperature()` is a convenience wrapper: exact/interior
  temperatures use the fitted `a_T`'s directly or log-linear
  interpolation between the two nearest fitted temperatures (no
  functional form imposed); temperatures outside the fitted range
  raise `ValueError` unless the caller passes `shift_law=<callable T ->
  a_T>` -- WLF, Arrhenius, VFT, or anything else, entirely the caller's
  choice, never assumed internally. (`MasterCurve.predict_aT`, which IS
  explicitly WLF-specific, is unchanged and can itself be passed as
  `shift_law=mc.predict_aT` if you want that.) `MultiRecordSPGP.__init__`
  gained an optional `temps=` kwarg (per-record temperature labels,
  same order as `records`) that `predict_at_temperature` requires;
  nothing else uses it. Also added `MultiRecordSPGP.tau_ref`. Tests:
  `tests/test_multi_predict.py` (9 tests) +
  `test_mastercurve_predict_at_shift_and_temperature` in
  `tests/test_mastercurve.py`; full suite 61/61 passing, no
  regressions in the pre-existing 51.
- `_strain_shift_exponents()` hook + FKV-family joint-TTS shift fix
  (prefactor rescaling by `a_T**p` for power-law/memoryless strain
  features) -- see inline docstrings in `kernels.py`/`rheogp_multi.py`.
  Prototyped and validated on independent synthetic FKV data (shift
  factors recovered to within 2%, vs. pinned-at-1/R²=0.34 for the
  unmodified mechanism); not yet exercised with the same noise-sweep/
  AIC-BIC rigor as the paper's two validated families.

## 0.4.1 — 2026-08-14

### Fixed
- `SPGP._count_parameters()` (used by every `metrics_['n_params']` /
  AIC / BIC computation): for `rheogp.rheogp_multi.MultiRecordSPGP`
  (a joint fit across M records via `shift_kernel_class`), the
  per-record horizontal shift factors `a_T` are trainable
  `phys.log_aT` parameters that are *not* part of
  `phys.named_phys_params()` (only the shared reference-record shape
  parameters are), so they were silently excluded from both `n_shape`
  and `n_gp_hyper` and therefore from `k_total` entirely — `k_total`
  undercounted every joint/master-curve fit by the number of records
  in the fit. Added an explicit `n_shift` category: `log_aT` has M raw
  entries, but `aT = exp(log_aT - log_aT[i_ref])` is invariant to a
  common additive shift, so only M-1 are identifiable; that one gauge
  direction is subtracted off. For an ordinary single-record kernel
  `n_shift = 0` and `k_total` is unchanged (verified against
  `tests/test_rheogp.py`'s exact-formula assertion). This mainly
  affects the *absolute* AIC/BIC reported by `MasterCurve` /
  `compare_master_curve_models` and by `MultiRecordSPGP.report()`;
  rankings from `compare_master_curve_models` are unaffected when
  comparing candidates fitted to the identical set of records, since
  every candidate gets the same additive `n_shift` correction and it
  cancels in `ΔAIC`/`ΔBIC`.

## 0.4.0 — 2026-08-14

### Added
- **Fractional Zener family** (6 new models, `control="strain"` only):
  `Zener`, `FractionalZenerSolidS` (`FZSS`), `FractionalZenerLiquidS`
  (`FZLS`), `FractionalZenerLiquidD` (`FZLD`), `FractionalZenerS`
  (`FZS`), and the general `FractionalZener` (`FZ`). Every model is one
  parallel element (spring / dashpot / springpot) plus one Maxwell-type
  arm (Maxwell / FMG / FML / FMM); complex moduli of parallel branches
  add directly, `G*(ω) = G*_parallel(ω) + G*_arm(ω)`, which is also how
  the strain-control features are built — the same pattern already used
  by the FKV family. `_MaxwellArmMixin` in `kernels.py` is the one
  shared building block reused by all six.
  Constitutive equations, creep compliance, relaxation modulus and
  storage/loss moduli transcribed from the pyRheo Supplementary
  Information; the `FractionalZenerSolidS` storage/loss modulus was
  additionally verified algebraically equivalent to the FMG-based
  parametrisation used internally, and all six models pass low-/
  high-frequency `Gstar` limit tests (`tests/test_zener.py`).
  **Stress control (creep) is intentionally NOT implemented** for this
  family in this release: unlike the FKV family (a single
  Mittag-Leffler creep kernel with its own learnable timescale), a
  parallel spring/dashpot here changes the *decay rate itself* of the
  Maxwell arm's creep response (e.g. classical Zener:
  `J(t) = 1/G0 - G1/(G0(G0+G1))·exp(-t/τc')` — the two prefactors and
  the exponential rate are coupled), so it needs its own
  `features_stress()` derivation rather than reusing the strain-control
  machinery. `control="stress"` raises `NotImplementedError` immediately
  at model construction with a message pointing here; contributions
  welcome.
- **`rheogp.mastercurve`**: `MasterCurve` (joint multi-temperature
  fitting via `MultiRecordSPGP` + WLF regression of the recovered shift
  factors + an analytic reduced-frequency master curve with a credible
  band propagated from the GP posterior prefactor samples through
  `Gstar()` — no repeated FFT / Monte-Carlo resampling needed) and
  `compare_master_curve_models(...)` (fits several candidate
  constitutive laws jointly to the *same* set of chirps and ranks them
  by AIC/BIC, the master-curve analogue of `rheogp.utils.compare_models`
  — this works because `MultiRecordSPGP` is itself an `SPGP` subclass
  with the same `.metrics_`). Only a **horizontal** shift a_T is
  modelled — this is an explicit, documented scope decision (see the
  module docstring and `rheogp.rheogp_multi`), not an oversight; a
  vertical shift b_T would need folding into the feature construction
  itself and is left for later.
- **`rheogp.preprocessing`**: `clean_chirp` / `rolling_rms` /
  `parse_temperature` — the onset-detection / baseline / detrend /
  taper cleaning previously living only inside an analysis script,
  now importable and unit-tested.
- **`rheogp.utils.make_tts_dataset`**: synthetic multi-temperature
  chirp dataset with a known WLF shift, for validating the joint /
  master-curve pipeline (any Maxwell-arm model: Maxwell, FMG, FML, FMM,
  or the new Zener family). Plus `make_synthetic_zener` /
  `_fzss` / `_fzls` / `_fzld` / `_fzs` / `_fz` generators for the new
  models, matching the existing "physical component values as
  arguments" convention.
- `examples/multichirp_master_curve.py`: end-to-end example (falls back
  to a synthetic demo dataset if no CSVs are found, so it always runs).

### Fixed
- `MultiRecordSPGP.report()`: removed dead `... if False else ...` code
  left over from a previous refactor; guarded the per-record
  RMSE/std ratio and the truth-comparison relative error against
  division by zero (a degenerate record with `sd_i = 0`, or a `truth`
  value of exactly 0) instead of raising or silently propagating `inf`.

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
