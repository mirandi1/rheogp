"""
RheoGP — Physics-Informed Gaussian Process for Rheology
========================================================

Supported models
----------------
  Maxwell                   classical Maxwell (spring + dashpot series)
  Springpot                 σ = 𝕍·D^α[γ]
  FractionalMaxwellGel      spring + springpot series           (FMG)
  FractionalMaxwellLiquid   springpot + dashpot series          (FML)
  FractionalMaxwell         two-springpot series  0≤β≤α≤1       (FMM)
  FractionalKelvinVoigtS    springpot ∥ spring                  (FKVS)
  FractionalKelvinVoigtD    springpot ∥ dashpot                 (FKVD)
  FractionalKelvinVoigt     two-springpot parallel  0≤β≤α≤1     (FKV)
  KernelFree                no constitutive model (diagnostics)

  Fractional Zener family (control="strain" only -- see rheogp.kernels)
  Zener                     spring ∥ Maxwell                    (standard linear solid)
  FractionalZenerSolidS     spring ∥ FMG                        (FZSS)
  FractionalZenerLiquidS    spring ∥ FML                        (FZLS)
  FractionalZenerLiquidD    dashpot ∥ FML                       (FZLD)
  FractionalZenerS          spring ∥ FMM                        (FZS)
  FractionalZener           springpot ∥ FMM  (most general)     (FZ)

New in 0.3 (released as v1.0.0 / v1.0.1)
----------
  control="strain" | "stress"      strain- or stress-controlled chirps
  convolution="causal" | "steady"  integral from 0  or  from -inf
  quadrature="exact"               transient-free causal features
  model.results()                  every quantity as numpy (FitResult)
  rheogp.fit(...)                  one-line fitting

New in 0.4 (released in v1.1.0)
----------
  Fractional Zener family (6 models, strain control) -- see above
  rheogp.mastercurve.MasterCurve            joint multi-temperature TTS
                                             fitting + WLF + AIC/BIC
                                             model comparison for master
                                             curves (see that module)
  rheogp.preprocessing.clean_chirp          reusable chirp-cleaning
                                             (onset/baseline/detrend/taper)
  rheogp.utils.make_tts_dataset             synthetic WLF-shifted
                                             multi-temperature chirps

New in 0.5 (released in v1.1.0)
----------
  MultiRecordSPGP.predict_at_shift(records, a_T)     real prediction on
  MultiRecordSPGP.predict_at_temperature(records, T)  NEW records for a
  MasterCurve.predict_at_shift / predict_at_temperature   trained joint
                                             fit -- the inherited
                                             SPGP.predict() silently
                                             mis-sliced anything that
                                             wasn't byte-identical to
                                             the training record layout;
                                             these derive bounds/dt
                                             fresh from whatever new
                                             records you give them and
                                             reuse the trained GP
                                             posterior (mean + std, no
                                             retraining). No functional
                                             form for shift-vs-temperature
                                             is ever assumed internally:
                                             exact/interpolated
                                             temperatures use the fitted
                                             a_T's directly; extrapolating
                                             past the fitted range
                                             requires YOU to pass
                                             shift_law=<callable T ->
                                             a_T> (WLF, Arrhenius, VFT,
                                             whatever you trust) or it
                                             raises rather than guess.
                                             See rheogp.rheogp_multi.
  MultiRecordSPGP(..., temps=[...])         optional per-record
                                             temperature labels, enables
                                             predict_at_temperature().
  Prototyped (not yet validated with the same rigor as the Maxwell/
  Zener families): a joint-TTS shift fix for Kelvin-Voigt-type
  (parallel/power-law) kernels -- see _strain_shift_exponents in
  rheogp.kernels and CHANGELOG.md.

Quick start
-----------
>>> import rheogp
>>> m = rheogp.fit(t, strain, stress, model="Springpot",
...                control="strain", convolution="steady")
>>> print(m.summary())
>>> res = m.results()          # arrays for your own figures
>>> res.save("outputs/run1")

Master curves from several chirps at different temperatures
--------------------------------------------------------------
>>> from rheogp.mastercurve import MasterCurve, compare_master_curve_models
>>> mc = MasterCurve(records, temperatures).fit(model="FractionalZenerS")
>>> mc.fit_wlf()
>>> best, table = compare_master_curve_models(
...     records, temperatures,
...     ["FractionalMaxwell", "FractionalZenerS", "FractionalZener"])

Predicting a NEW chirp from a trained master curve
----------------------------------------------------
>>> pred, std, bounds = mc.predict_at_shift([(t_new, gamma_new, 1.0)])
>>> pred, std, bounds = mc.predict_at_temperature([(t_new, gamma_new, 62.0)])
>>> # extrapolating past the fitted temperature range needs YOUR shift law:
>>> pred, std, bounds = mc.predict_at_temperature(
...     [(t_new, gamma_new, 120.0)], shift_law=mc.predict_aT)  # e.g. this MC's own WLF
"""

from .model import SPGP, fit
from .kernels import KERNELS
from .results import FitResult
from . import utils, features, plots, preprocessing

__all__ = ["SPGP", "fit", "FitResult", "KERNELS"]
__version__ = "1.1.0"
