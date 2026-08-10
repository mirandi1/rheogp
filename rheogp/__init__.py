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

New in 0.3
----------
  control="strain" | "stress"      strain- or stress-controlled chirps
  convolution="causal" | "steady"  integral from 0  or  from -inf
  quadrature="exact"               transient-free causal features
  model.results()                  every quantity as numpy (FitResult)
  rheogp.fit(...)                  one-line fitting

Quick start
-----------
>>> import rheogp
>>> m = rheogp.fit(t, strain, stress, model="Springpot",
...                control="strain", convolution="steady")
>>> print(m.summary())
>>> res = m.results()          # arrays for your own figures
>>> res.save("outputs/run1")
"""

from .model import SPGP, fit
from .kernels import KERNELS
from .results import FitResult
from . import utils, features, plots

__all__ = ["SPGP", "fit", "FitResult", "KERNELS"]
__version__ = "0.3.0"
