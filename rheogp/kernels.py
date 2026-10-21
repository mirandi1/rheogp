"""
rheogp.kernels
==============
Differentiable physics kernels for all supported rheological models.

Every kernel now supports:

  control      = "strain"  (imposed γ, features filter γ, GP target σ)
               | "stress"  (imposed σ, features filter σ, GP target γ)

  convolution  = "causal"  (integral from 0 — material at rest before t=0)
               | "steady"  (integral from -inf — periodic steady state,
                            computed exactly in frequency space)

  quadrature   = "exact" | "midpoint" | "left"   (causal mode only;
                            "left" reproduces rheogp ≤ 0.2 numbers)

Strain control (relaxation representation)
------------------------------------------
    σ(t) = Σ_i c_i x_i(t),   x_i = ∫ φ_i(t-t') γ̇(t') dt'
    Parallel elements add moduli → FKV family has 2 features,
    Maxwell family collapses to a single relaxation kernel.

Stress control (creep representation)
--------------------------------------
    γ(t) = Σ_i j_i x_i(t),   x_i built from the compliance kernels.
    Series elements add compliances → Maxwell family splits into
    simple fractional-integral features (linear in 1/modulus),
    while the parallel FKV family becomes a single Mittag-Leffler
    creep kernel whose internal timescale τ_c is a learnable shape
    parameter.

    Model      strain-control features        stress-control features
    ---------  ----------------------------   -----------------------------------
    Maxwell    [exp(-s/τ_c) ★ γ̇]             [σ,  I¹σ]              (no shape p.)
    Springpot  [D^α γ]                        [I^α σ]
    FMG        [K_FMM(α,0,τ_c) ★ γ̇]          [I^α σ,  σ]
    FML        [K_FMM(1,β,τ_c) ★ γ̇]          [I^β σ,  I¹σ]
    FMM        [K_FMM(α,β,τ_c) ★ γ̇]          [I^α σ,  I^β σ]
    FKV-S      [D^α γ,  γ]                    [J_FKV(α,0,τ_c) ★ σ̇]
    FKV-D      [D^β γ,  γ̇]                   [J_FKV(1,β,τ_c) ★ σ̇]
    FKV        [D^α γ,  D^β γ]                [J_FKV(α,β,τ_c) ★ σ̇]

    J_FKV(s; α, β, τ_c) = s^α · E_{α-β, 1+α}( -(s/τ_c)^{α-β} )
    (the FKV creep compliance shape; prefactor 1/𝕍).

Nomenclature
------------
  𝕍, α   — liquid-like springpot  (α → 1 : viscous)
  𝔾, β   — solid-like  springpot  (β → 0 : elastic)
  constraint:  0 ≤ β ≤ α ≤ 1
  In stress control the recovered sensitivities are compliances
  (1/𝕍, 1/G, 1/η, ...); moduli are reported in derived parameters.
"""

import re

import numpy as np
import torch
import torch.nn.functional as F

from .features import (
    rate, causal_convolve, steady_convolve,
    frac_deriv_causal, frac_integral_causal,
    frac_deriv_steady, frac_integral_steady,
)

# ----------------------------------------------------------------
# Back-compatibility: private helpers from rheogp <= 0.2 that
# existing notebooks import as `rk._caputo_kernel(...)` etc.
# These now live in kernels_legacy.py; prefer rheogp.features for
# new code.
# ----------------------------------------------------------------
from .kernels_legacy import (          # noqa: F401
    _gamma_dot, _fft_convolve, _fft_convolve_analytic,
    _lag_vector, _caputo_kernel, _fmm_kernel, _maxwell_kernel,
)

__all__ = [
    "KERNELS", "PREFACTOR_COLS",
    "KernelFreeKernel",
    "MaxwellKernel", "SpringpotKernel",
    "FractionalMaxwellGelKernel", "FractionalMaxwellLiquidKernel",
    "FractionalMaxwellKernel",
    "FKVSKernel", "FKVDKernel", "FractionalKelvinVoigtKernel",
    "ZenerKernel", "FractionalZenerSolidSKernel",
    "FractionalZenerLiquidSKernel", "FractionalZenerLiquidDKernel",
    "FractionalZenerSKernel", "FractionalZenerKernel",
    "GeneralizedMaxwellKernel",
]

# ================================================================
# Numerics
# ================================================================
_SERIES_TERMS = 30
_Z_THRESH     = 10.0


def _mittag_leffler(z, a, b):
    """
    Real-valued E_{a,b}(z) for scalar tensors a, b and tensor z ≤ 0.
    Hybrid truncated series / 1-term asymptotic for z << -1.
    """
    ml = torch.zeros_like(z)
    for k in range(_SERIES_TERMS):
        denom = torch.exp(torch.lgamma(a * float(k) + b))
        ml    = ml + (z ** float(k)) / (denom + 1e-30)
    ml_asymp  = -1.0 / (z * torch.exp(torch.lgamma(b - a + 1e-8)) + 1e-30)
    use_asymp = (z < -_Z_THRESH).float()
    return (1.0 - use_asymp) * ml + use_asymp * ml_asymp


def _fmm_kernel(s, alpha, beta, tau_c):
    """FMM relaxation kernel  K(s) = (s/τ_c)^{-β} E_{α-β,1-β}(-(s/τ_c)^{α-β})."""
    a = alpha - beta
    b = 1.0   - beta
    u = (s / tau_c).clamp(min=1e-8)
    return (u ** (-beta)) * _mittag_leffler(-(u ** a), a, b)


def _fkv_creep_kernel(s, alpha, beta, tau_c):
    """
    FKV creep-compliance shape (prefactor 1/𝕍 stripped):

        J_shape(s) = s^α · E_{α-β, 1+α}( -(s/τ_c)^{α-β} )
    """
    a = alpha - beta
    b = 1.0 + alpha
    u = (s / tau_c).clamp(min=1e-8)
    return (s ** alpha) * _mittag_leffler(-(u ** a), a, b)


def _to64(p, device):
    if torch.is_tensor(p):
        return p.to(torch.float64)
    return torch.tensor(float(p), dtype=torch.float64, device=device)


# ================================================================
# Scaling helpers  (unchanged semantics from v0.2)
# ================================================================

def _scale(x):
    """Standardise to zero mean / unit std; return (x_scaled, μ, σ)."""
    mu  = x.mean()
    std = x.std()
    return (x - mu) / std, mu.detach(), std.detach()


def _normalize(x):
    """Normalise by max |x| (input-signal scale)."""
    scale = torch.max(torch.abs(x))
    zero  = torch.zeros((), device=x.device, dtype=x.dtype)
    return x / scale, zero, scale


def _raw_to_sigmoid(value: float) -> float:
    value = float(np.clip(value, 1e-4, 1.0 - 1e-4))
    return float(np.log(value / (1.0 - value)))


def _raw_to_softplus(value: float) -> float:
    value = max(value, 1e-4)
    return float(np.log(np.exp(value) - 1.0))


def _no_stress_control(name, extra=""):
    """Raise a clear, consistent error for the Zener-family models,
    whose stress-controlled (creep) representation is not yet
    implemented -- see the 'Fractional Zener family' section below."""
    raise NotImplementedError(
        f"control='stress' (creep) is not yet implemented for "
        f"'{name}'. {extra}Use control='strain' (the default) for "
        "relaxation / SAOS / chirp fitting. Contributions deriving "
        "features_stress() from the closed-form creep law are "
        "welcome -- see CHANGELOG.md.")


# ================================================================
# Base class
# ================================================================

class _BaseKernel(torch.nn.Module):
    """
    Shared machinery for all rheological kernels.

    Subclasses implement:
        _features_strain(u, dt)  -> list[tensor]   (input u = γ)
        _features_stress(u, dt)  -> list[tensor]   (input u = σ)
        _prefactors_strain()     -> list[str]
        _prefactors_stress()     -> list[str]
        _phys_params()           -> dict           (active shape params)
        Gstar(omega, prefactors) -> (G', G'')      (always in Pa)
    """

    def __init__(self):
        super().__init__()
        self.control     = "strain"
        self.convolution = "causal"
        self.quadrature  = "exact"
        self._gamma_scale_frozen  = False
        self._feature_stds_frozen = False

    # -- configuration ------------------------------------------------
    def configure(self, control="strain", convolution="causal",
                  quadrature="exact"):
        if control not in ("strain", "stress"):
            raise ValueError("control must be 'strain' or 'stress'")
        if convolution not in ("causal", "steady"):
            raise ValueError("convolution must be 'causal' or 'steady'")
        if quadrature not in ("exact", "midpoint", "left"):
            raise ValueError("quadrature must be 'exact'|'midpoint'|'left'")
        self.control, self.convolution = control, convolution
        self.quadrature = quadrature
        return self

    # -- public interface ---------------------------------------------
    @property
    def n_features(self):
        return len(self.prefactor_names())

    def _strain_shift_exponents(self):
        """Per-feature fractional order p_j, for kernels whose STRAIN
        features are power-law / memoryless (no tau_c dependence) --
        currently the FKV family. A pure horizontal TTS shift of a
        power-law branch c*t^{-p} is algebraically identical to
        rescaling that branch's prefactor by a_T**p (see CHANGELOG),
        so `rheogp_multi.shift_kernel_class` uses this to multiply
        feature j by a_T(record)**p_j instead of (or in addition to)
        substituting tau_c, for records in a joint/master-curve fit.
        Returning None (the default) means this kernel's strain
        features already depend on tau_c directly and are shifted by
        the existing tau_c-substitution mechanism instead.
        """
        return None

    def prefactor_names(self):
        return (self._prefactors_strain() if self.control == "strain"
                else self._prefactors_stress())

    def param_labels(self):
        return list(self._phys_params().keys())

    def named_phys_params(self):
        return {k: (v.item() if torch.is_tensor(v) else float(v))
                for k, v in self._phys_params().items()}

    def compute_x_physical(self, u, dt):
        """
        Features in PHYSICAL units on the raw (un-normalised) input.
        This is what appears in  target(t) = Σ_i c_i x_i(t)  and what
        should be used for plotting x_i(t) as in the paper figures.
        Returns an (N, n_features) tensor.
        """
        feats = self._dispatch_features(u, dt)
        return torch.stack(feats, dim=1)

    def compute_x(self, u, dt):
        """
        GP input features: input normalised by max|u|, each feature
        standardised.  Freezing flags make predict() reuse the
        training normalisation.
        """
        # ---- input scale --------------------------------------------
        if not self._gamma_scale_frozen:
            u_norm, _, scale_u = _normalize(u)
            self._gamma_scale  = scale_u
        else:
            scale_u = self._gamma_scale
            u_norm  = u / scale_u.clamp(min=1e-12)

        feats = self._dispatch_features(u_norm, dt)

        # ---- feature standardisation --------------------------------
        cols, stds = [], []
        for i, x in enumerate(feats):
            if not self._feature_stds_frozen:
                xs, _, std = _scale(x)
                stds.append(std.item())
            else:
                std = torch.tensor(self._feature_stds[i],
                                   dtype=x.dtype, device=x.device)
                xs  = (x - x.mean()) / std.clamp(min=1e-12)
                stds.append(self._feature_stds[i])
            cols.append(xs)
        if not self._feature_stds_frozen:
            self._feature_stds = stds
        return torch.stack(cols, dim=1)

    # -- dispatch ------------------------------------------------------
    def _dispatch_features(self, u, dt):
        if self.control == "strain":
            return self._features_strain(u, dt)
        return self._features_stress(u, dt)

    # -- building blocks shared by subclasses --------------------------
    def _D(self, u, dt, alpha):
        """Fractional derivative D^α[u] in the active convolution mode."""
        if self.convolution == "steady":
            return frac_deriv_steady(u, dt, alpha)
        return frac_deriv_causal(rate(u, dt), dt, alpha,
                                 quadrature=self.quadrature)

    def _I(self, u, dt, alpha):
        """Fractional integral I^α[u] in the active convolution mode."""
        if self.convolution == "steady":
            return frac_integral_steady(u, dt, alpha)
        return frac_integral_causal(u, dt, alpha,
                                    quadrature=self.quadrature)

    def _rate(self, u, dt):
        """du/dt — spectral in steady mode, forward difference in causal."""
        if self.convolution == "steady":
            return steady_convolve(u, lambda w: 1j * w, dt)
        return rate(u, dt)

    def _conv_rate(self, u, dt, K_fn, H_fn):
        """
        Convolve u̇ with the time kernel K(s) (causal) or apply the
        transfer H(ω) = iω·K̂*(ω) to u (steady).
        """
        if self.convolution == "steady":
            return steady_convolve(u, H_fn, dt)
        return causal_convolve(rate(u, dt), K_fn, dt,
                               quadrature=self.quadrature)

    # -- misc ----------------------------------------------------------
    def _stress_prefactors_to_moduli(self, prefactors):
        raise NotImplementedError

    def Jstar(self, omega, prefactors):
        """Complex compliance J* = 1/G* (available for every model)."""
        Gp, Gdp = self.Gstar(omega, prefactors)
        Gc      = Gp + 1j * Gdp
        Jc      = 1.0 / Gc
        return Jc.real, -Jc.imag        # J', J''


# ================================================================
# Kernel-free  (no constitutive model — diagnostics only)
# ================================================================

class KernelFreeKernel(_BaseKernel):
    """Raw features: [u, u̇].  No physics parameters, no G*(ω)."""

    def __init__(self, use_strain_rate=True):
        super().__init__()
        self.use_strain_rate = use_strain_rate

    def _features_strain(self, u, dt):
        return [u, self._rate(u, dt)] if self.use_strain_rate else [u]

    _features_stress = _features_strain

    def _prefactors_strain(self):
        return (["dsig_dgamma", "dsig_dgammadot"] if self.use_strain_rate
                else ["dsig_dgamma"])

    def _prefactors_stress(self):
        return (["dgam_dsigma", "dgam_dsigmadot"] if self.use_strain_rate
                else ["dgam_dsigma"])

    def _phys_params(self):
        return {}

    def Gstar(self, omega, prefactors):
        raise NotImplementedError(
            "KernelFreeKernel has no constitutive model — G*(ω) is "
            "undefined. Use the time-resolved sensitivities instead.")


# ================================================================
# 1. Maxwell
#    σ + τ_c σ̇ = G_c τ_c γ̇          G* = G_c iωτ_c / (1 + iωτ_c)
# ================================================================

class MaxwellKernel(_BaseKernel):

    def __init__(self, tau_c_init=1.0):
        super().__init__()
        self.raw_tau_c = torch.nn.Parameter(
            torch.tensor(_raw_to_softplus(tau_c_init)))

    @property
    def tau_c(self):
        return F.softplus(self.raw_tau_c) + 1e-4

    # strain: single exponential relaxation feature
    def _features_strain(self, u, dt):
        tau = self.tau_c
        t64 = _to64(tau, u.device)
        x = self._conv_rate(
            u, dt,
            K_fn=lambda s: torch.exp(-s / tau),
            H_fn=lambda w: (1j * w * t64) / (1.0 + 1j * w * t64),
        )
        return [x]

    # stress: γ = σ/G_c + I¹σ/η   → two features, NO shape parameters
    def _features_stress(self, u, dt):
        return [u, self._I(u, dt, torch.ones((), dtype=u.dtype,
                                             device=u.device))]

    def _prefactors_strain(self):
        return ["Gc"]

    def _prefactors_stress(self):
        return ["invGc", "invEta"]

    def _phys_params(self):
        if self.control == "stress":
            return {}                    # τ_c is derived, not fitted
        return {"tau_c": self.tau_c}

    def _stress_prefactors_to_moduli(self, pf):
        Gc  = 1.0 / pf["invGc"]
        eta = 1.0 / pf["invEta"]
        return dict(Gc=Gc, tau_c=eta / Gc)

    def Gstar(self, omega, prefactors):
        if self.control == "stress":
            p  = self._stress_prefactors_to_moduli(prefactors)
            Gc, tc = p["Gc"], p["tau_c"]
        else:
            Gc, tc = prefactors["Gc"], self.tau_c.item()
        z  = 1j * omega * tc
        Gs = Gc * z / (1.0 + z)
        return Gs.real, Gs.imag


# ================================================================
# 2. Springpot        σ = 𝕍 D^α[γ]         G* = 𝕍 (iω)^α
# ================================================================

class SpringpotKernel(_BaseKernel):

    def __init__(self, alpha_init=0.5):
        super().__init__()
        self.raw_alpha = torch.nn.Parameter(
            torch.tensor(_raw_to_sigmoid(alpha_init)))

    @property
    def alpha(self):
        return torch.sigmoid(self.raw_alpha)

    def _features_strain(self, u, dt):
        return [self._D(u, dt, self.alpha)]

    def _strain_shift_exponents(self):
        return [self.alpha]

    def _features_stress(self, u, dt):
        return [self._I(u, dt, self.alpha)]

    def _prefactors_strain(self):
        return ["V"]

    def _prefactors_stress(self):
        return ["invV"]

    def _phys_params(self):
        return {"alpha": self.alpha}

    def _stress_prefactors_to_moduli(self, pf):
        return dict(V=1.0 / pf["invV"])

    def Gstar(self, omega, prefactors):
        V = (1.0 / prefactors["invV"] if self.control == "stress"
             else prefactors["V"])
        Gs = V * (1j * omega) ** self.alpha.item()
        return Gs.real, Gs.imag


# ================================================================
# 3. Fractional Maxwell Gel (FMG)  — spring + springpot in series
# ================================================================

class FractionalMaxwellGelKernel(_BaseKernel):

    def __init__(self, alpha_init=0.5, tau_c_init=1.0):
        super().__init__()
        self.raw_alpha = torch.nn.Parameter(
            torch.tensor(_raw_to_sigmoid(alpha_init)))
        self.raw_tau_c = torch.nn.Parameter(
            torch.tensor(_raw_to_softplus(tau_c_init)))

    @property
    def alpha(self):
        return torch.sigmoid(self.raw_alpha)

    @property
    def tau_c(self):
        return F.softplus(self.raw_tau_c) + 1e-4

    def _features_strain(self, u, dt):
        a, tau = self.alpha, self.tau_c
        a64, t64 = _to64(a, u.device), _to64(tau, u.device)
        beta0 = torch.zeros((), device=u.device, dtype=u.dtype)
        x = self._conv_rate(
            u, dt,
            K_fn=lambda s: _fmm_kernel(s, a, beta0, tau),
            H_fn=lambda w: (1j * w * t64) ** a64
                           / (1.0 + (1j * w * t64) ** a64),
        )
        return [x]

    # stress: γ = (1/𝕍) I^α σ + σ/G   (series → compliances add)
    def _features_stress(self, u, dt):
        return [self._I(u, dt, self.alpha), u]

    def _prefactors_strain(self):
        return ["Gc"]

    def _prefactors_stress(self):
        return ["invV", "invG"]

    def _phys_params(self):
        if self.control == "stress":
            return {"alpha": self.alpha}   # τ_c derived from 𝕍/G
        return {"alpha": self.alpha, "tau_c": self.tau_c}

    def _stress_prefactors_to_moduli(self, pf):
        V = 1.0 / pf["invV"]; G = 1.0 / pf["invG"]
        a = self.alpha.item()
        tc = (V / G) ** (1.0 / a)
        return dict(V=V, G=G, tau_c=tc, Gc=G)

    def Gstar(self, omega, prefactors):
        a = self.alpha.item()
        if self.control == "stress":
            p = self._stress_prefactors_to_moduli(prefactors)
            Gc, tc = p["Gc"], p["tau_c"]
        else:
            Gc, tc = prefactors["Gc"], self.tau_c.item()
        wtc   = omega * tc
        cos_a = np.cos(np.pi * a / 2)
        denom = 1.0 + wtc**(2*a) + 2.0 * wtc**a * cos_a
        Gp    = Gc * (wtc**(2*a) + wtc**a * cos_a) / denom
        Gdp   = Gc * wtc**a * np.sin(np.pi * a / 2) / denom
        return Gp, Gdp


# ================================================================
# 4. Fractional Maxwell Liquid (FML) — springpot + dashpot in series
# ================================================================

class FractionalMaxwellLiquidKernel(_BaseKernel):

    def __init__(self, beta_init=0.5, tau_c_init=1.0):
        super().__init__()
        self.raw_beta  = torch.nn.Parameter(
            torch.tensor(_raw_to_sigmoid(beta_init)))
        self.raw_tau_c = torch.nn.Parameter(
            torch.tensor(_raw_to_softplus(tau_c_init)))

    @property
    def beta(self):
        return torch.sigmoid(self.raw_beta)

    @property
    def tau_c(self):
        return F.softplus(self.raw_tau_c) + 1e-4

    def _features_strain(self, u, dt):
        b, tau = self.beta, self.tau_c
        b64, t64 = _to64(b, u.device), _to64(tau, u.device)
        alpha1 = torch.ones((), device=u.device, dtype=u.dtype) * 0.9999
        x = self._conv_rate(
            u, dt,
            K_fn=lambda s: _fmm_kernel(s, alpha1, b, tau),
            H_fn=lambda w: (1j * w * t64)
                           / (1.0 + (1j * w * t64) ** (1.0 - b64)),
        )
        return [x]

    # stress: γ = (1/𝔾) I^β σ + (1/η) I¹ σ
    def _features_stress(self, u, dt):
        one = torch.ones((), dtype=u.dtype, device=u.device)
        return [self._I(u, dt, self.beta), self._I(u, dt, one)]

    def _prefactors_strain(self):
        return ["Gc"]

    def _prefactors_stress(self):
        return ["invG_bb", "invEta"]

    def _phys_params(self):
        if self.control == "stress":
            return {"beta": self.beta}
        return {"beta": self.beta, "tau_c": self.tau_c}

    def _stress_prefactors_to_moduli(self, pf):
        G_bb = 1.0 / pf["invG_bb"]; eta = 1.0 / pf["invEta"]
        b  = self.beta.item()
        tc = (eta / G_bb) ** (1.0 / (1.0 - b))
        return dict(G_bb=G_bb, eta=eta, tau_c=tc, Gc=eta / tc)

    def Gstar(self, omega, prefactors):
        b = self.beta.item()
        if self.control == "stress":
            p = self._stress_prefactors_to_moduli(prefactors)
            Gc, tc = p["Gc"], p["tau_c"]
        else:
            Gc, tc = prefactors["Gc"], self.tau_c.item()
        wtc    = omega * tc
        cos_1b = np.cos(np.pi * (1 - b) / 2)
        denom  = 1.0 + wtc**(2*(1-b)) + 2.0 * wtc**(1-b) * cos_1b
        Gp  = Gc * wtc**(2-b) * np.cos(np.pi * b / 2) / denom
        Gdp = Gc * (wtc + wtc**(2-b) * np.sin(np.pi * b / 2)) / denom
        return Gp, Gdp


# ================================================================
# 5. Fractional Maxwell (FMM) — two springpots in series
# ================================================================

class FractionalMaxwellKernel(_BaseKernel):

    def __init__(self, alpha_init=0.7, beta_init=0.3, tau_c_init=1.0):
        super().__init__()
        self.raw_alpha = torch.nn.Parameter(
            torch.tensor(_raw_to_sigmoid(alpha_init)))
        # β = α · sigmoid(raw_gap)  →  0 < β < α always
        gap0 = beta_init / max(alpha_init, 1e-3)
        self.raw_gap = torch.nn.Parameter(
            torch.tensor(_raw_to_sigmoid(gap0)))
        self.raw_tau_c = torch.nn.Parameter(
            torch.tensor(_raw_to_softplus(tau_c_init)))

    @property
    def alpha(self):
        return torch.sigmoid(self.raw_alpha)

    @property
    def beta(self):
        return self.alpha * torch.sigmoid(self.raw_gap)

    @property
    def tau_c(self):
        return F.softplus(self.raw_tau_c) + 1e-4

    def _features_strain(self, u, dt):
        a, b, tau = self.alpha, self.beta, self.tau_c
        a64 = _to64(a, u.device); b64 = _to64(b, u.device)
        t64 = _to64(tau, u.device)
        x = self._conv_rate(
            u, dt,
            K_fn=lambda s: _fmm_kernel(s, a, b, tau),
            H_fn=lambda w: (1j * w * t64) ** a64
                           / (1.0 + (1j * w * t64) ** (a64 - b64)),
        )
        return [x]

    # stress: γ = (1/𝕍) I^α σ + (1/𝔾) I^β σ
    def _features_stress(self, u, dt):
        return [self._I(u, dt, self.alpha), self._I(u, dt, self.beta)]

    def _prefactors_strain(self):
        return ["Gc"]

    def _prefactors_stress(self):
        return ["invV", "invG_bb"]

    def _phys_params(self):
        if self.control == "stress":
            return {"alpha": self.alpha, "beta": self.beta}
        return {"alpha": self.alpha, "beta": self.beta,
                "tau_c": self.tau_c}

    def _stress_prefactors_to_moduli(self, pf):
        V = 1.0 / pf["invV"]; G_bb = 1.0 / pf["invG_bb"]
        a, b = self.alpha.item(), self.beta.item()
        tc = (V / G_bb) ** (1.0 / max(a - b, 1e-6))
        return dict(V=V, G_bb=G_bb, tau_c=tc, Gc=V * tc ** (-a))

    def Gstar(self, omega, prefactors):
        a, b = self.alpha.item(), self.beta.item()
        if self.control == "stress":
            p = self._stress_prefactors_to_moduli(prefactors)
            Gc, tc = p["Gc"], p["tau_c"]
        else:
            Gc, tc = prefactors["Gc"], self.tau_c.item()
        wtc = omega * tc
        cos_ab = np.cos(np.pi * (a - b) / 2)
        denom  = 1.0 + wtc**(2*(a-b)) + 2.0 * wtc**(a-b) * cos_ab
        Gp  = Gc * (wtc**a * np.cos(np.pi*a/2)
                    + wtc**(2*a-b) * np.cos(np.pi*b/2)) / denom
        Gdp = Gc * (wtc**a * np.sin(np.pi*a/2)
                    + wtc**(2*a-b) * np.sin(np.pi*b/2)) / denom
        return Gp, Gdp


# ================================================================
# Shared creep machinery for the FKV family (parallel elements)
# ================================================================

class _FKVCreepMixin:
    """Single Mittag-Leffler creep feature  x = J_shape ★ σ̇."""

    def _fkv_creep_feature(self, u, dt, alpha, beta):
        tau = self.tau_c
        a64 = _to64(alpha, u.device); b64 = _to64(beta, u.device)
        t64 = _to64(tau, u.device)
        return self._conv_rate(
            u, dt,
            K_fn=lambda s: _fkv_creep_kernel(s, alpha, beta, tau),
            # J*(ω)·iω = 1 / [ (iω)^α + τ_c^{-(α-β)} (iω)^β ]
            H_fn=lambda w: 1.0 / ((1j * w) ** a64
                                  + t64 ** (-(a64 - b64)) * (1j * w) ** b64),
        )


# ================================================================
# 6. FKV-S  — springpot(𝕍,α) ∥ spring(G)
# ================================================================

class FKVSKernel(_BaseKernel, _FKVCreepMixin):

    def __init__(self, alpha_init=0.5, tau_c_init=1.0):
        super().__init__()
        self.raw_alpha = torch.nn.Parameter(
            torch.tensor(_raw_to_sigmoid(alpha_init)))
        # τ_c only active under stress control
        self.raw_tau_c = torch.nn.Parameter(
            torch.tensor(_raw_to_softplus(tau_c_init)))

    @property
    def alpha(self):
        return torch.sigmoid(self.raw_alpha)

    @property
    def tau_c(self):
        return F.softplus(self.raw_tau_c) + 1e-4

    def _features_strain(self, u, dt):
        return [self._D(u, dt, self.alpha), u]

    def _strain_shift_exponents(self):
        # branch 1: springpot(alpha) -> shifts as a_T**alpha
        # branch 2: plain spring (order 0) -> a_T**0 = 1, no shift
        return [self.alpha, torch.zeros((), device=self.raw_alpha.device,
                                        dtype=self.raw_alpha.dtype)]

    def _features_stress(self, u, dt):
        beta0 = torch.zeros((), device=u.device, dtype=u.dtype)
        return [self._fkv_creep_feature(u, dt, self.alpha, beta0)]

    def _prefactors_strain(self):
        return ["V", "G"]

    def _prefactors_stress(self):
        return ["invV"]

    def _phys_params(self):
        if self.control == "stress":
            return {"alpha": self.alpha, "tau_c": self.tau_c}
        return {"alpha": self.alpha}

    def _stress_prefactors_to_moduli(self, pf):
        V = 1.0 / pf["invV"]
        a = self.alpha.item(); tc = self.tau_c.item()
        return dict(V=V, G=V * tc ** (-a), tau_c=tc)

    def Gstar(self, omega, prefactors):
        a = self.alpha.item()
        if self.control == "stress":
            p = self._stress_prefactors_to_moduli(prefactors)
            V, G = p["V"], p["G"]
        else:
            V, G = prefactors["V"], prefactors["G"]
        Gp  = V * omega**a * np.cos(np.pi * a / 2) + G
        Gdp = V * omega**a * np.sin(np.pi * a / 2)
        return Gp, Gdp


# ================================================================
# 7. FKV-D  — dashpot(η) ∥ springpot(𝔾,β)
# ================================================================

class FKVDKernel(_BaseKernel, _FKVCreepMixin):

    def __init__(self, beta_init=0.5, tau_c_init=1.0):
        super().__init__()
        self.raw_beta = torch.nn.Parameter(
            torch.tensor(_raw_to_sigmoid(beta_init)))
        self.raw_tau_c = torch.nn.Parameter(
            torch.tensor(_raw_to_softplus(tau_c_init)))

    @property
    def beta(self):
        return torch.sigmoid(self.raw_beta)

    @property
    def tau_c(self):
        return F.softplus(self.raw_tau_c) + 1e-4

    def _features_strain(self, u, dt):
        return [self._D(u, dt, self.beta), self._rate(u, dt)]

    def _strain_shift_exponents(self):
        # branch 1: springpot(beta) -> a_T**beta
        # branch 2: plain dashpot (order 1, the alpha->1 limit) -> a_T**1
        return [self.beta, torch.ones((), device=self.raw_beta.device,
                                      dtype=self.raw_beta.dtype)]

    def _features_stress(self, u, dt):
        alpha1 = torch.ones((), device=u.device, dtype=u.dtype) * 0.9999
        return [self._fkv_creep_feature(u, dt, alpha1, self.beta)]

    def _prefactors_strain(self):
        return ["G_bb", "eta"]

    def _prefactors_stress(self):
        return ["invEta"]

    def _phys_params(self):
        if self.control == "stress":
            return {"beta": self.beta, "tau_c": self.tau_c}
        return {"beta": self.beta}

    def _stress_prefactors_to_moduli(self, pf):
        eta = 1.0 / pf["invEta"]
        b = self.beta.item(); tc = self.tau_c.item()
        return dict(eta=eta, G_bb=eta * tc ** (-(1.0 - b)), tau_c=tc)

    def Gstar(self, omega, prefactors):
        b = self.beta.item()
        if self.control == "stress":
            p = self._stress_prefactors_to_moduli(prefactors)
            G_bb, eta = p["G_bb"], p["eta"]
        else:
            G_bb, eta = prefactors["G_bb"], prefactors["eta"]
        Gp  = G_bb * omega**b * np.cos(np.pi * b / 2)
        Gdp = eta * omega + G_bb * omega**b * np.sin(np.pi * b / 2)
        return Gp, Gdp


# ================================================================
# 8. FKV  — springpot(𝕍,α) ∥ springpot(𝔾,β)
# ================================================================

class FractionalKelvinVoigtKernel(_BaseKernel, _FKVCreepMixin):

    def __init__(self, alpha_init=0.7, beta_init=0.3, tau_c_init=1.0):
        super().__init__()
        self.raw_alpha = torch.nn.Parameter(
            torch.tensor(_raw_to_sigmoid(alpha_init)))
        gap0 = beta_init / max(alpha_init, 1e-3)
        self.raw_gap = torch.nn.Parameter(
            torch.tensor(_raw_to_sigmoid(gap0)))
        self.raw_tau_c = torch.nn.Parameter(
            torch.tensor(_raw_to_softplus(tau_c_init)))

    @property
    def alpha(self):
        return torch.sigmoid(self.raw_alpha)

    @property
    def beta(self):
        return self.alpha * torch.sigmoid(self.raw_gap)

    @property
    def tau_c(self):
        return F.softplus(self.raw_tau_c) + 1e-4

    def _features_strain(self, u, dt):
        return [self._D(u, dt, self.alpha), self._D(u, dt, self.beta)]

    def _strain_shift_exponents(self):
        # branch 1: springpot(alpha) -> a_T**alpha
        # branch 2: springpot(beta)  -> a_T**beta
        return [self.alpha, self.beta]

    def _features_stress(self, u, dt):
        return [self._fkv_creep_feature(u, dt, self.alpha, self.beta)]

    def _prefactors_strain(self):
        return ["V", "G_bb"]

    def _prefactors_stress(self):
        return ["invV"]

    def _phys_params(self):
        if self.control == "stress":
            return {"alpha": self.alpha, "beta": self.beta,
                    "tau_c": self.tau_c}
        return {"alpha": self.alpha, "beta": self.beta}

    def _stress_prefactors_to_moduli(self, pf):
        V = 1.0 / pf["invV"]
        a, b = self.alpha.item(), self.beta.item()
        tc = self.tau_c.item()
        return dict(V=V, G_bb=V * tc ** (-(a - b)), tau_c=tc)

    def Gstar(self, omega, prefactors):
        a, b = self.alpha.item(), self.beta.item()
        if self.control == "stress":
            p = self._stress_prefactors_to_moduli(prefactors)
            V, G_bb = p["V"], p["G_bb"]
        else:
            V, G_bb = prefactors["V"], prefactors["G_bb"]
        Gp  = (V * omega**a * np.cos(np.pi*a/2)
               + G_bb * omega**b * np.cos(np.pi*b/2))
        Gdp = (V * omega**a * np.sin(np.pi*a/2)
               + G_bb * omega**b * np.sin(np.pi*b/2))
        return Gp, Gdp


# ================================================================
# Fractional Zener family
#
#   Every model here is ONE parallel element -- a spring, a dashpot,
#   or a springpot -- in parallel with ONE "Maxwell arm": a classical
#   Maxwell element, or one of FMG / FML / FMM (spring/springpot +
#   springpot/dashpot in series).  Complex moduli of parallel branches
#   simply ADD, G*(ω) = G*_parallel(ω) + G*_arm(ω), and so -- in
#   strain control -- do the two convolution FEATURES, each with its
#   own independent prefactor.  This is exactly the pattern already
#   used by the FKV family (springpot ∥ springpot) above; the only
#   difference is that the second branch here is a *relaxing* Maxwell
#   arm instead of a second bare springpot.
#
#   _MaxwellArmMixin provides that one shared building block, used
#   unmodified by all six models below:
#
#     (alpha, beta) = (1, 0)   ->  classical Maxwell arm  (Zener)
#     (alpha, beta) = (a, 0)   ->  FMG arm   (FractionalZenerSolidS)
#     (alpha, beta) = (~1, b)  ->  FML arm   (FractionalZener{LiquidS,LiquidD})
#     (alpha, beta) = (a, b)   ->  FMM arm   (FractionalZenerS, FractionalZener)
#
#   because E_{1,1}(z) = exp(z), i.e. `_fmm_kernel` with alpha=1,
#   beta=0 IS the classical Maxwell kernel exactly.
#
#   Stress control (creep) is NOT implemented for this family: unlike
#   the FKV family (a single Mittag-Leffler creep kernel whose OWN
#   internal timescale is learnable), a parallel spring/dashpot here
#   changes the *decay rate itself* of the Maxwell-arm's creep response
#   (e.g. classical Zener: J(t) = 1/G0 - G1/(G0(G0+G1))·exp(-t/τc'),
#   τc' = τc·G0/(G0+G1)), so features_stress() needs its own dedicated
#   derivation rather than a direct reuse of the strain-control
#   machinery.  All six models raise NotImplementedError for
#   control="stress" with a message pointing here; see CHANGELOG.md.
# ================================================================

class _MaxwellArmMixin:
    """
    Shared building block for the Fractional Zener family: one
    convolution feature for a generalised Maxwell arm (spring/
    springpot + springpot/dashpot in series), plus the matching
    contribution to G*(ω).  Reused, unmodified, by every model below.
    """

    def _maxwell_arm_feature(self, u, dt, alpha, beta, tau_c):
        a64 = _to64(alpha, u.device)
        b64 = _to64(beta, u.device)
        t64 = _to64(tau_c, u.device)
        return self._conv_rate(
            u, dt,
            K_fn=lambda s: _fmm_kernel(s, alpha, beta, tau_c),
            H_fn=lambda w: (1j * w * t64) ** a64
                           / (1.0 + (1j * w * t64) ** (a64 - b64)),
        )

    @staticmethod
    def _maxwell_arm_Gstar(omega, Gc, tau_c, alpha, beta):
        """
        G'(ω), G''(ω) of the generalised Maxwell arm ALONE (no
        parallel branch added yet) -- the same formula used by
        Maxwell / FMG / FML / FMM above, written once for general
        (alpha, beta).  Reduces to each of their Gstar() formulas
        exactly at (1,0) / (a,0) / (~1,b) / (a,b).
        """
        wtc = omega * tau_c
        d_ab = alpha - beta
        cos_ab = np.cos(np.pi * d_ab / 2)
        denom = 1.0 + wtc**(2 * d_ab) + 2.0 * wtc**d_ab * cos_ab
        Gp = Gc * (wtc**alpha * np.cos(np.pi * alpha / 2)
                   + wtc**(2 * alpha - beta) * np.cos(np.pi * beta / 2)) / denom
        Gdp = Gc * (wtc**alpha * np.sin(np.pi * alpha / 2)
                    + wtc**(2 * alpha - beta) * np.sin(np.pi * beta / 2)) / denom
        return Gp, Gdp


# ================================================================
# 9. Zener (standard linear solid)   spring(G0) ∥ Maxwell(Gc, τc)
#
#    σ + τc σ̇ = G0 γ + (G0+Gc)τc γ̇          τc = η/G1,  Gc = G1
#    G* = G0 + Gc·iωτc/(1+iωτc)
# ================================================================

class ZenerKernel(_BaseKernel, _MaxwellArmMixin):

    def __init__(self, tau_c_init=1.0):
        super().__init__()
        self.raw_tau_c = torch.nn.Parameter(
            torch.tensor(_raw_to_softplus(tau_c_init)))

    @property
    def tau_c(self):
        return F.softplus(self.raw_tau_c) + 1e-4

    def _features_strain(self, u, dt):
        one  = torch.ones((),  dtype=u.dtype, device=u.device)
        zero = torch.zeros((), dtype=u.dtype, device=u.device)
        x_arm = self._maxwell_arm_feature(u, dt, one, zero, self.tau_c)
        return [x_arm, u]

    def _features_stress(self, u, dt):
        _no_stress_control("Zener")

    def _prefactors_strain(self):
        return ["Gc", "G0"]

    def _prefactors_stress(self):
        _no_stress_control("Zener")

    def _phys_params(self):
        if self.control == "stress":
            _no_stress_control("Zener")
        return {"tau_c": self.tau_c}

    def _stress_prefactors_to_moduli(self, pf):
        _no_stress_control("Zener")

    def Gstar(self, omega, prefactors):
        Gc, G0 = prefactors["Gc"], prefactors["G0"]
        tc = self.tau_c.item()
        Gp_arm, Gdp_arm = self._maxwell_arm_Gstar(omega, Gc, tc, 1.0, 0.0)
        return G0 + Gp_arm, Gdp_arm


# ================================================================
# 10. Fractional Zener Solid-S      spring(G0) ∥ FMG(𝕍, α, G1)
#     (the Maxwell arm is a Fractional Maxwell Gel: springpot + spring)
# ================================================================

class FractionalZenerSolidSKernel(_BaseKernel, _MaxwellArmMixin):

    def __init__(self, alpha_init=0.5, tau_c_init=1.0):
        super().__init__()
        self.raw_alpha = torch.nn.Parameter(
            torch.tensor(_raw_to_sigmoid(alpha_init)))
        self.raw_tau_c = torch.nn.Parameter(
            torch.tensor(_raw_to_softplus(tau_c_init)))

    @property
    def alpha(self):
        return torch.sigmoid(self.raw_alpha)

    @property
    def tau_c(self):
        return F.softplus(self.raw_tau_c) + 1e-4

    def _features_strain(self, u, dt):
        beta0 = torch.zeros((), dtype=u.dtype, device=u.device)
        x_arm = self._maxwell_arm_feature(u, dt, self.alpha, beta0, self.tau_c)
        return [x_arm, u]

    def _features_stress(self, u, dt):
        _no_stress_control("FractionalZenerSolidS")

    def _prefactors_strain(self):
        return ["Gc", "G0"]

    def _prefactors_stress(self):
        _no_stress_control("FractionalZenerSolidS")

    def _phys_params(self):
        if self.control == "stress":
            _no_stress_control("FractionalZenerSolidS")
        return {"alpha": self.alpha, "tau_c": self.tau_c}

    def _stress_prefactors_to_moduli(self, pf):
        _no_stress_control("FractionalZenerSolidS")

    def Gstar(self, omega, prefactors):
        Gc, G0 = prefactors["Gc"], prefactors["G0"]
        a, tc = self.alpha.item(), self.tau_c.item()
        Gp_arm, Gdp_arm = self._maxwell_arm_Gstar(omega, Gc, tc, a, 0.0)
        return G0 + Gp_arm, Gdp_arm


# ================================================================
# 11. Fractional Zener Liquid-S     spring(G0) ∥ FML(𝔾, β, η)
#     (the Maxwell arm is a Fractional Maxwell Liquid: springpot + dashpot)
# ================================================================

class FractionalZenerLiquidSKernel(_BaseKernel, _MaxwellArmMixin):

    def __init__(self, beta_init=0.5, tau_c_init=1.0):
        super().__init__()
        self.raw_beta = torch.nn.Parameter(
            torch.tensor(_raw_to_sigmoid(beta_init)))
        self.raw_tau_c = torch.nn.Parameter(
            torch.tensor(_raw_to_softplus(tau_c_init)))

    @property
    def beta(self):
        return torch.sigmoid(self.raw_beta)

    @property
    def tau_c(self):
        return F.softplus(self.raw_tau_c) + 1e-4

    def _features_strain(self, u, dt):
        alpha1 = torch.ones((), dtype=u.dtype, device=u.device) * 0.9999
        x_arm = self._maxwell_arm_feature(u, dt, alpha1, self.beta, self.tau_c)
        return [x_arm, u]

    def _features_stress(self, u, dt):
        _no_stress_control("FractionalZenerLiquidS")

    def _prefactors_strain(self):
        return ["Gc", "G0"]

    def _prefactors_stress(self):
        _no_stress_control("FractionalZenerLiquidS")

    def _phys_params(self):
        if self.control == "stress":
            _no_stress_control("FractionalZenerLiquidS")
        return {"beta": self.beta, "tau_c": self.tau_c}

    def _stress_prefactors_to_moduli(self, pf):
        _no_stress_control("FractionalZenerLiquidS")

    def Gstar(self, omega, prefactors):
        Gc, G0 = prefactors["Gc"], prefactors["G0"]
        b, tc = self.beta.item(), self.tau_c.item()
        Gp_arm, Gdp_arm = self._maxwell_arm_Gstar(omega, Gc, tc, 1.0, b)
        return G0 + Gp_arm, Gdp_arm


# ================================================================
# 12. Fractional Zener Liquid-D     dashpot(η_p) ∥ FML(𝔾, β, η_s)
#     (a parallel dashpot instead of a spring -> terminal liquid)
# ================================================================

class FractionalZenerLiquidDKernel(_BaseKernel, _MaxwellArmMixin):

    def __init__(self, beta_init=0.5, tau_c_init=1.0):
        super().__init__()
        self.raw_beta = torch.nn.Parameter(
            torch.tensor(_raw_to_sigmoid(beta_init)))
        self.raw_tau_c = torch.nn.Parameter(
            torch.tensor(_raw_to_softplus(tau_c_init)))

    @property
    def beta(self):
        return torch.sigmoid(self.raw_beta)

    @property
    def tau_c(self):
        return F.softplus(self.raw_tau_c) + 1e-4

    def _features_strain(self, u, dt):
        alpha1 = torch.ones((), dtype=u.dtype, device=u.device) * 0.9999
        x_arm = self._maxwell_arm_feature(u, dt, alpha1, self.beta, self.tau_c)
        return [x_arm, self._rate(u, dt)]

    def _features_stress(self, u, dt):
        _no_stress_control("FractionalZenerLiquidD")

    def _strain_shift_exponents(self):
        # feature 0: Maxwell arm, shifted through tau_c      -> a_T**0
        # feature 1: plain dashpot eta_p * du/dt, transfer i w, which
        #            under w -> a_T w scales as a_T**1
        zero = torch.zeros((), dtype=self.raw_tau_c.dtype,
                           device=self.raw_tau_c.device)
        return [zero, zero + 1.0]

    def _prefactors_strain(self):
        return ["Gc", "eta_p"]

    def _prefactors_stress(self):
        _no_stress_control("FractionalZenerLiquidD")

    def _phys_params(self):
        if self.control == "stress":
            _no_stress_control("FractionalZenerLiquidD")
        return {"beta": self.beta, "tau_c": self.tau_c}

    def _stress_prefactors_to_moduli(self, pf):
        _no_stress_control("FractionalZenerLiquidD")

    def Gstar(self, omega, prefactors):
        Gc, eta_p = prefactors["Gc"], prefactors["eta_p"]
        b, tc = self.beta.item(), self.tau_c.item()
        Gp_arm, Gdp_arm = self._maxwell_arm_Gstar(omega, Gc, tc, 1.0, b)
        return Gp_arm, eta_p * omega + Gdp_arm


# ================================================================
# 13. Fractional Zener-S     spring(G0) ∥ FMM(𝕍, α, 𝔾, β)
#     (the Maxwell arm is a full Fractional Maxwell: two springpots)
# ================================================================

class FractionalZenerSKernel(_BaseKernel, _MaxwellArmMixin):

    def __init__(self, alpha_init=0.7, beta_init=0.3, tau_c_init=1.0):
        super().__init__()
        self.raw_alpha = torch.nn.Parameter(
            torch.tensor(_raw_to_sigmoid(alpha_init)))
        gap0 = beta_init / max(alpha_init, 1e-3)
        self.raw_gap = torch.nn.Parameter(
            torch.tensor(_raw_to_sigmoid(gap0)))
        self.raw_tau_c = torch.nn.Parameter(
            torch.tensor(_raw_to_softplus(tau_c_init)))

    @property
    def alpha(self):
        return torch.sigmoid(self.raw_alpha)

    @property
    def beta(self):
        return self.alpha * torch.sigmoid(self.raw_gap)

    @property
    def tau_c(self):
        return F.softplus(self.raw_tau_c) + 1e-4

    def _features_strain(self, u, dt):
        x_arm = self._maxwell_arm_feature(u, dt, self.alpha, self.beta,
                                          self.tau_c)
        return [x_arm, u]

    def _features_stress(self, u, dt):
        _no_stress_control("FractionalZenerS")

    def _prefactors_strain(self):
        return ["Gc", "G0"]

    def _prefactors_stress(self):
        _no_stress_control("FractionalZenerS")

    def _phys_params(self):
        if self.control == "stress":
            _no_stress_control("FractionalZenerS")
        return {"alpha": self.alpha, "beta": self.beta, "tau_c": self.tau_c}

    def _stress_prefactors_to_moduli(self, pf):
        _no_stress_control("FractionalZenerS")

    def Gstar(self, omega, prefactors):
        Gc, G0 = prefactors["Gc"], prefactors["G0"]
        a, b, tc = self.alpha.item(), self.beta.item(), self.tau_c.item()
        Gp_arm, Gdp_arm = self._maxwell_arm_Gstar(omega, Gc, tc, a, b)
        return G0 + Gp_arm, Gdp_arm


# ================================================================
# 14. Fractional Zener (general)     springpot(𝕂, κ) ∥ FMM(𝕍, α, 𝔾, β)
#     The most general model in the family; Zener, FKV-type parallels,
#     FMG, FML and FMM all arise as special/limiting cases.
# ================================================================

class FractionalZenerKernel(_BaseKernel, _MaxwellArmMixin):
    """Three springpots: a fractional Maxwell arm -- springpot(alpha) in
    series with springpot(beta) -- in parallel with springpot(kappa).

        G*(w) = K (i w)^kappa
                + Gc (i w tau_c)^alpha / [1 + (i w tau_c)^(alpha-beta)]

    Any of the three exponents can be HELD FIXED instead of fitted, via
    `fix_alpha` / `fix_beta` / `fix_kappa`. A fixed exponent is stored as
    a buffer, never sees a gradient, is excluded from `_phys_params()` so
    it does not count towards k in AIC/BIC, and is reported separately by
    `fixed_phys_params()`. Passing None (the default) leaves that
    exponent free, which reproduces the original behaviour exactly.

    Fixing exponents is what makes this one kernel cover the whole
    family, including the degenerate members that the free
    parameterisation can only approach asymptotically:

        fix_kappa=0    parallel branch is a SPRING        -> FZS
        fix_kappa=1    parallel branch is a DASHPOT       -> FZLD
        fix_beta=0     arm is springpot + spring          -> FZSS-like
        fix_alpha=1    arm is dashpot + springpot         -> FZLS-like
        fix_beta=0, fix_kappa=0                           -> Zener-like

    The sigmoid parameterisation cannot reach 0 or 1, so those limits are
    only available by fixing. That matters for model selection: a fitted
    exponent pinned against a boundary is not evidence that the boundary
    is right, whereas fixing it makes the restricted model an explicit
    candidate with one fewer parameter.

    The ordering constraint beta <= alpha of the Maxwell arm is preserved
    in every combination; see `_ORDER` note in __init__.
    """

    def __init__(self, alpha_init=0.7, beta_init=0.3, kappa_init=0.1,
                 tau_c_init=1.0, fix_alpha=None, fix_beta=None,
                 fix_kappa=None):
        super().__init__()

        def _chk(v, name):
            if v is None:
                return None
            v = float(v)
            if not (0.0 <= v <= 1.0):
                raise ValueError(
                    f"{name} must lie in [0, 1] (got {v}); it is the order "
                    "of a springpot, 0 = spring, 1 = dashpot.")
            return v

        fa, fb, fk = (_chk(fix_alpha, "fix_alpha"), _chk(fix_beta, "fix_beta"),
                      _chk(fix_kappa, "fix_kappa"))
        if fa is not None and fb is not None and fb > fa:
            raise ValueError(
                f"fix_beta ({fb}) must not exceed fix_alpha ({fa}): the "
                "Maxwell arm requires beta <= alpha.")
        self.fix_alpha, self.fix_beta, self.fix_kappa = fa, fb, fk

        # -- alpha / beta ------------------------------------------------
        # beta <= alpha has to hold however many of the two are fixed, so
        # the free one is parameterised against the other:
        #   both free      alpha = sig(a),  beta = alpha * sig(g)
        #   alpha fixed    beta  = alpha_fix * sig(g)
        #   beta fixed     alpha = beta_fix + (1 - beta_fix) * sig(a)
        #   both fixed     nothing to parameterise
        if fa is None:
            lo = fb if fb is not None else 0.0
            init = max(min(float(alpha_init), 1.0 - 1e-6), lo + 1e-6)
            self.raw_alpha = torch.nn.Parameter(
                torch.tensor(_raw_to_sigmoid((init - lo) / max(1.0 - lo, 1e-6))))
        else:
            self.register_buffer("_alpha_fixed", torch.tensor(fa))
        if fb is None:
            a_ref = fa if fa is not None else float(alpha_init)
            gap0 = float(beta_init) / max(a_ref, 1e-3)
            self.raw_gap = torch.nn.Parameter(
                torch.tensor(_raw_to_sigmoid(min(max(gap0, 1e-6), 1 - 1e-6))))
        else:
            self.register_buffer("_beta_fixed", torch.tensor(fb))

        # -- kappa -------------------------------------------------------
        if fk is None:
            self.raw_kappa = torch.nn.Parameter(
                torch.tensor(_raw_to_sigmoid(kappa_init)))
        else:
            self.register_buffer("_kappa_fixed", torch.tensor(fk))

        self.raw_tau_c = torch.nn.Parameter(
            torch.tensor(_raw_to_softplus(tau_c_init)))

    @property
    def alpha(self):
        if self.fix_alpha is not None:
            return self._alpha_fixed
        lo = self.fix_beta if self.fix_beta is not None else 0.0
        return lo + (1.0 - lo) * torch.sigmoid(self.raw_alpha)

    @property
    def beta(self):
        if self.fix_beta is not None:
            return self._beta_fixed
        return self.alpha * torch.sigmoid(self.raw_gap)

    @property
    def kappa(self):
        if self.fix_kappa is not None:
            return self._kappa_fixed
        return torch.sigmoid(self.raw_kappa)

    @property
    def tau_c(self):
        return F.softplus(self.raw_tau_c) + 1e-4

    def _features_strain(self, u, dt):
        x_arm = self._maxwell_arm_feature(u, dt, self.alpha, self.beta,
                                          self.tau_c)
        x_kappa = self._D(u, dt, self.kappa)
        return [x_arm, x_kappa]

    def _features_stress(self, u, dt):
        _no_stress_control("FractionalZener")

    def _prefactors_strain(self):
        return ["Gc", "K"]

    def _prefactors_stress(self):
        _no_stress_control("FractionalZener")

    def _phys_params(self):
        if self.control == "stress":
            _no_stress_control("FractionalZener")
        # a FIXED exponent is not a fitted parameter: it must not be
        # counted in k, or a restricted model would be penalised for a
        # number it never learned
        p = {}
        if self.fix_alpha is None:
            p["alpha"] = self.alpha
        if self.fix_beta is None:
            p["beta"] = self.beta
        if self.fix_kappa is None:
            p["kappa"] = self.kappa
        p["tau_c"] = self.tau_c
        return p

    def fixed_phys_params(self):
        """Exponents held fixed rather than fitted -- reporting only;
        these are excluded from the parameter count."""
        out = {}
        for name in ("alpha", "beta", "kappa"):
            v = getattr(self, f"fix_{name}")
            if v is not None:
                out[name] = float(v)
        return out

    def _strain_shift_exponents(self):
        # Feature 0 is the Maxwell arm, which carries tau_c and is shifted
        # by the tau_c substitution -> exponent 0, i.e. no extra factor.
        # Feature 1 is a bare springpot K (i w)^kappa with NO tau_c to
        # substitute; under w -> a_T w it shifts as
        #     K (i a_T w)^kappa = K a_T^kappa (i w)^kappa,
        # so its feature must be scaled by a_T**kappa. Without this the
        # parallel branch would stay at the reference temperature while
        # the arm shifted, which is not the TTS-shifted model at all.
        zero = torch.zeros((), dtype=self.raw_tau_c.dtype,
                           device=self.raw_tau_c.device)
        return [zero, self.kappa]

    def _stress_prefactors_to_moduli(self, pf):
        _no_stress_control("FractionalZener")

    def Gstar(self, omega, prefactors):
        Gc, K = prefactors["Gc"], prefactors["K"]
        a, b, tc = self.alpha.item(), self.beta.item(), self.tau_c.item()
        k = self.kappa.item()
        Gp_arm, Gdp_arm = self._maxwell_arm_Gstar(omega, Gc, tc, a, b)
        Gp  = K * omega**k * np.cos(np.pi * k / 2) + Gp_arm
        Gdp = K * omega**k * np.sin(np.pi * k / 2) + Gdp_arm
        return Gp, Gdp


# ================================================================
# Registry
# ================================================================


# ================================================================
# 15. Generalized Maxwell / Prony series
#
#     G*(w) = G_e + SUM_k G_k * (i w tau_k) / (1 + i w tau_k)
#
#     i.e. `n_modes` Maxwell arms in parallel, optionally with an
#     equilibrium spring G_e across them. Rationalising each arm gives
#     the form usually written in the TTS literature,
#
#        G*(w) = SUM_k G_k [ (w tau_k)^2 + i w tau_k ] / (1 + (w tau_k)^2)
#
#     so  G' = SUM_k G_k (w tau_k)^2 / (1 + (w tau_k)^2)
#         G'' = SUM_k G_k (w tau_k)   / (1 + (w tau_k)^2).
#
#     THE SPECTRUM IS ANCHORED TO ONE tau_c. The relaxation times are
#     tau_k = tau_c * rho_k, where rho_k is a FIXED log-spaced ladder
#     fixed at construction and never fitted; only tau_c is a trainable
#     shape parameter. This is deliberate and has three consequences:
#
#       1. Time-temperature superposition works unmodified. The
#          multi-record shift machinery substitutes tau_c -> a_T tau_c
#          per record, which slides the WHOLE ladder rigidly. That is
#          exactly the statement of thermorheological simplicity --
#          one horizontal shift moves every mode by the same factor --
#          rather than an approximation of it. Fitting n independent
#          tau_k would break that: each mode could drift on its own and
#          the single a_T would stop meaning anything.
#       2. There is exactly ONE nonlinear parameter (tau_c) however
#          many modes are used. Every modulus G_k (and G_e) enters
#          linearly, which is the regime this library's prefactor
#          machinery is built for -- they are recovered as linear
#          readouts with credible intervals rather than pushed through
#          gradient descent.
#       3. Choosing `n_modes` and the ladder span replaces choosing
#          each tau_k. A discrete Prony spectrum is famously
#          ill-conditioned when modes are packed closely: neighbouring
#          exp(-t/tau_k) features become collinear, the linear readout
#          becomes unstable, and -- since prefactors here are
#          unconstrained -- individual G_k can come out NEGATIVE, which
#          is unphysical. Two modes per decade (the default) is the
#          usual compromise. Treat negative G_k in a fit as a signal
#          that the ladder is finer than the data can resolve, and
#          coarsen it.
#
#     Only strain control is implemented: the creep compliance of a
#     Prony series is not itself a Prony series, so `_features_stress`
#     has no closed form of the same shape.
# ================================================================

class GeneralizedMaxwellKernel(_BaseKernel, _MaxwellArmMixin):
    """Prony series with `n_modes` arms, optionally plus an equilibrium
    spring. Select it with `model="prony"` and set the geometry through
    arguments -- `n_modes`, `with_spring`, `modes_per_decade`,
    `freeze_tau_c`. The registry also parses `prony<N>` / `prony<N>s`
    for callers that can only pass a string; "gmm" is accepted as an
    alias but is ambiguous in a GP library (Gaussian mixture model), so
    prefer "prony"."""

    # overridden by the classes the registry builds
    _N_MODES = 8
    _WITH_SPRING = False
    _MODES_PER_DECADE = 2.0

    def __init__(self, tau_c_init=1.0, n_modes=None, with_spring=None,
                 modes_per_decade=None, freeze_tau_c=False):
        super().__init__()
        self.n_modes = int(n_modes if n_modes is not None else self._N_MODES)
        self.with_spring = bool(with_spring if with_spring is not None
                                else self._WITH_SPRING)
        mpd = float(modes_per_decade if modes_per_decade is not None
                    else self._MODES_PER_DECADE)
        if self.n_modes < 1:
            raise ValueError("n_modes must be >= 1")
        if mpd <= 0:
            raise ValueError("modes_per_decade must be > 0")
        self.modes_per_decade = mpd
        # tau_c positions the ladder. FREEZING IT removes the one
        # degeneracy this family has: sliding the ladder and
        # redistributing the moduli leaves G*(w) almost unchanged, so
        # tau_c and the G_k trade off against each other and neither is
        # individually identified. With tau_c fixed there are NO
        # nonlinear physics parameters left -- every modulus is a linear
        # prefactor of a known feature, so the spectrum is uniquely
        # determined by the data. This is also what standard Prony
        # practice does (the relaxation times are chosen, not fitted),
        # and it costs nothing under TTS: in a joint multi-record fit
        # the per-record a_T still slide the whole ladder freely, they
        # are simply measured against a fixed reference ladder instead
        # of against one that drifts with them.
        self.freeze_tau_c = bool(freeze_tau_c)
        _raw = torch.tensor(_raw_to_softplus(tau_c_init))
        if self.freeze_tau_c:
            self.register_buffer("raw_tau_c", _raw)
        else:
            self.raw_tau_c = torch.nn.Parameter(_raw)
        # The fixed dimensionless ladder, centred on 1 so that tau_c is
        # the CENTRE of the spectrum rather than one end of it -- that
        # keeps tau_c interpretable and comparable with the single-mode
        # models' tau_c. Registered as a buffer, not a Parameter: it
        # moves with the module but is never trained.
        span = (self.n_modes - 1) / mpd          # decades, end to end
        self.register_buffer(
            "rho", torch.logspace(-span / 2.0, span / 2.0, self.n_modes,
                                  dtype=torch.get_default_dtype())
            if self.n_modes > 1 else torch.ones(1))

    @property
    def tau_c(self):
        return F.softplus(self.raw_tau_c) + 1e-4

    def taus(self):
        """The n relaxation times [s], tau_k = tau_c * rho_k."""
        return (self.tau_c * self.rho.to(self.raw_tau_c.device))

    # -- features ------------------------------------------------------
    def _features_strain(self, u, dt):
        one = torch.ones((), dtype=u.dtype, device=u.device)
        zero = torch.zeros((), dtype=u.dtype, device=u.device)
        # `self.tau_c` is read once per arm on purpose: under a joint
        # multi-record fit the shift wrapper intercepts that property
        # and returns a_T * tau_c for the record being built, so the
        # whole ladder shifts together.
        tau_c = self.tau_c
        feats = [self._maxwell_arm_feature(u, dt, one, zero, tau_c * r)
                 for r in self.rho.to(u.device)]
        if self.with_spring:
            feats.append(u)          # G_e * gamma(t)
        return feats

    def _features_stress(self, u, dt):
        _no_stress_control(
            "GeneralizedMaxwell",
            "The creep compliance of a Prony series is not itself a "
            "Prony series. ")

    def _prefactors_strain(self):
        names = [f"G{k + 1}" for k in range(self.n_modes)]
        if self.with_spring:
            names.append("Ge")
        return names

    def _prefactors_stress(self):
        _no_stress_control("GeneralizedMaxwell")

    def _phys_params(self):
        if self.control == "stress":
            _no_stress_control("GeneralizedMaxwell")
        # tau_c only -- rho is fixed, and every G_k is a prefactor.
        # When tau_c is frozen it is not a fitted parameter at all, so it
        # is reported through `fixed_phys_params()` instead and must not
        # be counted in k for AIC/BIC.
        if self.freeze_tau_c:
            return {}
        return {"tau_c": self.tau_c}

    def fixed_phys_params(self):
        """Physics quantities that were HELD FIXED, not fitted -- for
        reporting only; these are excluded from the parameter count."""
        return {"tau_c": float(self.tau_c)} if self.freeze_tau_c else {}

    def _stress_prefactors_to_moduli(self, pf):
        _no_stress_control("GeneralizedMaxwell")

    def Gstar(self, omega, prefactors):
        tau_c = self.tau_c.item()
        rho = self.rho.detach().cpu().numpy()
        Gp = np.zeros_like(np.asarray(omega, dtype=float))
        Gdp = np.zeros_like(Gp)
        for k, r in enumerate(rho):
            Gk = prefactors[f"G{k + 1}"]
            p, dp = self._maxwell_arm_Gstar(omega, Gk, tau_c * float(r),
                                            1.0, 0.0)
            Gp = Gp + p
            Gdp = Gdp + dp
        if self.with_spring:
            Gp = Gp + prefactors["Ge"]
        return Gp, Gdp


def _gmm_class(n_modes, with_spring, modes_per_decade=2.0):
    """Build (and cache) a concrete GeneralizedMaxwellKernel subclass.

    The model factories in `model.py` / `rheogp_multi.py` instantiate a
    CLASS with only the shared `*_init` keywords, so the mode count and
    the presence of the spring have to be baked into the class rather
    than passed at call time -- that is what the registry names
    `gmm<N>` / `gmm<N>s` select.
    """
    key = (int(n_modes), bool(with_spring), float(modes_per_decade))
    cls = _GMM_CACHE.get(key)
    if cls is None:
        cls = type(
            f"GeneralizedMaxwell{int(n_modes)}"
            f"{'S' if with_spring else ''}Kernel",
            (GeneralizedMaxwellKernel,),
            {"_N_MODES": int(n_modes), "_WITH_SPRING": bool(with_spring),
             "_MODES_PER_DECADE": float(modes_per_decade)})
        _GMM_CACHE[key] = cls
    return cls


_GMM_CACHE = {}

class _KernelRegistry(dict):
    """The kernel table, plus parametric names for the Prony series.

    The canonical name is **"prony"**. `"generalizedmaxwell"`,
    `"maxwellwiechert"`/`"wiechert"` and `"gmm"` are accepted aliases,
    but prefer `"prony"` in new code: inside a Gaussian-process
    library `GMM` reads as *Gaussian mixture model* (and in
    econometrics as *generalized method of moments*), so the
    abbreviation is ambiguous exactly where this package lives.

    The ladder geometry belongs in the arguments, not the name:

        model="prony", n_modes=12, with_spring=True

    A trailing mode count and an optional "s" are still parsed, for
    callers that can only pass a string:

        "prony"     ->  8 modes, no spring
        "prony12"   -> 12 modes, no spring
        "prony12s"  -> 12 modes, plus the equilibrium spring G_e

    An explicit `n_modes`/`with_spring` argument overrides whatever the
    name encoded.

    `.get()` is what the model factories call, so the parsing lives
    there rather than in `__missing__` (which `dict.get` never
    triggers).
    """

    _GMM_RE = re.compile(
        r"(?:prony|generali[sz]edmaxwell|(?:maxwell)?wiechert|gmm)"
        r"(\d+)?(s)?\Z")

    def get(self, key, default=None):
        if key in self:
            return self[key]
        m = self._GMM_RE.match(str(key))
        if m is not None:
            n = int(m.group(1)) if m.group(1) else 8
            return _gmm_class(n, m.group(2) == "s")
        return default


KERNELS = _KernelRegistry({
    "kernelfree": KernelFreeKernel,
    "kf":         KernelFreeKernel,
    "maxwell":                 MaxwellKernel,
    "springpot":               SpringpotKernel,
    "fractionalmaxwellgel":    FractionalMaxwellGelKernel,
    "fractionalmaxwellliquid": FractionalMaxwellLiquidKernel,
    "fractionalmaxwell":       FractionalMaxwellKernel,
    "fractionalkelvinvoigts":  FKVSKernel,
    "fractionalkelvinvoigtd":  FKVDKernel,
    "fractionalkelvinvoigt":   FractionalKelvinVoigtKernel,
    "fmg":  FractionalMaxwellGelKernel,
    "fml":  FractionalMaxwellLiquidKernel,
    "fmm":  FractionalMaxwellKernel,
    "fkvs": FKVSKernel,
    "fkvd": FKVDKernel,
    "fkv":  FractionalKelvinVoigtKernel,
    "zener":                  ZenerKernel,
    "fractionalzenersolids":  FractionalZenerSolidSKernel,
    "fractionalzenerliquids": FractionalZenerLiquidSKernel,
    "fractionalzenerliquidd": FractionalZenerLiquidDKernel,
    "fractionalzeners":       FractionalZenerSKernel,
    "fractionalzener":        FractionalZenerKernel,
    "fzss": FractionalZenerSolidSKernel,
    "fzls": FractionalZenerLiquidSKernel,
    "fzld": FractionalZenerLiquidDKernel,
    "fzs":  FractionalZenerSKernel,
    "fz":   FractionalZenerKernel,
    # Prony series / generalized Maxwell. `prony<N>` and `prony<N>s`
    # are also accepted, as are the aliases below -- see
    # `_KernelRegistry.get`. Prefer "prony": "gmm" collides with
    # Gaussian mixture model, which is a real ambiguity in a GP library.
    "prony":             _gmm_class(8, False),
    "pronys":            _gmm_class(8, True),
    "generalizedmaxwell":  _gmm_class(8, False),
    "generalisedmaxwell":  _gmm_class(8, False),
    "maxwellwiechert":   _gmm_class(8, False),
    "wiechert":          _gmm_class(8, False),
    "gmm":               _gmm_class(8, False),   # alias, see above
    "gmms":              _gmm_class(8, True),    # alias, see above
})

# Back-compat map (strain-control column layout of rheogp ≤ 0.2).
# New code should call kernel.prefactor_names() instead.
PREFACTOR_COLS = {
    KernelFreeKernel:              {"dsig_dgamma": 0, "dsig_dgammadot": 1},
    MaxwellKernel:                 {"Gc":   0},
    SpringpotKernel:               {"V":    0},
    FractionalMaxwellGelKernel:    {"Gc":   0},
    FractionalMaxwellLiquidKernel: {"Gc":   0},
    FractionalMaxwellKernel:       {"Gc":   0},
    FKVSKernel:                    {"V":    0, "G":   1},
    FKVDKernel:                    {"G_bb": 0, "eta": 1},
    FractionalKelvinVoigtKernel:   {"V":    0, "G_bb": 1},
    ZenerKernel:                   {"Gc": 0, "G0": 1},
    FractionalZenerSolidSKernel:   {"Gc": 0, "G0": 1},
    FractionalZenerLiquidSKernel:  {"Gc": 0, "G0": 1},
    FractionalZenerLiquidDKernel:  {"Gc": 0, "eta_p": 1},
    FractionalZenerSKernel:        {"Gc": 0, "G0": 1},
    FractionalZenerKernel:         {"Gc": 0, "K": 1},
}
