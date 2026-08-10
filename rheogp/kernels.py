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
# Registry
# ================================================================

KERNELS = {
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
}

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
}
