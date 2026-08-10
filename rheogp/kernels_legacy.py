"""
rheogp.kernels
==============
Differentiable convolution kernels for all supported rheological models.

Nomenclature (standard fractional rheology)
--------------------------------------------
  𝕍, α   — liquid-like springpot  (α closer to 1  →  more viscous)
  𝔾, β   — solid-like  springpot  (β closer to 0  →  more elastic)
  constraint:  0 ≤ β ≤ α ≤ 1

Each class exposes:
    compute_x(gamma, dt) -> (N, n_features)   normalised GP input tensor
    named_phys_params()  -> dict[str, float]   exponents + τ_c for logging
    Gstar(omega, prefactors) -> (G', G'')       analytic frequency response
    _feature_stds        -> list[float]         σ_x for G_c chain rule

Relaxation kernel quick-reference
----------------------------------
  Maxwell              : K(s) = exp(-s/τ_c)                           [1 feat]
  Springpot            : K(s) = s^{-α}/Γ(1-α)                        [1 feat]
  FractionalMaxwellGel : K_FMM(s; α, β=0)  →  E_{α,1}(-·)            [1 feat]
  FractionalMaxwellLiq : K_FMM(s; α=1, β)  →  (s/τ)^{-β}E_{1-β,1-β} [1 feat]
  FractionalMaxwell    : K_FMM(s; α, β)    full kernel                [1 feat]
  FKV-S                : [D^α[γ],  γ]      parallel springpot+spring  [2 feat]
  FKV-D                : [D^β[γ],  γ̇]     parallel springpot+dashpot [2 feat]
  FractionalKelvinVoigt: [D^α[γ],  D^β[γ]]                           [2 feat]
"""

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "KERNELS",
    "MaxwellKernel",
    "SpringpotKernel",
    "FractionalMaxwellGelKernel",
    "FractionalMaxwellLiquidKernel",
    "FractionalMaxwellKernel",
    "FKVSKernel",
    "FKVDKernel",
    "FractionalKelvinVoigtKernel",
]

# ================================================================
# Numerics
# ================================================================
_SERIES_TERMS = 30
_Z_THRESH     = 10.0


def _mittag_leffler(z, a, b):
    """
    Real-valued E_{a,b}(z) for scalar tensors a, b and tensor z.
    Hybrid truncated series / 1-term asymptotic for |z| >> 1.
    """
    ml = torch.zeros_like(z)
    for k in range(_SERIES_TERMS):
        denom = torch.exp(torch.lgamma(a * float(k) + b))
        ml    = ml + (z ** float(k)) / (denom + 1e-30)
    ml_asymp  = -1.0 / (z * torch.exp(torch.lgamma(b - a + 1e-8)) + 1e-30)
    use_asymp = (z < -_Z_THRESH).float()
    return (1.0 - use_asymp) * ml + use_asymp * ml_asymp


# ================================================================
# Shared signal helpers
# ================================================================

def _gamma_dot(gamma, dt):
    """Causal forward-difference strain rate, padded to length N."""
    return torch.cat([
        torch.zeros(1, device=gamma.device, dtype=gamma.dtype),
        (gamma[1:] - gamma[:-1]) / dt
    ])


def _fft_convolve(signal, K, dt):
    """O(N log N) causal convolution  x = K ★ signal · dt."""
    N     = signal.shape[0]
    n_fft = 2 * N
    return torch.fft.irfft(
        torch.fft.rfft(signal, n=n_fft) *
        torch.fft.rfft(K,      n=n_fft)
    )[:N] * dt


def _fft_convolve_analytic(signal, Kstar_fn, dt):
    """
    Causal convolution using analytic Fourier transform of kernel.
    Kstar_fn: callable (omega_tensor) -> complex tensor of same length
    as rfft(signal). Avoids discretizing K(t) in the time domain.
    """
    N      = signal.shape[0]
    n_fft  = 2 * N
    freqs  = torch.fft.rfftfreq(n_fft, d=dt).to(signal.device, signal.dtype)
    omega  = 2.0 * torch.pi * freqs
    Kstar  = Kstar_fn(omega)                        # (n_fft//2 + 1,) complex
    return torch.fft.irfft(
        Kstar * torch.fft.rfft(signal, n=n_fft)
    )[:N] * dt


def _scale(x):
    """Normalise to zero mean / unit std; return (x_scaled, μ, σ)."""
    mu  = x.mean()
    std = x.std() 
    return (x - mu) / std, mu.detach(), std.detach()

#def _scale(x):
#    scale = torch.sqrt(torch.mean(x**2))  # RMS scale
#    scale = scale.clamp(min=1e-6)
#    return x / scale, 0.0, scale

def _normalize(x):
    """Nomalize using strain scale"""
    scale = torch.max(torch.abs(x))
    zero = torch.zeros((), device=x.device, dtype=x.dtype)
    return x / scale, zero, scale

def _scale_physical(x):
    zero = torch.zeros((), device=x.device, dtype=x.dtype)
    one  = torch.ones((),  device=x.device, dtype=x.dtype)
    return x, zero, one

def _lag_vector(N, dt, device, dtype):
    """s = [dt, 2·dt, …, N·dt]  (strictly positive lags)."""
    return torch.arange(1, N + 1, device=device, dtype=dtype) * dt


# ================================================================
# Physical kernel functions
# ================================================================
def _caputo_shape(s, alpha):
    """
    Shape part of the Caputo kernel:  s^{-α}
    (without the 1/Γ(1-α) prefactor, which is handled analytically
    in the back-transform to avoid the Γ(0⁺) → ∞ singularity at α→1).
    """
    return s ** (-alpha)


def _caputo_gamma_factor(alpha):
    """
    The Γ(1-α) prefactor of the Caputo kernel.
    Stored separately so the back-transform can multiply it back in.
    Returns a scalar tensor.
    """
    return torch.exp(torch.lgamma(1.0 - alpha))


def _caputo_kernel(s, alpha):
    #alpha = alpha.clamp(0.0, 0.9)
    """Caputo power-law kernel  K(s) = s^{-α} / Γ(1-α)."""
    return s ** (-alpha) / torch.exp(torch.lgamma(1.0 - alpha))

def _fkv_kernel(s, alpha, beta, tau_c):
    """
    FKV kernel (pure two-term power-law form):

    K(s) = (s/τ_c)^(-α) / Γ(1-α) + (s/τ_c)^(-β) / Γ(1-β)

    Valid for 0 ≤ β < α ≤ 1, τ_c > 0.
    """

    u = s / tau_c

    return (u ** (-alpha)) / _caputo_gamma_factor(alpha) + (u ** (-beta)) / _caputo_gamma_factor(beta)


def _fmm_kernel(s, alpha, beta, tau_c):
    """
    Full FMM relaxation kernel:
      K(s) = (s/τ_c)^{-β} · E_{α-β, 1-β}( -(s/τ_c)^{α-β} )
    Valid for 0 ≤ β < α ≤ 1, τ_c > 0.
    """
    a = alpha - beta
    b = 1.0   - beta
    u = (s / tau_c).clamp(min=1e-8)
    return (u ** (-beta)) * _mittag_leffler(-(u ** a), a, b)


def _maxwell_kernel(s, tau_c):
    """Classical Maxwell exponential kernel  K(s) = exp(-s/τ_c)."""
    return torch.exp(-s / tau_c)


# ================================================================
# Constraint helpers
# ================================================================

def _raw_to_sigmoid(value: float) -> float:
    """inverse-sigmoid so sigmoid(raw) ≈ value."""
    value = float(np.clip(value, 1e-4, 1.0 - 1e-4))
    return float(np.log(value / (1.0 - value)))


def _raw_to_softplus(value: float) -> float:
    """inverse-softplus so softplus(raw) ≈ value."""
    value = max(value, 1e-4)
    return float(np.log(np.exp(value) - 1.0))


# ================================================================
# Base class
# ================================================================

class _BaseKernel(torch.nn.Module):
    """Shared interface for all rheological kernel modules."""

    def compute_x(self, gamma, dt):
        raise NotImplementedError

    def named_phys_params(self):
        raise NotImplementedError

    def Gstar(self, omega, prefactors):
        raise NotImplementedError

    def param_labels(self):
        raise NotImplementedError

# ================================================================
# KernelFree  —  no viscoelastic model, raw strain features only
#
#   Features: [x_1(t), x_2(t)] where
#       x_1(t) = gamma(t)          elastic-like (strain)
#       x_2(t) = gamma_dot(t)      viscous-like (strain rate)
#
#   No constitutive assumptions. The GP covariance kernel
#   (RBF, Matern, etc.) learns the stress-strain map directly.
#   Use with kernel="rbf" for nonparametric fitting.
#
#   Sensitivities dσ/dx_1(t) and dσ/dx_2(t) serve as
#   time-resolved diagnostics of stationarity:
#       flat  → stationary linear material
#       drift → mutation or non-stationarity
#
#   No Gstar() is defined — there is no constitutive model.
#   named_phys_params() returns empty dict (no physics params).
#   n_features = 2
# ================================================================

class KernelFreeKernel(_BaseKernel):
    """
    Kernel-free GP: no viscoelastic model.
    Features are the raw strain gamma(t) and strain rate gammadot(t),
    both standardised. Use with kernel='rbf' or 'matern32'.

    Parameters
    ----------
    use_strain_rate : bool
        If True (default), include gammadot as second feature.
        If False, use only gamma — purely elastic-like input.
    """

    def __init__(self, use_strain_rate=True):
        super().__init__()
        self.use_strain_rate = use_strain_rate
        self.n_features      = 2 if use_strain_rate else 1

    def compute_x(self, gamma, dt):
        # ── strain feature ──────────────────────────────────
        if not getattr(self, '_gamma_scale_frozen', False):
            g_norm, _, scale_g = _scale_physical(gamma)
            self._gamma_scale  = scale_g
        else:
            scale_g = self._gamma_scale
            g_norm  = gamma / scale_g.clamp(min=1e-8)

        if not getattr(self, '_feature_stds_frozen', False):
            x1, _, std1 = _scale(g_norm)
            stds = [std1.item()]
        else:
            std1 = torch.tensor(self._feature_stds[0],
                                dtype=g_norm.dtype, device=g_norm.device)
            x1   = (g_norm - g_norm.mean()) / std1.clamp(min=1e-8)
            stds = [self._feature_stds[0]]

        if not self.use_strain_rate:
            if not getattr(self, '_feature_stds_frozen', False):
                self._feature_stds = stds
            return x1.unsqueeze(-1)   # (N, 1)

        # ── strain rate feature ──────────────────────────────
        gd = _gamma_dot(g_norm, dt)

        if not getattr(self, '_feature_stds_frozen', False):
            x2, _, std2 = _scale(gd)
            stds.append(std2.item())
        else:
            std2 = torch.tensor(self._feature_stds[1],
                                dtype=gd.dtype, device=gd.device)
            x2   = (gd - gd.mean()) / std2.clamp(min=1e-8)

        if not getattr(self, '_feature_stds_frozen', False):
            self._feature_stds = stds

        return torch.stack([x1, x2], dim=1)   # (N, 2)

    def named_phys_params(self):
        return {}   # no physics parameters

    def Gstar(self, omega, prefactors):
        raise NotImplementedError(
            "KernelFreeKernel has no constitutive model — "
            "G*(omega) is not defined. Use plot_prefactors() "
            "to inspect the time-resolved sensitivities."
        )

    def param_labels(self):
        if self.use_strain_rate:
            return [r"$\gamma$", r"$\dot\gamma$"]
        return [r"$\gamma$"]


# ================================================================
# 1. Maxwell  (classical)
#
#   σ + τ_c · dσ/dt = τ_c · G_c · dγ/dt
#
#   Relaxation kernel : K(s) = exp(-s/τ_c)
#   Feature           : x = K ★ γ̇         (1 feature)
#   Prefactors        : Gc  (∂σ/∂x)
#   G*(ω)             : Gc · (iωτ_c)² / (1+(iωτ_c)²)
#                       + i · Gc · iωτ_c  / (1+(iωτ_c)²)
#   Free params       : τ_c > 0
# ================================================================

class MaxwellKernel(_BaseKernel):
    n_features = 1

    def __init__(self, tau_c_init=1.0):
        super().__init__()
        self.raw_tau_c = torch.nn.Parameter(
            torch.tensor(_raw_to_softplus(tau_c_init))
        )

    @property
    def tau_c(self):
        return F.softplus(self.raw_tau_c) + 1e-4

    def compute_x(self, gamma, dt):
        N  = gamma.shape[0]
        s  = _lag_vector(N, dt, gamma.device, gamma.dtype)

        # gamma normalisation — use frozen training scale during predict
        if not getattr(self, '_gamma_scale_frozen', False):
            g_norm, _, scale_g  = _normalize(gamma)
            self._gamma_scale   = scale_g
        else:
            scale_g = self._gamma_scale
            g_norm  = gamma / scale_g.clamp(min=1e-8)

        gd = _gamma_dot(g_norm, dt)
        K  = _maxwell_kernel(s, self.tau_c)
        x  = _fft_convolve(gd, K, dt)

        # feature standardisation — use frozen training std during predict
        if not getattr(self, '_feature_stds_frozen', False):
            xs, _, std         = _scale(x)
            self._feature_stds = [std.item()]
        else:
            std = torch.tensor(self._feature_stds[0],
                               dtype=x.dtype, device=x.device)
            xs  = (x - x.mean()) / std.clamp(min=1e-8)

        return xs.unsqueeze(-1)   # (N, 1)

    def named_phys_params(self):
        return {"tau_c": self.tau_c.item()}

    def Gstar(self, omega, prefactors):
        Gc = prefactors["Gc"]
        tc = self.tau_c.item()
        z  = 1j * omega * tc
        Gs = Gc * (z**2 / (1.0 + z**2)) + 1j * Gc * (z / (1.0 + z**2))
        # equivalently: Gc · z / (1 + z)  but written explicitly:
        Gs = Gc * z / (1.0 + z)
        return Gs.real, Gs.imag

    def param_labels(self):
        return ["τ_c"]


# ================================================================
# 2. Springpot
#
#   σ = 𝕍 · D^α[γ]
#
#   Kernel  : K(s) = s^{-α} / Γ(1-α)
#   Feature : x_α = D^α[γ]             (1 feature)
#   Prefactor: 𝕍  (∂σ/∂x)
#   G*(ω)   : 𝕍 · (iω)^α
#   Params  : α ∈ (0,1)
# ================================================================

class SpringpotKernel(_BaseKernel):
    n_features = 1

    def __init__(self, alpha_init=0.5):
        super().__init__()
        self.raw_alpha = torch.nn.Parameter(
            torch.tensor(_raw_to_sigmoid(alpha_init))
        )

    @property
    def alpha(self):
        return torch.sigmoid(self.raw_alpha)

    def compute_x(self, gamma, dt):
        N  = gamma.shape[0]
        s  = _lag_vector(N, dt, gamma.device, gamma.dtype)
        g_norm, _, scale_g  = _normalize(gamma)
        self._gamma_scale = scale_g
        gd = _gamma_dot(g_norm, dt)
        x  = _fft_convolve(gd, _caputo_kernel(s, self.alpha), dt)
        xs, _, std = _scale(x)
        self._feature_stds = [std.item()]
        #self._gamma_factors = [
        #    _caputo_gamma_factor(self.alpha).item(),   # Γ(1-α)
        #]
        return xs.unsqueeze(-1)

    def named_phys_params(self):
        return {"alpha": self.alpha.item()}

    def Gstar(self, omega, prefactors):
        V  = prefactors["V"]
        Gs = V * (1j * omega) ** self.alpha.item()
        return Gs.real, Gs.imag

    def param_labels(self):
        return ["α"]

'''
class SpringpotKernel(_BaseKernel):
    n_features = 1

    def __init__(self, alpha_init=0.5):
        super().__init__()
        self.raw_alpha = torch.nn.Parameter(
            torch.tensor(_raw_to_sigmoid(alpha_init))
        )

    @property
    def alpha(self):
        return torch.sigmoid(self.raw_alpha)

    def compute_x(self, gamma, dt):
        N  = gamma.shape[0]
        g_norm, _, scale_g = _normalize(gamma)
        self._gamma_scale  = scale_g
        gd = _gamma_dot(g_norm, dt)

    # analytic K*(omega) = (i*omega)^alpha / omega  [from Caputo definition]
    # G*(omega) = V * (i*omega)^alpha
    # K*(omega) for the convolution with gamma_dot:
    # sigma = V * (i*omega)^alpha * gamma_tilde
    # but feature x = int K(s) * gammadot ds  where K(s) = s^{-alpha}/Gamma(1-alpha)
    # K*(omega) = Gamma(1-alpha) * (i*omega)^{alpha-1}  ... but we absorb Gamma
    # Simplest: K*(omega) = (i*omega)^alpha / (i*omega) ... 
    # Actually for convolution with gammadot:
    # x_tilde(omega) = K*(omega) * gammadot_tilde(omega)
    # K*(omega) = (i*omega)^{-alpha} * ... 
    # 
    # Cleanest: convolve gamma directly with the Springpot kernel in freq domain
    # sigma = V * (i*omega)^alpha * gamma_tilde  so feature = (i*omega)^alpha * gamma_tilde

        def Kstar(omega):
            mask  = omega > 0
            out   = torch.zeros(len(omega), dtype=torch.cdouble, device=omega.device)
            out[mask] = (1j * omega[mask].double()) ** self.alpha.double()
            return out.to(gamma.dtype)

        N_fft = 2 * N
        freqs = torch.fft.rfftfreq(N_fft, d=dt).to(gamma.device, gamma.dtype)
        omega = 2.0 * torch.pi * freqs
        Ks    = Kstar(omega)
        x     = torch.fft.irfft(Ks * torch.fft.rfft(g_norm, n=N_fft))[:N]
        # no dt multiplication needed — this is sigma/V directly

        xs, _, std         = _scale(x)
        self._feature_stds = [std.item()]
        return xs.unsqueeze(-1)

    def named_phys_params(self):
        return {"alpha": self.alpha.item()}

    def Gstar(self, omega, prefactors):
        V  = prefactors["V"]
        Gs = V * (1j * omega) ** self.alpha.item()
        return Gs.real, Gs.imag

    def param_labels(self):
        return ["α"]
'''

# ================================================================
# 3. Fractional Maxwell Gel  (FMG)
#
#   Spring + Springpot in series
#   σ + (𝕍/G) · D^α[σ] = 𝕍 · D^α[γ]
#
#   τ_c = (𝕍/G)^{1/α},   G_c = 𝕍 · τ_c^{-α}
#
#   Kernel  : K_FMM(s; α, β→0) = E_{α,1}(-(s/τ_c)^α)
#             i.e. _fmm_kernel with β=0
#   Feature : single x              (1 feature)
#   Prefactor: Gc
#   G*(ω)   : Gc · [(ωτ_c)^{2α} + (ωτ_c)^α cos(πα/2)]
#                 / [1 + (ωτ_c)^{2α} + 2(ωτ_c)^α cos(πα/2)]
#           + i · Gc · (ωτ_c)^α sin(πα/2)
#                 / [1 + (ωτ_c)^{2α} + 2(ωτ_c)^α cos(πα/2)]
#   Params  : α ∈ (0,1),  τ_c > 0
# ================================================================

class FractionalMaxwellGelKernel(_BaseKernel):
    n_features = 1

    def __init__(self, alpha_init=0.5, tau_c_init=1.0):
        super().__init__()
        self.raw_alpha = torch.nn.Parameter(
            torch.tensor(_raw_to_sigmoid(alpha_init))
        )
        self.raw_tau_c = torch.nn.Parameter(
            torch.tensor(_raw_to_softplus(tau_c_init))
        )

    @property
    def alpha(self):
        return torch.sigmoid(self.raw_alpha)

    @property
    def tau_c(self):
        return F.softplus(self.raw_tau_c) + 1e-4

    def compute_x(self, gamma, dt):
        N    = gamma.shape[0]
        s    = _lag_vector(N, dt, gamma.device, gamma.dtype)
        beta = torch.zeros(1, device=gamma.device, dtype=gamma.dtype).squeeze()
        g_norm, _, scale_g  = _normalize(gamma)
        self._gamma_scale = scale_g
        gd = _gamma_dot(g_norm, dt)
        K    = _fmm_kernel(s, self.alpha, beta, self.tau_c)
        x    = _fft_convolve(gd, K, dt)
        xs, _, std = _scale(x)
        self._feature_stds = [std.item()]
        return xs.unsqueeze(-1)

    def named_phys_params(self):
        return {"alpha": self.alpha.item(), "tau_c": self.tau_c.item()}

    def Gstar(self, omega, prefactors):
        Gc = prefactors["Gc"]
        a  = self.alpha.item()
        tc = self.tau_c.item()
        wtc   = omega * tc
        cos_a = np.cos(np.pi * a / 2)
        denom = 1.0 + wtc**(2*a) + 2.0 * wtc**a * cos_a
        Gp    = Gc * (wtc**(2*a) + wtc**a * cos_a) / denom
        Gdp   = Gc * wtc**a * np.sin(np.pi * a / 2) / denom
        return Gp, Gdp

    def param_labels(self):
        return ["α", "τ_c"]


# ================================================================
# 4. Fractional Maxwell Liquid  (FML)
#
#   Springpot (𝔾,β) + Dashpot (η) in series
#   σ + (η/𝔾) · D^{1-β}[σ] = η · dγ/dt
#
#   τ_c = (η/𝔾)^{1/(1-β)},   G_c = η · τ_c^{-1}
#
#   Kernel  : K_FMM(s; α=1, β)  →  (s/τ_c)^{-β} · E_{1-β,1-β}(-(s/τ_c)^{1-β})
#   Feature : single x              (1 feature)
#   Prefactor: Gc
#   G*(ω)   : see latex equations
#   Params  : β ∈ (0,1),  τ_c > 0
# ================================================================

class FractionalMaxwellLiquidKernel(_BaseKernel):
    n_features = 1

    def __init__(self, beta_init=0.5, tau_c_init=1.0):
        super().__init__()
        self.raw_beta  = torch.nn.Parameter(
            torch.tensor(_raw_to_sigmoid(beta_init))
        )
        self.raw_tau_c = torch.nn.Parameter(
            torch.tensor(_raw_to_softplus(tau_c_init))
        )

    @property
    def beta(self):
        return torch.sigmoid(self.raw_beta)

    @property
    def tau_c(self):
        return F.softplus(self.raw_tau_c) + 1e-4

    def compute_x(self, gamma, dt):
        N     = gamma.shape[0]
        s     = _lag_vector(N, dt, gamma.device, gamma.dtype)
        # α=1 exactly: use _fmm_kernel with alpha clamped to ~1
        alpha = torch.ones(1, device=gamma.device, dtype=gamma.dtype
                           ).squeeze() * 0.9999
        g_norm, _, scale_g  = _normalize(gamma)
        self._gamma_scale = scale_g
        gd = _gamma_dot(g_norm, dt)
        K     = _fmm_kernel(s, alpha, self.beta, self.tau_c)
        x     = _fft_convolve(gd, K, dt)
        xs, _, std = _scale(x)
        self._feature_stds = [std.item()]
        return xs.unsqueeze(-1)

    def named_phys_params(self):
        return {"beta": self.beta.item(), "tau_c": self.tau_c.item()}

    def Gstar(self, omega, prefactors):
        Gc  = prefactors["Gc"]
        b   = self.beta.item()
        tc  = self.tau_c.item()
        wtc = omega * tc
        cos_1b = np.cos(np.pi * (1 - b) / 2)
        denom  = 1.0 + wtc**(2*(1-b)) + 2.0 * wtc**(1-b) * cos_1b
        Gp  = Gc * wtc**(2-b) * np.cos(np.pi * b / 2) / denom
        Gdp = Gc * (wtc + wtc**(2-b) * np.sin(np.pi * b / 2)) / denom
        return Gp, Gdp

    def param_labels(self):
        return ["β", "τ_c"]


# ================================================================
# 5. Fractional Maxwell  (full)
#
#   Springpot (𝕍,α) + Springpot (𝔾,β) in series, 0 ≤ β ≤ α ≤ 1
#   σ + (𝕍/𝔾) · D^{α-β}[σ] = 𝕍 · D^α[γ]
#
#   τ_c = (𝕍/𝔾)^{1/(α-β)},   G_c = 𝕍 · τ_c^{-α}
#
#   Kernel  : K(s) = (s/τ_c)^{-β} · E_{α-β, 1-β}(-(s/τ_c)^{α-β})
#   Feature : single x              (1 feature)
#   Prefactor: Gc  (encodes G_c = 𝕍 · τ_c^{-α})
#   G*(ω)   : see latex
#   Params  : α ∈ (0,1),  β = α·sigmoid(raw_gap) ∈ (0,α),  τ_c > 0
# ================================================================

class FractionalMaxwellKernel(_BaseKernel):
    n_features = 1

    def __init__(self, alpha_init=0.7, beta_init=0.3, tau_c_init=1.0):
        super().__init__()
        self.raw_alpha = torch.nn.Parameter(
            torch.tensor(_raw_to_sigmoid(alpha_init))
        )
        # β parameterised as  α · sigmoid(raw_gap)  →  0 < β < α always
        self.raw_gap   = torch.nn.Parameter(torch.tensor(0.0))
        self.raw_tau_c = torch.nn.Parameter(
            torch.tensor(_raw_to_softplus(tau_c_init))
        )

    @property
    def alpha(self):
        return torch.sigmoid(self.raw_alpha)

    @property
    def beta(self):
        return self.alpha * torch.sigmoid(self.raw_gap)

    @property
    def tau_c(self):
        return F.softplus(self.raw_tau_c) + 1e-4

    def compute_x(self, gamma, dt):
        N  = gamma.shape[0]
        s  = _lag_vector(N, dt, gamma.device, gamma.dtype)
        g_norm, _, scale_g  = _normalize(gamma)
        self._gamma_scale = scale_g
        gd = _gamma_dot(g_norm, dt)
        K  = _fmm_kernel(s, self.alpha, self.beta, self.tau_c)
        x  = _fft_convolve(gd, K, dt)
        xs, _, std = _scale(x)
        self._feature_stds = [std.item()]
        return xs.unsqueeze(-1)

    def named_phys_params(self):
        return {
            "alpha": self.alpha.item(),
            "beta":  self.beta.item(),
            "tau_c": self.tau_c.item(),
        }

    def Gstar(self, omega, prefactors):
        Gc  = prefactors["Gc"]
        a   = self.alpha.item()
        b   = self.beta.item()
        tc  = self.tau_c.item()
        wtc = omega * tc
        cos_ab = np.cos(np.pi * (a - b) / 2)
        denom  = 1.0 + wtc**(2*(a-b)) + 2.0 * wtc**(a-b) * cos_ab
        Gp  = Gc * (wtc**a * np.cos(np.pi*a/2)
                    + wtc**(2*a-b) * np.cos(np.pi*b/2)) / denom
        Gdp = Gc * (wtc**a * np.sin(np.pi*a/2)
                    + wtc**(2*a-b) * np.sin(np.pi*b/2)) / denom
        return Gp, Gdp

    def param_labels(self):
        return ["α", "β", "τ_c"]


# ================================================================
# 6. Fractional Kelvin-Voigt S  (FKV-S)
#
#   Springpot (𝕍,α) ∥ Spring (G)
#   σ = 𝕍 · D^α[γ] + G · γ
#
#   τ_c = (𝕍/G)^{1/α},   G_c = 𝕍 · τ_c^{-α}
#
#   Features: [x_α = D^α[γ],   x_0 = γ]      (2 features)
#   Prefactors: 𝕍 = ∂σ/∂x_α,   G = ∂σ/∂x_0
#   G*(ω): G'  = 𝕍·ω^α·cos(πα/2) + G
#          G'' = 𝕍·ω^α·sin(πα/2)
#   Params: α ∈ (0,1)
# ================================================================

class FKVSKernel(_BaseKernel):
    n_features = 2

    def __init__(self, alpha_init=0.5):
        super().__init__()
        self.raw_alpha = torch.nn.Parameter(
            torch.tensor(_raw_to_sigmoid(alpha_init))
        )
        # fixed 2×2 PCA / orthogonalization matrix; start as identity
        # buffer = not a Parameter, keeps dtype/device, survives .to(), save/load
        self.register_buffer("_P", torch.eye(2))
        self._pca_initialized = False

    @property
    def alpha(self):
        return torch.sigmoid(self.raw_alpha)

    def _init_pca_if_needed(self, X):
        """
        Compute a fixed 2×2 orthogonalization matrix on first call.
        X: (N, 2) torch tensor of features [x_a, x_b] at the *current* α,β.
        The result is stored in self._P and never recomputed.
        """
        if self._pca_initialized:
            return

        # Center
        Xc = X - X.mean(dim=0, keepdim=True)

        # SVD: Xc = U S Vᵀ; columns of V are principal directions
        # Use double precision for stability, cast back to model dtype
        U, S, Vt = torch.linalg.svd(Xc.double(), full_matrices=False)
        P = Vt.T[:, :2].to(X.dtype)  # (2,2), columns orthonormal

        # Store as fixed transform
        # (register_buffer already done in __init__, so just copy into it)
        self._P.copy_(P)
        self._pca_initialized = True

    
    def compute_x(self, gamma, dt):
        N  = gamma.shape[0]
        s  = _lag_vector(N, dt, gamma.device, gamma.dtype)
        g_norm, _, scale_g  = _normalize(gamma)
        self._gamma_scale = scale_g
        gd = _gamma_dot(g_norm, dt)
        
        # fractional branch: D^α[γ]
        x_a = _fft_convolve(gd, _caputo_kernel(s, self.alpha), dt)

        # elastic branch: γ itself  (D^0[γ] = γ)
        x_0 = g_norm

        x_as, _, std_a = _scale_physical(x_a)
        x_0s, _, std_0 = _scale_physical(x_0)
        self._feature_stds = [std_a.item(), std_0.item()]
        #self._gamma_factors = [
        #    _caputo_gamma_factor(self.alpha).item(),   # Γ(1-α)
        #    1,    # Γ(1-β) ≈ 1 when β≈0
        #]
        #return torch.stack([x_as, x_0s], dim=1)           # (N, 2)
        # stack raw features before PCA
        X = torch.stack([x_as, x_0s], dim=1)   # (N, 2)

        # Initialize PCA / orthogonalization once, at first call
        # use .detach() so PCA is not part of the autograd graph
        self._init_pca_if_needed(X.detach())

        # Apply fixed 2×2 transform
        Z = X @ self._P   # (N, 2)

        return Z
    

    def named_phys_params(self):
        return {"alpha": self.alpha.item()}

    def Gstar(self, omega, prefactors):
        V = prefactors["V"]    # ∂σ/∂x_α  →  𝕍
        G = prefactors["G"]    # ∂σ/∂x_0  →  elastic modulus G
        a = self.alpha.item()
        Gp  = V * omega**a * np.cos(np.pi * a / 2) + G
        Gdp = V * omega**a * np.sin(np.pi * a / 2)
        return Gp, Gdp

    def param_labels(self):
        return ["α"]


# ================================================================
# 7. Fractional Kelvin-Voigt D  (FKV-D)
#
#   Dashpot (η) ∥ Springpot (𝔾,β)
#   σ = η · dγ/dt + 𝔾 · D^β[γ]
#
#   τ_c = (η/𝔾)^{1/(1-β)},   G_c = η · τ_c^{-1}
#
#   Features: [x_β = D^β[γ],   x_1 = γ̇]     (2 features)
#   Prefactors: 𝔾 = ∂σ/∂x_β,   η = ∂σ/∂x_1
#   G*(ω): G'  = 𝔾·ω^β·cos(πβ/2)
#          G'' = η·ω + 𝔾·ω^β·sin(πβ/2)
#   Params: β ∈ (0,1)
# ================================================================

class FKVDKernel(_BaseKernel):
    n_features = 2

    def __init__(self, beta_init=0.5):
        super().__init__()
        self.raw_beta = torch.nn.Parameter(
            torch.tensor(_raw_to_sigmoid(beta_init))
        )
        # fixed 2×2 PCA / orthogonalization matrix; start as identity
        # buffer = not a Parameter, keeps dtype/device, survives .to(), save/load
        self.register_buffer("_P", torch.eye(2))
        self._pca_initialized = False

    @property
    def beta(self):
        return torch.sigmoid(self.raw_beta)

    def _init_pca_if_needed(self, X):
        """
        Compute a fixed 2×2 orthogonalization matrix on first call.
        X: (N, 2) torch tensor of features [x_a, x_b] at the *current* α,β.
        The result is stored in self._P and never recomputed.
        """
        if self._pca_initialized:
            return

        # Center
        Xc = X - X.mean(dim=0, keepdim=True)

        # SVD: Xc = U S Vᵀ; columns of V are principal directions
        # Use double precision for stability, cast back to model dtype
        U, S, Vt = torch.linalg.svd(Xc.double(), full_matrices=False)
        P = Vt.T[:, :2].to(X.dtype)  # (2,2), columns orthonormal

        # Store as fixed transform
        # (register_buffer already done in __init__, so just copy into it)
        self._P.copy_(P)
        self._pca_initialized = True

    def compute_x(self, gamma, dt):
        N  = gamma.shape[0]
        s  = _lag_vector(N, dt, gamma.device, gamma.dtype)
        g_norm, _, scale_g  = _normalize(gamma)
        self._gamma_scale = scale_g
        gd = _gamma_dot(g_norm, dt)
        
        # fractional branch: D^β[γ]
        x_b = _fft_convolve(gd, _caputo_kernel(s, self.beta),  dt)

        # viscous branch: γ̇  (D^1[γ])
        x_1 = gd

        x_bs, _, std_b = _scale_physical(x_b)
        x_1s, _, std_1 = _scale_physical(x_1)
        self._feature_stds = [std_b.item(), std_1.item()]
        #self._gamma_factors = [
        #    _caputo_gamma_factor(self.beta).item(),    # Γ(1-β) ≈ 1 when β≈0
        #    1
        #]
        #return torch.stack([x_bs, x_1s], dim=1)           # (N, 2)
        # stack raw features before PCA
        X = torch.stack([x_bs, x_1s], dim=1)   # (N, 2)

        # Initialize PCA / orthogonalization once, at first call
        # use .detach() so PCA is not part of the autograd graph
        self._init_pca_if_needed(X.detach())

        # Apply fixed 2×2 transform
        Z = X @ self._P   # (N, 2)

        return Z
        
    
    def named_phys_params(self):
        return {"beta": self.beta.item()}

    def Gstar(self, omega, prefactors):
        G_bb = prefactors["G_bb"]  # 𝔾  (solid-like springpot)
        eta  = prefactors["eta"]   # η  (dashpot)
        b    = self.beta.item()
        Gp   = G_bb * omega**b * np.cos(np.pi * b / 2)
        Gdp  = eta * omega + G_bb * omega**b * np.sin(np.pi * b / 2)
        return Gp, Gdp

    def param_labels(self):
        return ["β"]


class FractionalKelvinVoigtKernel(_BaseKernel):
    """
    FKV: σ = 𝕍·D^α[γ] + 𝔾·D^β[γ],   0 ≤ β < α ≤ 1

    Root cause of inflated 𝕍 when α ≈ 1, β ≈ 0
    ---------------------------------------------
    The Caputo kernel is  K_α(s) = s^{-α} / Γ(1-α).

    As α → 1:  Γ(1-α) → Γ(0⁺) → ∞,  so K_α → 0 pointwise.
    The physical feature  x_α = K_α ★ γ̇  therefore has a very small
    amplitude — its empirical std  σ_{x,α}  ≈ ε  (tiny).

    StandardScaler divides by σ_{x,α}, giving a unit-variance GP input.
    The GP is well-conditioned.  But the back-transform to physical units
    is:
        𝕍 = ∂σ_scaled/∂x_scaled · (s_y / σ_{x,α})

    Dividing by a tiny σ_{x,α} inflates 𝕍 enormously even when the GP
    gradient is modest and correct.  The GP is *not* wrong — it learned
    that x_α is the dominant predictor — but the chain rule through a
    near-zero σ_{x,α} explodes numerically.

    Fix: separate gamma function
    ---------------------------------
    Γ(1-α) and Γ(1-β) are stored in _gamma_factors and multiplied
    back in _extract_prefactors.  This completely removes the Γ(0⁺)
    singularity from the chain rule without introducing sampling artefacts.
    """
    
    n_features = 2

    def __init__(self, alpha_init=0.7, beta_init=0.3):
        super().__init__()
        
        # independent parameters in (0,1)
        self.raw_alpha = torch.nn.Parameter(
            torch.tensor(_raw_to_sigmoid(alpha_init))
        )
        self.raw_beta = torch.nn.Parameter(
            torch.tensor(_raw_to_sigmoid(beta_init))
        )
        # fixed 2×2 PCA / orthogonalization matrix; start as identity
        # buffer = not a Parameter, keeps dtype/device, survives .to(), save/load
        self.register_buffer("_P", torch.eye(2))
        self._pca_initialized = False

    @property
    def alpha(self):
        return torch.sigmoid(self.raw_alpha)

    @property
    def beta(self):
        return torch.sigmoid(self.raw_beta)

    def _init_pca_if_needed(self, X):
        """
        Compute a fixed 2×2 orthogonalization matrix on first call.
        X: (N, 2) torch tensor of features [x_a, x_b] at the *current* α,β.
        The result is stored in self._P and never recomputed.
        """
        if self._pca_initialized:
            return

        # Center
        Xc = X - X.mean(dim=0, keepdim=True)

        # SVD: Xc = U S Vᵀ; columns of V are principal directions
        # Use double precision for stability, cast back to model dtype
        U, S, Vt = torch.linalg.svd(Xc.double(), full_matrices=False)
        P = Vt.T[:, :2].to(X.dtype)  # (2,2), columns orthonormal

        # Store as fixed transform
        # (register_buffer already done in __init__, so just copy into it)
        self._P.copy_(P)
        self._pca_initialized = True

    def compute_x(self, gamma, dt):
        '''
        Γ-split approach — identical to SpringpotKernel but for two branches.

        Feed  s^{-α}  and  s^{-β}  (shape only, no Γ) to the convolution.
        Store  Γ(1-α)  and  Γ(1-β)  in  _gamma_factors  for the back-transform.

        Why Γ-split works for both extremes:
            α → 1: Γ(1-α) → ∞  →  large correction  (fixes 𝕍 blow-up)
            β → 0: Γ(1-β) → 1  →  correction ≈ 1    (𝔾 unchanged, correct)
        '''
        N  = gamma.shape[0]
        s  = _lag_vector(N, dt, gamma.device, gamma.dtype)
        g_norm, _, scale_g  = _normalize(gamma)
        self._gamma_scale = scale_g
        gd = _gamma_dot(g_norm, dt)

        # shape-only kernels  s^{-α}  and  s^{-β}  (Γ factors stripped)
        x_a = _fft_convolve(gd, _caputo_kernel(s, self.alpha), dt)
        x_b = _fft_convolve(gd, _caputo_kernel(s, self.beta),  dt)
        

        x_as, _, std_a = _scale_physical(x_a)
        x_bs, _, std_b = _scale_physical(x_b)

        self._feature_stds  = [std_a.item(), std_b.item()]
        #self._gamma_factors = [
        #    _caputo_gamma_factor(self.alpha).item(),   # Γ(1-α)
        #    _caputo_gamma_factor(self.beta).item(),    # Γ(1-β) ≈ 1 when β≈0
        #]
        X = torch.stack([x_as, x_bs], dim=1)   # (N, 2)

        # Initialize PCA / orthogonalization once, at first call
        # use .detach() so PCA is not part of the autograd graph
        self._init_pca_if_needed(X.detach())

        # Apply fixed 2×2 transform
        Z = X @ self._P   # (N, 2)

        return Z


        #return torch.stack([x_as, x_bs], dim=1)   # (N, 2)

    def named_phys_params(self):
        return {"alpha": self.alpha.item(), "beta": self.beta.item()}
        
    def Gstar(self, omega, prefactors):
        V    = prefactors["V"]     # 𝕍  (liquid-like springpot)
        G_bb = prefactors["G_bb"] # 𝔾  (solid-like springpot)
        a    = self.alpha.item()
        b    = self.beta.item()
        Gp   = (V * omega**a * np.cos(np.pi*a/2)
                + G_bb * omega**b * np.cos(np.pi*b/2))
        Gdp  = (V * omega**a * np.sin(np.pi*a/2)
                + G_bb * omega**b * np.sin(np.pi*b/2))
        return Gp, Gdp

    def param_labels(self):
        return ["α", "β"]
'''

class FractionalKelvinVoigtKernel_old(_BaseKernel):
    """
    FKV: σ = 𝕍·D^α[γ] + 𝔾·D^β[γ],   0 ≤ β < α ≤ 1
    """
    n_features = 1

    def __init__(self, alpha_init=0.7, beta_init=0.3, tau_c_init=1):
        super().__init__()
        self.raw_alpha = torch.nn.Parameter(
            torch.tensor(_raw_to_sigmoid(alpha_init))
        )
        # β = α · sigmoid(raw_gap)  →  β < α always
        self.raw_gap   = torch.nn.Parameter(torch.tensor(0.0))

        self.raw_tau_c = torch.nn.Parameter(
            torch.tensor(_raw_to_softplus(tau_c_init))
        )

        # fixed 2×2 PCA / orthogonalization matrix; start as identity
        # buffer = not a Parameter, keeps dtype/device, survives .to(), save/load
        #self.register_buffer("_P", torch.eye(2))
        #self._pca_initialized = False

    @property
    def alpha(self):
        return torch.sigmoid(self.raw_alpha)

    @property
    def beta(self):
        return self.alpha * torch.sigmoid(self.raw_gap)

        self.raw_tau_c = torch.nn.Parameter(
            torch.tensor(_raw_to_softplus(tau_c_init))
        )

    @property
    def tau_c(self):
        return F.softplus(self.raw_tau_c) + 1e-4


    def _init_pca_if_needed(self, X):
        """
        Compute a fixed 2×2 orthogonalization matrix on first call.
        X: (N, 2) torch tensor of features [x_a, x_b] at the *current* α,β.
        The result is stored in self._P and never recomputed.
        """
        if self._pca_initialized:
            return

        # Center
        Xc = X - X.mean(dim=0, keepdim=True)

        # SVD: Xc = U S Vᵀ; columns of V are principal directions
        # Use double precision for stability, cast back to model dtype
        U, S, Vt = torch.linalg.svd(Xc.double(), full_matrices=False)
        P = Vt.T[:, :2].to(X.dtype)  # (2,2), columns orthonormal

        # Store as fixed transform
        # (register_buffer already done in __init__, so just copy into it)
        self._P.copy_(P)
        self._pca_initialized = True

    def compute_x(self, gamma, dt):
        """
        Γ-split approach — identical to SpringpotKernel but for two branches.

        Here we also apply a fixed 2D orthogonalization (PCA) to [x_a, x_b]
        to reduce collinearity between the two features. The PCA transform
        is computed once (on first call) and then held fixed.
        """
        N  = gamma.shape[0]
        s  = _lag_vector(N, dt, gamma.device, gamma.dtype)
        g_norm, _, scale_g  = _scale_physical(gamma)
        self._gamma_scale = scale_g
        gd = _gamma_dot(g_norm, dt)

        # shape-only kernels s^{-α}, s^{-β} (Γ factors stripped)
        x_a = _fft_convolve(gd, _caputo_kernel(s, self.alpha) * self.tau_c**self.alpha, dt)
        x_b = _fft_convolve(gd, _caputo_kernel(s, self.beta) * self.tau_c**self.beta,  dt)
        x_ab = x_a + x_b

        # no-op physical scaling (keeps interface consistent)
        x_as, _, std_a = _scale_physical(x_a)
        x_bs, _, std_b = _scale_physical(x_b)
        x_abs, _, std_ab = _scale(x_ab)
        # store feature stds as list of floats (as expected by SPGP._extract_prefactors)
        #self._feature_stds = [float(std_a), float(std_b)]
        self._feature_stds = [std_ab.item()]
        return x_abs.unsqueeze(-1)
        #self._gamma_factors = [
        #    _caputo_gamma_factor(self.alpha).item(),   # Γ(1-α)
        #    _caputo_gamma_factor(self.beta).item(),    # Γ(1-β) ≈ 1 when β≈0
        #]

        # stack raw features before PCA
        #X = torch.stack([x_as, x_bs], dim=1)   # (N, 2)

        # Initialize PCA / orthogonalization once, at first call
        # use .detach() so PCA is not part of the autograd graph
        #self._init_pca_if_needed(X.detach())

        # Apply fixed 2×2 transform
        #Z = X @ self._P   # (N, 2)

        #return Z

        #self._feature_stds  = [std_a.item(), std_b.item()]
        #self._gamma_factors = [
        #    _caputo_gamma_factor(self.alpha).item(),   # Γ(1-α)
        #    _caputo_gamma_factor(self.beta).item(),    # Γ(1-β) ≈ 1 when β≈0
        #]


        #return torch.stack([x_as, x_bs], dim=1)   # (N, 2)

    
    def named_phys_params(self):
        return {
            "alpha": self.alpha.item(),
            "beta":  self.beta.item(),
            "tau_c": self.tau_c.item(),
        }

    def Gstar(self, omega, prefactors):
        Gc = prefactors["Gc"]
        #V    = prefactors["V"]    # 𝕍  (liquid-like springpot)
        #G_bb = prefactors["G_bb"]  # 𝔾  (solid-like springpot)
        tau = self.tau_c.item()
        a    = self.alpha.item()
        b    = self.beta.item()
        Gp   = (Gc*tau**a * omega**a * np.cos(np.pi*a/2)
                + Gc*tau**b * omega**b * np.cos(np.pi*b/2))
        Gdp  = (Gc*tau**a * omega**a * np.sin(np.pi*a/2)
                + Gc*tau**b * omega**b * np.sin(np.pi*b/2))
        return Gp, Gdp

    def param_labels(self):
        return ["α", "β", "τ_c"]

'''



'''
class FractionalKelvinVoigtKernel(_BaseKernel):
    """
    FKV: σ = 𝕍·D^α[γ] + 𝔾·D^β[γ],   0 ≤ β < α ≤ 1

    Root cause of inflated 𝕍 when α ≈ 1, β ≈ 0
    ---------------------------------------------
    The Caputo kernel is  K_α(s) = s^{-α} / Γ(1-α).

    As α → 1:  Γ(1-α) → Γ(0⁺) → ∞,  so K_α → 0 pointwise.
    The physical feature  x_α = K_α ★ γ̇  therefore has a very small
    amplitude — its empirical std  σ_{x,α}  ≈ ε  (tiny).

    StandardScaler divides by σ_{x,α}, giving a unit-variance GP input.
    The GP is well-conditioned.  But the back-transform to physical units
    is:
        𝕍 = ∂σ_scaled/∂x_scaled · (s_y / σ_{x,α})

    Dividing by a tiny σ_{x,α} inflates 𝕍 enormously even when the GP
    gradient is modest and correct.  The GP is *not* wrong — it learned
    that x_α is the dominant predictor — but the chain rule through a
    near-zero σ_{x,α} explodes numerically.

    Fix: separate gamma function
    ---------------------------------
    Γ(1-α) and Γ(1-β) are stored in _gamma_factors and multiplied
    back in _extract_prefactors.  This completely removes the Γ(0⁺)
    singularity from the chain rule without introducing sampling artefacts.
    """
    n_features = 2

    def __init__(self, alpha_init=0.7, beta_init=0.3):
        super().__init__()
        self.raw_alpha = torch.nn.Parameter(
            torch.tensor(_raw_to_sigmoid(alpha_init))
        )
        # β = α · sigmoid(raw_gap)  →  β < α always
        self.raw_gap   = torch.nn.Parameter(torch.tensor(0.0))

    @property
    def alpha(self):
        return torch.sigmoid(self.raw_alpha)

    @property
    def beta(self):
        return self.alpha * torch.sigmoid(self.raw_gap)

    def compute_x(self, gamma, dt):
        """
        Γ-split approach — identical to SpringpotKernel but for two branches.

        Feed  s^{-α}  and  s^{-β}  (shape only, no Γ) to the convolution.
        Store  Γ(1-α)  and  Γ(1-β)  in  _gamma_factors  for the back-transform.

        Why Γ-split works for both extremes:
            α → 1: Γ(1-α) → ∞  →  large correction  (fixes 𝕍 blow-up)
            β → 0: Γ(1-β) → 1  →  correction ≈ 1    (𝔾 unchanged, correct)
        """
        N  = gamma.shape[0]
        s  = _lag_vector(N, dt, gamma.device, gamma.dtype)
        g_norm, _, scale_g  = _normalize(gamma)
        self._gamma_scale = scale_g
        gd = _gamma_dot(g_norm, dt)

        # shape-only kernels  s^{-α}  and  s^{-β}  (Γ factors stripped)
        x_a = _fft_convolve(gd, _caputo_kernel(s, self.alpha), dt)
        x_b = _fft_convolve(gd, _caputo_kernel(s, self.beta),  dt)
        

        x_as, _, std_a = _scale_physical(x_a)
        x_bs, _, std_b = _scale_physical(x_b)


        self._feature_stds  = [std_a.item(), std_b.item()]
        #self._gamma_factors = [
        #    _caputo_gamma_factor(self.alpha).item(),   # Γ(1-α)
        #    _caputo_gamma_factor(self.beta).item(),    # Γ(1-β) ≈ 1 when β≈0
        #]


        return torch.stack([x_as, x_bs], dim=1)   # (N, 2)

    def named_phys_params(self):
        return {"alpha": self.alpha.item(), "beta": self.beta.item()}
        
    def Gstar(self, omega, prefactors):
        V    = prefactors["V"]     # 𝕍  (liquid-like springpot)
        G_bb = prefactors["G_bb"] # 𝔾  (solid-like springpot)
        a    = self.alpha.item()
        b    = self.beta.item()
        Gp   = (V * omega**a * np.cos(np.pi*a/2)
                + G_bb * omega**b * np.cos(np.pi*b/2))
        Gdp  = (V * omega**a * np.sin(np.pi*a/2)
                + G_bb * omega**b * np.sin(np.pi*b/2))
        return Gp, Gdp

    def param_labels(self):
        return ["α", "β"]
'''

'''

    def compute_x(self, gamma, dt):
        N  = gamma.shape[0]
        s  = _lag_vector(N, dt, gamma.device, gamma.dtype)
        g_norm, _, scale_g  = _scale_physical(gamma)
        self._gamma_scale = scale_g
        gd = _gamma_dot(g_norm, dt)
        K  = _fmm_kernel(s, self.alpha, self.beta, self.tau_c)
        x  = _fft_convolve(gd, K, dt)
        xs, _, std = _normalize(x)
        self._feature_stds = [std.item()]
        return xs.unsqueeze(-1)

    def named_phys_params(self):
        return {
            "alpha": self.alpha.item(),
            "beta":  self.beta.item(),
            "tau_c": self.tau_c.item(),
        }

    def Gstar(self, omega, prefactors):
        Gc  = prefactors["Gc"]
        a   = self.alpha.item()
        b   = self.beta.item()
        tc  = self.tau_c.item()
        wtc = omega * tc
        cos_ab = np.cos(np.pi * (a - b) / 2)
        denom  = 1.0 + wtc**(2*(a-b)) + 2.0 * wtc**(a-b) * cos_ab
        Gp  = Gc * (wtc**a * np.cos(np.pi*a/2)
                    + wtc**(2*a-b) * np.cos(np.pi*b/2)) / denom
        Gdp = Gc * (wtc**a * np.sin(np.pi*a/2)
                    + wtc**(2*a-b) * np.sin(np.pi*b/2)) / denom
        return Gp, Gdp

    def param_labels(self):
        return ["α", "β", "τ_c"]

# ================================================================
# 8. Fractional Kelvin-Voigt  (FKV, full)
#
#   Springpot (𝕍,α) ∥ Springpot (𝔾,β),   0 ≤ β ≤ α ≤ 1
#   σ = 𝕍 · D^α[γ] + 𝔾 · D^β[γ]
#
#   τ_c = (𝕍/𝔾)^{1/(α-β)},   G_c = 𝕍 · τ_c^{-α}
#
#   Features: [x_α = D^α[γ],   x_β = D^β[γ]]   (2 features)
#   Prefactors: 𝕍 = ∂σ/∂x_α,   𝔾 = ∂σ/∂x_β
#   G*(ω): G'  = 𝕍·ω^α·cos(πα/2) + 𝔾·ω^β·cos(πβ/2)
#          G'' = 𝕍·ω^α·sin(πα/2) + 𝔾·ω^β·sin(πβ/2)
#   Params: α ∈ (0,1),  β = α·sigmoid(raw_gap) ∈ (0,α)
# ================================================================
class FractionalKelvinVoigtKernel(_BaseKernel):
    n_features = 2

    def __init__(self, alpha_init=0.7, beta_init=0.3):
        super().__init__()
        self.raw_alpha = torch.nn.Parameter(
            torch.tensor(_raw_to_sigmoid(alpha_init))
        )
        # β = α · sigmoid(raw_gap)  →  β < α always
        self.raw_gap   = torch.nn.Parameter(torch.tensor(0.0))

    @property
    def alpha(self):
        return torch.sigmoid(self.raw_alpha)

    @property
    def beta(self):
        return self.alpha * torch.sigmoid(self.raw_gap)

    def compute_x(self, gamma, dt):
        N  = gamma.shape[0]
        s  = _lag_vector(N, dt, gamma.device, gamma.dtype)
        gd = _gamma_dot(gamma, dt)

        x_a = _fft_convolve(gd, _caputo_kernel(s, self.alpha), dt)
        x_b = _fft_convolve(gd, _caputo_kernel(s, self.beta),  dt)

        x_as, _, std_a = _scale(x_a)
        x_bs, _, std_b = _scale(x_b)
        self._feature_stds = [std_a.item(), std_b.item()]
        return torch.stack([x_as, x_bs], dim=1)           # (N, 2)

    def named_phys_params(self):
        return {"alpha": self.alpha.item(), "beta": self.beta.item()}

    def Gstar(self, omega, prefactors):
        V    = prefactors["V"]     # 𝕍  (liquid-like springpot)
        G_bb = prefactors["G_bb"] # 𝔾  (solid-like springpot)
        a    = self.alpha.item()
        b    = self.beta.item()
        Gp   = (V * omega**a * np.cos(np.pi*a/2)
                + G_bb * omega**b * np.cos(np.pi*b/2))
        Gdp  = (V * omega**a * np.sin(np.pi*a/2)
                + G_bb * omega**b * np.sin(np.pi*b/2))
        return Gp, Gdp

    def param_labels(self):
        return ["α", "β"]
'''
        
'''
class FractionalKelvinVoigtKernel(_BaseKernel):
    n_features = 2

    def __init__(self, alpha_init=0.7, beta_init=0.3):
        super().__init__()
        self.raw_alpha = torch.nn.Parameter(
            torch.tensor(_raw_to_sigmoid(alpha_init))
        )
        # β = α · sigmoid(raw_gap)  →  β < α always
        self.raw_gap   = torch.nn.Parameter(torch.tensor(0.0))

    @property
    def alpha(self):
        return torch.sigmoid(self.raw_alpha)

    @property
    def beta(self):
        return self.alpha * torch.sigmoid(self.raw_gap)

    def compute_x(self, gamma, dt):
        N  = gamma.shape[0]
        s  = _lag_vector(N, dt, gamma.device, gamma.dtype)
        gd = _gamma_dot(gamma, dt)

        # Raw Caputo features
        x_a = _fft_convolve(gd, _caputo_kernel(s, self.alpha), dt)  # D^α[γ]
        x_b = _fft_convolve(gd, _caputo_kernel(s, self.beta),  dt)  # D^β[γ]

        # α-/β-aware pre-scaling to stabilize κ→1
        # S(κ,dt) = Γ(1-κ) * dt^(κ-1); cancels the small-gain as κ→1
        S_a = torch.exp(torch.lgamma(1.0 - self.alpha)) * (dt ** (self.alpha - 1.0))
        # Option A: stabilize β near 1 as well
        #S_b = torch.exp(torch.lgamma(1.0 - self.beta))  * (dt ** (self.beta  - 1.0))
        # Option B (if you want β→0 to keep γ-scale): uncomment the next line
        S_b = torch.ones((), device=gamma.device, dtype=gamma.dtype)

        x_a = x_a * S_a
        x_b = x_b * S_b

        # z-score for optimization
        x_as, _, std_a = _scale(x_a)
        x_bs, _, std_b = _scale(x_b)

        # cache scales (tensors) for recovery; keep stds for chain rule
        self._feature_stds  = [std_a.item(), std_b.item()]
        self._feature_stds_t = (std_a.detach(), std_b.detach())
        self._alpha_scales   = (S_a.detach(), S_b.detach())

        return torch.stack([x_as, x_bs], dim=1)  # (N, 2)

    def named_phys_params(self):
        return {"alpha": self.alpha.item(), "beta": self.beta.item()}

    def Gstar(self, omega, prefactors):
        V    = prefactors["V"]     # 𝕍  (liquid-like springpot)
        G_bb = prefactors["G_bb"]  # 𝔾  (solid-like springpot)
        a    = self.alpha.item()
        b    = self.beta.item()
        Gp   = (V * omega**a * np.cos(np.pi*a/2)
                + G_bb * omega**b * np.cos(np.pi*b/2))
        Gdp  = (V * omega**a * np.sin(np.pi*a/2)
                + G_bb * omega**b * np.sin(np.pi*b/2))
        return Gp, Gdp

    def param_labels(self):
        return ["α", "β"]


class FractionalKelvinVoigtKernel(_BaseKernel):
    """
    FKV: σ = 𝕍·D^α[γ] + 𝔾·D^β[γ],   0 ≤ β < α ≤ 1

    Root cause of inflated 𝕍 when α ≈ 1, β ≈ 0
    ---------------------------------------------
    The Caputo kernel is  K_α(s) = s^{-α} / Γ(1-α).

    As α → 1:  Γ(1-α) → Γ(0⁺) → ∞,  so K_α → 0 pointwise.
    The physical feature  x_α = K_α ★ γ̇  therefore has a very small
    amplitude — its empirical std  σ_{x,α}  ≈ ε  (tiny).

    StandardScaler divides by σ_{x,α}, giving a unit-variance GP input.
    The GP is well-conditioned.  But the back-transform to physical units
    is:
        𝕍 = ∂σ_scaled/∂x_scaled · (s_y / σ_{x,α})

    Dividing by a tiny σ_{x,α} inflates 𝕍 enormously even when the GP
    gradient is modest and correct.  The GP is *not* wrong — it learned
    that x_α is the dominant predictor — but the chain rule through a
    near-zero σ_{x,α} explodes numerically.

    Fix: separate gamma function
    ---------------------------------
    Γ(1-α) and Γ(1-β) are stored in _gamma_factors and multiplied
    back in _extract_prefactors.  This completely removes the Γ(0⁺)
    singularity from the chain rule without introducing sampling artefacts.
    """
    
    n_features = 2

    def __init__(self, alpha_init=0.7, beta_init=0.3):
        super().__init__()
        
        # independent parameters in (0,1)
        self.raw_alpha = torch.nn.Parameter(
            torch.tensor(_raw_to_sigmoid(alpha_init))
        )
        self.raw_beta = torch.nn.Parameter(
            torch.tensor(_raw_to_sigmoid(beta_init))
        )

    @property
    def alpha(self):
        return torch.sigmoid(self.raw_alpha)

    @property
    def beta(self):
        return torch.sigmoid(self.raw_beta)

    def compute_x(self, gamma, dt):
        """
        Γ-split approach — identical to SpringpotKernel but for two branches.

        Feed  s^{-α}  and  s^{-β}  (shape only, no Γ) to the convolution.
        Store  Γ(1-α)  and  Γ(1-β)  in  _gamma_factors  for the back-transform.

        Why Γ-split works for both extremes:
            α → 1: Γ(1-α) → ∞  →  large correction  (fixes 𝕍 blow-up)
            β → 0: Γ(1-β) → 1  →  correction ≈ 1    (𝔾 unchanged, correct)
        """
        N  = gamma.shape[0]
        s  = _lag_vector(N, dt, gamma.device, gamma.dtype)
        g_norm, _, scale_g  = _scale_physical(gamma)
        self._gamma_scale = scale_g
        gd = _gamma_dot(g_norm, dt)

        # shape-only kernels  s^{-α}  and  s^{-β}  (Γ factors stripped)
        x_a = _fft_convolve(gd, _caputo_kernel(s, self.alpha), dt)
        x_b = _fft_convolve(gd, _caputo_kernel(s, self.beta),  dt)
        

        x_as, _, std_a = _normalize(x_a)
        x_bs, _, std_b = _normalize(x_b)

        self._feature_stds  = [std_a.item(), std_b.item()]
        #self._gamma_factors = [
        #    _caputo_gamma_factor(self.alpha).item(),   # Γ(1-α)
        #    _caputo_gamma_factor(self.beta).item(),    # Γ(1-β) ≈ 1 when β≈0
        #]


        return torch.stack([x_as, x_bs], dim=1)   # (N, 2)

    def named_phys_params(self):
        return {"alpha": self.alpha.item(), "beta": self.beta.item()}
        
    def Gstar(self, omega, prefactors):
        V    = prefactors["V"]     # 𝕍  (liquid-like springpot)
        G_bb = prefactors["G_bb"] # 𝔾  (solid-like springpot)
        a    = self.alpha.item()
        b    = self.beta.item()
        Gp   = (V * omega**a * np.cos(np.pi*a/2)
                + G_bb * omega**b * np.cos(np.pi*b/2))
        Gdp  = (V * omega**a * np.sin(np.pi*a/2)
                + G_bb * omega**b * np.sin(np.pi*b/2))
        return Gp, Gdp

    def param_labels(self):
        return ["α", "β"]
'''

# ================================================================
# Registry
# ================================================================

KERNELS = {
    # canonical names
    "kernelfree":   KernelFreeKernel,
    "kf":           KernelFreeKernel,
    "maxwell":                       MaxwellKernel,
    "springpot":                     SpringpotKernel,
    "fractionalmaxwellgel":          FractionalMaxwellGelKernel,
    "fractionalmaxwellliquid":       FractionalMaxwellLiquidKernel,
    "fractionalmaxwell":             FractionalMaxwellKernel,
    "fractionalkelvinvoigts":        FKVSKernel,
    "fractionalkelvinvoigtd":        FKVDKernel,
    "fractionalkelvinvoigt":         FractionalKelvinVoigtKernel,
    # short aliases
    "fmg":                           FractionalMaxwellGelKernel,
    "fml":                           FractionalMaxwellLiquidKernel,
    "fmm":                           FractionalMaxwellKernel,
    "fkvs":                          FKVSKernel,
    "fkvd":                          FKVDKernel,
    "fkv":                           FractionalKelvinVoigtKernel,
}

# ----------------------------------------------------------------
# Prefactor column map  { kernel_class: { name: feature_col } }
# ----------------------------------------------------------------
PREFACTOR_COLS = {
    KernelFreeKernel: {"sigma_gamma": 0, "sigma_gammadot": 1},
    MaxwellKernel:                   {"Gc":  0},
    SpringpotKernel:                 {"V":   0},
    FractionalMaxwellGelKernel:      {"Gc":  0},
    FractionalMaxwellLiquidKernel:   {"Gc":  0},
    FractionalMaxwellKernel:         {"Gc":  0},
    FKVSKernel:                      {"V":   0, "G":    1},
    FKVDKernel:                      {"G_bb":0, "eta":  1},
#    FractionalKelvinVoigtKernel:     {"Gc":   0},
    FractionalKelvinVoigtKernel:     {"V":   0, "G_bb":  1},

}