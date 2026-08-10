"""
rheogp.features
===============
Unified, differentiable feature-construction engine.

Two convolution modes
---------------------

``causal``  — integral from 0 to t  (material at rest for t < 0)

    x(t) = ∫₀ᵗ K(t - t') u(t') dt'

    The material is assumed undeformed before the first sample.
    Correct when the experiment truly starts from rest, but a poor
    quadrature of the singular kernel  s^{-α}  near s → 0 creates a
    *start-up transient*.  Three quadrature rules are provided:

      * ``"exact"``     (default)  — the kernel is integrated
        analytically over each sampling bin (product-integration /
        L1-type rule for power laws, closed forms for exp; midpoint
        otherwise).  This removes the s→0 singularity error and
        strongly suppresses the start-up transient.
      * ``"midpoint"``  — kernel sampled at s = (m + 1/2)·dt.
      * ``"left"``      — legacy rule, s = (m + 1)·dt (reproduces
        rheogp ≤ 0.2 numbers exactly).

``steady``  — integral from -inf to t  (periodic steady state)

    x(t) = ∫_{-∞}^{t} K(t - t') u(t') dt'
         = IFFT[ K̂*(ω) · û(ω) ]        (N-point FFT, NO zero padding)

    The signal is assumed to repeat with period T = N·dt, i.e. the
    material has been driven by the same waveform forever.  For an
    optimally-windowed chirp (which tapers to zero at both ends) the
    periodic extension is continuous, so this is an excellent
    approximation and there is NO start-up transient at all.

    IMPORTANT — why zero padding must not be used here:
    padding to 2N and using the analytic kernel transform implies a
    periodic history of "N zeros followed by the signal".  A
    long-memory kernel then leaks response into the padded half and
    the amplitude of the retained N samples is systematically
    UNDERESTIMATED.  This is the bug the v0.2 experimental
    frequency-space feature suffered from.  The steady engine below
    uses n_fft = N and zeroes the DC bin instead.

All functions are written in torch and are differentiable with
respect to the kernel shape parameters (α, β, τ_c, ...).
"""

import torch

__all__ = [
    "rate",
    "causal_convolve",
    "steady_convolve",
    "frac_deriv_causal",
    "frac_integral_causal",
    "frac_deriv_steady",
    "frac_integral_steady",
]


# ----------------------------------------------------------------
# signal helpers
# ----------------------------------------------------------------

def rate(u, dt):
    """Causal forward-difference derivative, zero-padded to length N."""
    return torch.cat([
        torch.zeros(1, device=u.device, dtype=u.dtype),
        (u[1:] - u[:-1]) / dt
    ])


def _linear_fft_convolve(u, w):
    """Linear (a-cyclic) convolution of u with weight vector w, first N samples."""
    N = u.shape[0]
    n_fft = 2 * N
    return torch.fft.irfft(
        torch.fft.rfft(u, n=n_fft) * torch.fft.rfft(w, n=n_fft),
        n=n_fft,
    )[:N]


# ----------------------------------------------------------------
# causal mode  (integral from 0)
# ----------------------------------------------------------------

def _lag_left(N, dt, device, dtype):
    """s = dt, 2dt, ..., N·dt (legacy left rule)."""
    return torch.arange(1, N + 1, device=device, dtype=dtype) * dt


def _lag_mid(N, dt, device, dtype):
    """s = dt/2, 3dt/2, ... midpoint of each bin."""
    return (torch.arange(0, N, device=device, dtype=dtype) + 0.5) * dt


def causal_convolve(u, kernel_fn, dt, quadrature="exact"):
    """
    Causal convolution  x_i = Σ_m w_m u_{i-m}  approximating
    ∫₀ᵗ K(s) u(t-s) ds  with the material at rest for t < 0.

    Parameters
    ----------
    u          : (N,) tensor — the signal being filtered (γ̇, σ̇, σ, ...)
    kernel_fn  : callable(s_tensor) -> K(s) values (differentiable)
    dt         : float
    quadrature : "exact" | "midpoint" | "left"
        "exact" here means midpoint sampling of a *regular* kernel;
        for singular power-law kernels use frac_deriv_causal /
        frac_integral_causal, which integrate the singularity exactly.
    """
    N = u.shape[0]
    if quadrature == "left":
        s = _lag_left(N, dt, u.device, u.dtype)
    else:  # "midpoint" and "exact" coincide for regular kernels
        s = _lag_mid(N, dt, u.device, u.dtype)
    w = kernel_fn(s) * dt
    return _linear_fft_convolve(u, w)


def _powerlaw_bin_weights(N, dt, p, device, dtype):
    """
    Exact bin-integrated weights for the kernel  K(s) = s^{-p},  p < 1:

        w_m = ∫_{m·dt}^{(m+1)·dt} s^{-p} ds
            = dt^{1-p} · ((m+1)^{1-p} - m^{1-p}) / (1-p)

    Differentiable w.r.t. p.  Valid for p < 1 (integrable singularity).
    """
    m = torch.arange(0, N, device=device, dtype=dtype)
    q = 1.0 - p
    return dt ** q * ((m + 1.0) ** q - m ** q) / q


def frac_deriv_causal(gamma_dot, dt, alpha, quadrature="exact"):
    """
    Caputo fractional derivative  D^α[γ](t) = ∫₀ᵗ (t-t')^{-α}/Γ(1-α) γ̇ dt'.

    "exact"    — product integration: the s^{-α} singularity is
                 integrated analytically over each bin (removes the
                 start-up transient of the left rule).
    "midpoint" — s sampled at bin midpoints.
    "left"     — legacy rheogp ≤ 0.2 rule (s starts at dt).
    """
    N = gamma_dot.shape[0]
    dev, dtp = gamma_dot.device, gamma_dot.dtype
    if quadrature == "exact":
        w = _powerlaw_bin_weights(N, dt, alpha, dev, dtp) \
            / torch.exp(torch.lgamma(1.0 - alpha))
        return _linear_fft_convolve(gamma_dot, w)
    s = _lag_left(N, dt, dev, dtp) if quadrature == "left" \
        else _lag_mid(N, dt, dev, dtp)
    K = s ** (-alpha) / torch.exp(torch.lgamma(1.0 - alpha))
    return _linear_fft_convolve(gamma_dot, K * dt)


def frac_integral_causal(sigma, dt, alpha, quadrature="exact"):
    """
    Riemann–Liouville fractional integral (creep feature)

        I^α[σ](t) = ∫₀ᵗ (t-t')^{α-1}/Γ(α) σ(t') dt' ,   0 < α ≤ 1.

    Note: convolved directly with σ (not σ̇).
    """
    N = sigma.shape[0]
    dev, dtp = sigma.device, sigma.dtype
    if quadrature == "exact":
        # kernel s^{α-1} = s^{-(1-α)}  → p = 1-α
        w = _powerlaw_bin_weights(N, dt, 1.0 - alpha, dev, dtp) \
            / torch.exp(torch.lgamma(alpha))
        return _linear_fft_convolve(sigma, w)
    s = _lag_left(N, dt, dev, dtp) if quadrature == "left" \
        else _lag_mid(N, dt, dev, dtp)
    K = s ** (alpha - 1.0) / torch.exp(torch.lgamma(alpha))
    return _linear_fft_convolve(sigma, K * dt)


# ----------------------------------------------------------------
# steady mode  (integral from -inf, periodic assumption)
# ----------------------------------------------------------------

def _omega_grid(N, dt, device):
    """Angular frequencies of the N-point rFFT (double precision)."""
    freqs = torch.fft.rfftfreq(N, d=dt).to(device=device, dtype=torch.float64)
    return 2.0 * torch.pi * freqs


def steady_convolve(u, transfer_fn, dt):
    """
    Steady-state (from -inf) response of the filter with complex
    transfer function H(ω) to the periodic signal u:

        x = IFFT[ H(ω) · û(ω) ]   with n_fft = N (no zero padding)

    The DC bin is zeroed: the steady-state response to the mean of u
    is either zero (fractional derivatives), divergent (fractional
    integrals of a biased signal), or should be handled by an explicit
    elastic/identity branch — never by the convolution feature.

    Parameters
    ----------
    u           : (N,) tensor, assumed periodic with period N·dt
    transfer_fn : callable(omega_float64_tensor) -> complex tensor.
                  For a strain-controlled branch this is the modulus
                  shape g*(ω) = iω·K̂*(ω); for a stress-controlled
                  branch it is the compliance shape 1/g*(ω).
    """
    N = u.shape[0]
    omega = _omega_grid(N, dt, u.device)
    H = transfer_fn(omega[1:])                       # skip DC
    U = torch.fft.rfft(u.to(torch.float64))
    X = torch.zeros_like(U)
    X[1:] = H.to(torch.complex128) * U[1:]           # DC bin = 0
    return torch.fft.irfft(X, n=N).to(u.dtype)


def frac_deriv_steady(gamma, dt, alpha):
    """Steady-state D^α[γ]:  transfer  (iω)^α  acting on γ."""
    a = alpha.to(torch.float64) if torch.is_tensor(alpha) else torch.tensor(
        float(alpha), dtype=torch.float64, device=gamma.device)
    return steady_convolve(
        gamma, lambda w: (1j * w) ** a, dt
    )


def frac_integral_steady(sigma, dt, alpha):
    """Steady-state I^α[σ]:  transfer  (iω)^{-α}  acting on σ (zero-mean)."""
    a = alpha.to(torch.float64) if torch.is_tensor(alpha) else torch.tensor(
        float(alpha), dtype=torch.float64, device=sigma.device)
    return steady_convolve(
        sigma, lambda w: (1j * w) ** (-a), dt
    )
