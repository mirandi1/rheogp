"""
rheogp.utils
============
Model comparison helper + synthetic data generators for all 8 models.
"""

import numpy as np
from scipy.special import gamma as sc_gamma

__all__ = [
    "compare_models",
    "make_chirp",
    "make_synthetic_maxwell",
    "make_synthetic_springpot",
    "make_synthetic_fmg",
    "make_synthetic_fml",
    "make_synthetic_fmm",
    "make_synthetic_fkvs",
    "make_synthetic_fkvd",
    "make_synthetic_fkv",
]


# ----------------------------------------------------------------
# Shared signal / convolution helpers  (numpy only)
# ----------------------------------------------------------------

def make_chirp(n=2000, dt=1e-3, omega1=0.1, omega2=10.0, taper=0.0):
    """Linear-sweep (chirp) signal.

    Parameters
    ----------
    taper : float in [0, 0.5]
        Tukey-window fraction applied to each end (OWCh-style).
        ``taper=0`` (default) reproduces the historic un-windowed chirp;
        ``taper=0.1`` gives a signal that starts and ends at zero, for
        which the ``convolution="steady"`` periodicity assumption holds.
    """
    t = np.arange(n) * dt
    f = (omega1 + (omega2 - omega1) * t / t[-1]) / (2 * np.pi)
    wave = np.sin(2 * np.pi * f * t)
    if taper > 0:
        m = max(int(taper * n), 1)
        ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(m) / m))
        win = np.ones(n)
        win[:m] = ramp
        win[-m:] = ramp[::-1]
        wave = wave * win
    return t, wave


def _gd(strain, dt):
    return np.gradient(strain, dt)


def _ml_np(z, a, b, n_terms=30):
    """Numpy Mittag-Leffler E_{a,b}(z), no autograd needed."""
    result = np.zeros_like(z, dtype=float)
    for k in range(n_terms):
        result += z**k / sc_gamma(a * k + b)
    return result


def _caputo_np(gd, dt, alpha):
    """Caputo derivative via exact product integration of the
    power-law kernel (bin-integrated weights — no quadrature bias)."""
    N = len(gd)
    m = np.arange(N)
    q = 1.0 - alpha
    w = dt**q * ((m + 1)**q - m**q) / q / sc_gamma(1.0 - alpha)
    return np.convolve(gd, w)[:N]


def _fmm_np(gd, dt, alpha, beta, tau_c):
    N = len(gd)
    s = (np.arange(N) + 0.5) * dt          # midpoint rule
    a = alpha - beta
    b = 1.0   - beta
    u = s / tau_c
    K = u**(-beta) * _ml_np(-u**a, a, b)
    return np.convolve(gd, K)[:N] * dt


def _noisy(stress, noise, rng):
    return stress + rng.normal(0, noise * np.std(stress), size=len(stress))


# ----------------------------------------------------------------
# Synthetic generators
# ----------------------------------------------------------------

def make_synthetic_maxwell(Gc=100.0, tau_c=1.0,
                            n=2000, dt=1e-3,
                            omega1=0.1, omega2=10.0,
                            noise=0.01, seed=42):
    """σ = Gc · (exp(-·/τ_c) ★ γ̇)"""
    rng = np.random.default_rng(seed)
    t, strain = make_chirp(n, dt, omega1, omega2)
    gd = _gd(strain, dt)
    s  = np.arange(1, n + 1) * dt
    K  = np.exp(-s / tau_c)
    x  = np.convolve(gd, K)[:n] * dt
    return t, strain, _noisy(Gc * x, noise, rng)


def make_synthetic_springpot(V=100.0, alpha=0.5,
                              n=2000, dt=1e-3,
                              omega1=0.1, omega2=10.0,
                              noise=0.01, seed=42, taper=0.0):
    """σ = 𝕍 · D^α[γ]"""
    rng = np.random.default_rng(seed)
    t, strain = make_chirp(n, dt, omega1, omega2, taper=taper)
    x = _caputo_np(_gd(strain, dt), dt, alpha)
    return t, strain, _noisy(V * x, noise, rng)


def make_synthetic_fmg(V=100.0, G=100.0, alpha=0.5,
                        n=2000, dt=1e-3,
                        omega1=0.1, omega2=10.0,
                        noise=0.01, seed=42):
    """
    Fractional Maxwell Gel
    τ_c = (𝕍/G)^{1/α},  G_c = 𝕍·τ_c^{-α}
    Kernel: FMM with β=0
    """
    rng = np.random.default_rng(seed)
    t, strain = make_chirp(n, dt, omega1, omega2)
    tau_c = (V / G) ** (1.0 / alpha)
    Gc    = V * tau_c**(-alpha)
    x     = _fmm_np(_gd(strain, dt), dt, alpha, 0.0, tau_c)
    return t, strain, _noisy(Gc * x, noise, rng)


def make_synthetic_fml(G_bb=100.0, eta=100.0, beta=0.5,
                        n=2000, dt=1e-3,
                        omega1=0.1, omega2=10.0,
                        noise=0.01, seed=42):
    """
    Fractional Maxwell Liquid
    τ_c = (η/𝔾)^{1/(1-β)},  G_c = η·τ_c^{-1}
    Kernel: FMM with α→1
    """
    rng = np.random.default_rng(seed)
    t, strain = make_chirp(n, dt, omega1, omega2)
    tau_c = (eta / G_bb) ** (1.0 / (1.0 - beta))
    Gc    = eta * tau_c**(-1)
    x     = _fmm_np(_gd(strain, dt), dt, 0.9999, beta, tau_c)
    return t, strain, _noisy(Gc * x, noise, rng)


def make_synthetic_fmm(V=100.0, G_bb=100.0, alpha=0.7, beta=0.3,
                        n=2000, dt=1e-3,
                        omega1=0.1, omega2=10.0,
                        noise=0.01, seed=42):
    """
    Fractional Maxwell (full)
    τ_c = (𝕍/𝔾)^{1/(α-β)},  G_c = 𝕍·τ_c^{-α}
    """
    rng = np.random.default_rng(seed)
    t, strain = make_chirp(n, dt, omega1, omega2)
    tau_c = (V / G_bb) ** (1.0 / (alpha - beta))
    Gc    = V * tau_c**(-alpha)
    x     = _fmm_np(_gd(strain, dt), dt, alpha, beta, tau_c)
    return t, strain, _noisy(Gc * x, noise, rng)


def make_synthetic_fkvs(V=100.0, G=100.0, alpha=0.5,
                         n=2000, dt=1e-3,
                         omega1=0.1, omega2=10.0,
                         noise=0.01, seed=42):
    """
    FKV-S  :  σ = 𝕍·D^α[γ] + G·γ
    """
    rng = np.random.default_rng(seed)
    t, strain = make_chirp(n, dt, omega1, omega2)
    xa = _caputo_np(_gd(strain, dt), dt, alpha)
    stress = V * xa + G * strain
    return t, strain, _noisy(stress, noise, rng)


def make_synthetic_fkvd(G_bb=100.0, eta=100.0, beta=0.5,
                         n=2000, dt=1e-3,
                         omega1=0.1, omega2=10.0,
                         noise=0.01, seed=42):
    """
    FKV-D  :  σ = 𝔾·D^β[γ] + η·dγ/dt
    """
    rng = np.random.default_rng(seed)
    t, strain = make_chirp(n, dt, omega1, omega2)
    xb = _caputo_np(_gd(strain, dt), dt, beta)
    stress = G_bb * xb + eta * _gd(strain, dt)
    return t, strain, _noisy(stress, noise, rng)


def make_synthetic_fkv(V=100.0, G_bb=100.0, alpha=0.7, beta=0.3,
                        n=2000, dt=1e-3,
                        omega1=0.1, omega2=10.0,
                        noise=0.01, seed=42):
    """
    FKV (full)  :  σ = 𝕍·D^α[γ] + 𝔾·D^β[γ],  0 ≤ β ≤ α ≤ 1
    """
    rng = np.random.default_rng(seed)
    t, strain = make_chirp(n, dt, omega1, omega2)
    gd = _gd(strain, dt)
    xa = _caputo_np(gd, dt, alpha)
    xb = _caputo_np(gd, dt, beta)
    stress = V * xa + G_bb * xb
    return t, strain, _noisy(stress, noise, rng)


# ----------------------------------------------------------------
# Model comparison
# ----------------------------------------------------------------

def compare_models(fitted_models: dict, omega=None, criterion="bic"):
    """
    Print an AIC/BIC comparison table and overlay G*(ω) for all models.

    Parameters
    ----------
    fitted_models : dict  { label: fitted SPGP instance }
    omega         : array-like or None
    criterion     : "bic" (default) or "aic" — which criterion ranks
                    the table and selects the returned best model.

    Returns
    -------
    best : SPGP  (lowest ``criterion``)
    """
    import matplotlib.pyplot as plt

    crit = criterion.lower()
    if crit not in ("bic", "aic"):
        raise ValueError("criterion must be 'bic' or 'aic'")

    header = (f"{'Model':<24} {'AIC':>10} {'BIC':>10} {'k':>4}"
              f" {'RMSE [Pa]':>12} {'R²':>8}  Params")
    print("\n" + "=" * 78)
    print(header); print("-" * 78)
    rows = sorted(fitted_models.items(),
                  key=lambda kv: kv[1].metrics_[crit])
    for label, m in rows:
        pstr = "  ".join(
            f"{k}={v:.3f}"
            for k, v in m.gp_model_.phys.named_phys_params().items()
        )
        k_tot = m.metrics_.get("n_params", {}).get("k_total", "-")
        aic = m.metrics_.get("aic", float("nan"))
        print(f"{label:<24} {aic:>10.2f} {m.metrics_['bic']:>10.2f}"
              f" {k_tot:>4}"
              f" {m.metrics_['rmse']:>12.4f}"
              f" {m.metrics_['r2']:>8.4f}  {pstr}")
    print("=" * 78)
    print(f"  ✅  Best model by {crit.upper()}: {rows[0][0]}\n")

    if omega is None:
        omega = np.logspace(-3, 3, 400)

    cmap = plt.get_cmap("tab10")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for i, (label, m) in enumerate(fitted_models.items()):
        col      = cmap(i % 10)
        scalar_pf = {k: v["scalar_mean"] for k, v in m.prefactors_.items()}
        Gp, Gdp  = m.gp_model_.phys.Gstar(omega, scalar_pf)
        ax1.loglog(omega, np.abs(Gp),  color=col, lw=2.0, label=label)
        ax2.loglog(omega, np.abs(Gdp), color=col, lw=2.0, label=label)

    for ax, title in [(ax1, "G′(ω) storage"), (ax2, "G″(ω) loss")]:
        ax.set_xlabel("ω [rad/s]"); ax.set_ylabel("[Pa]")
        ax.set_title(title); ax.legend(fontsize=9, frameon=False)
        ax.grid(True, which="both", ls=":", alpha=0.4)

    fig.suptitle("G*(ω) — model comparison", fontweight="bold")
    fig.tight_layout(); plt.show()

    return rows[0][1]

# ----------------------------------------------------------------
# Stress-controlled synthetic generators  (creep representation)
# ----------------------------------------------------------------

def _frac_integral_np(sig, dt, alpha):
    """Exact product-integration fractional integral I^alpha[sig]."""
    N = len(sig)
    m = np.arange(N)
    w = dt**alpha * ((m + 1)**alpha - m**alpha) / alpha / sc_gamma(alpha)
    return np.convolve(sig, w)[:N]


def make_synthetic_springpot_stress(V=100.0, alpha=0.5,
                                    n=2000, dt=1e-3,
                                    omega1=0.1, omega2=10.0,
                                    noise=0.01, seed=42,
                                    sigma0=10.0):
    """Stress-controlled springpot:  γ = (1/𝕍) I^α[σ]."""
    rng = np.random.default_rng(seed)
    t, wave = make_chirp(n, dt, omega1, omega2)
    stress = sigma0 * wave
    strain = _frac_integral_np(stress, dt, alpha) / V
    return t, _noisy(strain, noise, rng), stress


def make_synthetic_maxwell_stress(Gc=100.0, tau_c=1.0,
                                  n=2000, dt=1e-3,
                                  omega1=0.1, omega2=10.0,
                                  noise=0.01, seed=42,
                                  sigma0=10.0):
    """Stress-controlled Maxwell:  γ = σ/G_c + (1/η) I¹[σ],  η = G_c τ_c."""
    rng = np.random.default_rng(seed)
    t, wave = make_chirp(n, dt, omega1, omega2)
    stress = sigma0 * wave
    eta    = Gc * tau_c
    strain = stress / Gc + np.cumsum(stress) * dt / eta
    return t, _noisy(strain, noise, rng), stress


def make_synthetic_fkvs_stress(V=100.0, G=50.0, alpha=0.8,
                               n=2000, dt=1e-3,
                               omega1=0.1, omega2=10.0,
                               noise=0.01, seed=42,
                               sigma0=10.0):
    """
    Stress-controlled FKV-S:  J(t) = (t^α/𝕍) E_{α,1+α}(-(G/𝕍) t^α),
    γ = ∫ J(t-t') σ̇ dt'  (computed with the ML creep kernel).
    """
    rng = np.random.default_rng(seed)
    t, wave = make_chirp(n, dt, omega1, omega2)
    stress = sigma0 * wave
    s = (np.arange(n) + 0.5) * dt
    J = s**alpha * _ml_np(-(G / V) * s**alpha, alpha, 1.0 + alpha) / V
    sd = np.gradient(stress, dt)
    strain = np.convolve(sd, J)[:n] * dt
    return t, _noisy(strain, noise, rng), stress
