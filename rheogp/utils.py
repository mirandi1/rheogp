"""
rheogp.utils
============
Model comparison helper + synthetic data generators for all 14 models
(the original 8 plus the Fractional Zener family), plus
`make_tts_dataset` -- a synthetic multi-temperature chirp dataset with
a known WLF shift, for validating the joint / master-curve pipeline in
`rheogp.mastercurve`.
"""

import numpy as np
from scipy.special import gamma as sc_gamma

__all__ = [
    "compare_models",
    "integrated_autocorr_time",
    "effective_sample_size",
    "effective_sample_size_segments",
    "make_chirp",
    "make_synthetic_maxwell",
    "make_synthetic_springpot",
    "make_synthetic_fmg",
    "make_synthetic_fml",
    "make_synthetic_fmm",
    "make_synthetic_fkvs",
    "make_synthetic_fkvd",
    "make_synthetic_fkv",
    "make_synthetic_zener",
    "make_synthetic_fzss",
    "make_synthetic_fzls",
    "make_synthetic_fzld",
    "make_synthetic_fzs",
    "make_synthetic_fz",
    "make_tts_dataset",
    "wlf_aT",
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
# Fractional Zener family -- synthetic generators
#
#   Every model = ONE parallel element (spring / dashpot / springpot)
#   + ONE "Maxwell arm" built with `_fmm_np` (which reduces to the
#   classical Maxwell kernel exactly at alpha=1, beta=0 -- see
#   rheogp.kernels module docstring for the same identity used by the
#   torch implementation).  Matches `rheogp.kernels`'s convention that
#   the arm's own characteristic modulus is reported as the "Gc"
#   prefactor, equal to G1 = G_infty - G0 in the paper's notation.
# ----------------------------------------------------------------

def make_synthetic_zener(G0=30.0, G1=70.0, eta=70.0,
                          n=2000, dt=1e-3,
                          omega1=0.1, omega2=10.0,
                          noise=0.01, seed=42):
    """
    Zener (standard linear solid) :  spring(G0) ∥ Maxwell(G1, tau_c)
    tau_c = eta / G1
    """
    rng = np.random.default_rng(seed)
    t, strain = make_chirp(n, dt, omega1, omega2)
    gd = _gd(strain, dt)
    tau_c = eta / G1
    x_arm = _fmm_np(gd, dt, 1.0, 0.0, tau_c)
    stress = G1 * x_arm + G0 * strain
    return t, strain, _noisy(stress, noise, rng)


def make_synthetic_fzss(G0=30.0, V=100.0, G1=100.0, alpha=0.5,
                         n=2000, dt=1e-3,
                         omega1=0.1, omega2=10.0,
                         noise=0.01, seed=42):
    """
    Fractional Zener Solid-S :  spring(G0) ∥ FMG(V, alpha, G1)
    tau_c = (V/G1)^{1/alpha};  the FMG arm's own Gc equals G1 exactly.
    """
    rng = np.random.default_rng(seed)
    t, strain = make_chirp(n, dt, omega1, omega2)
    gd = _gd(strain, dt)
    tau_c = (V / G1) ** (1.0 / alpha)
    x_arm = _fmm_np(gd, dt, alpha, 0.0, tau_c)
    stress = G1 * x_arm + G0 * strain
    return t, strain, _noisy(stress, noise, rng)


def make_synthetic_fzls(G0=30.0, G_bb=100.0, eta=100.0, beta=0.5,
                         n=2000, dt=1e-3,
                         omega1=0.1, omega2=10.0,
                         noise=0.01, seed=42):
    """
    Fractional Zener Liquid-S :  spring(G0) ∥ FML(G_bb, beta, eta)
    tau_c = (eta/G_bb)^{1/(1-beta)},  Gc_arm = eta/tau_c
    """
    rng = np.random.default_rng(seed)
    t, strain = make_chirp(n, dt, omega1, omega2)
    gd = _gd(strain, dt)
    tau_c  = (eta / G_bb) ** (1.0 / (1.0 - beta))
    Gc_arm = eta / tau_c
    x_arm = _fmm_np(gd, dt, 0.9999, beta, tau_c)
    stress = Gc_arm * x_arm + G0 * strain
    return t, strain, _noisy(stress, noise, rng)


def make_synthetic_fzld(eta_p=10.0, G_bb=100.0, eta_s=100.0, beta=0.5,
                         n=2000, dt=1e-3,
                         omega1=0.1, omega2=10.0,
                         noise=0.01, seed=42):
    """
    Fractional Zener Liquid-D :  dashpot(eta_p) ∥ FML(G_bb, beta, eta_s)
    tau_c = (eta_s/G_bb)^{1/(1-beta)},  Gc_arm = eta_s/tau_c
    """
    rng = np.random.default_rng(seed)
    t, strain = make_chirp(n, dt, omega1, omega2)
    gd = _gd(strain, dt)
    tau_c  = (eta_s / G_bb) ** (1.0 / (1.0 - beta))
    Gc_arm = eta_s / tau_c
    x_arm = _fmm_np(gd, dt, 0.9999, beta, tau_c)
    stress = Gc_arm * x_arm + eta_p * gd
    return t, strain, _noisy(stress, noise, rng)


def make_synthetic_fzs(G0=30.0, V=100.0, G_bb=100.0, alpha=0.7, beta=0.3,
                        n=2000, dt=1e-3,
                        omega1=0.1, omega2=10.0,
                        noise=0.01, seed=42):
    """
    Fractional Zener-S :  spring(G0) ∥ FMM(V, alpha, G_bb, beta)
    tau_c = (V/G_bb)^{1/(alpha-beta)},  Gc_arm = V*tau_c^{-alpha}
    """
    rng = np.random.default_rng(seed)
    t, strain = make_chirp(n, dt, omega1, omega2)
    gd = _gd(strain, dt)
    tau_c  = (V / G_bb) ** (1.0 / (alpha - beta))
    Gc_arm = V * tau_c ** (-alpha)
    x_arm = _fmm_np(gd, dt, alpha, beta, tau_c)
    stress = Gc_arm * x_arm + G0 * strain
    return t, strain, _noisy(stress, noise, rng)


def make_synthetic_fz(K=20.0, kappa=0.2, V=100.0, G_bb=100.0,
                       alpha=0.7, beta=0.3,
                       n=2000, dt=1e-3,
                       omega1=0.1, omega2=10.0,
                       noise=0.01, seed=42):
    """
    Fractional Zener (general) :  springpot(K, kappa) ∥ FMM(V, alpha, G_bb, beta)
    tau_c = (V/G_bb)^{1/(alpha-beta)},  Gc_arm = V*tau_c^{-alpha}
    """
    rng = np.random.default_rng(seed)
    t, strain = make_chirp(n, dt, omega1, omega2)
    gd = _gd(strain, dt)
    tau_c  = (V / G_bb) ** (1.0 / (alpha - beta))
    Gc_arm = V * tau_c ** (-alpha)
    x_arm   = _fmm_np(gd, dt, alpha, beta, tau_c)
    x_kappa = _caputo_np(gd, dt, kappa)
    stress = Gc_arm * x_arm + K * x_kappa
    return t, strain, _noisy(stress, noise, rng)


# ----------------------------------------------------------------
# Time-temperature-superposition synthetic dataset
# ----------------------------------------------------------------

def wlf_aT(T, C1, C2, T_ref):
    """WLF horizontal shift factor  log10(aT) = -C1(T-Tref)/(C2+T-Tref)."""
    return 10.0 ** (-C1 * (T - T_ref) / (C2 + (T - T_ref)))


# {model key: (generator, tau_c-generator-kwarg-builder, held-params)}
# each builder maps (held, tau_c) -> full kwargs for the generator, by
# solving that model's own tau_c(...) relation for the ONE parameter
# not held fixed (see the docstrings above).
def _tts_kwargs_maxwell(held, tau_c):
    return dict(Gc=held["Gc"], tau_c=tau_c)


def _tts_kwargs_fmg(held, tau_c):
    return dict(V=held["G1"] * tau_c ** held["alpha"],
               G=held["G1"], alpha=held["alpha"])


def _tts_kwargs_fml(held, tau_c):
    return dict(G_bb=held["G_bb"],
               eta=held["G_bb"] * tau_c ** (1.0 - held["beta"]),
               beta=held["beta"])


def _tts_kwargs_fmm(held, tau_c):
    V = held["G_bb"] * tau_c ** (held["alpha"] - held["beta"])
    return dict(V=V, G_bb=held["G_bb"],
               alpha=held["alpha"], beta=held["beta"])


def _tts_kwargs_zener(held, tau_c):
    return dict(G0=held["G0"], G1=held["G1"], eta=held["G1"] * tau_c)


def _tts_kwargs_fzss(held, tau_c):
    return dict(G0=held["G0"], G1=held["G1"], alpha=held["alpha"],
               V=held["G1"] * tau_c ** held["alpha"])


def _tts_kwargs_fzls(held, tau_c):
    return dict(G0=held["G0"], G_bb=held["G_bb"], beta=held["beta"],
               eta=held["G_bb"] * tau_c ** (1.0 - held["beta"]))


def _tts_kwargs_fzld(held, tau_c):
    return dict(eta_p=held["eta_p"], G_bb=held["G_bb"], beta=held["beta"],
               eta_s=held["G_bb"] * tau_c ** (1.0 - held["beta"]))


def _tts_kwargs_fzs(held, tau_c):
    V = held["G_bb"] * tau_c ** (held["alpha"] - held["beta"])
    return dict(G0=held["G0"], G_bb=held["G_bb"], V=V,
               alpha=held["alpha"], beta=held["beta"])


def _tts_kwargs_fz(held, tau_c):
    V = held["G_bb"] * tau_c ** (held["alpha"] - held["beta"])
    return dict(K=held["K"], kappa=held["kappa"], G_bb=held["G_bb"], V=V,
               alpha=held["alpha"], beta=held["beta"])


_TTS_REGISTRY = {
    "maxwell":  (make_synthetic_maxwell, _tts_kwargs_maxwell,
                {"Gc": 100.0}),
    "fmg":      (make_synthetic_fmg, _tts_kwargs_fmg,
                {"G1": 100.0, "alpha": 0.5}),
    "fml":      (make_synthetic_fml, _tts_kwargs_fml,
                {"G_bb": 100.0, "beta": 0.5}),
    "fmm":      (make_synthetic_fmm, _tts_kwargs_fmm,
                {"G_bb": 100.0, "alpha": 0.7, "beta": 0.3}),
    "zener":    (make_synthetic_zener, _tts_kwargs_zener,
                {"G0": 30.0, "G1": 70.0}),
    "fzss":     (make_synthetic_fzss, _tts_kwargs_fzss,
                {"G0": 30.0, "G1": 100.0, "alpha": 0.5}),
    "fzls":     (make_synthetic_fzls, _tts_kwargs_fzls,
                {"G0": 30.0, "G_bb": 100.0, "beta": 0.5}),
    "fzld":     (make_synthetic_fzld, _tts_kwargs_fzld,
                {"eta_p": 10.0, "G_bb": 100.0, "beta": 0.5}),
    "fzs":      (make_synthetic_fzs, _tts_kwargs_fzs,
                {"G0": 30.0, "G_bb": 100.0, "alpha": 0.7, "beta": 0.3}),
    "fz":       (make_synthetic_fz, _tts_kwargs_fz,
                {"K": 20.0, "kappa": 0.2, "G_bb": 100.0,
                 "alpha": 0.7, "beta": 0.3}),
}


def make_tts_dataset(model="fmm", temps=(50, 60, 70, 80, 90, 100),
                      T_ref=None, C1=8.0, C2=130.0, tau_c_ref=1.0,
                      held_params=None,
                      n=1500, dt=2e-3, omega1=0.1, omega2=10.0,
                      noise=0.01, seed=42):
    """
    Synthetic multi-temperature chirp dataset for validating the
    joint / master-curve (time-temperature-superposition) pipeline in
    `rheogp.mastercurve`.

    ONE set of shape parameters and prefactors (`held_params`, or the
    model's own defaults) is shared across every temperature; only
    tau_c is shifted, following the WLF law

        log10 aT(T) = -C1 (T - T_ref) / (C2 + T - T_ref)
        tau_c(T)    = aT(T) * tau_c_ref

    -- exactly what `MultiRecordSPGP` assumes when it recovers a_T as
    a per-record fitted parameter.  Any model with a `tau_c` (i.e. any
    Maxwell-arm-containing model: "maxwell", "fmg", "fml", "fmm", or
    the Zener family "zener"/"fzss"/"fzls"/"fzld"/"fzs"/"fz") is
    supported; Springpot and the FKV family have no strain-mode tau_c
    to shift and are intentionally not included here.

    Returns
    -------
    truth   : dict -- {"model", "T_ref", "C1", "C2", "tau_c_ref",
              **held_params, "temperatures": [...], "aT": [...]}
    records : list[(t, strain, stress)], one per temperature, in the
              same order as truth["temperatures"].
    """
    key = model.lower().replace(" ", "").replace("-", "").replace("_", "")
    if key not in _TTS_REGISTRY:
        raise ValueError(f"Unknown/unsupported model '{model}' for "
                         f"make_tts_dataset. Available: "
                         f"{sorted(_TTS_REGISTRY)}")
    gen_fn, kwargs_fn, defaults = _TTS_REGISTRY[key]
    held = dict(defaults)
    if held_params:
        held.update(held_params)

    temps = sorted(float(T) for T in temps)
    T_ref = float(T_ref) if T_ref is not None else temps[len(temps) // 2]

    records, aT_list = [], []
    for i, T in enumerate(temps):
        aT = float(wlf_aT(T, C1, C2, T_ref))
        tau_c = aT * tau_c_ref
        kw = kwargs_fn(held, tau_c)
        t, strain, stress = gen_fn(
            n=n, dt=dt, omega1=omega1, omega2=omega2,
            noise=noise, seed=seed + i, **kw)
        records.append((t, strain, stress))
        aT_list.append(aT)

    truth = dict(model=model, T_ref=T_ref, C1=C1, C2=C2,
                tau_c_ref=tau_c_ref, temperatures=temps, aT=aT_list,
                **held)
    return truth, records


# ----------------------------------------------------------------
# Effective sample size from residual autocorrelation
# ----------------------------------------------------------------
#
# AIC/BIC (see `_train_loop` in rheogp.model) score a fit with the RAW
# number of time samples N. For a chirp sampled far above its own
# relaxation bandwidth, adjacent residuals are almost the same number
# repeated -- N vastly overstates how much INDEPENDENT evidence the fit
# has, so the k*ln(N) penalty in BIC is too small. The fix is to replace
# N by N_eff = N / tau_int, where tau_int is the integrated
# autocorrelation time of the residuals -- estimated here by Geyer's
# (1992) initial-positive-sequence rule, matching the convention already
# used in `demos/demo_cnf.ipynb`'s `residual_neff` module (rather than
# Sokal's windowed estimator, a different but equally standard choice):
# accumulate consecutive lag PAIRS rho[k]+rho[k+1] while their running
# sum stays positive. The pairwise truncation is what keeps this
# well-defined for oscillatory residuals -- a chirp's residuals ring at
# the excitation frequency, so the raw ACF does not have a single clean
# zero-crossing a simpler rule could cut at.

def _acf(x, nlags=None):
    """Normalised autocorrelation function of a 1-D real series via FFT
    (zero-padded to avoid circular wrap-around), rho[0] == 1."""
    x = np.asarray(x, float)
    x = x - x.mean()
    n = x.size
    if n < 2:
        return np.array([1.0])
    if nlags is None:
        nlags = min(n - 1, 5000)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(x, nfft)
    r = np.fft.irfft(f * np.conjugate(f), nfft)[:nlags + 1].real
    return r / r[0] if r[0] > 0 else np.zeros_like(r)


def integrated_autocorr_time(x, nlags=5000):
    """
    Integrated autocorrelation time tau_int = 1 + 2*sum_k rho_k of a
    1-D series, via Geyer's initial-positive-sequence rule: accumulate
    consecutive lag PAIRS while their sum stays positive (see module
    comment above). Matches `demos/demo_cnf.ipynb`'s `residual_neff.
    tau_int`.

    Returns tau_int >= 1.0 (>= 1 always: an i.i.d. series is the floor,
    never "more independent than independent").
    """
    x = np.asarray(x, float)
    n = x.size
    if n < 4:
        return 1.0
    rho = _acf(x, nlags=min(nlags, n - 1))
    s, k = 0.0, 1
    while k + 1 < rho.size:
        pair = rho[k] + rho[k + 1]
        if pair <= 0:
            break
        s += pair
        k += 2
    return float(max(1.0 + 2.0 * s, 1.0))


def effective_sample_size(residuals, nlags=5000):
    """
    (n_eff, tau_int) for one contiguous, autocorrelated residual series
    -- n_eff = len(residuals) / tau_int. See `integrated_autocorr_time`
    for the estimator.
    """
    residuals = np.asarray(residuals, float)
    tau = integrated_autocorr_time(residuals, nlags=nlags)
    n_eff = len(residuals) / tau
    return float(max(n_eff, 1.0)), tau


def effective_sample_size_segments(segments, nlags=5000):
    """
    Effective sample size across several INDEPENDENT residual segments
    (e.g. one per record/chirp in a joint `MultiRecordSPGP` fit).

    Autocorrelation only means something WITHIN one segment -- the
    residual at the end of one chirp tells you nothing about the start
    of the next, unrelated experiment -- so tau_int/n_eff is computed
    PER SEGMENT and summed, never on the naively concatenated residual
    vector (which would read the segment boundary as if it were memory
    within a single process, and mix together segments whose noise
    amplitude/spectral content generally differ).

    Returns
    -------
    n_eff_total : float -- sum of each segment's n_eff
    n_raw_total : int   -- sum of each segment's raw length
    per_segment : list[(n_eff_i, tau_int_i, n_i)], same order as `segments`
    """
    per_segment = []
    n_eff_total = 0.0
    n_raw_total = 0
    for seg in segments:
        seg = np.asarray(seg, float)
        n_i = len(seg)
        n_eff_i, tau_i = effective_sample_size(seg, nlags=nlags)
        per_segment.append((n_eff_i, tau_i, n_i))
        n_eff_total += n_eff_i
        n_raw_total += n_i
    return n_eff_total, n_raw_total, per_segment


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
