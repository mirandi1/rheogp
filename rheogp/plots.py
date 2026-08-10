"""
rheogp.plots
============
All matplotlib visualisation helpers.
Accepts optional `ax` so users can embed into their own layouts.
"""

import numpy as np
import matplotlib.pyplot as plt

_C = plt.get_cmap("tab10").colors

__all__ = [
    "plot_fit", "plot_prefactors", "plot_Gstar", "plot_Gstar_result",
    "plot_convergence", "plot_kernel",
]

# ── pretty symbols for prefactor names ──────────────────────
_SYM = {
    "V":       r"$\mathbb{V}(t)$",
    "G_bb":    r"$\mathbb{G}(t)$",
    "G":       r"$G(t)$",
    "eta":     r"$\eta(t)$",
    "Gc":      r"$G_c(t)$",
    "invV":    r"$1/\mathbb{V}(t)$",
    "invG_bb": r"$1/\mathbb{G}(t)$",
    "invG":    r"$1/G(t)$",
    "invEta":  r"$1/\eta(t)$",
    "invGc":   r"$1/G_c(t)$",
}
_UNIT = {
    "V":       "Pa·s^α",
    "G_bb":    "Pa·s^β",
    "G":       "Pa",
    "eta":     "Pa·s",
    "Gc":      "Pa",
    "invV":    "1/(Pa·s^α)",
    "invG_bb": "1/(Pa·s^β)",
    "invG":    "1/Pa",
    "invEta":  "1/(Pa·s)",
    "invGc":   "1/Pa",
}


def plot_fit(t, sigma_obs, sigma_pred, model_name, metrics,
             ax=None, figsize=(9, 4), ylabel="σ [Pa]"):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.plot(t, sigma_obs,  color="black", lw=0.8, alpha=0.5, label="data")
    ax.plot(t, sigma_pred, color=_C[0],   lw=1.5,
            label=f"{model_name}  R²={metrics['r2']:.4f}")
    ax.set_xlabel("t [s]"); ax.set_ylabel(ylabel)
    ax.set_title(
        f"{model_name}  |  "
        f"RMSE={metrics['rmse']:.4f} Pa   "
        f"R²={metrics['r2']:.4f}   BIC={metrics['bic']:.1f}"
    )
    ax.legend(fontsize=9, frameon=False)
    if fig is not None:
        fig.tight_layout()
    return ax


def plot_prefactors(t, prefactors, model_name, ax=None, figsize=None):
    names = list(prefactors.keys())
    n     = len(names)
    if figsize is None:
        figsize = (5.5 * n, 3.8)

    if ax is None:
        fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
        axes = axes[0]
    else:
        axes = ax if hasattr(ax, "__len__") else [ax]
        fig  = None

    for i, name in enumerate(names):
        pf  = prefactors[name]
        col = _C[i % 10]
        sym = _SYM.get(name, name)
        unit = _UNIT.get(name, "?")

        axi = axes[i]
        axi.plot(t, pf["mean"],  color=col, lw=1.5, label=sym)
        axi.fill_between(t, pf["lower"], pf["upper"],
                         color=col, alpha=0.20, label="95% CI")
        axi.axhline(pf["scalar_mean"], color="black", lw=0.9, ls="--",
                    label=f"mean={pf['scalar_mean']:.3f}")
        axi.set_xlabel("t [s]"); axi.set_ylabel(f"[{unit}]")
        axi.set_title(rf"$\partial\sigma/\partial x$  →  {sym}")
        axi.legend(fontsize=8, frameon=False)

    if fig is not None:
        fig.suptitle(f"{model_name} — prefactors", fontweight="bold")
        fig.tight_layout()
    return axes


def plot_Gstar(omega, Gp, Gdp, phys_params, prefactors,
               model_name, ax=None, figsize=(12, 4)):
    fig = None
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
    else:
        axes = ax

    ax1, ax2 = axes

    ax1.loglog(omega, np.abs(Gp),  color=_C[0], lw=2.0, label="G′ storage")
    ax1.loglog(omega, np.abs(Gdp), color=_C[1], lw=2.0, label="G″ loss")

    # power-law slope guides at α and β
    mid = len(omega) // 2
    ref = np.abs(Gp[mid]) if np.abs(Gp[mid]) > 0 else 1.0
    for exp, col, lbl in _slope_guides(phys_params):
        ax1.loglog(omega, ref * (omega / omega[mid]) ** exp,
                   color=col, lw=0.9, ls=":", alpha=0.6,
                   label=f"ω^{exp:.2f}")

    if "tau_c" in phys_params and phys_params["tau_c"] > 0:
        wc = 1.0 / phys_params["tau_c"]
        ax1.axvline(wc, color="gray", lw=1.0, ls="--",
                    label=f"ω*=1/τ_c={wc:.2f}")

    ax1.set_xlabel("ω [rad/s]"); ax1.set_ylabel("[Pa]")
    ax1.set_title(f"{model_name}  G*(ω)")
    ax1.legend(fontsize=8, frameon=False)
    ax1.grid(True, which="both", ls=":", alpha=0.4)

    delta = np.degrees(np.arctan2(np.abs(Gdp), np.abs(Gp)))
    ax2.semilogx(omega, delta, color=_C[2], lw=2.0, label="δ(ω)")

    for exp, col, _ in _slope_guides(phys_params):
        ax2.axhline(exp * 90, color=col, lw=0.9, ls="--",
                    label=f"{exp:.2f}·90°={exp*90:.1f}°")
    if "tau_c" in phys_params and phys_params["tau_c"] > 0:
        ax2.axvline(1.0 / phys_params["tau_c"], color="gray",
                    lw=1.0, ls="--")
    ax2.axhline(45, color="black", lw=0.7, ls=":")
    ax2.set_xlabel("ω [rad/s]"); ax2.set_ylabel("δ [°]")
    ax2.set_title("Phase angle δ(ω)")
    ax2.set_ylim(0, 90)
    ax2.legend(fontsize=8, frameon=False)
    ax2.grid(True, which="both", ls=":", alpha=0.4)

    pstr = "  ".join(f"{k}={v:.3f}" for k, v in phys_params.items())
    gstr = "  ".join(
        f"{_SYM.get(k, k)}={v:.3f}" for k, v in prefactors.items()
    )
    if fig is not None:
        fig.suptitle(f"{model_name}  |  {pstr}\n{gstr}",
                     fontweight="bold", fontsize=9)
        fig.tight_layout()
    return axes


def plot_convergence(loss_hist, param_hist, model_name, figsize=(12, 4)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].plot(loss_hist, lw=1.0, color=_C[0], alpha=0.7)
    axes[0].set_xlabel("step"); axes[0].set_ylabel("−ELBO")
    axes[0].set_title(f"{model_name} — training loss")

    ax2  = axes[1]
    ax2r = ax2.twinx()
    tau_h = param_hist.pop("tau_c", None)

    for i, (k, v) in enumerate(param_hist.items()):
        ax2.plot(v, lw=1.5, color=_C[i % 10], label=f"${k}$")
        ax2.axhline(v[-1], color=_C[i % 10], lw=0.7, ls="--")

    if tau_h is not None:
        ax2r.plot(tau_h, lw=1.5, ls="--", color=_C[4], label=r"$\tau_c$")
        ax2r.set_ylabel(r"$\tau_c$ [s]", color=_C[4])
        param_hist["tau_c"] = tau_h

    ax2.set_xlabel("step"); ax2.set_ylabel("exponent")
    ax2.set_title("Parameter convergence")
    lines1, labs1 = ax2.get_legend_handles_labels()
    if tau_h is not None:
        lines2, labs2 = ax2r.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labs1 + labs2, fontsize=8, frameon=False)
    else:
        ax2.legend(fontsize=8, frameon=False)

    fig.suptitle(f"{model_name} — convergence", fontweight="bold")
    fig.tight_layout()
    return axes


def plot_kernel(phys, dt, model_name, ax=None, figsize=(6, 3.5)):
    """Plot the relaxation kernel K(s)."""
    import torch
    from .kernels import (
        _fmm_kernel,
        MaxwellKernel, SpringpotKernel,
        FractionalMaxwellGelKernel, FractionalMaxwellLiquidKernel,
        FractionalMaxwellKernel,
        FKVSKernel, FKVDKernel, FractionalKelvinVoigtKernel,
    )
    def _maxwell_kernel(s, tau_c):
        return torch.exp(-s / tau_c)
    def _caputo_kernel(s, alpha):
        return s ** (-alpha) / torch.exp(torch.lgamma(1.0 - alpha))

    # choose a sensible lag range
    tau_c = phys.tau_c.item() if hasattr(phys, "tau_c") else 1.0
    s_np  = np.linspace(dt, 6.0 * tau_c, 600)
    s_t   = torch.tensor(s_np, dtype=torch.float32, device=next(phys.parameters()).device)

    with torch.no_grad():
        if isinstance(phys, MaxwellKernel):
            K_np = _maxwell_kernel(s_t, phys.tau_c).cpu().numpy()
            xlabel = r"$s$ [s]"
        elif isinstance(phys, SpringpotKernel):
            K_np = _caputo_kernel(s_t, phys.alpha).cpu().numpy()
            xlabel = r"$s$ [s]"
        elif isinstance(phys, FractionalMaxwellGelKernel):
            beta = torch.zeros(1, device=s_t.device).squeeze()
            K_np = _fmm_kernel(s_t, phys.alpha, beta, phys.tau_c).cpu().numpy()
            xlabel = r"$s\,/\,\tau_c$"
            s_np   = s_np / tau_c
        else:
            # Full FMM or FML
            alpha = phys.alpha if hasattr(phys, "alpha") else torch.tensor(0.9999)
            beta  = phys.beta  if hasattr(phys, "beta")  else torch.tensor(0.0)
            K_np  = _fmm_kernel(s_t, alpha, beta, phys.tau_c).cpu().numpy()
            xlabel = r"$s\,/\,\tau_c$"
            s_np   = s_np / tau_c

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.plot(s_np, K_np, color=_C[0], lw=2.0)
    if hasattr(phys, "tau_c"):
        ax.axvline(1.0, color="gray", lw=1.0, ls="--", label=r"$s=\tau_c$")
        ax.legend(fontsize=8, frameon=False)

    p = {k: f"{v:.3f}" for k, v in phys.named_phys_params().items()}
    ax.set_xlabel(xlabel); ax.set_ylabel(r"$K(s)$")
    ax.set_title(f"{model_name} — relaxation kernel\n{p}")
    if fig is not None:
        fig.tight_layout()
    return ax


def _slope_guides(phys_params):
    """Return (exponent, colour, label) pairs for log-log slope lines."""
    pairs = []
    if "alpha" in phys_params:
        pairs.append((phys_params["alpha"], _C[0], "α"))
    if "beta"  in phys_params:
        pairs.append((phys_params["beta"],  _C[1], "β"))
    return pairs

def plot_Gstar_result(res, ax=None, add_fft=False, fft_kwargs=None,
                      figsize=(12, 4)):
    """G*(ω) figure driven entirely by a FitResult."""
    axes = plot_Gstar(res.omega, res.Gp, res.Gdp,
                      res.phys_params, res.scalar_prefactors(),
                      res.model, ax=ax, figsize=figsize)
    if add_fft and res.fft_omega is not None:
        m = ((res.fft_omega >= res.omega.min())
             & (res.fft_omega <= res.omega.max()))
        kw = dict(s=20, alpha=0.9)
        if fft_kwargs:
            kw.update(fft_kwargs)
        ax1 = axes[0] if hasattr(axes, "__len__") else axes
        ax1.scatter(res.fft_omega[m], np.abs(res.fft_Gp[m]), marker="o",
                    facecolors="none", edgecolors="#1f77b4",
                    label="FFT G′", **kw)
        ax1.scatter(res.fft_omega[m], np.abs(res.fft_Gdp[m]), marker="s",
                    facecolors="none", edgecolors="#ff7f0e",
                    label="FFT G″", **kw)
        ax1.legend(fontsize=8, frameon=False)
    return axes
