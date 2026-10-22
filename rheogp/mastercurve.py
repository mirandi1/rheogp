"""
rheogp.mastercurve
===================
Time-temperature-superposition (TTS) master curves from several chirps
(or any transient records) taken at different temperatures, fitted
jointly with ONE shared fractional constitutive law plus a per-record
horizontal shift factor a_T(T).

This module is the "paper-ready" entry point for that workflow. It is
a thin, tested wrapper around `rheogp.rheogp_multi.MultiRecordSPGP`
(which does the actual joint fitting) that adds:

  * `MasterCurve`               one object: fit, WLF-regress the
                                 recovered shift factors, and pull out
                                 master-curve arrays -- both the raw
                                 FFT of the measured chirps and the
                                 model's own G'(ω)/G''(ω), the latter
                                 with a closed-form credible band
                                 (propagated from the GP posterior
                                 prefactor samples through the
                                 analytic Gstar formula -- no repeated
                                 FFT / Monte-Carlo resampling needed).
  * `compare_master_curve_models`  fit several candidate constitutive
                                 laws to the SAME set of chirps and
                                 rank them by AIC/BIC, exactly as
                                 `rheogp.utils.compare_models` does for
                                 a single record -- because
                                 `MultiRecordSPGP` is itself an `SPGP`
                                 subclass with the same `.metrics_`,
                                 `compare_models` already works
                                 unmodified on a dict of fitted
                                 `MultiRecordSPGP` instances; this is
                                 just a convenience that fits the
                                 candidates for you.

Only a HORIZONTAL shift a_T is modelled here
(τc(T) = a_T(T)·τ_ref); a vertical shift b_T would rescale the
prefactor per record and -- because prefactors are recovered as GP
sensitivities rather than directly optimised -- needs folding into the
feature construction itself.  That is a deliberate, documented scope
decision (see `rheogp.rheogp_multi`), not an oversight.

Example
-------
>>> from rheogp.mastercurve import MasterCurve, compare_master_curve_models
>>> mc = MasterCurve(records, temperatures, T_ref=25.0)
>>> mc.fit(model="FractionalZenerS", kernel="linear", n_epochs=4000)
>>> mc.fit_wlf()
>>> print(mc.summary())
>>> omega, Gp, Gdp, Gp_ci, Gdp_ci = mc.model_curve()
>>>
>>> best, table = compare_master_curve_models(
...     records, temperatures,
...     ["FractionalMaxwell", "FractionalZenerS", "FractionalZener"],
...     T_ref=25.0)
"""

import numpy as np

from .rheogp_multi import MultiRecordSPGP
from .utils import wlf_aT, compare_models

__all__ = ["MasterCurve", "compare_master_curve_models",
          "wlf_aT", "fft_transfer"]


# ----------------------------------------------------------------
# FFT master-curve points from the RAW (measured) signals
# ----------------------------------------------------------------

def fft_transfer(t, u, y, min_snr_db=30.0):
    """
    (omega, Y*(ω) = FFT(y)/FFT(u)), gating out frequency bins where the
    DRIVING signal `u` (e.g. strain) is weakly excited.

    Used only for visualising/exporting the raw measured chirps on a
    reduced-frequency axis -- the joint fit itself is entirely
    time-domain and does not depend on this.

    Parameters
    ----------
    t, u, y    : (N,) arrays, uniform dt.  u drives, y responds.
    min_snr_db : float -- bins more than this many dB below the peak
                 of |FFT(u)| are dropped.

    Returns
    -------
    omega : (M,) rad/s,  Ystar : (M,) complex
    """
    t = np.asarray(t, float); u = np.asarray(u, float); y = np.asarray(y, float)
    if len(t) < 4:
        return np.array([]), np.array([], dtype=complex)
    dt = float(np.median(np.diff(t)))
    Uw = np.fft.rfft(u)
    Yw = np.fft.rfft(y)
    f = np.fft.rfftfreq(len(t), dt)
    omega = 2 * np.pi * f
    magU = np.abs(Uw)
    eps = np.finfo(float).eps
    magU_db = 20 * np.log10(np.maximum(magU / (magU.max() + eps), eps))
    keep = (f > 0) & (magU_db > (magU_db.max() - min_snr_db))
    return omega[keep], (Yw[keep] / Uw[keep])


def _logspace_bins(x, n_bins=60):
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size == 0:
        return np.array([])
    xmin, xmax = np.min(x), np.max(x)
    if xmin == xmax:
        return np.array([xmin, xmax])
    return np.logspace(np.log10(xmin), np.log10(xmax), n_bins + 1)


def _posterior_scalar_draws(prefactor_info):
    """
    One scalar draw per GP posterior sample from a prefactor's
    time-resolved sensitivity samples (S, N): the mean, over time, of
    the finite & positive entries in that draw (falling back to the
    plain finite mean if a draw has no positive entries).  Used to
    propagate prefactor uncertainty through the analytic Gstar(ω)
    formula without any FFT / signal resampling.
    """
    samples = np.asarray(prefactor_info["samples"])
    draws = np.empty(samples.shape[0])
    for s in range(samples.shape[0]):
        row = samples[s]
        finite = row[np.isfinite(row)]
        pos = finite[finite > 0]
        if pos.size:
            draws[s] = pos.mean()
        elif finite.size:
            draws[s] = finite.mean()
        else:
            draws[s] = np.nan
    return draws


# ================================================================
# MasterCurve
# ================================================================

class MasterCurve:
    """
    One shared constitutive law jointly fitted to chirps recorded at
    several temperatures, with the reduced-frequency master curve and
    the WLF shift-factor regression built in.

    Parameters
    ----------
    records      : list[(t, strain, stress)], one per temperature
    temperatures : list[float], same length and order as `records`
    i_ref, T_ref : choose ONE to pick the reference record; default is
                   the median temperature.  `i_ref` indexes into
                   `records`/`temperatures` as given (NOT after
                   sorting -- records are re-sorted by temperature
                   internally, see `.temperatures`/`.records`).
    """

    def __init__(self, records, temperatures, i_ref=None, T_ref=None):
        records = list(records)
        temperatures = [float(T) for T in temperatures]
        if len(records) != len(temperatures):
            raise ValueError("records and temperatures must have the "
                             "same length.")
        if len(records) < 2:
            raise ValueError("MasterCurve needs at least 2 records "
                             "(chirps at 2+ temperatures).")
        if i_ref is not None and T_ref is not None:
            raise ValueError("Pass at most one of i_ref, T_ref.")

        order = np.argsort(temperatures)
        self.temperatures = [temperatures[i] for i in order]
        self.records = [records[i] for i in order]

        if T_ref is not None:
            self.i_ref = int(np.argmin(np.abs(
                np.array(self.temperatures) - T_ref)))
        elif i_ref is not None:
            # i_ref given against the ORIGINAL (unsorted) order
            orig_T = temperatures[i_ref]
            self.i_ref = int(np.argmin(np.abs(
                np.array(self.temperatures) - orig_T)))
        else:
            self.i_ref = len(self.temperatures) // 2
        self.T_ref = self.temperatures[self.i_ref]

        self.model_ = None      # fitted MultiRecordSPGP
        self.aT_ = None
        self.wlf_ = None        # dict(C1, C2, cov) once fit_wlf() is called

    # ------------------------------------------------------------
    def fit(self, model="FractionalMaxwell", **spgp_kwargs):
        """Joint fit across all records. Returns self."""
        spgp_kwargs.setdefault("verbose", False)
        mj = MultiRecordSPGP(self.records, i_ref=self.i_ref, model=model,
                             temps=self.temperatures, **spgp_kwargs)
        mj.fit_all()
        self.model_ = mj
        self.aT_ = mj.shift_factors()
        self.wlf_ = None
        return self

    def _check_fitted(self):
        if self.model_ is None:
            raise RuntimeError("Call .fit(model=...) before using this "
                              "method.")

    @property
    def metrics_(self):
        self._check_fitted()
        return self.model_.metrics_

    @property
    def model_name(self):
        self._check_fitted()
        return self.model_.model_name

    # ------------------------------------------------------------
    def fit_wlf(self, p0=(8.0, 130.0)):
        """
        WLF-regress the recovered shift factors:
        log10(aT) = -C1(T-T_ref) / (C2+T-T_ref).  Stores and returns
        `self.wlf_ = {"C1", "C2", "C1_std", "C2_std", "cov"}`.
        """
        self._check_fitted()
        from scipy.optimize import curve_fit

        if len(self.temperatures) < 3:
            raise RuntimeError("fit_wlf() needs at least 3 temperatures "
                              "to fit 2 WLF constants.")
        Tv = np.asarray(self.temperatures, float)
        aT = np.asarray(self.aT_, float)
        if not np.all(aT > 0):
            raise RuntimeError("fit_wlf(): non-positive shift factor(s) "
                              "recovered -- inspect .aT_ before "
                              "trusting a WLF fit.")

        def _wlf(TT, C1, C2):
            return -C1 * (TT - self.T_ref) / (C2 + TT - self.T_ref)

        (C1, C2), cov = curve_fit(_wlf, Tv, np.log10(aT), p0=p0,
                                  maxfev=40000)
        sd = np.sqrt(np.diag(cov))
        self.wlf_ = dict(C1=float(C1), C2=float(C2),
                         C1_std=float(sd[0]), C2_std=float(sd[1]), cov=cov)
        return self.wlf_

    def predict_aT(self, T):
        """WLF-predicted shift factor(s) at temperature(s) `T`, using
        the constants from `fit_wlf()`. This assumes WLF specifically
        -- if your material follows a different law (Arrhenius, VFT,
        ...) or you don't want to commit to one, use
        `predict_at_temperature(..., shift_law=<your callable>)`
        instead, which never assumes any particular functional form."""
        if self.wlf_ is None:
            raise RuntimeError("Call .fit_wlf() first.")
        return wlf_aT(np.asarray(T, float), self.wlf_["C1"],
                      self.wlf_["C2"], self.T_ref)

    # ------------------------------------------------------------ predict
    def predict_at_shift(self, records_with_aT):
        """Predict new records given their shift factors directly --
        see `MultiRecordSPGP.predict_at_shift`. No assumption about
        temperature dependence at all."""
        self._check_fitted()
        return self.model_.predict_at_shift(records_with_aT)

    def predict_at_temperature(self, records_with_T, shift_law=None):
        """Predict new records given their temperatures -- see
        `MultiRecordSPGP.predict_at_temperature`. Exact/interior
        temperatures use the fitted a_T's or interpolation between them
        (no functional form imposed); temperatures outside the fitted
        range raise unless you pass `shift_law=<callable T -> a_T>`
        (e.g. `lambda T: self.predict_aT(T)` if you explicitly want
        this MasterCurve's own WLF fit, or any other law you trust)."""
        self._check_fitted()
        return self.model_.predict_at_temperature(records_with_T,
                                                   shift_law=shift_law)

    # ------------------------------------------------------------
    def reduced_data(self, min_snr_db=30.0):
        """
        FFT master-curve points from the MEASURED chirps, reduced to
        the reference temperature: ω_ref = a_T(T)·ω.

        Returns
        -------
        omega_ref : (M,) concatenated across all records
        Gstar     : (M,) complex  (= FFT(stress)/FFT(strain) per bin)
        record_index : (M,) int -- which record each point came from
        """
        self._check_fitted()
        omegas, Gs, idx = [], [], []
        for i, (t, strain, stress) in enumerate(self.records):
            omega, G = fft_transfer(t, strain, stress, min_snr_db=min_snr_db)
            if omega.size == 0:
                continue
            omegas.append(omega * self.aT_[i])
            Gs.append(G)
            idx.append(np.full(omega.shape, i))
        if not omegas:
            return (np.array([]), np.array([], dtype=complex),
                   np.array([], dtype=int))
        return (np.concatenate(omegas), np.concatenate(Gs),
               np.concatenate(idx))

    def model_curve(self, omega=None, n_omega=400, credible_interval=True):
        """
        Model G'(ω_ref), G''(ω_ref) on the reference-temperature
        reduced-frequency axis -- purely analytic (the fitted shape
        parameters + prefactor posterior samples through Gstar()), no
        FFT or signal resampling.

        Returns
        -------
        omega : (n_omega,)
        Gp, Gdp : (n_omega,) posterior-mean model curves
        Gp_ci, Gdp_ci : (2, n_omega) [2.5%, 97.5%] bands, or None if
                        `credible_interval=False`
        """
        self._check_fitted()
        if omega is None:
            omega = np.logspace(np.log10(self.model_.omega_min),
                                np.log10(self.model_.omega_max), n_omega)
        phys = self.model_.gp_model_.phys
        scalar_pf = {k: v["scalar_mean"] for k, v in
                    self.model_.prefactors_.items()}
        Gp, Gdp = phys.Gstar(omega, scalar_pf)

        if not credible_interval:
            return omega, Gp, Gdp, None, None

        names = list(self.model_.prefactors_)
        draws = {n: _posterior_scalar_draws(self.model_.prefactors_[n])
                for n in names}
        S = len(next(iter(draws.values())))
        Gp_s = np.empty((S, len(omega))); Gdp_s = np.empty((S, len(omega)))
        for s in range(S):
            pf_s = {n: draws[n][s] for n in names}
            if not all(np.isfinite(v) and v > 0 for v in pf_s.values()):
                Gp_s[s], Gdp_s[s] = np.nan, np.nan
                continue
            Gp_s[s], Gdp_s[s] = phys.Gstar(omega, pf_s)

        Gp_ci = np.nanquantile(Gp_s, [0.025, 0.975], axis=0)
        Gdp_ci = np.nanquantile(Gdp_s, [0.025, 0.975], axis=0)
        return omega, Gp, Gdp, Gp_ci, Gdp_ci

    # ------------------------------------------------------------
    def summary(self):
        self._check_fitted()
        p = self.model_.gp_model_.phys.named_phys_params()
        pf = self.model_.prefactors_
        lines = [
            "=" * 62,
            f"  MasterCurve  —  {self.model_name}  "
            f"({len(self.records)} records, T_ref = {self.T_ref:g})",
            "=" * 62,
            "  shared shape parameters",
        ]
        for k, v in p.items():
            lines.append(f"    {k:<10} = {v:.6g}")
        lines.append("  shared prefactors")
        for k, v in pf.items():
            lines.append(f"    {k:<10} = {v['scalar_mean']:.6g}  "
                        f"[{v['scalar_lower']:.6g}, {v['scalar_upper']:.6g}]")
        m = self.metrics_
        lines += [
            "",
            f"  RMSE = {m['rmse']:.6g}   R2 = {m['r2']:.6f}   "
            f"AIC = {m['aic']:.1f}   BIC = {m['bic']:.1f}   "
            f"k = {m['n_params']['k_total']}",
            "",
            "  shift factors a_T (aT[i_ref] = 1 by construction)",
        ]
        for T, aT in zip(self.temperatures, self.aT_):
            lines.append(f"    T = {T:>8.2f}    a_T = {aT:.6g}")
        if self.wlf_ is not None:
            lines += [
                "",
                f"  WLF fit:  C1 = {self.wlf_['C1']:.4f} +/- "
                f"{self.wlf_['C1_std']:.4f}    "
                f"C2 = {self.wlf_['C2']:.2f} +/- "
                f"{self.wlf_['C2_std']:.2f} K",
            ]
        lines.append("=" * 62)
        return "\n".join(lines)


# ================================================================
# Model comparison across the SAME set of chirps
# ================================================================

def compare_master_curve_models(records, temperatures, model_names,
                                i_ref=None, T_ref=None, criterion="bic",
                                **spgp_kwargs):
    """
    Fit several candidate constitutive laws JOINTLY to the same set of
    chirps (one `MasterCurve` per candidate, all sharing the same
    reference record) and rank them by AIC/BIC -- the master-curve
    analogue of `rheogp.utils.compare_models`.

    Parameters
    ----------
    records, temperatures, i_ref, T_ref : see `MasterCurve`
    model_names : list[str] -- candidate model names (e.g.
        ["FractionalMaxwell", "FractionalZenerS", "FractionalZener"])
    criterion   : "bic" (default) or "aic"
    **spgp_kwargs : forwarded to `MasterCurve.fit()` /
        `MultiRecordSPGP` (e.g. kernel="linear", n_epochs=4000)

    Returns
    -------
    best  : the best-ranked MasterCurve (lowest `criterion`)
    curves : {model_name: MasterCurve}, all fitted -- prints the same
             AIC/BIC/R²/params comparison table as
             `rheogp.utils.compare_models`
    """
    curves, fitted = {}, {}
    for name in model_names:
        mc = MasterCurve(records, temperatures, i_ref=i_ref, T_ref=T_ref)
        mc.fit(model=name, **spgp_kwargs)
        curves[name] = mc
        fitted[name] = mc.model_        # SPGP-compatible -> reuse as-is

    best_model = compare_models(fitted, criterion=criterion)
    best_name = next(name for name, m in fitted.items() if m is best_model)
    return curves[best_name], curves
