"""
rheogp.multi -- joint inference across several records
======================================================
One constitutive law fitted simultaneously to a set of records that differ
only by a horizontal shift of the relaxation spectrum.

    shared   : the memory-shape parameters (alpha, beta) and the prefactors
    per record : a shift factor a_T(i), so that tau_c(i) = a_T(i) * tau_ref

The reference record has a_T = 1 by construction, so tau_ref is the
characteristic time at the reference condition and the remaining a_T are
the horizontal shift factors of time-temperature superposition -- obtained
here as fitted parameters of the constitutive model rather than by sliding
spectra.

Why a joint fit
---------------
A single chirp covers a fixed frequency window. If the crossover lies
outside that window the record constrains only one asymptote, so one of the
exponents is unidentifiable from that record alone. Pooling lets each
parameter be constrained by whichever records actually see it, while the
shift factors absorb the movement of the spectrum.

Implementation
--------------
Features are built PER RECORD, each with its own dt and its own tau_c, and
then concatenated. The causal convolution therefore never crosses a record
boundary, and no fictitious memory is carried from one experiment into the
next. Everything downstream -- the GP, the ELBO, the sensitivities, the
information criteria -- is the unmodified rheogp machinery acting on the
stacked design matrix.

Only the horizontal shift is implemented. A vertical shift b_T would scale
the prefactor per record; because the prefactors are recovered as GP
sensitivities rather than optimized, that requires folding b_T into the
feature and is left for a later version.

Usage
-----
    from rheogp_multi import MultiRecordSPGP
    m = MultiRecordSPGP(records, i_ref=4, model="FMM",
                        kernel="linear", convolution="causal").fit_all()
    m.report()
"""

import numpy as np
import torch
import gpytorch

from rheogp import SPGP
from rheogp.kernels import KERNELS

__all__ = ["MultiRecordSPGP", "shift_kernel_class"]


# ================================================================
# kernel wrapper: per-record features with a per-record tau_c
# ================================================================
def shift_kernel_class(base_cls):
    """Build a kernel class that evaluates `base_cls` once per record,
    each with its own dt and its own shifted tau_c."""

    class _ShiftedKernel(base_cls):
        def __init__(self, bounds, dts, i_ref=0, **kw):
            super().__init__(**kw)
            self.bounds = bounds
            self.dts = dts
            self.i_ref = i_ref
            self.log_aT = torch.nn.Parameter(torch.zeros(len(bounds)))

        # ---- shift factors, normalised so a_T(i_ref) = 1 -------------
        @property
        def aT(self):
            return torch.exp(self.log_aT - self.log_aT[self.i_ref])

        # ---- features: one block per record --------------------------
        def _features_strain(self, u, dt):
            aT = self.aT
            tau_ref = self.tau_c                # the base class property
            # power-law (memoryless) branches -- e.g. the FKV family --
            # have no tau_c to substitute; a pure horizontal shift of
            # such a branch is instead a rescale of its feature by
            # a_T**p (see _strain_shift_exponents docstring).
            shift_exponents = super()._strain_shift_exponents()
            blocks = None
            for i, ((lo, hi), dti) in enumerate(zip(self.bounds, self.dts)):
                # temporarily present the shifted tau_c to the parent
                # (a no-op for kernels that never read self.tau_c)
                self._tau_override = tau_ref * aT[i]
                feats_i = super()._features_strain(u[lo:hi], dti)
                if shift_exponents is not None:
                    feats_i = [f * (aT[i] ** p)
                               for f, p in zip(feats_i, shift_exponents)]
                if blocks is None:
                    blocks = [[] for _ in feats_i]
                for j, f in enumerate(feats_i):
                    blocks[j].append(f)
            self._tau_override = None
            return [torch.cat(b) for b in blocks]

        # the parent reads self.tau_c; intercept it when an override is set
        @property
        def tau_c(self):
            ov = getattr(self, "_tau_override", None)
            if ov is not None:
                return ov
            return super().tau_c

        def _phys_params(self):
            p = super()._phys_params()
            # only report a (single, shared) reference tau_c for
            # kernels that actually use it to shift -- for power-law
            # kernels (_strain_shift_exponents() is not None) the raw
            # tau_c parameter is untrained/unused in strain control.
            # ...and not when tau_c was frozen: it is then a fixed
            # ladder position, not a fitted parameter, and counting it
            # in k would penalise a model for a number it never learned.
            if (super()._strain_shift_exponents() is None
                    and not getattr(self, "freeze_tau_c", False)):
                p["tau_c"] = super().tau_c      # unshifted reference value
            return p

    _ShiftedKernel.__name__ = f"Shifted{base_cls.__name__}"
    return _ShiftedKernel


# ================================================================
# driver
# ================================================================
class MultiRecordSPGP(SPGP):
    """SPGP fitted jointly to a list of (t, strain, stress) records.

    `temps`, if given, are the per-record temperatures (or any other
    single scalar labelling condition) in the SAME order as `records`.
    They are optional and are used for nothing during fitting -- they
    only enable `predict_at_temperature()` afterward. Everything else
    (`predict_at_shift()`, `fit_all()`, `report()`) works without them.
    """

    def __init__(self, records, i_ref=0, temps=None, **kw):
        super().__init__(**kw)
        self.records, self.i_ref = records, i_ref
        if temps is not None and len(temps) != len(records):
            raise ValueError(
                f"temps has {len(temps)} entries but records has "
                f"{len(records)}; they must be the same length and order.")
        self.temps = list(temps) if temps is not None else None
        gs, ss, self._bounds, self._dts, n = [], [], [], [], 0
        for t, g, s in records:
            gs.append(np.asarray(g, float))
            ss.append(np.asarray(s, float))
            self._dts.append(float(np.mean(np.diff(np.asarray(t, float)))))
            self._bounds.append((n, n + len(g)))
            n += len(g)
        self._g = np.concatenate(gs)
        self._s = np.concatenate(ss)

    def _build_phys_kernel(self):
        key = self.model_name.lower().replace(" ", "").replace("-", "").replace("_", "")
        base = KERNELS.get(key)
        if base is None:
            raise ValueError(f"Unknown model '{self.model_name}'")

        kw = {}
        varnames = base.__init__.__code__.co_varnames
        if self.alpha_init is not None and "alpha_init" in varnames:
            kw["alpha_init"] = self.alpha_init
        if self.beta_init is not None and "beta_init" in varnames:
            kw["beta_init"] = self.beta_init
        if getattr(self, "kappa_init", None) is not None \
           and "kappa_init" in varnames:
            kw["kappa_init"] = self.kappa_init
        if self.tau_c_init is not None and "tau_c_init" in varnames:
            kw["tau_c_init"] = self.tau_c_init
        # Generalized-Maxwell ladder geometry. The registry name
        # ("gmm12s") already fixes the mode count and the spring, but
        # passing these explicitly is what lets the RESOLUTION be chosen
        # from the data -- n_modes = modes_per_decade * (decades the
        # master curve actually spans) -- instead of from a name.
        for _gmm_kw in ("n_modes", "with_spring", "modes_per_decade",
                        "freeze_tau_c",
                        "fix_alpha", "fix_beta", "fix_kappa"):
            _v = getattr(self, _gmm_kw, None)
            if _v is not None and _gmm_kw in varnames:
                kw[_gmm_kw] = _v

        cls = shift_kernel_class(base)
        phys = cls(self._bounds, self._dts, self.i_ref, **kw)
        phys.configure(control=self.control, convolution=self.convolution,
                       quadrature=self.quadrature)
        return phys

    def fit_all(self):
        t_fake = np.arange(len(self._g), dtype=np.float64) * self._dts[0]
        return self.fit(t_fake, self._g, self._s)

    # ------------------------------------------------------------ output
    def shift_factors(self):
        return self.gp_model_.phys.aT.detach().cpu().numpy()

    @property
    def tau_ref(self):
        """The reference-record relaxation time tau_c (seconds), i.e.
        the timescale a_T=1 corresponds to. Convert your own tau_c into
        an a_T for `predict_at_shift()` via `aT = tau_c / m.tau_ref`."""
        self._check_fitted()
        p = self.gp_model_.phys._phys_params()
        v = p.get("tau_c")
        if v is None:
            raise ValueError(
                "This kernel family has no explicit tau_c (its strain "
                "features are power-law, not relaxation-time-weighted -- "
                "see _strain_shift_exponents in kernels.py). Pass a_T "
                "directly; there is no tau_ref to divide by.")
        return float(v.item() if torch.is_tensor(v) else v)

    # ------------------------------------------------------------ predict
    def _predict_core(self, new_records, aT_values):
        """
        Evaluate the TRAINED joint GP posterior (same inducing points,
        variational distribution, covariance hyperparameters -- nothing
        retrained) on a new list of (t, gamma) records, each assigned
        its own shift factor a_T in `aT_values`.

        This makes NO assumption about how a_T relates to temperature
        (or to anything else) -- it is the physics-only, shift-agnostic
        core that `predict_at_shift()` and `predict_at_temperature()`
        both delegate to. The new records may have entirely different
        bounds/lengths/count than the training records: bounds/dt are
        (re)derived fresh from THIS call's records, exactly as
        `__init__` derives them for training data, rather than reusing
        whatever was baked in at training time.

        Returns
        -------
        pred, std : np.ndarray, np.ndarray -- concatenated across all
            new records, in physical units (Pa, or strain units under
            stress control). Split by the returned `bounds` to recover
            per-record arrays.
        bounds : list[(lo, hi)] -- per-record index ranges into pred/std.
        """
        self._check_fitted()
        if len(new_records) != len(aT_values):
            raise ValueError(
                f"{len(new_records)} records but {len(aT_values)} shift "
                "factors -- must be one a_T per record.")

        gp, phys = self.gp_model_, self.gp_model_.phys
        gs, dts, bounds, n = [], [], [], 0
        for t, g in new_records:
            g = np.asarray(g, float)
            gs.append(g)
            dts.append(float(np.mean(np.diff(np.asarray(t, float)))))
            bounds.append((n, n + len(g)))
            n += len(g)
        u_all = np.concatenate(gs)
        u_t = torch.tensor(u_all, dtype=torch.float32, device=self.device)
        aT_arr = np.asarray(aT_values, dtype=float)

        # save everything this touches so a failed/aborted call can't
        # leave the trained model in a mutated state
        save = dict(bounds=phys.bounds, dts=phys.dts, gamma=gp.gamma.clone(),
                    dt=gp.dt, gsf=phys._gamma_scale_frozen,
                    fsf=phys._feature_stds_frozen)
        orig_aT = type(phys).aT
        try:
            phys.bounds, phys.dts = bounds, dts   # freshly derived, not training's
            # freeze the input-normalisation / feature-standardisation
            # to their TRAINING values: a new record must be scaled the
            # same way the training data was, not renormalised against
            # its own (possibly very different) amplitude/length.
            phys._gamma_scale_frozen = phys._feature_stds_frozen = True
            gp.gamma, gp.dt = u_t, dts[0]

            aT_tensor = torch.tensor(aT_arr, dtype=torch.float32,
                                     device=self.device)

            class _CallAT:
                def __get__(self, obj, objtype=None):
                    return aT_tensor
            type(phys).aT = _CallAT()

            gp.eval(); self.likelihood_.eval()
            with torch.no_grad(), gpytorch.settings.cholesky_jitter(1e-6):
                x = gp.compute_x()
                dist = self.likelihood_(gp(x))
                mu = dist.mean.cpu().numpy()
                var = dist.variance.cpu().numpy()
        finally:
            type(phys).aT = orig_aT
            phys.bounds, phys.dts = save["bounds"], save["dts"]
            gp.gamma, gp.dt = save["gamma"], save["dt"]
            phys._gamma_scale_frozen  = save["gsf"]
            phys._feature_stds_frozen = save["fsf"]

        sy, sm = self.scaler_y_.scale_[0], self.scaler_y_.mean_[0]
        pred = mu * sy + sm
        std  = np.sqrt(np.maximum(var, 0.0)) * sy
        return pred, std, bounds

    def predict_at_shift(self, records_with_aT):
        """
        Predict new records given their shift factors directly.

        records_with_aT : list of (t, gamma, a_T) -- a_T relative to
            this fit's own reference record (a_T=1 at the reference).
            If you have a relaxation time tau_c instead, pass
            `a_T = tau_c / m.tau_ref`.

        Makes no assumption about temperature dependence at all -- use
        this directly if you don't trust (or don't want to commit to)
        any particular shift law.

        Returns (pred, std, bounds) -- see `_predict_core`.
        """
        recs   = [(t, g) for t, g, _ in records_with_aT]
        aT_vals = [a for _, _, a in records_with_aT]
        return self._predict_core(recs, aT_vals)

    def predict_at_temperature(self, records_with_T, shift_law=None):
        """
        Predict new records given their temperatures.

        records_with_T : list of (t, gamma, T).
        shift_law : callable T -> a_T, or None.

            If given, it is used for EVERY record, full stop -- fit
            your own WLF, Arrhenius, VFT, or anything else you trust
            from the recovered `shift_factors()`/temps, and pass it
            in. This method never picks a functional form for you.

            If None (default), a_T is obtained WITHOUT assuming any
            functional form:
              * T exactly matching a fitted temperature -> that
                record's own fitted a_T (no model at all).
              * T strictly between two fitted temperatures -> log-linear
                interpolation of a_T between those two neighbours only
                (data-only, no WLF/Arrhenius/etc. imposed).
              * T outside the fitted temperature range -> raises
                ValueError. Extrapolation is a physical modelling
                choice (which law? how far is it valid?) that this
                method refuses to make silently -- pass `shift_law` if
                you want it extrapolated.

        Requires `temps=` to have been given when this MultiRecordSPGP
        was constructed. Returns (pred, std, bounds).
        """
        if self.temps is None:
            raise ValueError(
                "predict_at_temperature() needs to know the fitted "
                "records' temperatures. Either re-construct this "
                "MultiRecordSPGP with temps=[...] (same order as "
                "records), or use predict_at_shift() with a_T values "
                "you already know.")

        temps_arr = np.asarray(self.temps, dtype=float)
        aT_fit    = self.shift_factors()
        order     = np.argsort(temps_arr)
        temps_s, aT_s = temps_arr[order], aT_fit[order]
        log_aT_s = np.log10(aT_s)
        T_lo, T_hi = temps_s[0], temps_s[-1]

        aT_new = []
        for _, _, T in records_with_T:
            if shift_law is not None:
                aT_new.append(float(shift_law(T)))
                continue
            hit = np.isclose(temps_s, T, rtol=0, atol=1e-9)
            if hit.any():
                aT_new.append(float(aT_s[hit][0]))
            elif T_lo <= T <= T_hi:
                log_aT = np.interp(T, temps_s, log_aT_s)
                aT_new.append(float(10.0 ** log_aT))
            else:
                raise ValueError(
                    f"T={T!r} is outside the fitted temperature range "
                    f"[{T_lo:g}, {T_hi:g}] -- extrapolating a shift "
                    "factor past the data requires committing to a "
                    "physical law (WLF, Arrhenius, VFT, ...), which "
                    "this method will not choose for you. Fit whichever "
                    "law you trust to shift_factors()/temps yourself "
                    "and pass it as shift_law=<callable T -> a_T>.")

        recs = [(t, g) for t, g, _ in records_with_T]
        return self._predict_core(recs, aT_new)

    def per_record_rmse(self):
        res = self.results()
        out = []
        for lo, hi in self._bounds:
            r = res.stress[lo:hi] - res.prediction[lo:hi]
            out.append((float(np.sqrt((r ** 2).mean())),
                        float(res.stress[lo:hi].std())))
        return out

    def report(self, labels=None, truth=None):
        """
        Print a joint-fit report and return the fitted shift factors.

        `truth`, when given, is an optional {param_or_prefactor_name:
        value} dict of ground truth (e.g. for synthetic-data
        validation); comparisons against a zero truth value are shown
        as an absolute, not relative, error to avoid a division by
        zero.  Missing or degenerate per-record statistics (e.g. a
        record whose stress happens to be exactly constant, giving
        sd_i = 0) are reported as "n/a" rather than raising.
        """
        phys = self.gp_model_.phys
        labels = labels or [f"rec {i}" for i in range(len(self._bounds))]
        aT = self.shift_factors()
        tau_ref_raw = phys._phys_params()["tau_c"]
        tau_ref = float(tau_ref_raw.item() if torch.is_tensor(tau_ref_raw)
                        else tau_ref_raw)

        def _fmt_truth(name, value):
            if not truth or name not in truth:
                return ""
            tval = truth[name]
            if isinstance(tval, (int, float)) and abs(tval) > 1e-12:
                return (f"      (truth {tval:.6g}, "
                        f"error {100 * (value - tval) / tval:+.2f}%)")
            return f"      (truth {tval:.6g}, abs err {value - tval:+.3g})"

        print(f"\n{'='*66}")
        print(f"  JOINT FIT   {self.model_name}   {len(self._bounds)} records   "
              f"N = {len(self._g)}")
        print(f"{'='*66}")
        print("  shared parameters")
        for k, v in phys.named_phys_params().items():
            print(f"    {k:8s} = {v:.6f}" + _fmt_truth(k, v))
        for k, v in self.prefactors_.items():
            line = (f"    {k:8s} = {v['scalar_mean']:.5g}  "
                    f"[{v['scalar_lower']:.5g}, {v['scalar_upper']:.5g}]")
            print(line + _fmt_truth(k, v["scalar_mean"]))
        print(f"\n  overall  RMSE = {self.metrics_['rmse']:.5g}   "
              f"R2 = {self.metrics_['r2']:.6f}   "
              f"BIC = {self.metrics_['bic']:.1f}   "
              f"k = {self.metrics_['n_params']['k_total']}")

        rm = self.per_record_rmse()
        print(f"\n  {'record':>8} {'a_T':>12} {'tau_c [s]':>12} "
              f"{'RMSE_i':>12} {'RMSE_i/sd_i':>12}")
        rel = []
        for i, lab in enumerate(labels):
            rmse_i, sd_i = rm[i]
            ratio = (rmse_i / sd_i) if sd_i > 1e-300 else float("nan")
            if np.isfinite(ratio):
                rel.append(ratio)
            print(f"  {str(lab):>8} {aT[i]:12.5g} {aT[i]*tau_ref:12.5g} "
                  f"{rmse_i:12.5g} "
                  f"{(f'{ratio:12.4f}' if np.isfinite(ratio) else '        n/a')}")
        if rel:
            spread = f"{min(rel):.4f} - {max(rel):.4f}"
            ratio_txt = (f"{max(rel)/min(rel):.2f}" if min(rel) > 1e-300
                        else "n/a")
            print(f"\n  relative RMSE spread {spread}  (ratio {ratio_txt})")
        else:
            print("\n  relative RMSE spread: n/a (all records had sd_i = 0)")
        return aT
