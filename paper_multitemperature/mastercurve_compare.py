"""
mastercurve_compare.py
======================
Cross-check a `rheogp` joint-fit master curve against the one the
`mastercurves` package builds from the same spectra, and against a
`pyRheo` fit of each.

Why this exists
---------------
`rheogp` recovers the shift factors a_T as *fitted parameters of one
constitutive model* applied jointly to every chirp: the shifts and the
physics are inferred together, and the master curve is a by-product.
The `mastercurves` package (Lennon, McKinley & Swan) does the opposite
-- it superposes the data with Gaussian process regression and maximum
a posteriori estimation, with NO constitutive model at all, and the
shift factors are the only output. Those are genuinely different
routes to the same object, so agreement between them is evidence for
both, and disagreement localises which step is responsible.

The comparison here is deliberately run in three layers, weakest
assumption first:

  1. **Shift factors.** a_T from each route, on the same reference
     state. No model, no convention issues -- directly comparable.
  2. **Master curves.** Each route's a_T applied to the SAME raw
     G'/G'' spectra. Still model-free.
  3. **Parameters.** `pyRheo` fits the SAME constitutive model to each
     of the two master curves. Both parameter sets then live in
     pyRheo's own quasi-property convention, so they are directly
     comparable to each other, and the only thing that differs between
     them is how the master curve was built. This sidesteps having to
     hand-convert rheogp's (alpha, beta, tau_c, G_c) parametrisation
     into pyRheo's (G, V, alpha, beta) one just to compare two
     numbers -- though `fmm_quasiproperties()` below does that exact
     conversion, with a numerical self-check, where it is wanted.

What gets shifted
-----------------
Following the standard TTS practice this module defaults to superposing
**tan(delta) = G''/G'**, not G' or |G*|: tan(delta) is invariant to any
vertical shift b_T, so a horizontal-only superposition of it recovers
a_T without vertical-shift contamination. `|G*|` is available via
`shift_kind="gstar"` for comparison. Either way the resulting a_T is
then applied to G' and G'' to build the master curve -- the shifted
quantity is only the vehicle for finding a_T.

Dependencies
------------
`numpy` and `matplotlib` at module level. `mastercurves` (which brings
`scikit-learn`) and `pyRheo` are imported LAZILY, inside the functions
that need them, so the plotting half of this module keeps working on a
machine that has neither installed:

    pip install mastercurves pyRheo

References
----------
Lennon, McKinley & Swan, "A data-driven method for automated data
superposition with applications in soft matter science", PNAS 120
(2023).  https://github.com/krlennon/mastercurves
Miranda-Valdez et al., "pyRheo: an open-source Python package for
complex rheology", Digital Discovery 4, 1075 (2025).
https://github.com/mirandi1/pyRheo
"""
import json

import numpy as np


__all__ = [
    "spectra_from_export", "shift_quantity", "run_mastercurves",
    "build_master_curve", "build_master_curve_raw", "align_reference",
    "fit_pyrheo", "fmm_quasiproperties",
    "export_mastercurve_comparison", "load_mastercurve_comparison",
    "plot_shift_factor_comparison", "plot_master_curve_comparison",
    "plot_parameter_comparison",
]

_EXPONENTS = ("alpha", "beta", "kappa")     # dimensionless, compared directly


# ================================================================
# 1) read the rheogp export, and bin it down to a GP-sized spectrum
# ================================================================
def _log_bin(omega, *values, n_bins=60, min_count=1):
    """Median-bin `values` into `n_bins` log-spaced omega bins.

    Only worth doing when a record has far more points than the GP can
    take: the `mastercurves` GP is O(n^3) per state, and the DGEBA-style
    chirp spectra run to a few thousand points per record. Log spacing
    is the right choice there because it equalises points per decade,
    which is what a GP over log(omega) wants.

    Two things to know before switching it on. A chirp FFT grid is
    LINEARLY spaced (`np.fft.rfftfreq`), so log-spaced bins are nearly
    empty at low omega and swallow several points each at high omega:
    the output is always SHORTER than `n_bins` (empty bins are dropped),
    and the thinning falls entirely on the high-frequency end. On a
    spectrum that the GP could have taken whole that is pure loss,
    concentrated exactly where a power-law exponent is best
    constrained -- which is why `spectra_from_export` does not bin
    unless asked. The median (not the mean) keeps a surviving bin from
    being dragged by the occasional near-zero-SNR point.
    """
    omega = np.asarray(omega, float)
    edges = np.logspace(np.log10(omega.min()), np.log10(omega.max()), n_bins + 1)
    idx = np.clip(np.digitize(omega, edges) - 1, 0, n_bins - 1)
    w_out, v_out = [], [[] for _ in values]
    for b in range(n_bins):
        m = idx == b
        if m.sum() < min_count:
            continue
        w_out.append(np.median(omega[m]))
        for j, v in enumerate(values):
            v_out[j].append(np.median(np.asarray(v, float)[m]))
    return np.array(w_out), [np.array(v) for v in v_out]


def spectra_from_export(path, n_bins=None, min_points=8, require="positive"):
    """
    Pull the per-record (unshifted) FFT spectra and both the fitted and
    (if synthetic) true shift factors out of an `export_master_curve()`
    bundle, keeping only points where G' and G'' are both positive --
    logs are taken downstream, and the low-SNR tail of a chirp spectrum
    routinely carries a negative G''.

    `n_bins=None` (the default) hands the GP every point that survives
    that positivity filter. Pass an integer ONLY when a record has more
    points than the `mastercurves` GP can take in reasonable time
    (it is O(n^3) per state, so a few thousand points per record across
    a dozen-plus records is not viable) -- see `_log_bin` for what the
    binning costs. As a guide: the 61-point synthetic chirp spectra need
    no binning at all; the ~2700-point DGEBA ones do.

    Returns a dict with per-record lists `omega`/`Gp`/`Gdp`, the
    temperatures, the reference index, rheogp's own a_T, the
    ground-truth a_T when the bundle has them, rheogp's analytic master
    curve and parameters, and `point_counts` -- (raw, positive, used)
    per record, so any thinning is visible rather than silent.
    """
    with open(f"{path}.json") as fh:
        meta = json.load(fh)
    data = np.load(f"{path}.npz")

    n_rec = int(meta["n_records"])
    temps = np.asarray(data["temperatures"], float)
    omegas, Gps, Gdps, kept, counts = [], [], [], [], []
    for i in range(n_rec):
        w = np.asarray(data[f"rec{i}_fft_omega"], float)
        gp = np.asarray(data[f"rec{i}_fft_Gp"], float)
        gd = np.asarray(data[f"rec{i}_fft_Gdp"], float)
        # Only drop what the log transform genuinely cannot take.
        # require="finite": omega > 0 and all values finite. That is
        #   enough for |G*| = hypot(G', G''), which is positive by
        #   construction, so NO data point is discarded.
        # require="positive": additionally G' > 0 and G'' > 0, which
        #   tan(delta), G' and G'' each need individually before a log.
        ok = np.isfinite(w) & np.isfinite(gp) & np.isfinite(gd) & (w > 0)
        if require == "positive":
            ok &= (gp > 0) & (gd > 0)
        elif require != "finite":
            raise ValueError(f"require must be 'finite' or 'positive', got {require!r}")
        if ok.sum() < min_points:
            continue
        if n_bins is None or ok.sum() <= n_bins:
            wb, gpb, gdb = w[ok], gp[ok], gd[ok]
        else:
            wb, (gpb, gdb) = _log_bin(w[ok], gp[ok], gd[ok], n_bins=n_bins)
        if wb.size < min_points:
            continue
        order = np.argsort(wb)
        omegas.append(wb[order]); Gps.append(gpb[order]); Gdps.append(gdb[order])
        counts.append((int(w.size), int(ok.sum()), int(wb.size)))
        kept.append(i)

    if len(kept) < 2:
        raise RuntimeError(
            f"only {len(kept)} record(s) survived the positivity/size filter -- "
            "superposition needs at least 2.")

    i_ref_full = int(meta.get("i_ref", 0))
    if i_ref_full not in kept:
        raise RuntimeError(
            f"the reference record (index {i_ref_full}, T={temps[i_ref_full]:g}) "
            "was dropped by the positivity filter; pick another reference or "
            "loosen `min_points`.")
    i_ref = kept.index(i_ref_full)

    out = dict(
        path=path, model_name=meta.get("model_name"),
        record_index=kept, temps=temps[kept],
        i_ref=i_ref, T_ref=float(temps[i_ref_full]),
        omega=omegas, Gp=Gps, Gdp=Gdps, point_counts=counts,
        aT_rheogp=np.asarray(data["shift_factors"], float)[kept],
        phys_params=meta.get("phys_params", {}),
        prefactors=meta.get("prefactors", {}),
        metrics=meta.get("metrics", {}),
        truth=meta.get("truth"),
        wlf_fit=meta.get("wlf_fit"),
    )
    if "shift_factors_truth" in data.files:
        out["aT_truth"] = np.asarray(data["shift_factors_truth"], float)[kept]
    if "model_curve_omega" in data.files:
        out["model_curve"] = dict(
            omega=np.asarray(data["model_curve_omega"], float),
            Gp=np.asarray(data["model_curve_Gp"], float),
            Gdp=np.asarray(data["model_curve_Gdp"], float),
        )
    # rheogp normalises a_T to 1 at the reference record; re-normalise
    # after the record subset above in case the reference moved.
    out["aT_rheogp"] = out["aT_rheogp"] / out["aT_rheogp"][i_ref]
    if "aT_truth" in out:
        out["aT_truth"] = out["aT_truth"] / out["aT_truth"][i_ref]
    return out


# ================================================================
# 2) the quantity that actually gets superposed
# ================================================================
def shift_quantity(Gp, Gdp, kind="tandelta"):
    """The scalar curve handed to `mastercurves` to be superposed.

    "tandelta" : G''/G' -- invariant to a vertical shift b_T, so a
                 horizontal-only superposition of it recovers a_T
                 uncontaminated by any vertical shift. The default.
    "gstar"    : |G*| = sqrt(G'^2 + G''^2) -- carries the modulus
                 scale, so a horizontal-only superposition of it
                 implicitly assumes b_T = 1.
    "gp"/"gdp" : G' or G'' alone, same caveat as |G*|.
    """
    Gp = np.asarray(Gp, float); Gdp = np.asarray(Gdp, float)
    if kind == "tandelta":
        return Gdp / Gp, r"$\tan\delta$"
    if kind == "gstar":
        return np.hypot(Gp, Gdp), r"$|G^*|$ [Pa]"
    if kind == "gp":
        return Gp, r"$G'$ [Pa]"
    if kind == "gdp":
        return Gdp, r"$G''$ [Pa]"
    raise ValueError(f"unknown shift_kind {kind!r} -- use 'tandelta', "
                     "'gstar', 'gp' or 'gdp'")


# ================================================================
# 2b) keep the mastercurves GP well-conditioned
# ================================================================
def _gp_conditioning(logx, logy, fixed_noise=0.04):
    """Fit `mastercurves`' own GP kernel to each state and report whether
    it stayed numerically sound.

    This is not a cosmetic check. The package fits
    `RationalQuadratic*Constant + Constant + White + White(fixed)` per
    state with no `normalize_y`, and a chirp FFT grid is LINEARLY spaced
    in omega -- so in log(omega), which is what the GP sees, the points
    bunch ever more tightly as omega grows. Past a few dozen points per
    state the kernel matrix goes effectively singular, K^-1 y overflows,
    and the GP returns garbage that silently corrupts the pairwise
    superposition loss. On the 61-point synthetic chirps that produced
    thousands of numpy overflow/divide-by-zero warnings and a shift
    factor error ~35x worse than a well-conditioned run.

    Returns (ok, n_warnings).
    """
    import warnings as _warnings
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        RationalQuadratic, WhiteKernel, ConstantKernel)

    kernel = (RationalQuadratic() * ConstantKernel() + ConstantKernel()
              + WhiteKernel() + WhiteKernel(fixed_noise ** 2, "fixed"))
    n_warn = 0
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        for x, y in zip(logx, logy):
            x = np.asarray(x, float).reshape(-1, 1)
            gp = GaussianProcessRegressor(kernel=kernel).fit(x, np.asarray(y, float))
            if not np.all(np.isfinite(gp.predict(x))):
                return False, sum(1 for a in caught
                                  if issubclass(a.category, RuntimeWarning))
        n_warn = sum(1 for a in caught if issubclass(a.category, RuntimeWarning))
    return n_warn == 0, n_warn


def spectra_for_superposition(path, shift_kind="gstar", n_bins=None,
                              candidates=(48, 40, 32, 28, 24, 20, 16),
                              min_points=8, verbose=True):
    """
    Load an export bundle and pick a binning the `mastercurves` GP can
    actually handle, then return `(spec, values, label)` ready for
    `run_mastercurves`.

    `n_bins="auto"` walks `candidates` from most points to fewest and
    stops at the FIRST one that is well-conditioned -- i.e. it keeps as
    much data as the GP can take, and the choice is made purely on
    numerical conditioning, never by scoring against a known answer.
    That matters: on the synthetic set the error against the WLF truth
    happens to dip at one particular bin count, but only because the
    worst-determined state crosses the truth there on its way past;
    picking that number would be tuning to the answer and would not
    transfer to real data, which has no answer to tune to.

    Pass an integer to force a specific binning, or None to disable
    binning entirely (expect the GP to break on dense spectra).
    """
    # |G*| is positive by construction, so it needs no positivity filter
    # at all; only the quantities that take a log of G' or G'' alone do.
    require = "finite" if shift_kind == "gstar" else "positive"

    if n_bins != "auto":
        spec = spectra_from_export(path, n_bins=n_bins, min_points=min_points,
                                   require=require)
        vals, label = zip(*[shift_quantity(gp, gd, shift_kind)
                            for gp, gd in zip(spec["Gp"], spec["Gdp"])])
        spec["n_bins_used"] = n_bins
        if verbose:
            kept = sum(len(w) for w in spec["omega"])
            raw = sum(c[0] for c in spec["point_counts"])
            print(f"  using {kept} of {raw} raw FFT points "
                  f"({'unbinned, ' if n_bins is None else f'binned to {n_bins}, '}"
                  f"require={require})")
            logx = [np.log(w) for w in spec["omega"]]
            logy = [np.log(v) for v in vals]
            ok, n_warn = _gp_conditioning(logx, logy)
            if not ok:
                print(f"  NOTE: the mastercurves GP is ill-conditioned on this "
                      f"sampling ({n_warn} numerical warnings). Its shift "
                      f"factors may be unreliable; this is reported, not "
                      f"silently worked around.")
        return spec, list(vals), label[0]

    for cand in candidates:
        spec = spectra_from_export(path, n_bins=cand, min_points=min_points,
                                   require=require)
        vals, label = zip(*[shift_quantity(gp, gd, shift_kind)
                            for gp, gd in zip(spec["Gp"], spec["Gdp"])])
        logx = [np.log(w) for w in spec["omega"]]
        logy = [np.log(v) for v in vals]
        ok, n_warn = _gp_conditioning(logx, logy)
        pts = max(len(x) for x in logx)
        if verbose:
            print(f"  n_bins={cand:>3} -> {pts:>3} points/state: "
                  + ("well-conditioned" if ok
                     else f"ILL-CONDITIONED ({n_warn} numerical warnings)"))
        if ok:
            spec["n_bins_used"] = cand
            return spec, list(vals), label[0]

    raise RuntimeError(
        "no binning in `candidates` gave a well-conditioned GP. The spectra "
        "are probably too tightly spaced in log(omega) even after binning -- "
        "try smaller values, or a larger `fixed_noise` when constructing "
        "MasterCurve.")


def overlap_report(omegas, aT, temps, verbose=True):
    """
    Decades of overlap between each adjacent pair of records once shifted.

    Pairwise superposition determines a shift from how two neighbouring
    records overlap, and the shifts chain, so a pair that barely overlaps
    leaves its own shift poorly determined AND passes that looseness
    down the chain. This is the diagnostic that says WHICH state to
    distrust when the two routes disagree -- as opposed to concluding the
    superposition as a whole went wrong.
    """
    lo = np.array([np.log10(np.min(w) * a) for w, a in zip(omegas, aT)])
    hi = np.array([np.log10(np.max(w) * a) for w, a in zip(omegas, aT)])
    order = np.argsort(-np.asarray(aT, float))      # coldest/stiffest first
    rows = []
    for j in range(len(order) - 1):
        i1, i2 = order[j], order[j + 1]
        ov = min(hi[i1], hi[i2]) - max(lo[i1], lo[i2])
        rows.append((float(temps[i1]), float(temps[i2]), float(ov)))
    if verbose and rows:
        med = np.median([r[2] for r in rows])
        i_min = int(np.argmin([r[2] for r in rows]))
        print(f"  adjacent-pair overlap [decades], median {med:.2f}:")
        for j, (t1, t2, ov) in enumerate(rows):
            # The MINIMUM pair is always marked, not just pairs under a
            # threshold. On the synthetic set the weakest pair (30-40 C)
            # overlaps by 1.08 decades against a 1.39 median -- 78% of
            # median, so any fixed threshold loose enough to stay quiet
            # on healthy data would have missed it, yet it is precisely
            # the one state where the two routes disagree (by 0.17
            # decades, against 0.01 for every other state).
            flag = ""
            if j == i_min:
                flag = "   <- least overlap, shift least determined"
            elif ov < 0.75 * med:
                flag = "   <- low overlap"
            print(f"    {t1:>5g} - {t2:<5g} C : {ov:>6.2f}{flag}")
    return rows


# ================================================================
# 3) run the mastercurves package
# ================================================================
def run_mastercurves(omegas, values, temps, i_ref, vertical=False,
                     lamh=None, lamv=None, shift_bounds=(1e-8, 1.0),
                     verbose=True):
    """
    Superpose `values` vs `omegas` across states `temps` with the
    `mastercurves` package, and return shift factors normalised so
    a_T = 1 at `i_ref`.

    a_T is read back from `mc.xtransformed` -- the transformed
    coordinates the package itself produced -- rather than from
    `mc.hparams`, so it cannot be thrown off by the package's internal
    sign/direction convention for `Multiply`. `hparams` is still read
    and cross-checked against it, and any disagreement is reported.

    `vertical=True` additionally fits a vertical shift b_T (only
    meaningful for a modulus-valued curve; tan(delta) has no vertical
    shift to find).
    """
    try:
        from mastercurves import MasterCurve
        from mastercurves.transforms import Multiply
    except ImportError as exc:                       # pragma: no cover
        raise ImportError(
            "the `mastercurves` package is required for this comparison -- "
            "`pip install mastercurves` (it pulls in scikit-learn). See "
            "https://github.com/krlennon/mastercurves"
        ) from exc

    logx = [np.log(np.asarray(w, float)) for w in omegas]
    logy = [np.log(np.asarray(v, float)) for v in values]
    if not all(np.all(np.isfinite(a)) for a in logx + logy):
        raise ValueError("non-positive or non-finite values reached the log "
                         "transform -- filter the spectra first "
                         "(`spectra_from_export` does this).")

    states = [float(T) for T in temps]
    if verbose:
        print(f"  states handed to MasterCurve (T [C]): "
              f"{[f'{T:g}' for T in states]}")
        print(f"  points per state: {[len(x) for x in logx]}")
    # `Multiply` defaults to bounds=(1e-2, 1), i.e. AT MOST TWO DECADES
    # per pairwise shift. That is fine for a set spanning a couple of
    # decades, and silently wrong for one that does not: the optimiser
    # clamps at the bound and consecutive states come back with
    # identical shift factors. The DGEBA set needs 2.5 decades between
    # 150 and 155 C alone (14 decades overall), so the default truncates
    # it badly. Widened here, and checked for saturation below.
    mc = MasterCurve()
    mc.add_data(logx, logy, states)
    mc.add_htransform(Multiply(bounds=tuple(shift_bounds), scale="log"))
    if vertical:
        mc.add_vtransform(Multiply(bounds=tuple(shift_bounds), scale="log"))
    import warnings as _warnings
    with _warnings.catch_warnings(record=True) as _caught:
        _warnings.simplefilter("always")
        loss = mc.superpose(**({"lamh": lamh} if lamh is not None else {}),
                            **({"lamv": lamv} if lamv is not None else {}))
        n_gp_warnings = sum(1 for a in _caught
                            if issubclass(a.category, RuntimeWarning))
    if n_gp_warnings and verbose:
        print(f"  WARNING: the GP emitted {n_gp_warnings} numerical warnings "
              "(overflow / divide-by-zero in K^-1 y). Its predictions, and so "
              "these shift factors, are NOT trustworthy -- reduce the points "
              "per state (see `spectra_for_superposition`).")
    # `superpose` is documented as returning a float, but returns a LIST
    # (one entry per shifted pair) in the installed version -- reduce it
    # to a scalar either way rather than trusting the docstring.
    loss = float(np.sum(np.asarray(loss, float))) if np.ndim(loss) else float(loss)

    # a_T straight out of the transformed coordinates: x_shifted = x + ln a_T
    aT = np.array([float(np.exp(np.median(np.asarray(xt, float)
                                          - np.asarray(x0, float))))
                   for xt, x0 in zip(mc.xtransformed, logx)])
    bT = None
    if vertical and getattr(mc, "ytransformed", None) is not None:
        bT = np.array([float(np.exp(np.median(np.asarray(yt, float)
                                              - np.asarray(y0, float))))
                       for yt, y0 in zip(mc.ytransformed, logy)])

    aT_ref = aT[i_ref]
    aT = aT / aT_ref
    if bT is not None:
        bT = bT / bT[i_ref]

    # cross-check against the package's own reported parameters
    hp = np.asarray(getattr(mc, "hparams", [[]])[0], float) if getattr(mc, "hparams", None) else None
    agree = None
    if hp is not None and hp.size == aT.size:
        hp_n = hp / hp[i_ref]
        agree = bool(np.allclose(np.log(hp_n), np.log(aT), atol=1e-6))
        if verbose and not agree:
            print("  note: mc.hparams and mc.xtransformed imply different "
                  "shift factors (max |dlog| = "
                  f"{np.max(np.abs(np.log(hp_n) - np.log(aT))):.3g}); using "
                  "xtransformed, which is what the package actually plotted.")

    # Saturation check: a pairwise shift sitting exactly on a bound, or
    # two states returning the identical shift factor, means the search
    # ran out of room rather than finding an optimum.
    n_saturated = 0
    if hp is not None and hp.size:
        lo, hi = float(shift_bounds[0]), float(shift_bounds[1])
        rel = np.abs(np.log10(np.clip(hp, 1e-300, None))
                     - np.log10(np.array([[lo], [hi]])))
        n_saturated = int(np.sum(np.min(rel, axis=0) < 1e-6))
    n_dup = int(len(aT) - len(np.unique(np.round(np.log10(aT), 9))))
    if (n_saturated or n_dup) and verbose:
        print(f"  WARNING: {n_saturated} pairwise shift(s) sat on the search "
              f"bounds {shift_bounds} and {n_dup} state(s) share an identical "
              "shift factor. The superposition ran out of room, so these a_T "
              "are truncated, not converged -- widen `shift_bounds`.")

    unc = None
    hu = getattr(mc, "huncertainties", None)
    if hu is not None and len(hu) > 0:
        try:
            unc = np.asarray(hu[0], float)
        except (TypeError, ValueError):
            unc = None

    if verbose:
        print(f"  mastercurves: superposed {len(temps)} states, "
              f"loss = {float(loss):.6g}, reference T = {temps[i_ref]:g}")
        print(f"  a_T range: {aT.min():.4g} .. {aT.max():.4g} "
              f"({np.log10(aT.max() / aT.min()):.2f} decades)")
    return dict(aT=aT, bT=bT, aT_uncertainty=unc, loss=float(loss),
                hparams_agree=agree, n_gp_warnings=int(n_gp_warnings),
                n_saturated=n_saturated, n_duplicate_aT=n_dup,
                shift_bounds=tuple(shift_bounds), mc=mc)


def build_master_curve_raw(path, temps, aT, require="finite", min_points=8,
                           verbose=True):
    """
    Build a master curve from the FULL RAW spectra using shift factors
    determined elsewhere.

    Shift factors and the master curve are two separate jobs, and they
    must not share a sampling decision. If the `mastercurves` GP needs
    binned input to stay well-conditioned, that binning belongs to the
    step that FINDS a_T -- once a_T is known it is just a number per
    record, and the curve it shifts should be every raw point that was
    measured. Building the curve from the binned subset instead would
    throw away ~99% of the data for no reason, and would make this
    figure disagree with the master curve the same notebook plots
    elsewhere from the same export.

    `temps` and `aT` come from whatever produced the shifts; records are
    matched to them BY TEMPERATURE, so a record the superposition step
    dropped (too few usable points) simply gets no shift and is skipped
    rather than silently misaligning the rest.
    """
    spec = spectra_from_export(path, n_bins=None, min_points=min_points,
                               require=require)
    lookup = {float(T): float(a) for T, a in zip(temps, aT)}
    omegas, Gps, Gdps, used = [], [], [], []
    for i, T in enumerate(spec["temps"]):
        a = lookup.get(float(T))
        if a is None:
            continue
        omegas.append(spec["omega"][i]); Gps.append(spec["Gp"][i])
        Gdps.append(spec["Gdp"][i]); used.append(a)
    missing = [float(T) for T in spec["temps"] if float(T) not in lookup]
    if verbose:
        n = sum(len(w) for w in omegas)
        print(f"  master curve from {n} RAW points across {len(omegas)} records"
              + (f" (no shift for T={missing}, skipped)" if missing else ""))
    return build_master_curve(omegas, Gps, Gdps, used)


def build_master_curve(omegas, Gp, Gdp, aT):
    """Apply horizontal shift factors to the raw spectra: the master
    curve is (omega * a_T, G', G'') pooled over records, sorted by
    reduced frequency, with each point labelled by the record it came
    from."""
    w, gp, gd, rec = [], [], [], []
    for i, (wi, gpi, gdi) in enumerate(zip(omegas, Gp, Gdp)):
        w.append(np.asarray(wi, float) * float(aT[i]))
        gp.append(np.asarray(gpi, float))
        gd.append(np.asarray(gdi, float))
        rec.append(np.full(np.size(wi), i, dtype=int))
    w = np.concatenate(w); gp = np.concatenate(gp)
    gd = np.concatenate(gd); rec = np.concatenate(rec)
    order = np.argsort(w)
    return dict(omega=w[order], Gp=gp[order], Gdp=gd[order], record=rec[order])


# ================================================================
# 4) fit a master curve with pyRheo
# ================================================================
def fit_pyrheo(omega, Gp, Gdp, model, initial_guesses="random",
               num_initial_guesses=64, minimization_algorithm="Powell",
               cost_function="RSS", verbose=True, **kw):
    """
    Fit one constitutive model to one master curve with `pyRheo`'s
    `SAOSModel`, and return its parameters plus its prediction on the
    input grid.

    `model` takes pyRheo's own model names, which are spelled
    identically to rheogp's kernel names ("FractionalMaxwell",
    "FractionalMaxwellGel", "FractionalZener", ...), so a bundle's
    `model_name` can be passed straight through.
    """
    try:
        from pyRheo.oscillation_model import SAOSModel
    except ImportError:
        try:                                   # pyRheo < 1.0.1
            from pyRheo.oscillation_model import OscillationModel as SAOSModel
        except ImportError as exc:             # pragma: no cover
            raise ImportError(
                "the `pyRheo` package is required for this comparison -- "
                "`pip install pyRheo`. See https://github.com/mirandi1/pyRheo"
            ) from exc

    omega = np.asarray(omega, float)
    Gp = np.asarray(Gp, float); Gdp = np.asarray(Gdp, float)

    # pyRheo takes log10 of the modulus range when it builds its initial
    # guesses, so a non-positive G' or G'' makes it raise OverflowError.
    # A raw chirp master curve genuinely contains them: at the glassy end
    # G'' sits at essentially zero SNR and ~10% of its points come out
    # negative. Those points are real measurements and stay in whatever
    # is PLOTTED -- but they cannot be fitted by a routine that logs the
    # moduli, so the fit uses the positive subset and reports how many
    # points that was. A constraint of the fitting code, not a cleanup.
    ok = np.isfinite(omega) & np.isfinite(Gp) & np.isfinite(Gdp) \
        & (omega > 0) & (Gp > 0) & (Gdp > 0)
    n_drop = int((~ok).sum())
    if n_drop and verbose:
        print(f"  pyRheo: fitting {int(ok.sum())} of {ok.size} points "
              f"({100 * n_drop / ok.size:.1f}% dropped: non-positive G' or "
              f"G'', which pyRheo cannot take a log of)")
    if ok.sum() < 8:
        raise RuntimeError(
            f"only {int(ok.sum())} points have positive G' and G'' -- too few "
            "to fit. The master curve is dominated by sub-noise-floor points.")
    omega, Gp, Gdp = omega[ok], Gp[ok], Gdp[ok]
    order = np.argsort(omega)
    omega, Gp, Gdp = omega[order], Gp[order], Gdp[order]

    m = SAOSModel(model=model, initial_guesses=initial_guesses,
                  num_initial_guesses=num_initial_guesses,
                  minimization_algorithm=minimization_algorithm,
                  cost_function=cost_function, **kw)
    m.fit(omega, Gp, Gdp)
    params = m.get_parameters()
    Gp_hat, Gdp_hat = m.predict(omega)

    Gp_hat = np.asarray(Gp_hat, float); Gdp_hat = np.asarray(Gdp_hat, float)
    rel = np.concatenate([(Gp_hat - Gp) / Gp, (Gdp_hat - Gdp) / Gdp])
    out = dict(
        model=model,
        params={k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                for k, v in params.items()},
        omega=omega, Gp=Gp, Gdp=Gdp, Gp_fit=Gp_hat, Gdp_fit=Gdp_hat,
        mape=float(100 * np.mean(np.abs(rel))),
        rmse_log=float(np.sqrt(np.mean(
            np.concatenate([np.log(Gp_hat / Gp), np.log(Gdp_hat / Gdp)]) ** 2))),
    )
    if verbose:
        shown = ", ".join(f"{k}={v:.5g}" for k, v in out["params"].items()
                          if isinstance(v, float))
        print(f"  pyRheo [{model}]: {shown}")
        print(f"  pyRheo [{model}]: MAPE = {out['mape']:.2f}%, "
              f"log-RMSE = {out['rmse_log']:.4g}")
    return out


# ================================================================
# 5) rheogp -> pyRheo quasi-property conversion (FMM family only)
# ================================================================
def fmm_quasiproperties(Gc, alpha, beta, tau_c, check=True, atol=1e-8):
    """
    Convert rheogp's Fractional Maxwell parametrisation into pyRheo's
    quasi-property one, exactly.

    rheogp writes   G*(w) = G_c (i w tau_c)^alpha / (1 + (i w tau_c)^(alpha-beta))
    pyRheo writes   G*(w) = V(iw)^alpha G(iw)^beta / (V(iw)^alpha + G(iw)^beta)

    Equating the two gives

        V = G_c * tau_c**alpha        [Pa s^alpha]
        G = G_c * tau_c**beta         [Pa s^beta]

    (and back:  tau_c = (V/G)**(1/(alpha-beta)),  G_c = V / tau_c**alpha).
    With `check=True` the identity is verified numerically across five
    decades of frequency rather than taken on faith, and a mismatch
    raises instead of quietly returning wrong numbers. FMG is the
    beta -> 0 case, where G reduces to pyRheo's `G_s` = G_c.
    """
    Gc, alpha, beta, tau_c = float(Gc), float(alpha), float(beta), float(tau_c)
    V = Gc * tau_c ** alpha
    G = Gc * tau_c ** beta
    if check:
        w = np.logspace(-2, 3, 240)
        z = 1j * w * tau_c
        lhs = Gc * z ** alpha / (1.0 + z ** (alpha - beta))
        a = V * (1j * w) ** alpha
        b = G * (1j * w) ** beta
        rhs = a * b / (a + b)
        err = np.max(np.abs(lhs - rhs) / np.abs(lhs))
        if not np.isfinite(err) or err > max(atol, 1e-9):
            raise AssertionError(
                "fmm_quasiproperties(): the (G, V) conversion does not "
                f"reproduce rheogp's own G*(w) (max relative error {err:.3g}). "
                "Do not compare these parameters; compare the master curves "
                "instead.")
    return dict(G=G, V=V, alpha=alpha, beta=beta,
                G_s=Gc if abs(beta) < 1e-12 else None)


def align_reference(temps_dst, temps_src, aT_src, i_ref=None, tol=1.0):
    """
    Put an externally reported a_T series on the same record ordering and
    the same reference state as the fitted ones.

    Literature temperatures rarely match the nominal setpoints exactly
    (the DGEBA-IPDA values are quoted at 105.01, 135.01, 165.01 ... rather
    than 105, 135, 165), so records are matched to the nearest reported
    temperature within `tol` degrees rather than by index, and any record
    with no match within that window gets NaN instead of a silently
    misaligned value. The series is then renormalised to a_T = 1 at the
    same reference record used by the fits, so all three curves are
    anchored identically and the comparison is of shapes, not offsets.
    """
    temps_dst = np.asarray(temps_dst, float)
    temps_src = np.asarray(temps_src, float)
    aT_src = np.asarray(aT_src, float)
    if temps_src.size != aT_src.size:
        raise ValueError(f"{temps_src.size} reference temperatures but "
                         f"{aT_src.size} reference shift factors")
    out = np.full(temps_dst.size, np.nan)
    for i, T in enumerate(temps_dst):
        j = int(np.argmin(np.abs(temps_src - T)))
        if abs(temps_src[j] - T) <= tol:
            out[i] = aT_src[j]
    if i_ref is not None and np.isfinite(out[i_ref]) and out[i_ref] != 0:
        out = out / out[i_ref]
    n_missing = int(np.sum(~np.isfinite(out)))
    if n_missing:
        print(f"  note: {n_missing} record(s) had no reported a_T within "
              f"{tol} deg and are left blank")
    return out


# ================================================================
# 6) export / reload the comparison
# ================================================================
def export_mastercurve_comparison(path, spec, mcres, mc_curve, our_curve,
                                  fit_mc, fit_ours, shift_kind="tandelta",
                                  shift_values=None, aT_reference=None,
                                  ref_curve=None, reference_label="reported",
                                  extra=None):
    """
    Write the whole comparison to `<path>.json` + `<path>.npz`, in the
    same two-file shape as `export_master_curve()`, so the plotting
    half of this module (and any notebook cell) reloads it with numpy
    and json alone -- no `mastercurves`, no `pyRheo`, no `rheogp`.
    """
    arrays = {
        "temps": np.asarray(spec["temps"], float),
        "aT_rheogp": np.asarray(spec["aT_rheogp"], float),
        "aT_mastercurves": np.asarray(mcres["aT"], float),
    }
    if mcres.get("aT_uncertainty") is not None:
        arrays["aT_mastercurves_unc"] = np.asarray(mcres["aT_uncertainty"], float)
    if mcres.get("bT") is not None:
        arrays["bT_mastercurves"] = np.asarray(mcres["bT"], float)
    if "aT_truth" in spec:
        arrays["aT_truth"] = np.asarray(spec["aT_truth"], float)
    if aT_reference is not None:
        arrays["aT_reference"] = np.asarray(aT_reference, float)

    for i, (w, gp, gd) in enumerate(zip(spec["omega"], spec["Gp"], spec["Gdp"])):
        arrays[f"rec{i}_omega"] = np.asarray(w, float)
        arrays[f"rec{i}_Gp"] = np.asarray(gp, float)
        arrays[f"rec{i}_Gdp"] = np.asarray(gd, float)
        if shift_values is not None:
            arrays[f"rec{i}_shiftq"] = np.asarray(shift_values[i], float)

    curves = [("ours", our_curve), ("mc", mc_curve)]
    if ref_curve is not None:
        curves.append(("ref", ref_curve))
    for tag, curve in curves:
        for k in ("omega", "Gp", "Gdp", "record"):
            arrays[f"curve_{tag}_{k}"] = np.asarray(curve[k])
    for tag, fit in (("ours", fit_ours), ("mc", fit_mc)):
        if fit is None:
            continue
        for k in ("omega", "Gp_fit", "Gdp_fit"):
            arrays[f"pyrheo_{tag}_{k}"] = np.asarray(fit[k], float)

    if spec.get("model_curve"):
        for k, v in spec["model_curve"].items():
            arrays[f"rheogp_model_{k}"] = np.asarray(v, float)

    np.savez_compressed(f"{path}.npz", **arrays)

    meta = dict(
        source_export=spec.get("path"),
        model_name=spec.get("model_name"),
        shift_kind=shift_kind,
        n_records=len(spec["omega"]),
        record_index=[int(i) for i in spec["record_index"]],
        i_ref=int(spec["i_ref"]), T_ref=float(spec["T_ref"]),
        reference_label=reference_label,
        mastercurves_loss=mcres.get("loss"),
        mastercurves_hparams_agree=mcres.get("hparams_agree"),
        rheogp_phys_params=spec.get("phys_params"),
        rheogp_prefactors=spec.get("prefactors"),
        rheogp_metrics=spec.get("metrics"),
        pyrheo={
            tag: (None if fit is None else
                  dict(model=fit["model"], params=fit["params"],
                       mape=fit["mape"], rmse_log=fit["rmse_log"]))
            for tag, fit in (("ours", fit_ours), ("mc", fit_mc))
        },
    )
    if spec.get("truth") is not None:
        meta["truth"] = spec["truth"]
    if spec.get("wlf_fit") is not None:
        meta["wlf_fit"] = spec["wlf_fit"]
    if extra:
        meta.update(extra)

    with open(f"{path}.json", "w") as fh:
        json.dump(meta, fh, indent=2, default=float)
    print(f"exported comparison -> {path}.json + {path}.npz")
    return f"{path}.json", f"{path}.npz"


def load_mastercurve_comparison(path):
    """Reload `export_mastercurve_comparison()` -- pure numpy + json."""
    from types import SimpleNamespace
    with open(f"{path}.json") as fh:
        meta = json.load(fh)
    data = np.load(f"{path}.npz")
    arrays = {k: data[k] for k in data.files}

    def record(i):
        pfx = f"rec{i}_"
        return {k[len(pfx):]: v for k, v in arrays.items() if k.startswith(pfx)}

    def curve(tag):
        pfx = f"curve_{tag}_"
        return {k[len(pfx):]: v for k, v in arrays.items() if k.startswith(pfx)}

    return SimpleNamespace(meta=meta, arrays=arrays, record=record, curve=curve,
                           n_records=int(meta["n_records"]))


# ================================================================
# 7) figures
# ================================================================
def _style(plt):
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "nature", "bright"])
    except ImportError:
        print("scienceplots not found -- falling back to default matplotlib "
              "style (install with `pip install scienceplots` to match the "
              "paper exactly)")
    plt.rcParams.update({
        "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 6, "axes.grid": False,
        "figure.constrained_layout.use": True,
    })


def _panel_letter(ax, text, xoff=-20, yoff=9, fontsize=10):
    ax.annotate(text, xy=(0, 1), xycoords="axes fraction",
                xytext=(xoff, yoff), textcoords="offset points",
                ha="right", va="top", fontsize=fontsize, fontweight="bold",
                clip_on=False, zorder=100)


#: Raster resolution for PNG output. MUST match the `dpi=` used by the
#: plotting cells of fmm_plotting.ipynb / fzm_dgeba.ipynb (300), or these
#: figures come out several times larger in pixels than every other
#: figure in the same notebook and are scaled differently on the page.
FIG_DPI = 300


def _save(fig, out_name, plt):
    fig.savefig(f"{out_name}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_name}.png", dpi=FIG_DPI, bbox_inches="tight")
    print(f"wrote {out_name}.pdf and {out_name}.png")
    return fig


C_OURS = "tab:blue"      # rheogp joint fit
C_MC = "tab:orange"      # mastercurves package
C_TRUTH = "k"


def plot_shift_factor_comparison(cmp_path, out_name, plt=None):
    """
    Shift factors from the two routes, and the collapse the
    `mastercurves` package achieved on the quantity it superposed.

    (a) a_T vs T from rheogp's joint fit and from `mastercurves`, with
        the WLF ground truth overlaid when the source was synthetic.
    (b) the disagreement, as log10(a_T^mc / a_T^rheogp) -- a horizontal
        line at zero would mean the two routes recovered identical
        shifts. Plotted in log units because a_T spans decades and a
        ratio is the only scale-free way to read it.
    (c) the superposed quantity itself, before (open, faint) and after
        (filled) applying the `mastercurves` shifts, which is the
        evidence that the shifts in (a) actually collapse anything.
    """
    import matplotlib.pyplot as plt_mod
    plt = plt or plt_mod
    d = load_mastercurve_comparison(cmp_path)
    _style(plt)

    T = d.arrays["temps"]
    a_ours = d.arrays["aT_rheogp"]
    a_mc = d.arrays["aT_mastercurves"]
    unc = d.arrays.get("aT_mastercurves_unc")
    kind = d.meta.get("shift_kind", "tandelta")

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(7.0, 2.6))

    a_ref = d.arrays.get("aT_reference")
    ref_lab = d.meta.get("reference_label", "reported")
    if "aT_truth" in d.arrays:
        axA.plot(T, d.arrays["aT_truth"], "-", color=C_TRUTH, lw=1.0,
                 label="WLF truth", zorder=1)
    if a_ref is not None:
        axA.plot(T, a_ref, "^", ms=5, color=C_TRUTH, mfc="none", mew=1.0,
                 label=ref_lab, zorder=4)
    axA.plot(T, a_ours, "o", ms=4, color=C_OURS, mec="k", mew=0.4,
             label="rheogp joint fit", zorder=3)
    if unc is not None and np.size(unc) == np.size(a_mc):
        axA.errorbar(T, a_mc, yerr=np.abs(a_mc) * np.asarray(unc, float),
                     fmt="s", ms=4, color=C_MC, mec="k", mew=0.4,
                     elinewidth=0.7, capsize=1.5, label="mastercurves", zorder=2)
    else:
        axA.plot(T, a_mc, "s", ms=4, color=C_MC, mec="k", mew=0.4,
                 label="mastercurves", zorder=2)
    axA.set_yscale("log")
    axA.set_xlabel(r"$T~[^\circ\mathrm{C}]$"); axA.set_ylabel(r"$a_T$")
    axA.legend(frameon=False, loc="best")
    _panel_letter(axA, r"$\mathbf{a}$")

    axB.axhline(0.0, color="0.5", lw=0.6, ls=":")
    if a_ref is not None:
        # With an independently reported series available, it -- not one of
        # the two fits -- is the natural baseline: plotting each route
        # against it says how far each sits from the published answer,
        # rather than only how far they sit from each other.
        axB.plot(T, np.log10(a_ours / a_ref), "o-", ms=4, lw=1.0,
                 color=C_OURS, mec="k", mew=0.4, label="rheogp")
        axB.plot(T, np.log10(a_mc / a_ref), "s-", ms=4, lw=1.0,
                 color=C_MC, mec="k", mew=0.4, label="mastercurves")
        # \mathrm collapses spaces, so "Sheridan et al." rendered as
        # "Sheridanetal."; \ / inserts a real space inside math mode.
        _lab = ref_lab.replace(" ", r"\,")
        axB.set_ylabel(rf"$\log_{{10}}(a_T/a_T^{{\mathrm{{{_lab}}}}})$")
        axB.legend(frameon=False, loc="best")
    else:
        axB.plot(T, np.log10(a_mc / a_ours), "d-", ms=4, lw=1.0,
                 color="tab:purple", mec="k", mew=0.4)
        axB.set_ylabel(r"$\log_{10}(a_T^{\mathrm{mc}}/a_T^{\mathrm{rheogp}})$")
    axB.set_xlabel(r"$T~[^\circ\mathrm{C}]$")
    _panel_letter(axB, r"$\mathbf{b}$")

    cmap = plt.cm.viridis
    n = d.n_records
    for i in range(n):
        rec = d.record(i)
        if "shiftq" not in rec:
            continue
        col = cmap(i / max(n - 1, 1))
        axC.plot(rec["omega"], rec["shiftq"], "o", ms=1.6, mfc="none",
                 mec=col, mew=0.4, alpha=0.35, zorder=2)
        axC.plot(rec["omega"] * a_mc[i], rec["shiftq"], "o", ms=2.2,
                 color=col, mew=0.0, zorder=3)
    axC.set_xscale("log"); axC.set_yscale("log")
    axC.set_xlabel(r"$\omega$ or $a_T\omega$ [rad s$^{-1}$]")
    axC.set_ylabel(shift_quantity(np.array([1.0]), np.array([1.0]), kind)[1])
    axC.set_title("open: unshifted   filled: shifted", fontsize=6)
    _panel_letter(axC, r"$\mathbf{c}$")

    print(f"{'T [C]':>8} {'aT rheogp':>13} {'aT mastercurves':>17} "
          f"{'log10 ratio':>13}")
    for i in range(len(T)):
        print(f"{T[i]:>8g} {a_ours[i]:>13.5g} {a_mc[i]:>17.5g} "
              f"{np.log10(a_mc[i] / a_ours[i]):>13.4f}")
    rms = float(np.sqrt(np.mean(np.log10(a_mc / a_ours) ** 2)))
    print(f"RMS disagreement: {rms:.4f} decades")
    return _save(fig, out_name, plt)


def plot_master_curve_comparison(cmp_path, out_name, plt=None):
    """
    The two master curves, built from the SAME raw spectra and differing
    only in which set of shift factors was applied.

    (a) rheogp's shifts: reduced G' (filled) and G'' (open), colored by
        record, with rheogp's own analytic master curve over them.
    (b) the `mastercurves` shifts on the same spectra, with the pyRheo
        fit to THAT curve over them.

    No legend and no panel titles: at this width the two scatters plus
    their overlaid curves already fill the axes, and anything further
    competes with the data. Identify the series in the caption.

    The quantitative comparison the old panels (c) and (d) carried --
    the two pyRheo fits overlaid, and their relative difference per
    decade -- is printed below instead of plotted, since overlaying four
    model curves on one log-log axis obscured more than it showed.
    """
    import matplotlib.pyplot as plt_mod
    plt = plt or plt_mod
    d = load_mastercurve_comparison(cmp_path)
    _style(plt)

    has_ref = "curve_ref_omega" in d.arrays
    n_panel = 3 if has_ref else 2
    # Fixed 7.0in width, like every other figure in these notebooks --
    # NOT scaled with the panel count. A wider canvas has to be shrunk to
    # fit the same column, which silently reduces the effective font size
    # relative to its neighbours even though the point sizes match.
    # Adding a third panel therefore narrows the panels; it must not
    # widen the figure. Height follows the other 1x3 figures at 2.6in.
    fig, axes = plt.subplots(1, n_panel, figsize=(7.0, 2.6 if n_panel > 2 else 3.0))
    axA, axB = axes[0], axes[1]
    axC = axes[2] if has_ref else None
    cmap = plt.cm.viridis
    n = d.n_records

    def scatter(ax, curve):
        rec = curve["record"].astype(int)
        for i in range(n):
            m = rec == i
            if not m.any():
                continue
            col = cmap(i / max(n - 1, 1))
            ax.plot(curve["omega"][m], curve["Gp"][m], "o", ms=1.8,
                    color=col, mew=0.0, zorder=2)
            ax.plot(curve["omega"][m], np.abs(curve["Gdp"][m]), "o", ms=2.2,
                    mfc="none", mec=col, mew=0.4, zorder=2)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$a_T\omega$ [rad s$^{-1}$]")

    # Neither panel carries a model curve. This figure compares two
    # SUPERPOSITIONS of the same raw spectra, so the only thing that may
    # differ between the panels is the shift factors. Drawing rheogp's
    # analytic master curve over (a) while (b) has none would let one
    # route look better for a reason unrelated to its shifts.
    scatter(axA, d.curve("ours"))
    axA.set_ylabel(r"$G'$, $|G''|$ [Pa]")
    _panel_letter(axA, r"$\mathbf{a}$")

    # No pyRheo curve is drawn here. The fits are still computed and
    # exported -- see `print_parameter_comparison` -- but they are not
    # plotted: panel (b) is about how the mastercurves shifts collapse
    # the raw data, and overlaying a model fit on top invites reading
    # the figure as a statement about the fit instead.
    scatter(axB, d.curve("mc"))
    _panel_letter(axB, r"$\mathbf{b}$")

    if axC is not None:
        scatter(axC, d.curve("ref"))
        _panel_letter(axC, r"$\mathbf{c}$")

    # share the y range so the panels are directly comparable by eye
    live = [a for a in (axA, axB, axC) if a is not None]
    lo = min(a.get_ylim()[0] for a in live)
    hi = max(a.get_ylim()[1] for a in live)
    for a in live:
        a.set_ylim(lo, hi)

    if "pyrheo_mc_omega" in d.arrays and "pyrheo_ours_omega" in d.arrays:
        wl = max(d.arrays["pyrheo_ours_omega"].min(), d.arrays["pyrheo_mc_omega"].min())
        wh = min(d.arrays["pyrheo_ours_omega"].max(), d.arrays["pyrheo_mc_omega"].max())
        wg = np.logspace(np.log10(wl), np.log10(wh), 300)

        def interp(tag, key):
            return np.exp(np.interp(np.log(wg),
                                    np.log(d.arrays[f"pyrheo_{tag}_omega"]),
                                    np.log(d.arrays[f"pyrheo_{tag}_{key}"])))

        for key, lab in (("Gp_fit", "G'"), ("Gdp_fit", "G''")):
            rel = 100 * (interp("mc", key) - interp("ours", key)) / interp("ours", key)
            print(f"  {lab}: median |mc - ours| = {np.median(np.abs(rel)):.2f}%, "
                  f"max = {np.max(np.abs(rel)):.2f}%")
    return _save(fig, out_name, plt)


def print_parameter_comparison(cmp_path):
    """
    The pyRheo parameter comparison as a table, with no figure.

    The fits themselves are unchanged and still in the exported bundle;
    this only stops them being drawn. Keeping the numbers visible
    matters because they carry their own health check: a fit whose
    exponents sit at their bounds with a large MAPE has not converged,
    and a parameter ratio computed from two such fits means nothing.
    """
    d = load_mastercurve_comparison(cmp_path)
    pr = d.meta.get("pyrheo", {})
    f_ours, f_mc = pr.get("ours"), pr.get("mc")
    if not f_ours or not f_mc:
        print("  (no pyRheo fits in this bundle)")
        return
    p_ours = {k: v for k, v in f_ours["params"].items() if isinstance(v, float)}
    p_mc = {k: v for k, v in f_mc["params"].items() if isinstance(v, float)}
    shared = [k for k in p_ours if k in p_mc and k != "Cost"]
    q = {k: v for k, v in (d.meta.get("rheogp_quasiproperties") or {}).items()
         if isinstance(v, float)}
    qcol = f" {'rheogp analytic':>17}" if q else ""
    hdr = (f"{'parameter':>12} {'rheogp curve':>16} {'mastercurves':>16} "
           f"{'ratio':>12}{qcol}")
    print(f"pyRheo model: {f_ours['model']}")
    print(hdr); print("-" * len(hdr))
    for k in shared:
        qv = f" {q[k]:>17.6g}" if k in q else (" " * 18 if q else "")
        ratio = p_mc[k] / p_ours[k] if p_ours[k] else float("nan")
        print(f"{k:>12} {p_ours[k]:>16.6g} {p_mc[k]:>16.6g} {ratio:>12.4g}{qv}")
    print(f"{'MAPE [%]':>12} {f_ours['mape']:>16.3f} {f_mc['mape']:>16.3f}")
    print(f"{'log-RMSE':>12} {f_ours['rmse_log']:>16.5g} {f_mc['rmse_log']:>16.5g}")
    worst = max(f_ours["mape"], f_mc["mape"])
    if worst > 20:
        print(f"\n  NOTE: the worse fit has MAPE {worst:.0f}%. Parameters from a "
              "fit that poor,\n  and any ratio built from them, are not "
              "interpretable.")
    return dict(ours=p_ours, mastercurves=p_mc)


def plot_parameter_comparison(cmp_path, out_name, plt=None):
    """
    The parameters `pyRheo` recovered from each of the two master
    curves -- the same model, the same package, the same settings, so
    the only thing that differs is which superposition built the curve
    it was fitted to.

    (a) dimensionless exponents (alpha, beta, kappa) side by side, with
        the synthetic ground truth when there is one: these are
        directly comparable numbers on a linear axis.
    (b) the quasi-properties (G, V, K, ...), which span decades, as the
        ratio mastercurves/rheogp on a log axis against a line at 1.
    (c) each fit's own goodness of fit to the curve it was fitted to --
        MAPE and log-RMSE -- since a parameter gap only means something
        if both fits actually describe their data.
    """
    import matplotlib.pyplot as plt_mod
    plt = plt or plt_mod
    d = load_mastercurve_comparison(cmp_path)
    _style(plt)

    pr = d.meta.get("pyrheo", {})
    f_ours, f_mc = pr.get("ours"), pr.get("mc")
    if not f_ours or not f_mc:
        raise RuntimeError(
            "this comparison bundle has only one pyRheo fit -- re-run the "
            "compute cell with both fits enabled before plotting parameters.")
    p_ours = {k: v for k, v in f_ours["params"].items() if isinstance(v, float)}
    p_mc = {k: v for k, v in f_mc["params"].items() if isinstance(v, float)}
    shared = [k for k in p_ours if k in p_mc and k != "Cost"]
    exps = [k for k in shared if k in _EXPONENTS]
    quasi = [k for k in shared if k not in _EXPONENTS]
    truth = d.meta.get("truth") or {}

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(7.0, 2.6))

    x = np.arange(len(exps))
    w = 0.36
    axA.bar(x - w / 2, [p_ours[k] for k in exps], w, color=C_OURS,
            label="rheogp curve", edgecolor="k", linewidth=0.4)
    axA.bar(x + w / 2, [p_mc[k] for k in exps], w, color=C_MC,
            label="mastercurves curve", edgecolor="k", linewidth=0.4)
    drew_truth = False
    for j, k in enumerate(exps):
        if k in truth:
            axA.plot([j - w, j + w], [truth[k]] * 2, "-", color=C_TRUTH, lw=1.2,
                     zorder=5)
            drew_truth = True
    if drew_truth:                      # one proxy, whichever exponents had a truth
        axA.plot([], [], "-", color=C_TRUTH, lw=1.2, label="truth")
    axA.set_xticks(x)
    axA.set_xticklabels([rf"$\{k}$" for k in exps])
    axA.set_ylabel("exponent")
    axA.legend(frameon=False, loc="best")
    _panel_letter(axA, r"$\mathbf{a}$")

    # Quasi-properties span decades, so compare them as a ratio -- but as
    # log10(ratio) on a LINEAR axis rather than a ratio on a log one: a bar
    # on a log axis grows from the bottom of the view instead of from 1,
    # which would bury the agreement line under every bar.
    ratios = np.array([p_mc[k] / p_ours[k] for k in quasi], float)
    axB.axhline(0.0, color="0.5", lw=0.8, ls="--")
    axB.bar(np.arange(len(quasi)), np.log10(ratios), 0.5, color="tab:purple",
            edgecolor="k", linewidth=0.4)
    axB.set_xticks(np.arange(len(quasi)))
    axB.set_xticklabels([f"${k}$" for k in quasi])
    axB.set_ylabel(r"$\log_{10}$(mastercurves / rheogp)")
    _panel_letter(axB, r"$\mathbf{b}$")

    # MAPE is a percentage and log-RMSE is O(0.01); on one axis the second
    # pair of bars would be invisible, so each metric gets its own axis.
    pct = "MAPE [\\%]" if plt.rcParams.get("text.usetex") else "MAPE [%]"
    axC_r = axC.twinx()
    axC.bar([0 - w / 2], [f_ours["mape"]], w, color=C_OURS, edgecolor="k",
            linewidth=0.4, label="rheogp curve")
    axC.bar([0 + w / 2], [f_mc["mape"]], w, color=C_MC, edgecolor="k",
            linewidth=0.4, label="mastercurves curve")
    axC_r.bar([1 - w / 2], [f_ours["rmse_log"]], w, color=C_OURS,
              edgecolor="k", linewidth=0.4, hatch="///")
    axC_r.bar([1 + w / 2], [f_mc["rmse_log"]], w, color=C_MC,
              edgecolor="k", linewidth=0.4, hatch="///")
    axC.set_xticks([0, 1]); axC.set_xticklabels([pct, "log-RMSE"])
    axC.set_xlim(-0.6, 1.6)
    axC.set_ylabel(pct)
    axC_r.set_ylabel("log-RMSE", fontsize=7)
    axC_r.tick_params(axis="y", labelsize=7)
    # no legend here: (a) already carries the same blue/orange key, and a
    # second copy gets clipped by this panel's twin-axis tick labels
    _panel_letter(axC, r"$\mathbf{c}$")

    # A third column when the source bundle's own rheogp parameters were
    # converted into pyRheo's quasi-property convention (FMM family
    # only -- see `fmm_quasiproperties`). It is rheogp's ANALYTIC answer
    # in pyRheo's units, so it is the reference the two fitted columns
    # should both sit near, not a third fit.
    q = d.meta.get("rheogp_quasiproperties") or {}
    q = {k: v for k, v in q.items() if isinstance(v, float)}
    qcol = f" {'rheogp analytic':>17}" if q else ""
    hdr = (f"{'parameter':>12} {'rheogp curve':>16} {'mastercurves':>16} "
           f"{'ratio':>10}{qcol}")
    print(f"pyRheo model: {f_ours['model']}")
    print(hdr); print("-" * len(hdr))
    for k in shared:
        qv = f" {q[k]:>17.6g}" if k in q else (" " * 18 if q else "")
        print(f"{k:>12} {p_ours[k]:>16.6g} {p_mc[k]:>16.6g} "
              f"{p_mc[k] / p_ours[k]:>10.4f}{qv}")
    print(f"{'MAPE [%]':>12} {f_ours['mape']:>16.3f} {f_mc['mape']:>16.3f}")
    print(f"{'log-RMSE':>12} {f_ours['rmse_log']:>16.5g} {f_mc['rmse_log']:>16.5g}")
    return _save(fig, out_name, plt)
