"""
paper_utils.py
==============
Small helpers shared by the three "compute this yourself" notebooks
(synthetic FMM, synthetic FractionalZenerSolidS, real data) so that each
notebook, once run, drops two machine-generated fragments next to itself:

  * `results_<tag>.tex`      -- \\newcommand macros for every number the
                                 paper text refers to (so main.tex reads
                                 \\alphaFitFmm instead of a hard-coded
                                 number, and re-running the notebook
                                 after a data update just means
                                 recompiling the paper).
  * `table_<tag>.tex`        -- a full `table*` environment with the
                                 AIC/BIC model-comparison rows, in the
                                 same layout as Table I of the
                                 companion (single-chirp) paper.

Also provides `export_master_curve()` / `load_master_curve_export()`, a
pair for taking every array behind a master-curve figure (raw chirps,
fitted stress, FFT spectra, shift factors, prefactor sensitivities,
training history, analytic master curve + credible band) off of a
fitted notebook object and onto another machine for replotting.

Everything above this line has no heavy dependencies and is safe to
import anywhere. `export_master_curve()` is the one exception: it does a
local `import rheogp.mastercurve` (only when called, and only for two
small helper functions), since it must be called from inside a notebook
that already has rheogp fitted a model -- so this module stays free of a
module-level rheogp/torch dependency. `load_master_curve_export()`, the
reload side meant to run on a DIFFERENT machine, uses only numpy + json,
exactly like the rest of this file.
"""
import re
import numpy as np


# ----------------------------------------------------------------
# \newcommand macro file
# ----------------------------------------------------------------
def _sanitize_macro_name(name):
    """LaTeX macro names must be pure letters -- strip digits/underscores
    the caller may have used for readability (e.g. 'alpha_fit_fmm' ->
    'alphafitfmm') and camel-case is preserved as-is if already alpha."""
    if not re.fullmatch(r"[A-Za-z]+", name):
        raise ValueError(
            f"macro name {name!r} must be pure letters (LaTeX \\newcommand "
            f"names cannot contain digits/underscores) -- use camelCase")
    return name


def write_macros(path, macros, mode="w"):
    """
    Write a `\\newcommand{\\Name}{value}` line per (name, value) pair.

    `macros` : dict[str, Any] -- keys must be pure-letter camelCase
        (e.g. "alphaFitFmm"); values are formatted with `str()` (so
        pre-format floats/percentages before passing them in, e.g.
        f"{x:.3f}" -- keeps this function dumb and predictable).
    """
    lines = []
    for name, value in macros.items():
        _sanitize_macro_name(name)
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")
    with open(path, mode) as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {len(macros)} macros -> {path}")


def fmt(x, sig=3):
    """Format a float with `sig` significant figures, plain (no LaTeX math)."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x:.{sig}g}"


def pct(x, signed=True):
    fmtstr = "{:+.2f}" if signed else "{:.2f}"
    return fmtstr.format(x) + r"\%"


# ----------------------------------------------------------------
# model-comparison table (mirrors Table I / II of the companion paper)
# ----------------------------------------------------------------
def model_comparison_latex_table(curves, criterion="bic", caption="",
                                  label="tab:modelcomp",
                                  display_names=None, star=True):
    """
    Build a `table*` LaTeX environment ranking the candidates in
    `curves` (the `{name: MasterCurve}` dict returned by
    `compare_master_curve_models`) by `criterion`, in the same layout
    as the companion paper's Table I: 2U, 2k, k ln N, AIC, BIC,
    Delta-AIC, Delta-BIC, RMSE, R^2, and the shared shape parameters.

    Returns the LaTeX source as a string (write it to a .tex file
    yourself, or pass straight to `write_table`).
    """
    display_names = display_names or {}
    crit = criterion.lower()
    assert crit in ("aic", "bic")

    rows = sorted(curves.items(), key=lambda kv: kv[1].metrics_[crit])
    best_val = rows[0][1].metrics_[crit]

    # union of all shape-parameter names across candidates, in a stable
    # order (alpha, beta, kappa, tau_c first if present, then whatever
    # else -- matches how the constitutive tables read in the text)
    preferred = ["alpha", "beta", "kappa", "tau_c"]
    all_params = []
    for _, mc in rows:
        for k in mc.model_.gp_model_.phys.named_phys_params():
            if k not in all_params:
                all_params.append(k)
    ordered_params = [p for p in preferred if p in all_params] + \
                     [p for p in all_params if p not in preferred]

    _greek = {"alpha": r"\alpha", "beta": r"\beta", "kappa": r"\kappa",
              "tau": r"\tau", "eta": r"\eta"}

    def _param_symbol(p):
        if p in _greek:
            return _greek[p]
        if "_" in p:
            head, _, tail = p.partition("_")
            head = _greek.get(head, head)
            return f"{head}_{{{tail}}}"
        return p

    col_spec = "l" + "c" * (8 + len(ordered_params))
    header_params = " & ".join(f"${_param_symbol(p)}$" for p in ordered_params)

    lines = []
    env = "table*" if star else "table"
    lines.append(f"\\begin{{{env}}}[t]")
    lines.append("  \\centering")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append("  \\small")
    lines.append("  \\setlength{\\tabcolsep}{4pt}")
    lines.append("  \\renewcommand{\\arraystretch}{1.2}")
    lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    lines.append("    \\hline")
    lines.append(
        "    Model & $2\\mathcal{U}$ & $2k$ & $k\\ln N$ & AIC & BIC & "
        f"$\\Delta${criterion.upper()} & RMSE [Pa] & $R^2$ & {header_params} \\\\")
    lines.append("    \\hline")
    N_data = rows[0][1].metrics_["n_data"]
    for name, mc in rows:
        m = mc.metrics_
        k = m["n_params"]["k_total"]
        nll = m["nll"]
        twoU = 2 * nll
        twok = 2 * k
        klogn = k * np.log(N_data)
        delta = m[crit] - best_val
        p = mc.model_.gp_model_.phys.named_phys_params()
        pvals = " & ".join(fmt(p.get(pp), 3) for pp in ordered_params)
        disp = display_names.get(name, name).replace("_", r"\_")
        lines.append(
            f"    {disp} & {fmt(twoU,6)} & {twok:.0f} & {fmt(klogn,5)} & "
            f"{fmt(m['aic'],6)} & {fmt(m['bic'],6)} & {fmt(delta,3)} & "
            f"{fmt(m['rmse'],4)} & {m['r2']:.4f} & {pvals} \\\\")
    lines.append("    \\hline")
    lines.append(f"  \\end{{tabular}}")
    lines.append(f"\\end{{{env}}}")
    return "\n".join(lines)


def write_table(path, latex_source):
    with open(path, "w") as fh:
        fh.write(latex_source + "\n")
    print(f"wrote table -> {path}")


# ----------------------------------------------------------------
# Full-dataset export, for replotting on another machine
# ----------------------------------------------------------------
# `export_master_curve()` is called from INSIDE a notebook, where a
# fitted `rheogp.rheogp_multi.MultiRecordSPGP` (`mj` in the notebooks,
# or `best.model_` when you only have the `MasterCurve` wrapper) is
# already in memory -- it only reads attributes off that object, so it
# stays a thin export step rather than doing any fitting itself.
#
# `load_master_curve_export()` is the reload side, meant to run on a
# DIFFERENT machine that does not have rheogp/torch installed -- it only
# needs numpy + json (matplotlib/scienceplots for the actual plotting is
# up to you), mirroring the FitResult loader pattern used for the
# single-chirp figures.
def _model_curve_from_mj(mj, n_omega=400, credible_interval=True):
    """
    Analytic G'(omega_ref), G''(omega_ref) directly off a fitted
    MultiRecordSPGP `mj`, with a credible band propagated from the GP
    posterior prefactor samples through the kernel's own closed-form
    Gstar(omega, prefactors) -- exactly what `MasterCurve.model_curve()`
    does, but usable on a bare `mj` with no `MasterCurve` wrapper needed.
    Returns (omega, Gp, Gdp, Gp_ci, Gdp_ci); Gp_ci/Gdp_ci are None if
    `credible_interval=False` or if the fit has no positive-prefactor
    posterior draws to propagate (e.g. very early in a bad fit).
    """
    from rheogp.mastercurve import _posterior_scalar_draws

    phys = mj.gp_model_.phys
    omega = np.logspace(np.log10(mj.omega_min), np.log10(mj.omega_max), n_omega)
    scalar_pf = {k: v["scalar_mean"] for k, v in mj.prefactors_.items()}
    Gp, Gdp = phys.Gstar(omega, scalar_pf)

    if not credible_interval:
        return omega, Gp, Gdp, None, None

    names = list(mj.prefactors_)
    draws = {n: _posterior_scalar_draws(mj.prefactors_[n]) for n in names}
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


def _posterior_fft_ensemble(mj, records, aT_fit, n_samples=30,
                             min_snr_db=30.0, seed=None):
    """
    An alternative to the analytic Gstar credible band on the master
    curve: draw joint posterior-PREDICTIVE samples of the fitted stress
    trace -- correlated in time exactly as the GP posterior believes,
    NOT i.i.d. noise at pred_std_ (which would inject high-frequency
    content the posterior does not actually contain) -- FFT-transform
    EACH draw of EACH record with the same `fft_transfer()` used for
    the raw/measured spectra, and shift onto the reduced-frequency axis
    with the record's own fitted a_T.

    Pooling these across draws and records gives a scatter cloud whose
    spread reflects the model's actual predictive uncertainty in the
    frequency domain -- unlike `_model_curve_from_mj`'s scalar-prefactor
    band, which is a credible interval on the *mean* prefactor and
    mechanically shrinks as more data is pooled, regardless of how
    noisy any one record is.

    Returns four flat (n_points,) arrays: omega_reduced, Gp, Gdp,
    record_index, draw_index -- deliberately NOT pre-binned onto a
    common frequency grid, so how to summarize them (percentile bands,
    2-D histograms, ...) is left to the plotting code.
    """
    import torch
    import gpytorch
    from rheogp.mastercurve import fft_transfer

    model, lik = mj.gp_model_, mj.likelihood_
    sy, sm = mj.scaler_y_.scale_[0], mj.scaler_y_.mean_[0]
    bounds = mj._bounds

    model.eval(); lik.eval()
    if seed is not None:
        torch.manual_seed(seed)
    with torch.no_grad(), gpytorch.settings.cholesky_jitter(1e-6), \
         gpytorch.settings.fast_pred_var():
        x = model.compute_x()
        preds = lik(model(x))
        draws_std = preds.rsample(torch.Size([n_samples])).cpu().numpy()  # (S, N)
    draws = draws_std * sy + sm  # physical stress, (S, N)

    omega_list, Gp_list, Gdp_list, rec_list, draw_list = [], [], [], [], []
    for i, ((t, gamma, _sigma), (lo, hi)) in enumerate(zip(records, bounds)):
        t = np.asarray(t, float)
        gamma = np.asarray(gamma, float)
        for s in range(n_samples):
            sigma_s = draws[s, lo:hi]
            omega, Gs = fft_transfer(t, gamma, sigma_s, min_snr_db=min_snr_db)
            if omega.size == 0:
                continue
            omega_list.append(omega * aT_fit[i])
            Gp_list.append(np.real(Gs))
            Gdp_list.append(np.imag(Gs))
            rec_list.append(np.full(omega.size, i, dtype=np.int32))
            draw_list.append(np.full(omega.size, s, dtype=np.int32))

    if not omega_list:
        empty_f = np.array([], dtype=float)
        empty_i = np.array([], dtype=np.int32)
        return empty_f, empty_f, empty_f, empty_i, empty_i

    return (np.concatenate(omega_list), np.concatenate(Gp_list),
            np.concatenate(Gdp_list), np.concatenate(rec_list),
            np.concatenate(draw_list))


def export_master_curve(path, mj, records, temperatures, i_ref,
                         truth=None, wlf=None, aT_true=None,
                         include_samples=False, min_snr_db=30.0,
                         include_posterior_fft=True,
                         n_posterior_fft_samples=30,
                         posterior_fft_seed=None):
    """
    Export everything needed to reproduce every panel of a master-curve
    figure (raw chirps, fitted stress, FFT spectra at both the record's
    own and the reduced frequency axis, shift factors, prefactor
    sensitivities, training history, and the analytic master curve with
    its credible band) to `<path>.npz` + `<path>.json`.

    Parameters
    ----------
    path         : str -- files written are f"{path}.npz"/f"{path}.json"
    mj           : a FITTED `MultiRecordSPGP` (the `mj` variable in
                   notebooks 01/02/04, or `best.model_` in notebook 03
                   where you only built a `MasterCurve` wrapper `best`)
    records      : list[(t, gamma, sigma)], SAME order used to fit `mj`
    temperatures : list[float], same order/length as `records`
    i_ref        : int -- reference-record index into `records`
    truth        : optional dict of synthetic ground truth (e.g. the
                   `truth` dict from `make_dataset()`); stored as-is
                   (minus the bulky `_meta` key) for later comparison
    wlf          : optional dict(C1=, C2=, C1_std=, C2_std=) -- e.g.
                   `dict(C1=C1f, C2=C2f, C1_std=sd[0], C2_std=sd[1])`
                   from a manual WLF fit, or `best.wlf_` if you called
                   `best.fit_wlf()` on a `MasterCurve`
    aT_true      : optional (M,) array -- ground-truth shift factors,
                   for synthetic validations
    include_samples : if True, also save the full (S, N_record)
                   posterior sensitivity samples per prefactor per
                   record (bulky -- off by default; the mean/lower/upper
                   bands are always saved)
    min_snr_db   : forwarded to `fft_transfer` for every FFT computed
                   here
    include_posterior_fft : if True (default), also export the
                   posterior-predictive FFT ensemble (see
                   `_posterior_fft_ensemble`) as `postfft_omega_reduced`
                   / `postfft_Gp` / `postfft_Gdp` / `postfft_record` /
                   `postfft_draw` -- a scatter cloud built by drawing
                   `n_posterior_fft_samples` correlated posterior draws
                   of the fitted stress and FFT-transforming each one,
                   which (unlike the analytic Gstar band) reflects the
                   model's actual predictive uncertainty in the
                   frequency domain rather than uncertainty in the mean
                   prefactor. Costs about as much compute as the
                   prefactor extraction already done during `mj.fit()`.
    n_posterior_fft_samples : int -- posterior draws for the ensemble
                   above; ignored if `include_posterior_fft=False`.
    posterior_fft_seed : optional int -- `torch.manual_seed()` before
                   drawing the ensemble, for reproducibility.

    Returns
    -------
    (npz_path, json_path)
    """
    import json
    from rheogp.mastercurve import fft_transfer

    aT_fit = mj.shift_factors()
    bounds = mj._bounds
    pred, pred_std = mj.pred_, mj.pred_std_
    phys_params = mj.gp_model_.phys.named_phys_params()
    prefactors = mj.prefactors_
    metrics = mj.metrics_

    arrays = {
        "temperatures": np.asarray(temperatures, float),
        "shift_factors": np.asarray(aT_fit, float),
    }
    if aT_true is not None:
        arrays["shift_factors_truth"] = np.asarray(aT_true, float)

    for i, ((t, gamma, sigma), (lo, hi)) in enumerate(zip(records, bounds)):
        t = np.asarray(t, float)
        gamma = np.asarray(gamma, float)
        sigma = np.asarray(sigma, float)
        arrays[f"rec{i}_t"] = t
        arrays[f"rec{i}_gamma"] = gamma
        arrays[f"rec{i}_sigma"] = sigma
        arrays[f"rec{i}_pred"] = pred[lo:hi]
        arrays[f"rec{i}_pred_std"] = pred_std[lo:hi]

        # FFT of the RAW measured chirp, on its own (unshifted) frequency
        # axis, and on the reduced (shifted-to-T_ref) axis.
        omega, Graw = fft_transfer(t, gamma, sigma, min_snr_db=min_snr_db)
        arrays[f"rec{i}_fft_omega"] = omega
        arrays[f"rec{i}_fft_Gp"] = np.real(Graw)
        arrays[f"rec{i}_fft_Gdp"] = np.imag(Graw)
        arrays[f"rec{i}_fft_omega_reduced"] = omega * aT_fit[i]

        # FFT of the model's own (denoised) fitted prediction -- what the
        # paper's master-curve panels actually scatter-plot alongside the
        # analytic curve.
        omega_p, Gpred = fft_transfer(t, gamma, pred[lo:hi], min_snr_db=min_snr_db)
        arrays[f"rec{i}_fftpred_omega_reduced"] = omega_p * aT_fit[i]
        arrays[f"rec{i}_fftpred_Gp"] = np.real(Gpred)
        arrays[f"rec{i}_fftpred_Gdp"] = np.imag(Gpred)

        for name, info in prefactors.items():
            arrays[f"rec{i}_sens_{name}_mean"] = info["mean"][lo:hi]
            arrays[f"rec{i}_sens_{name}_lower"] = info["lower"][lo:hi]
            arrays[f"rec{i}_sens_{name}_upper"] = info["upper"][lo:hi]
            if include_samples:
                arrays[f"rec{i}_sens_{name}_samples"] = info["samples"][:, lo:hi]

    if "loss" in getattr(mj, "history_", {}):
        arrays["hist_loss"] = np.asarray(mj.history_["loss"])
        for name, vals in mj.history_.get("params", {}).items():
            arrays[f"hist_param_{name}"] = np.asarray(vals)

    try:
        omega_m, Gp_m, Gdp_m, Gp_ci, Gdp_ci = _model_curve_from_mj(mj)
        arrays["model_curve_omega"] = omega_m
        arrays["model_curve_Gp"] = Gp_m
        arrays["model_curve_Gdp"] = Gdp_m
        if Gp_ci is not None:
            arrays["model_curve_Gp_ci"] = Gp_ci
            arrays["model_curve_Gdp_ci"] = Gdp_ci
    except Exception as e:  # pragma: no cover -- e.g. a kernel with no Gstar
        print(f"  (skipping analytic model_curve: {e})")

    n_postfft_points = 0
    if include_posterior_fft:
        try:
            pf_omega, pf_Gp, pf_Gdp, pf_rec, pf_draw = _posterior_fft_ensemble(
                mj, records, aT_fit, n_samples=n_posterior_fft_samples,
                min_snr_db=min_snr_db, seed=posterior_fft_seed)
            arrays["postfft_omega_reduced"] = pf_omega
            arrays["postfft_Gp"] = pf_Gp
            arrays["postfft_Gdp"] = pf_Gdp
            arrays["postfft_record"] = pf_rec
            arrays["postfft_draw"] = pf_draw
            n_postfft_points = int(pf_omega.size)
        except Exception as e:  # pragma: no cover
            print(f"  (skipping posterior-predictive FFT ensemble: {e})")

    np.savez_compressed(f"{path}.npz", **arrays)

    def _floatify(d):
        return {k: (float(v) if not isinstance(v, dict) else _floatify(v))
                for k, v in d.items()}

    meta = dict(
        model_name=mj.model_name,
        n_records=len(records),
        i_ref=int(i_ref),
        T_ref=float(temperatures[i_ref]),
        record_bounds=[[int(lo), int(hi)] for lo, hi in bounds],
        phys_params=_floatify(phys_params),
        prefactors={k: dict(scalar_mean=float(v["scalar_mean"]),
                            scalar_lower=float(v["scalar_lower"]),
                            scalar_upper=float(v["scalar_upper"]))
                   for k, v in prefactors.items()},
        metrics={k: v for k, v in metrics.items() if k != "n_params"},
        n_params=metrics["n_params"],
        n_posterior_fft_samples=int(n_posterior_fft_samples) if include_posterior_fft else 0,
        n_postfft_points=n_postfft_points,
    )
    if truth is not None:
        meta["truth"] = {k: v for k, v in truth.items() if k != "_meta"}
    if wlf is not None:
        meta["wlf_fit"] = {k: float(v) if not isinstance(v, np.ndarray) else v.tolist()
                           for k, v in wlf.items() if k != "cov"}

    with open(f"{path}.json", "w") as fh:
        json.dump(meta, fh, indent=2, default=float)

    print(f"exported {len(records)} records -> {path}.npz + {path}.json")
    return f"{path}.npz", f"{path}.json"


def load_master_curve_export(path):
    """
    Reload everything `export_master_curve()` wrote -- pure numpy +
    json, NO rheogp/torch import required, so this runs on a machine
    that only has your plotting stack (matplotlib, scienceplots, ...)
    installed. Copy just this function (and its `json`/`numpy` imports)
    to the other machine if you don't want the rest of `paper_utils.py`.

    Returns a `SimpleNamespace` with:
      .meta       -- the full `<path>.json` dict (model name, physical
                     parameters, prefactor credible intervals, metrics,
                     ground truth / WLF fit if they were passed in)
      .arrays     -- the full `<path>.npz` as a plain dict
      .n_records  -- convenience shortcut to `.meta["n_records"]`
      .record(i)  -- dict of one chirp's arrays with the `rec{i}_`
                     prefix stripped, e.g. `.record(0)["t"]`,
                     `.record(0)["gamma"]`, `.record(0)["fft_Gp"]`,
                     `.record(0)["sens_Gc_mean"]`
      .model_curve -- (omega, Gp, Gdp, Gp_ci, Gdp_ci) tuple pulled out
                     of `.arrays`, or None if it wasn't exported
    """
    import json
    from types import SimpleNamespace

    with open(f"{path}.json") as fh:
        meta = json.load(fh)
    data = np.load(f"{path}.npz")
    arrays = {k: data[k] for k in data.files}

    def record(i):
        pfx = f"rec{i}_"
        return {k[len(pfx):]: v for k, v in arrays.items() if k.startswith(pfx)}

    model_curve = None
    if "model_curve_omega" in arrays:
        model_curve = (
            arrays["model_curve_omega"], arrays["model_curve_Gp"],
            arrays["model_curve_Gdp"], arrays.get("model_curve_Gp_ci"),
            arrays.get("model_curve_Gdp_ci"),
        )

    return SimpleNamespace(meta=meta, arrays=arrays, record=record,
                           n_records=meta["n_records"],
                           model_curve=model_curve)


def export_conventional_pipeline(path, records, temperatures, raw_spectra,
                                  aT_conv, i_ref, fit_params=None,
                                  aT_true=None, truth=None, wlf=None):
    """
    GP-free counterpart to `export_master_curve()`, for the conventional
    FFT-then-shift-then-fit pipeline notebook (`05_...ipynb`): raw
    chirps, each record's own (unshifted) FFT spectrum, the
    successive-overlap shift factors, and the post-hoc `curve_fit`
    parameters. Uses the SAME `rec{i}_...` array schema as
    `export_master_curve()` (minus the `pred`/`sens_*`/`model_curve_*`
    keys that only exist for a GP fit), so it reloads with the exact
    same `load_master_curve_export()` -- `.record(i)` just won't have a
    `"pred"`/`"sens_..."` entry, and `.model_curve` will be `None`.

    This pipeline has no rheogp/GP dependency of its own (only
    `fft_transfer`, already computed by the notebook into
    `raw_spectra` before this is called), so this function does not
    import rheogp either -- pure numpy/json, like the reload side.

    Parameters
    ----------
    path         : str -- files written are f"{path}.npz"/f"{path}.json"
    records      : list[(t, gamma, sigma)]
    temperatures : list[float], same order/length as `records`
    raw_spectra  : list[(omega, Gstar)] -- one FFT per record, e.g. the
                   `raw_spectra` list already built in Stage 1
    aT_conv      : (M,) shift factors from the successive-overlap
                   optimization (Stage 2)
    i_ref        : int -- reference-record index
    fit_params   : optional dict of the Stage-3 `curve_fit` result
                   (e.g. `dict(alpha=alpha_conv, Gc=Gc_conv,
                   tau_c=tau_c_conv, rmse=rmse_conv, r2=r2_conv)`)
    aT_true, truth, wlf : same meaning as in `export_master_curve()`
    """
    import json

    arrays = {
        "temperatures": np.asarray(temperatures, float),
        "shift_factors": np.asarray(aT_conv, float),
    }
    if aT_true is not None:
        arrays["shift_factors_truth"] = np.asarray(aT_true, float)

    for i, ((t, gamma, sigma), (omega, Gstar)) in enumerate(zip(records, raw_spectra)):
        arrays[f"rec{i}_t"] = np.asarray(t, float)
        arrays[f"rec{i}_gamma"] = np.asarray(gamma, float)
        arrays[f"rec{i}_sigma"] = np.asarray(sigma, float)
        omega = np.asarray(omega, float)
        arrays[f"rec{i}_fft_omega"] = omega
        arrays[f"rec{i}_fft_Gp"] = np.real(Gstar)
        arrays[f"rec{i}_fft_Gdp"] = np.imag(Gstar)
        arrays[f"rec{i}_fft_omega_reduced"] = omega * aT_conv[i]

    np.savez_compressed(f"{path}.npz", **arrays)

    meta = dict(n_records=len(records), i_ref=int(i_ref),
               T_ref=float(temperatures[i_ref]))
    if fit_params is not None:
        meta["fit_params"] = {k: float(v) for k, v in fit_params.items()}
    if truth is not None:
        meta["truth"] = {k: v for k, v in truth.items() if k != "_meta"}
    if wlf is not None:
        meta["wlf_fit"] = {k: float(v) for k, v in wlf.items()}

    with open(f"{path}.json", "w") as fh:
        json.dump(meta, fh, indent=2, default=float)

    print(f"exported {len(records)} records -> {path}.npz + {path}.json")
    return f"{path}.npz", f"{path}.json"

# ----------------------------------------------------------------
# noise-robustness sweep: export / reload
# ----------------------------------------------------------------
_SWEEP_CRITERIA = ("aic", "bic_raw", "bic_eff_pen", "bic_eff_full")

_SWEEP_MODEL_COLUMNS = [
    "noise_frac", "model", "is_true_model", "k", "n_data", "n_eff",
    "mean_tau_int", "elbo_per_point", "neg2_loglik",
    "aic", "d_aic", "bic_raw", "d_bic_raw",
    "bic_eff_pen", "d_bic_eff_pen", "bic_eff_full", "d_bic_eff_full",
    "rmse", "r2", "alpha", "beta", "tau_c",
    "Gc", "Gc_lower", "Gc_upper", "noise_sd_inferred", "early_stopped",
    "wins_aic", "wins_bic_raw", "wins_bic_eff_pen", "wins_bic_eff_full",
]

_SWEEP_PARAM_COLUMNS = [
    "noise_frac", "model",
    "alpha", "alpha_truth", "alpha_err_pct",
    "beta", "beta_truth",
    "tau_c", "tau_c_truth", "tau_c_err_pct",
    "Gc", "Gc_lower", "Gc_upper", "Gc_truth", "Gc_err_pct",
    "C1", "C1_std", "C1_truth", "C2", "C2_std", "C2_truth",
    "rmse", "r2", "aic", "bic", "k", "n_data",
    "noise_frac_nominal", "noise_sd_true", "noise_sd_realized",
    "noise_sd_inferred", "noise_frac_inferred", "noise_sd_ratio",
    "winner_aic", "winner_bic_raw", "winner_bic_eff_pen", "winner_bic_eff_full",
]


def _pct_err(fit, truth):
    """Percent error, or None when the truth is zero (β=0 in the FMM
    ground truth) and a relative error is meaningless."""
    if fit is None or truth in (None, 0) or not np.isfinite(truth):
        return None
    return 100.0 * (fit - truth) / truth


def export_noise_sweep(path, rows, truth=None, meta=None):
    """
    Export a noise-robustness sweep -- one entry of `rows` per injected
    noise level, in the schema the Sec. 9 loop of
    `01_synthetic_fmm_tts.ipynb` builds -- to three files:

      `<path>.json`         everything, nested: per noise level the
                            ground truth, the injected vs. inferred
                            noise statistics, the joint true-model fit
                            (shape parameters, prefactors with credible
                            intervals, shift factors, WLF constants),
                            and EVERY candidate's information criteria
                            under the raw `N` and under both `N_eff`
                            conventions, with its own recovered
                            parameters.
      `<path>_models.csv`   one row per (noise level, candidate) -- the
                            flat model-selection table.
      `<path>_params.csv`   one row per noise level -- the true-model
                            fit's parameters beside ground truth, plus
                            inferred vs. injected noise.

    Only the winner's NAME is a selection result; every candidate's
    parameters are exported too, so a downstream plotting notebook can
    trace any candidate against noise, not just whichever one won.

    Parameters
    ----------
    path  : str -- the three files above are written next to it
    rows  : list[dict] -- one per noise level. Required keys per row:
            `noise_frac`, `noise` (dict of injected/inferred noise
            statistics), `fit` (dict for the joint true-model fit),
            `candidates` ({name: dict}), `winners` ({criterion: name}),
            `delta` ({criterion: {name: Δ}}).
    truth : optional ground-truth dict (the `truth` from
            `make_dataset()`); the bulky `_meta`/`_chirp` keys are kept
            since they are small here and carry the per-record noise
            bookkeeping.
    meta  : optional dict merged into the JSON top level (candidate
            list, temperatures, `i_ref`, fit configuration, ...).

    The CSVs use the stdlib `csv` module -- no pandas dependency on
    either side -- and reload with `pandas.read_csv` or
    `numpy.genfromtxt`. The JSON reloads with `load_noise_sweep()`, or
    with a plain `json.load` on any machine without rheogp installed.
    """
    import csv
    import json

    rows = list(rows)
    truth = truth or {}
    true_model = (meta or {}).get("true_model")

    bundle = dict(
        tag=str(path),
        criteria=list(_SWEEP_CRITERIA),
        noise_levels=[float(r["noise_frac"]) for r in rows],
        truth={k: v for k, v in truth.items() if k != "_meta"},
        rows=rows,
    )
    if truth.get("_meta") is not None:
        bundle["truth_per_record"] = truth["_meta"]
    if meta:
        bundle.update(meta)

    with open(f"{path}.json", "w") as fh:
        json.dump(bundle, fh, indent=2, default=float)

    # ---- flat table 1: model selection, one row per candidate --------
    with open(f"{path}_models.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_SWEEP_MODEL_COLUMNS)
        w.writeheader()
        for r in rows:
            winners, delta = r["winners"], r["delta"]
            for name, c in r["candidates"].items():
                p, pf = c.get("params", {}), c.get("prefactors", {}).get("Gc", {})
                w.writerow({
                    "noise_frac": r["noise_frac"], "model": name,
                    "is_true_model": (name == true_model) if true_model else "",
                    "k": c["k"], "n_data": c["n_data"], "n_eff": c["n_eff"],
                    "mean_tau_int": c["mean_tau_int"],
                    "elbo_per_point": c["elbo_per_point"],
                    "neg2_loglik": c["neg2_loglik"],
                    "aic": c["aic"], "d_aic": delta["aic"][name],
                    "bic_raw": c["bic_raw"], "d_bic_raw": delta["bic_raw"][name],
                    "bic_eff_pen": c["bic_eff_pen"],
                    "d_bic_eff_pen": delta["bic_eff_pen"][name],
                    "bic_eff_full": c["bic_eff_full"],
                    "d_bic_eff_full": delta["bic_eff_full"][name],
                    "rmse": c["rmse"], "r2": c["r2"],
                    "alpha": p.get("alpha"), "beta": p.get("beta"),
                    "tau_c": p.get("tau_c"),
                    "Gc": pf.get("mean"), "Gc_lower": pf.get("lower"),
                    "Gc_upper": pf.get("upper"),
                    "noise_sd_inferred": c.get("noise_sd_inferred"),
                    "early_stopped": c.get("early_stopped"),
                    **{f"wins_{crit}": (winners[crit] == name)
                       for crit in _SWEEP_CRITERIA},
                })

    # ---- flat table 2: parameter recovery, one row per noise level ---
    with open(f"{path}_params.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_SWEEP_PARAM_COLUMNS)
        w.writeheader()
        for r in rows:
            fit, nz = r["fit"], r["noise"]
            p = fit.get("params", {})
            pf = fit.get("prefactors", {}).get("Gc", {})
            wlf = fit.get("wlf", {})
            a_t, t_t = truth.get("alpha"), truth.get("tau_ref")
            g_t = truth.get("Gc")
            w.writerow({
                "noise_frac": r["noise_frac"], "model": fit.get("model"),
                "alpha": p.get("alpha"), "alpha_truth": a_t,
                "alpha_err_pct": _pct_err(p.get("alpha"), a_t),
                "beta": p.get("beta"), "beta_truth": truth.get("beta"),
                "tau_c": p.get("tau_c"), "tau_c_truth": t_t,
                "tau_c_err_pct": _pct_err(p.get("tau_c"), t_t),
                "Gc": pf.get("mean"), "Gc_lower": pf.get("lower"),
                "Gc_upper": pf.get("upper"), "Gc_truth": g_t,
                "Gc_err_pct": _pct_err(pf.get("mean"), g_t),
                "C1": wlf.get("C1"), "C1_std": wlf.get("C1_std"),
                "C1_truth": truth.get("C1"),
                "C2": wlf.get("C2"), "C2_std": wlf.get("C2_std"),
                "C2_truth": truth.get("C2"),
                "rmse": fit.get("rmse"), "r2": fit.get("r2"),
                "aic": fit.get("aic"), "bic": fit.get("bic"),
                "k": fit.get("k"), "n_data": fit.get("n_data"),
                "noise_frac_nominal": nz.get("noise_frac_nominal"),
                "noise_sd_true": nz.get("sd_true"),
                "noise_sd_realized": nz.get("sd_realized"),
                "noise_sd_inferred": nz.get("sd_inferred"),
                "noise_frac_inferred": nz.get("frac_inferred"),
                "noise_sd_ratio": nz.get("sd_ratio"),
                **{f"winner_{crit}": r["winners"][crit]
                   for crit in _SWEEP_CRITERIA},
            })

    print(f"exported {len(rows)} noise levels -> {path}.json + "
          f"{path}_models.csv + {path}_params.csv")
    return f"{path}.json", f"{path}_models.csv", f"{path}_params.csv"


def load_noise_sweep(path):
    """
    Reload what `export_noise_sweep()` wrote -- pure json, NO
    rheogp/torch/pandas import required, so this runs on a machine that
    only has the plotting stack installed. Copy this function (and its
    `json` import) next to your exported data if you don't want the rest
    of `paper_utils.py`.

    Returns a `SimpleNamespace` with:
      .bundle       -- the full `<path>.json` dict
      .rows         -- list of per-noise-level dicts, ascending noise
      .noise_levels -- list[float]
      .truth        -- the ground-truth dict
      .criteria     -- ("aic", "bic_raw", "bic_eff_pen", "bic_eff_full")
      .series(*keys)        -- pull one nested scalar per noise level as
                               a numpy array, e.g.
                               `.series("fit", "params", "alpha")` or
                               `.series("noise", "sd_inferred")`
      .candidate_series(model, *keys)
                            -- same, but from one candidate's entry,
                               e.g. `.candidate_series("FractionalMaxwell",
                               "bic_eff_full")`
      .winners(criterion)   -- list[str], the winning model per noise level
    """
    import json
    from types import SimpleNamespace

    with open(f"{path}.json") as fh:
        bundle = json.load(fh)

    rows = sorted(bundle["rows"], key=lambda r: r["noise_frac"])
    noise_levels = [float(r["noise_frac"]) for r in rows]

    def _dig(d, keys):
        for k in keys:
            if d is None:
                return None
            d = d.get(k)
        return d

    def series(*keys):
        return np.array([_dig(r, keys) if _dig(r, keys) is not None else np.nan
                         for r in rows], dtype=float)

    def candidate_series(model, *keys):
        return np.array([_dig(r["candidates"].get(model), keys)
                         if _dig(r["candidates"].get(model), keys) is not None
                         else np.nan for r in rows], dtype=float)

    def winners(criterion):
        return [r["winners"][criterion] for r in rows]

    return SimpleNamespace(bundle=bundle, rows=rows,
                           noise_levels=noise_levels,
                           truth=bundle.get("truth", {}),
                           criteria=tuple(bundle.get("criteria", _SWEEP_CRITERIA)),
                           series=series, candidate_series=candidate_series,
                           winners=winners)


# ----------------------------------------------------------------
# candidate model comparison (real data): export / reload
# ----------------------------------------------------------------
_MC_COLUMNS = [
    "model", "k", "n_data", "n_eff", "mean_tau_int", "elbo_per_point",
    "neg2_loglik", "aic", "d_aic", "bic_raw", "d_bic_raw",
    "bic_eff_pen", "d_bic_eff_pen", "bic_eff_full", "d_bic_eff_full",
    "rmse", "r2", "noise_sd_inferred", "n_epochs_ran", "early_stopped",
    "wins_aic", "wins_bic_raw", "wins_bic_eff_pen", "wins_bic_eff_full",
]


def inferred_noise_sd(m):
    """Observation-noise std, in stress units, that a fitted SPGP's GP
    likelihood learned. The GP trains on a standardized target, so its
    likelihood variance is in standardized units; multiplying by the
    target scaler's scale puts it back into Pa."""
    nz = m.likelihood_.noise.detach().cpu().numpy().reshape(-1)[0]
    return float(np.sqrt(float(nz)) * m.scaler_y_.scale_[0])


def export_model_comparison(path, curves, temps=None, i_ref=None,
                            n_epochs_budget=None, extra=None):
    """
    Export a candidate-model comparison -- the `curves` dict that
    `compare_master_curve_models()` returns -- to `<path>.json` +
    `<path>_models.csv`.

    This is the real-data counterpart of `export_noise_sweep()`: same
    per-candidate schema (information criteria under the raw N and both
    N_eff conventions, parameter counts, recovered parameters,
    prefactors, shift factors, inferred observation noise, convergence),
    but with no noise axis and no ground truth, so a plotting notebook
    can render the model-selection figure without refitting anything.

    Every candidate's parameters are exported, not just the winner's --
    the winner's NAME is the only thing selection actually decides, and
    seeing what the runners-up recovered is usually what tells you
    whether the decision was real.

    `n_epochs_budget` (e.g. `FITKW["n_epochs"]`) lets the bundle record
    whether each candidate early-stopped or hit its budget; a candidate
    that hit the budget has provisional criteria.
    """
    import json
    from rheogp.utils import effective_sample_size_segments

    cand = {}
    for name, mc in curves.items():
        mj = mc.model_
        segs = [mj.stress_[lo:hi] - mj.pred_[lo:hi] for lo, hi in mj._bounds]
        n_eff, n_raw, per_seg = effective_sample_size_segments(segs)
        m = mc.metrics_
        k = int(m["n_params"]["k_total"])
        elbo = float(m["elbo_per_point"])
        n_ran = len(mj.history_.get("loss", []))
        cand[name] = dict(
            model=name, k=k, n_data=int(m["n_data"]),
            n_raw=int(n_raw), n_eff=float(n_eff),
            mean_tau_int=float(np.mean([t for _, t, _ in per_seg])),
            per_record_tau_int=[float(t) for _, t, _ in per_seg],
            elbo_per_point=elbo,
            neg2_loglik=float(2.0 * m["nll"]),
            aic=float(m["aic"]), bic_raw=float(m["bic"]),
            bic_eff_pen=float(2.0 * n_raw * elbo + k * np.log(n_eff)),
            bic_eff_full=float(2.0 * n_eff * elbo + k * np.log(n_eff)),
            rmse=float(m["rmse"]), r2=float(m["r2"]),
            params={kk: float(vv) for kk, vv
                    in mj.gp_model_.phys.named_phys_params().items()},
            prefactors={kk: dict(mean=float(v["scalar_mean"]),
                                 lower=float(v["scalar_lower"]),
                                 upper=float(v["scalar_upper"]))
                        for kk, v in mj.prefactors_.items()},
            aT=[float(a) for a in mj.shift_factors()],
            noise_sd_inferred=inferred_noise_sd(mj),
            n_epochs_ran=int(n_ran),
            n_epochs_budget=(int(n_epochs_budget) if n_epochs_budget else None),
            early_stopped=(bool(n_ran < n_epochs_budget)
                           if n_epochs_budget else None),
        )

    criteria = ["aic", "bic_raw", "bic_eff_pen", "bic_eff_full"]
    winners = {c: min(cand, key=lambda n: cand[n][c]) for c in criteria}
    delta = {c: {n: cand[n][c] - cand[winners[c]][c] for n in cand}
             for c in criteria}

    meta = dict(tag=str(path), criteria=criteria, candidates=list(cand),
                winners=winners, delta=delta, candidate_results=cand)
    if temps is not None:
        meta["temperatures"] = [float(T) for T in temps]
    if i_ref is not None:
        meta["i_ref"] = int(i_ref)
    if extra:
        meta.update(extra)

    with open(f"{path}.json", "w") as fh:
        json.dump(meta, fh, indent=2, default=float)

    import csv
    with open(f"{path}_models.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_MC_COLUMNS)
        w.writeheader()
        for name in sorted(cand, key=lambda n: cand[n]["bic_raw"]):
            c = cand[name]
            w.writerow({**{k: c.get(k) for k in _MC_COLUMNS if k in c},
                        **{f"d_{cr}": delta[cr][name] for cr in criteria},
                        **{f"wins_{cr}": winners[cr] == name for cr in criteria}})

    print(f"exported {len(cand)} candidate(s) -> {path}.json + {path}_models.csv")
    return f"{path}.json", f"{path}_models.csv"


def load_model_comparison(path):
    """Reload `export_model_comparison()` -- pure json + numpy, no
    rheogp/torch needed."""
    import json
    from types import SimpleNamespace
    with open(f"{path}.json") as fh:
        meta = json.load(fh)
    return SimpleNamespace(
        meta=meta,
        candidates=meta["candidates"],
        results=meta["candidate_results"],
        winners=meta["winners"],
        delta=meta["delta"],
        criteria=meta.get("criteria",
                          ["aic", "bic_raw", "bic_eff_pen", "bic_eff_full"]),
    )
