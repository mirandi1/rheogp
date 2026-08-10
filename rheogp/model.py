"""
rheogp.model
============
Public-facing SPGP class.

Quick start
-----------
>>> from rheogp import SPGP
>>> m = SPGP(model="Springpot")                     # strain-controlled,
>>> m.fit(t, strain, stress)                        # causal features
>>> res = m.results()                               # everything as numpy
>>> res.save("outputs/springpot")

Stress-controlled chirp
-----------------------
>>> m = SPGP(model="Springpot", control="stress")
>>> m.fit(t, strain, stress)          # same call — roles swap internally

Steady-state (from -inf) features
---------------------------------
>>> m = SPGP(model="Springpot", convolution="steady")
>>> m.fit(t, strain, stress)

Supported models (case-insensitive, spaces/dashes ignored)
----------------------------------------------------------
  "Maxwell", "Springpot",
  "FractionalMaxwellGel"/"FMG", "FractionalMaxwellLiquid"/"FML",
  "FractionalMaxwell"/"FMM",
  "FractionalKelvinVoigtS"/"FKVS", "FractionalKelvinVoigtD"/"FKVD",
  "FractionalKelvinVoigt"/"FKV", "KernelFree"/"KF"
"""

import warnings

import numpy as np
import torch
import gpytorch
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from .kernels import (
    KERNELS, PREFACTOR_COLS, KernelFreeKernel,
    MaxwellKernel, SpringpotKernel,
    FractionalMaxwellGelKernel, FractionalMaxwellLiquidKernel,
    FractionalMaxwellKernel,
    FKVSKernel, FKVDKernel, FractionalKelvinVoigtKernel,
)
from .gp import RheoGPModel
from .results import FitResult
from . import plots

__all__ = ["SPGP", "fit"]


def _normalise_key(name: str) -> str:
    return name.lower().replace(" ", "").replace("-", "").replace("_", "")


class SPGP:
    """
    Physics-Informed Sparse Gaussian Process for Rheology.

    Parameters
    ----------
    model : str
        Rheological model name (see module docstring).
    control : {"strain", "stress"}
        "strain": features filter the strain history, GP fits the stress
        (relaxation representation, default).
        "stress": features filter the stress history, GP fits the strain
        (creep representation — for stress-controlled chirps).
    convolution : {"causal", "steady"}
        "causal": convolution from 0 — material at rest before t = 0.
        "steady": convolution from -inf — periodic steady state, computed
        exactly in frequency space (no start-up transient).  Best for
        optimally-windowed chirps that taper to zero at both ends.
    quadrature : {"exact", "midpoint", "left"}
        Causal-mode discretisation.  "exact" integrates singular
        power-law kernels analytically over each bin (recommended —
        removes the start-up transient of the legacy rule).
        "left" reproduces rheogp <= 0.2 numbers exactly.
    kernel : str
        GP covariance: "rbf" | "matern12" | "matern32" | "matern52" |
        "linear" | "rbf+linear" (two-phase, recommended for prefactors).
    inducing_points, learning_rate, learning_rate_physics, n_epochs,
    patience, n_epochs_linear, patience_linear, learning_rate_linear :
        Training controls (as in v0.2).
    omega_min, omega_max, n_omega :
        Frequency grid for G*(ω) reconstruction.
    alpha_init, beta_init, tau_c_init : float or None
        Initial values for the shape parameters.
    num_posterior_samples : int
    seed : int
    device : str or None
    verbose : bool
    """

    def __init__(
        self,
        model                 = "Springpot",
        control               = "strain",
        convolution           = "causal",
        quadrature            = "exact",
        kernel                = "rbf+linear",
        inducing_points       = 300,
        learning_rate         = 0.005,
        learning_rate_physics = 0.005,
        n_epochs              = 10_000,
        patience              = 500,
        n_epochs_linear       = 3_000,
        patience_linear       = 200,
        learning_rate_linear  = 0.01,
        omega_min             = 1e-3,
        omega_max             = 1e3,
        n_omega               = 400,
        alpha_init            = None,
        beta_init             = None,
        tau_c_init            = None,
        use_strain_rate       = True,   # KernelFree only
        num_posterior_samples = 40,
        seed                  = 42,
        device                = None,
        verbose               = True,
    ):
        self.model_name  = model
        self.control     = control
        self.convolution = convolution
        self.quadrature  = quadrature
        self.kernel_name = kernel
        self.inducing_points       = inducing_points
        self.learning_rate         = learning_rate
        self.learning_rate_physics = learning_rate_physics
        self.n_epochs              = n_epochs
        self.patience              = patience
        self.n_epochs_linear       = n_epochs_linear
        self.patience_linear       = patience_linear
        self.learning_rate_linear  = learning_rate_linear
        self.omega_min = omega_min
        self.omega_max = omega_max
        self.n_omega   = n_omega
        self.alpha_init = alpha_init
        self.beta_init  = beta_init
        self.tau_c_init = tau_c_init
        self.use_strain_rate       = use_strain_rate
        self.num_posterior_samples = num_posterior_samples
        self.seed    = seed
        self.verbose = verbose
        self._two_phase = kernel.lower() == "rbf+linear"

        self.device = (torch.device("cuda" if torch.cuda.is_available()
                                    else "cpu")
                       if device is None else torch.device(device))

        self._is_fitted  = False
        self.gp_model_   = None
        self.likelihood_ = None
        self.scaler_y_   = None
        self.dt_ = self.t_ = None
        self.strain_ = self.stress_ = None
        self.input_ = self.target_ = None
        self.pred_ = self.pred_std_ = None
        self.prefactors_ = None
        self.metrics_ = {}
        self.history_ = {}

    # ============================================================
    # fit
    # ============================================================
    def fit(self, time, strain, stress):
        """
        Fit the model.  The same call is used for both control modes:

          control="strain": features are built from `strain`,
                            the GP fits `stress`  (σ = f(x[γ])).
          control="stress": features are built from `stress`,
                            the GP fits `strain`  (γ = f(x[σ])).

        Parameters
        ----------
        time   : array-like (N,)  [s]
        strain : array-like (N,)  [-]
        stress : array-like (N,)  [Pa]
        """
        self._seed()

        time   = np.asarray(time,   dtype=np.float64)
        strain = np.asarray(strain, dtype=np.float64)
        stress = np.asarray(stress, dtype=np.float64)
        if not (time.shape == strain.shape == stress.shape):
            raise ValueError("time, strain, stress must have the same shape.")

        self.t_, self.strain_, self.stress_ = time, strain, stress
        self.dt_ = float(np.mean(np.diff(time)))

        if self.control == "strain":
            self.input_, self.target_ = strain, stress
        else:
            self.input_, self.target_ = stress, strain

        self.scaler_y_ = StandardScaler()
        y_np    = self.scaler_y_.fit_transform(
                      self.target_.reshape(-1, 1)).flatten()
        y_torch = torch.tensor(y_np, dtype=torch.float32, device=self.device)
        u_torch = torch.tensor(self.input_, dtype=torch.float32,
                               device=self.device)

        phys = self._build_phys_kernel().to(self.device)

        self.gp_model_ = RheoGPModel(
            phys, u_torch, self.dt_,
            num_inducing=self.inducing_points,
            gp_kernel_name=self.kernel_name,
        ).to(self.device)
        self.likelihood_ = gpytorch.likelihoods.GaussianLikelihood(
                           ).to(self.device)

        self._train(y_torch, self.target_)
        self.prefactors_ = self._extract_prefactors()
        self.pred_std_   = self._posterior_std()
        self._is_fitted  = True
        return self

    # ============================================================
    # predict
    # ============================================================
    def predict(self, input_signal, time=None, return_std=False,
                convolution=None):
        """
        Predict the target for a new input history.

        control="strain": input_signal is a strain history → returns σ(t)
        control="stress": input_signal is a stress history → returns γ(t)

        Parameters
        ----------
        input_signal : array-like (M,)
        time         : array-like (M,) or None — used for dt only
        return_std   : bool
        convolution  : None | "causal" | "steady"
            Override the feature mode for this prediction.  A trained
            "steady" model evaluated on a non-periodic protocol (e.g.
            a step strain) should use convolution="causal" here.
        """
        self._check_fitted()
        u  = np.asarray(input_signal, dtype=np.float64)
        dt = (float(np.mean(np.diff(time))) if time is not None
              else self.dt_)

        phys = self.gp_model_.phys
        old_mode = phys.convolution
        if convolution is not None:
            phys.convolution = convolution
        elif old_mode == "steady":
            warnings.warn(
                "Predicting with steady (periodic) features: the new "
                "input is assumed periodic. For step/one-shot protocols "
                "pass convolution='causal'.")

        u_t = torch.tensor(u, dtype=torch.float32, device=self.device)
        old_u, old_dt = self.gp_model_.gamma.clone(), self.gp_model_.dt
        self.gp_model_.gamma, self.gp_model_.dt = u_t, dt

        phys._gamma_scale_frozen  = True
        phys._feature_stds_frozen = True

        self.gp_model_.eval(); self.likelihood_.eval()
        with torch.no_grad(), gpytorch.settings.cholesky_jitter(1e-6):
            x    = self.gp_model_.compute_x()
            dist = self.likelihood_(self.gp_model_(x))
            mu   = dist.mean.cpu().numpy()
            var  = dist.variance.cpu().numpy()

        phys._gamma_scale_frozen  = False
        phys._feature_stds_frozen = False
        phys.convolution = old_mode
        self.gp_model_.gamma, self.gp_model_.dt = old_u, old_dt

        sy, sm = self.scaler_y_.scale_[0], self.scaler_y_.mean_[0]
        pred   = mu * sy + sm
        std    = np.sqrt(np.maximum(var, 0.0)) * sy
        return (pred, std) if return_std else pred

    # ============================================================
    # results — everything as numpy, for external plotting
    # ============================================================
    def results(self):
        """Return a `FitResult` with every fitted quantity."""
        self._check_fitted()
        phys = self.gp_model_.phys

        # physical features on the raw input
        u_t = torch.tensor(self.input_, dtype=torch.float32,
                           device=self.device)
        with torch.no_grad():
            feats = phys.compute_x_physical(u_t, self.dt_).cpu().numpy()

        # model G*
        omega = np.logspace(np.log10(self.omega_min),
                            np.log10(self.omega_max), self.n_omega)
        scalar_pf = {k: v["scalar_mean"] for k, v in self.prefactors_.items()}
        try:
            Gp, Gdp = phys.Gstar(omega, scalar_pf)
        except NotImplementedError:
            omega = Gp = Gdp = None

        # FFT G* from the raw chirp
        try:
            f_om, f_Gp, f_Gdp = self._chirp_fft_Gstar()
        except Exception:
            f_om = f_Gp = f_Gdp = None

        return FitResult(
            model=self.model_name, gp_kernel=self.kernel_name,
            control=self.control, convolution=self.convolution,
            quadrature=self.quadrature,
            t=self.t_, strain=self.strain_, stress=self.stress_,
            input=self.input_, target=self.target_,
            prediction=self.pred_, prediction_std=self.pred_std_,
            features=feats, feature_names=phys.prefactor_names(),
            sensitivities=self.prefactors_,
            phys_params=phys.named_phys_params(),
            derived_params=self._derived_params(),
            metrics=dict(self.metrics_),
            omega=omega, Gp=Gp, Gdp=Gdp,
            fft_omega=f_om, fft_Gp=f_Gp, fft_Gdp=f_Gdp,
            history={k: (dict(v) if isinstance(v, dict) else list(v))
                     for k, v in self.history_.items()},
        )

    def get_features(self, input_signal=None, time=None, physical=True):
        """
        Convolutional features for plotting.

        physical=True  → raw physical units (what multiplies the
                         prefactors in the constitutive law).
        physical=False → the standardised GP inputs.
        """
        self._check_fitted()
        u  = (self.input_ if input_signal is None
              else np.asarray(input_signal, dtype=np.float64))
        dt = (self.dt_ if time is None
              else float(np.mean(np.diff(time))))
        u_t = torch.tensor(u, dtype=torch.float32, device=self.device)
        phys = self.gp_model_.phys
        with torch.no_grad():
            if physical:
                x = phys.compute_x_physical(u_t, dt)
            else:
                phys._gamma_scale_frozen = phys._feature_stds_frozen = True
                x = phys.compute_x(u_t, dt)
                phys._gamma_scale_frozen = phys._feature_stds_frozen = False
        return x.cpu().numpy()

    # ============================================================
    # summary
    # ============================================================
    def summary(self):
        self._check_fitted()
        p = self.gp_model_.phys.named_phys_params()
        y_unit = "Pa" if self.control == "strain" else "-"
        lines = [
            "=" * 60,
            f"  RheoGP  —  {self.model_name}",
            "=" * 60,
            f"  Control        : {self.control}-controlled "
            f"({'relaxation' if self.control == 'strain' else 'creep'} "
            f"representation)",
            f"  Convolution    : {self.convolution}"
            + (f"  (quadrature: {self.quadrature})"
               if self.convolution == "causal" else "  (from -inf, periodic)"),
            f"  GP kernel      : {self.kernel_name}",
            f"  Training mode  : "
            f"{'two-phase (RBF → Linear)' if self._two_phase else 'single-phase'}",
            f"  Inducing pts   : {self.inducing_points}",
            "",
            "  Learned shape parameters",
            "  ------------------------",
        ]
        if p:
            for k, v in p.items():
                lines.append(f"    {k:<10} = {v:.6f}")
        else:
            lines.append("    (none — fully determined by prefactors)")

        lines += ["", "  Prefactors / sensitivities  (mean  [95% CI])",
                  "  --------------------------------------------"]
        for name, info in self.prefactors_.items():
            lines.append(
                f"    {name:<8} = {info['scalar_mean']:>12.6g}"
                f"  [{info['scalar_lower']:>12.6g},"
                f" {info['scalar_upper']:>12.6g}]")

        derived = self._derived_params()
        if derived:
            lines += ["", "  Derived physical parameters",
                      "  ---------------------------"]
            for k, v in derived.items():
                lines.append(f"    {k:<10} = {v:.6g}")

        np_ = self.metrics_.get("n_params", {})
        lines += [
            "", "  Fit metrics", "  -----------",
            f"    RMSE  = {self.metrics_['rmse']:.6g} {y_unit}",
            f"    R²    = {self.metrics_['r2']:.6f}",
            f"    NLL   = {self.metrics_['nll']:.2f}   "
            f"(variational bound, N = {self.metrics_['n_data']})",
            f"    AIC   = {self.metrics_['aic']:.2f}",
            f"    AICc  = {self.metrics_['aicc']:.2f}",
            f"    BIC   = {self.metrics_['bic']:.2f}",
            "",
            "  Parameters counted in AIC/BIC "
            f"(k = {np_.get('k_total', '?')})",
            "  " + "-" * 44,
            f"    viscoelastic shape params : {np_.get('n_shape', '?')}",
            f"    prefactors (sensitivities): {np_.get('n_prefactors', '?')}",
            f"    GP hyperparameters        : {np_.get('n_gp_hyper', '?')}",
            f"    [variational params       : {np_.get('n_variational', '?')}"
            "  — approximate posterior, excluded]",
            "=" * 60,
        ]
        return "\n".join(lines)

    # ============================================================
    # plotting (thin wrappers; use results() for custom figures)
    # ============================================================
    def plot_fit(self, ax=None, **kw):
        self._check_fitted()
        return plots.plot_fit(self.t_, self.target_, self.pred_,
                              self.model_name, self.metrics_, ax=ax,
                              ylabel=("σ [Pa]" if self.control == "strain"
                                      else "γ [-]"), **kw)

    def plot_prefactors(self, ax=None, **kw):
        self._check_fitted()
        return plots.plot_prefactors(self.t_, self.prefactors_,
                                     self.model_name, ax=ax, **kw)

    def plot_Gstar(self, ax=None, add_fft=False, fft_kwargs=None, **kw):
        self._check_fitted()
        if isinstance(self.gp_model_.phys, KernelFreeKernel):
            print("KernelFreeKernel has no G*(ω). "
                  "Use plot_prefactors() instead.")
            return
        res = self.results()
        return plots.plot_Gstar_result(res, ax=ax, add_fft=add_fft,
                                       fft_kwargs=fft_kwargs, **kw)

    def plot_convergence(self, **kw):
        self._check_fitted()
        return plots.plot_convergence(
            self.history_["loss"], dict(self.history_["params"]),
            self.model_name, **kw)

    # ============================================================
    # save
    # ============================================================
    def save(self, path):
        """Save all results (delegates to FitResult.save)."""
        self._check_fitted()
        root = self.results().save(path)
        with open(Path(root) / "summary.txt", "w") as f:
            f.write(self.summary())
        if self.verbose:
            print(f"  Results saved to {Path(root).resolve()}")
        return root

    # ============================================================
    # internals
    # ============================================================
    def _seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _build_phys_kernel(self):
        key = _normalise_key(self.model_name)
        cls = KERNELS.get(key)
        if cls is None:
            raise ValueError(f"Unknown model '{self.model_name}'.\n"
                             f"Available: {sorted(set(KERNELS.keys()))}")
        kwargs = {}
        varnames = cls.__init__.__code__.co_varnames
        if self.alpha_init is not None and "alpha_init" in varnames:
            kwargs["alpha_init"] = self.alpha_init
        if self.beta_init is not None and "beta_init" in varnames:
            kwargs["beta_init"] = self.beta_init
        if self.tau_c_init is not None and "tau_c_init" in varnames:
            kwargs["tau_c_init"] = self.tau_c_init
        if cls is KernelFreeKernel:
            kwargs["use_strain_rate"] = self.use_strain_rate
        phys = cls(**kwargs)
        phys.configure(control=self.control,
                       convolution=self.convolution,
                       quadrature=self.quadrature)
        return phys

    # ── training ─────────────────────────────────────────────────
    def _train(self, y_tensor, target_obs):
        if self._two_phase:
            self._train_phase1(y_tensor, target_obs)
            self._train_phase2(y_tensor, target_obs)
        else:
            self.history_ = {}
            self._train_loop(y_tensor, target_obs,
                             lr_gp=self.learning_rate,
                             lr_phys=self.learning_rate_physics,
                             n_epochs=self.n_epochs,
                             patience=self.patience,
                             phase_tag="", extra_params=None)

    @staticmethod
    def _count_parameters(model, lik):
        """
        Count model parameters for the information criteria.

        Included in ``k_total`` (all documented in ``metrics_['n_params']``):

        * ``n_shape``      — viscoelastic shape parameters
                             (α, β, τ_c — whatever the chosen kernel learns)
        * ``n_prefactors`` — one scalar per physical feature: the
                             prefactors (𝕍, G, η, or their inverses in
                             stress control) inferred through the GP
                             posterior sensitivities
        * ``n_gp_hyper``   — GP hyperparameters: mean constant, kernel
                             output-scale / lengthscales / variances and
                             the likelihood noise

        Excluded from ``k_total`` but reported for transparency:

        * ``n_variational`` — inducing-point variational parameters
                              (mean + Cholesky factor).  These
                              parametrise the *approximate posterior*,
                              not the model, so they do not enter
                              AIC/BIC (they play the role of the
                              posterior itself in an exact GP, which
                              contributes no parameters either).
        """
        n_shape      = len(model.phys.named_phys_params())
        n_prefactors = len(model.phys.prefactor_names())

        n_gp_hyper, n_variational = 0, 0
        for name, p in model.named_parameters():
            if name.startswith("variational_strategy"):
                n_variational += p.numel()
            elif name.startswith("phys."):
                continue                       # counted as n_shape above
            else:
                n_gp_hyper += p.numel()        # mean, scales, lengthscales
        for _, p in lik.named_parameters():
            n_gp_hyper += p.numel()            # observation noise

        return dict(n_shape=n_shape,
                    n_prefactors=n_prefactors,
                    n_gp_hyper=n_gp_hyper,
                    n_variational=n_variational,
                    k_total=n_shape + n_prefactors + n_gp_hyper)

    def _train_loop(self, y_tensor, target_obs, lr_gp, lr_phys,
                    n_epochs, patience, phase_tag, extra_params):
        model, lik = self.gp_model_, self.likelihood_
        mll = gpytorch.mlls.VariationalELBO(lik, model,
                                            num_data=y_tensor.numel())

        if extra_params is not None:
            opt = torch.optim.Adam(
                extra_params + list(lik.parameters()), lr=lr_gp)
        else:
            phys_params = list(model.phys.parameters())
            gp_params   = [p for n, p in model.named_parameters()
                           if not n.startswith("phys.")]
            opt = torch.optim.Adam([
                {"params": phys_params, "lr": lr_phys},
                {"params": gp_params,   "lr": lr_gp},
                {"params": list(lik.parameters()), "lr": lr_gp},
            ])

        model.train(); lik.train()
        best_loss, wait = np.inf, 0
        loss_h  = []
        param_h = {k: [] for k in model.phys.named_phys_params()}
        tag = f"[{self.model_name}{phase_tag}]"

        with gpytorch.settings.cholesky_jitter(1e-6):
            for i in range(n_epochs):
                opt.zero_grad()
                x    = model.compute_x()
                loss = -mll(model(x), y_tensor)
                loss.backward()
                opt.step()

                lv = loss.item()
                loss_h.append(lv)
                for k, v in model.phys.named_phys_params().items():
                    param_h[k].append(v)

                if lv < best_loss - 1e-4:
                    best_loss, wait = lv, 0
                else:
                    wait += 1

                if self.verbose and i % 200 == 0:
                    pstr = "  ".join(
                        f"{k}={v:.3f}" for k, v in
                        model.phys.named_phys_params().items())
                    print(f"  {tag} step {i:5d} | -ELBO {lv:.4f} | "
                          f"{pstr} | wait {wait}")
                if wait >= patience:
                    if self.verbose:
                        print(f"  {tag} early stopping at step {i}")
                    break

        if "loss" not in self.history_:
            self.history_ = {"loss": loss_h, "params": param_h}
        else:
            self.history_["loss"].extend(loss_h)
            for k in param_h:
                self.history_["params"].setdefault(k, []).extend(param_h[k])

        model.eval(); lik.eval()
        with torch.no_grad(), gpytorch.settings.cholesky_jitter(1e-6):
            xf     = model.compute_x()
            pred_s = lik(model(xf)).mean.cpu().numpy()

        sy, sm = self.scaler_y_.scale_[0], self.scaler_y_.mean_[0]
        self.pred_ = pred_s * sy + sm

        rmse = float(np.sqrt(np.mean((self.pred_ - target_obs)**2)))
        r2   = float(1 - np.sum((self.pred_ - target_obs)**2)
                       / np.sum((target_obs - target_obs.mean())**2))

        # ---- information criteria -------------------------------------
        # gpytorch's VariationalELBO is *per data point*, so the total
        # negative log-likelihood surrogate is  N * (-ELBO_per_point).
        # (The ELBO is a lower bound on log p(y); using it in place of
        #  the exact marginal likelihood is standard for variational GPs
        #  and makes the criteria conservative.)
        N_data = len(y_tensor)
        counts = self._count_parameters(model, lik)
        k      = counts["k_total"]
        nll    = float(N_data * best_loss)
        bic    = float(2.0 * nll + k * np.log(N_data))
        aic    = float(2.0 * nll + 2.0 * k)
        aicc   = float(aic + 2.0 * k * (k + 1) / max(N_data - k - 1, 1))

        self.metrics_ = dict(rmse=rmse, r2=r2,
                             nll=nll, elbo_per_point=float(best_loss),
                             aic=aic, aicc=aicc, bic=bic,
                             n_params=counts, n_data=N_data)

        if self.verbose:
            print(f"\n  {tag} RMSE={rmse:.4g}  R²={r2:.4f}  "
                  f"AIC={aic:.1f}  BIC={bic:.1f}  "
                  f"(k={k}: shape {counts['n_shape']} + prefactors "
                  f"{counts['n_prefactors']} + GP {counts['n_gp_hyper']})")
            print(f"  params: {model.phys.named_phys_params()}")
        return best_loss

    def _train_phase1(self, y_tensor, target_obs):
        if self.verbose:
            print("\n  ── Phase 1: RBF  (physics discovery) ──────────────")
        self.history_ = {}
        self._train_loop(y_tensor, target_obs,
                         lr_gp=self.learning_rate,
                         lr_phys=self.learning_rate_physics,
                         n_epochs=self.n_epochs, patience=self.patience,
                         phase_tag=" Ph1-RBF", extra_params=None)

    def _train_phase2(self, y_tensor, target_obs):
        if self.verbose:
            print("\n  ── Phase 2: Linear  (prefactor refinement) ─────────")
        trainable = self.gp_model_.swap_to_linear()
        self._train_loop(y_tensor, target_obs,
                         lr_gp=self.learning_rate_linear, lr_phys=None,
                         n_epochs=self.n_epochs_linear,
                         patience=self.patience_linear,
                         phase_tag=" Ph2-Linear", extra_params=trainable)

    # ── prefactor extraction ─────────────────────────────────────
    def _extract_prefactors(self):
        model, lik = self.gp_model_, self.likelihood_
        sig_y = self.scaler_y_.scale_[0]
        names = model.phys.prefactor_names()

        model.eval(); lik.eval()
        with torch.no_grad(), gpytorch.settings.cholesky_jitter(1e-6):
            _ = model.compute_x()
        stds = model.phys._feature_stds

        with gpytorch.settings.cholesky_jitter(1e-6), \
             gpytorch.settings.fast_pred_var():
            x_raw   = model.compute_x().detach().clone().requires_grad_(True)
            preds   = lik(model(x_raw))
            S       = self.num_posterior_samples
            samples = preds.rsample(torch.Size([S]))

            grads = []
            for s in range(S):
                g = torch.autograd.grad(
                    samples[s], x_raw,
                    grad_outputs=torch.ones_like(samples[s]),
                    retain_graph=True)[0]
                grads.append(g.detach())
            grads = torch.stack(grads).cpu().numpy()   # (S, N, n_feat)

        u_scale = getattr(model.phys, "_gamma_scale", 1.0)
        if torch.is_tensor(u_scale):
            u_scale = u_scale.item()

        out = {}
        for col, name in enumerate(names):
            phys = grads[:, :, col] * (sig_y / stds[col]) / u_scale

            mean  = phys.mean(axis=0)
            lower = np.quantile(phys, 0.025, axis=0)
            upper = np.quantile(phys, 0.975, axis=0)

            m = np.isfinite(phys) & (phys > 0)
            if m.any():
                pos = phys[m]
                scalar_mean  = float(np.mean(pos))
                scalar_lower = float(np.quantile(pos, 0.025))
                scalar_upper = float(np.quantile(pos, 0.975))
            else:
                scalar_mean = scalar_lower = scalar_upper = float("nan")

            out[name] = dict(samples=phys, mean=mean, lower=lower,
                             upper=upper, scalar_mean=scalar_mean,
                             scalar_lower=scalar_lower,
                             scalar_upper=scalar_upper)
            if self.verbose:
                print(f"  {name}: {scalar_mean:.6g}  "
                      f"95%CI [{scalar_lower:.6g}, {scalar_upper:.6g}]")
        return out

    def _posterior_std(self):
        model, lik = self.gp_model_, self.likelihood_
        try:
            model.eval(); lik.eval()
            with torch.no_grad(), gpytorch.settings.cholesky_jitter(1e-6):
                xf   = model.compute_x()
                var  = lik(model(xf)).variance.cpu().numpy()
            return np.sqrt(np.maximum(var, 0.0)) * self.scaler_y_.scale_[0]
        except Exception as e:
            warnings.warn(f"Could not compute posterior std: {e}")
            return np.full_like(self.pred_, np.nan)

    # ── derived physical parameters ──────────────────────────────
    def _derived_params(self):
        phys = self.gp_model_.phys
        pf   = {k: v["scalar_mean"] for k, v in self.prefactors_.items()}
        out  = {}

        if self.control == "stress":
            try:
                out.update(phys._stress_prefactors_to_moduli(pf))
            except (NotImplementedError, KeyError, ZeroDivisionError):
                pass
            return {k: float(v) for k, v in out.items()
                    if np.isfinite(v)}

        # strain control — same relations as v0.2
        p   = phys.named_phys_params()
        cls = type(phys)
        if cls is MaxwellKernel:
            out.update(tau_c=p["tau_c"], Gc=pf.get("Gc", np.nan))
        elif cls is FractionalMaxwellGelKernel:
            tc, a = p["tau_c"], p["alpha"]; Gc = pf.get("Gc", np.nan)
            out.update(tau_c=tc, Gc=Gc, V=Gc * tc**a, G=Gc)
        elif cls is FractionalMaxwellLiquidKernel:
            out.update(tau_c=p["tau_c"], Gc=pf.get("Gc", np.nan))
        elif cls is FractionalMaxwellKernel:
            tc, a, b = p["tau_c"], p["alpha"], p["beta"]
            Gc = pf.get("Gc", np.nan)
            out.update(tau_c=tc, Gc=Gc, V=Gc * tc**a, G_bb=Gc * tc**b)
        elif cls in (FKVSKernel, FractionalKelvinVoigtKernel):
            V    = pf.get("V", np.nan)
            G_bb = pf.get("G_bb", pf.get("G", np.nan))
            a    = p.get("alpha", np.nan)
            b    = p.get("beta", 0.0)
            if np.isfinite(V) and np.isfinite(G_bb) and G_bb > 0 \
               and (a - b) > 0:
                tc = float((V / G_bb) ** (1.0 / (a - b)))
                out.update(tau_c=tc, Gc=float(V * tc ** (-a)))
        elif cls is FKVDKernel:
            G_bb, eta = pf.get("G_bb", np.nan), pf.get("eta", np.nan)
            b = p.get("beta", np.nan)
            if np.isfinite(G_bb) and np.isfinite(eta) and G_bb > 0 \
               and (1 - b) > 0:
                out["tau_c"] = float((eta / G_bb) ** (1.0 / (1.0 - b)))
        return {k: float(v) for k, v in out.items() if np.isfinite(v)}

    # ── FFT reference estimate ───────────────────────────────────
    def _chirp_fft_Gstar(self, eps=1e-12):
        t, e, s = self.t_, self.strain_, self.stress_
        N  = len(t)
        dt = float(np.median(np.diff(t)))
        E, S = np.fft.rfft(e), np.fft.rfft(s)
        f = np.fft.rfftfreq(N, d=dt)
        omega = 2 * np.pi * f
        magE = np.abs(E)
        m = (f > 0) & (magE > eps * magE.max())
        Gstar = S[m] / (E[m] + 0j)
        return omega[m], np.real(Gstar), np.imag(Gstar)

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Call fit() before using this method.")


# ================================================================
# One-line convenience API
# ================================================================

def fit(time, strain, stress, model="Springpot", **kwargs):
    """
    One-liner:  ``res_model = rheogp.fit(t, γ, σ, model="FKV",
    control="stress", convolution="steady")``.

    Returns the fitted SPGP instance; call ``.results()`` on it for
    all arrays, ``.summary()`` for a report.
    """
    return SPGP(model=model, **kwargs).fit(time, strain, stress)
