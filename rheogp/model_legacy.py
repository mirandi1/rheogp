"""
rheogp.model
============
Public-facing SPGP class.

Supported models (case-insensitive, spaces ignored)
----------------------------------------------------
  "Maxwell"
  "Springpot"
  "FractionalMaxwellGel"    / "FMG"
  "FractionalMaxwellLiquid" / "FML"
  "FractionalMaxwell"       / "FMM"
  "FractionalKelvinVoigtS"  / "FKVS"
  "FractionalKelvinVoigtD"  / "FKVD"
  "FractionalKelvinVoigt"   / "FKV"

Training modes
--------------
  kernel="rbf"          — single-phase, RBF kernel (default)
  kernel="matern32"     — single-phase, Matérn 3/2
  kernel="rbf+linear"   — two-phase:
                            Phase 1 (RBF)    : optimise α, β, τ_c + GP hypers
                            Phase 2 (Linear) : freeze physics, retrain GP only
                          The LinearKernel enforces the constitutively correct
                          linear σ = Σⱼ θⱼ xⱼ relationship exactly and gives
                          cleaner prefactor recovery.
"""

import numpy as np
import torch
import gpytorch
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler

class DummyScaler:
    def fit(self, X, y=None):
        X = np.asarray(X)
        self.mean_ = np.zeros(X.shape[1])
        self.scale_ = np.ones(X.shape[1])
        self.var_ = np.ones(X.shape[1])
        self.n_samples_seen_ = X.shape[0]
        return self

    def transform(self, X):
        return np.asarray(X)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        return np.asarray(X)

from .kernels import (
    KERNELS, PREFACTOR_COLS, KernelFreeKernel,
    MaxwellKernel, SpringpotKernel,
    FractionalMaxwellGelKernel, FractionalMaxwellLiquidKernel,
    FractionalMaxwellKernel,
    FKVSKernel, FKVDKernel, FractionalKelvinVoigtKernel,
)
from .gp   import RheoGPModel
from .     import plots

__all__ = ["SPGP"]

_DEFAULTS = {
    KernelFreeKernel: dict(use_strain_rate=True),
    MaxwellKernel:                 dict(tau_c_init=1.0),
    SpringpotKernel:               dict(alpha_init=0.5),
    FractionalMaxwellGelKernel:    dict(alpha_init=0.5, tau_c_init=1.0),
    FractionalMaxwellLiquidKernel: dict(beta_init=0.5,  tau_c_init=1.0),
    FractionalMaxwellKernel:       dict(alpha_init=0.7, beta_init=0.3, tau_c_init=1.0),
    FKVSKernel:                    dict(alpha_init=0.5),
    FKVDKernel:                    dict(beta_init=0.5),
    FractionalKelvinVoigtKernel:   dict(alpha_init=0.7, beta_init=0.3, #tau_c_init=1.0
                                       ),
}


def _normalise_key(name: str) -> str:
    return name.lower().replace(" ", "").replace("-", "").replace("_", "")


class SPGP:
    """
    Physics-Informed Sparse Gaussian Process for Rheology.

    Parameters
    ----------
    model : str
        Rheological model name.
    kernel : str
        GP covariance kernel.
        "rbf" | "matern12" | "matern32" | "matern52" | "linear" | "rbf+linear"
        Use "rbf+linear" for two-phase training (recommended).
    inducing_points : int
        Variational inducing points. Default 300.
    learning_rate : float
        Adam lr for GP hyperparameters. Default 0.005.
    learning_rate_physics : float
        Adam lr for physics parameters (α, β, τ_c). Default 0.05.
        A higher rate here helps physics converge faster than GP hypers.
    n_epochs : int
        Max training steps (Phase 1 or single-phase). Default 10 000.
    patience : int
        Early-stopping patience. Default 500.
    n_epochs_linear : int
        Max steps for Phase 2 (LinearKernel). Default 3 000.
    patience_linear : int
        Early-stopping patience for Phase 2. Default 200.
    learning_rate_linear : float
        Adam lr for Phase 2. Can be higher since it is a convex problem.
        Default 0.01.
    omega_min, omega_max : float
        Frequency range [rad/s] for G*(ω) plots.
    n_omega : int
        Number of frequency points. Default 400.
    alpha_init, beta_init, tau_c_init : float or None
    num_posterior_samples : int  Default 40.
    seed : int
    device : str or None
    verbose : bool
    """

    def __init__(
        self,
        use_strain_rate=True,
        model                 = "Springpot",
        kernel                = "matern32",
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
        feature               = 'strain',   # ← add this
        num_posterior_samples = 40,
        seed                  = 42,
        device                = None,
        verbose               = True,
    ):
        self.model_name           = model
        self.feature = feature   # for KernelFreeKernel
        self.kernel_name          = kernel
        self.inducing_points      = inducing_points
        self.learning_rate        = learning_rate
        self.learning_rate_physics= learning_rate_physics
        self.n_epochs             = n_epochs
        self.patience             = patience
        self.n_epochs_linear      = n_epochs_linear
        self.patience_linear      = patience_linear
        self.learning_rate_linear = learning_rate_linear
        self.omega_min            = omega_min
        self.omega_max            = omega_max
        self.n_omega              = n_omega
        self.alpha_init           = alpha_init
        self.beta_init            = beta_init
        self.tau_c_init           = tau_c_init
        self.num_posterior_samples= num_posterior_samples
        self.seed                 = seed
        self.verbose              = verbose
        self._two_phase           = kernel.lower() == "rbf+linear"

        self.device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
                       if device is None else torch.device(device))

        self._is_fitted   = False
        self.gp_model_    = None
        self.likelihood_  = None
        self.scaler_y_    = None
        self.dt_          = None
        self.t_           = None
        self.strain_      = None
        self.stress_      = None
        self.pred_        = None
        self.prefactors_  = None
        self.metrics_     = {}
        self.history_     = {}

    # ============================================================
    # fit
    # ============================================================
    def fit(self, time, strain, stress):
        """
        Fit the model to strain-controlled data.

        Parameters
        ----------
        time   : array-like (N,)  [s]
        strain : array-like (N,)
        stress : array-like (N,)  [Pa]
        """
        self._seed()

        time   = np.asarray(time,   dtype=np.float64)
        strain = np.asarray(strain, dtype=np.float64)
        stress = np.asarray(stress, dtype=np.float64)
        if not (time.shape == strain.shape == stress.shape):
            raise ValueError("time, strain, stress must have the same shape.")

        self.t_      = time
        self.strain_ = strain
        self.stress_ = stress
        self.dt_     = float(np.mean(np.diff(time)))

        self.scaler_y_ = StandardScaler()
        y_np    = self.scaler_y_.fit_transform(stress.reshape(-1, 1)).flatten()
        y_torch = torch.tensor(y_np,    dtype=torch.float32, device=self.device)
        g_torch = torch.tensor(strain,  dtype=torch.float32, device=self.device)

        phys = self._build_phys_kernel().to(self.device)

        self.gp_model_  = RheoGPModel(
            phys, g_torch, self.dt_,
            num_inducing   = self.inducing_points,
            gp_kernel_name = self.kernel_name,
        ).to(self.device)
        self.likelihood_ = gpytorch.likelihoods.GaussianLikelihood().to(self.device)

        self._train(y_torch, stress)
        self.prefactors_ = self._extract_prefactors()
        self._is_fitted  = True
        return self

    # ============================================================
    # predict
    # ============================================================
    '''
    def predict(self, strain, time=None, return_std=False):
        """
        Predict stress for new strain input.

        Parameters
        ----------
        strain     : array-like (N,)
        time       : array-like (N,) or None
        return_std : bool
        """
        self._check_fitted()
        strain = np.asarray(strain, dtype=np.float64)
        dt     = (float(np.mean(np.diff(time)))
                  if time is not None else self.dt_)

        g_t = torch.tensor(strain, dtype=torch.float32, device=self.device)

        old_g, old_dt        = self.gp_model_.gamma.clone(), self.gp_model_.dt
        self.gp_model_.gamma = g_t
        self.gp_model_.dt    = dt

        self.gp_model_.eval(); self.likelihood_.eval()
        with torch.no_grad(), gpytorch.settings.cholesky_jitter(1e-6):
            x    = self.gp_model_.compute_x()
            dist = self.likelihood_(self.gp_model_(x))
            mu   = dist.mean.cpu().numpy()
            var  = dist.variance.cpu().numpy()

        self.gp_model_.gamma, self.gp_model_.dt = old_g, old_dt

        sy, sm = self.scaler_y_.scale_[0], self.scaler_y_.mean_[0]
        pred   = mu * sy + sm
        std    = np.sqrt(np.maximum(var, 0.0)) * sy
        return (pred, std) if return_std else pred
        '''
    def predict(self, strain, time=None, return_std=False):
        self._check_fitted()
        strain = np.asarray(strain, dtype=np.float64)
        dt     = (float(np.mean(np.diff(time)))
                  if time is not None else self.dt_)

        g_t = torch.tensor(strain, dtype=torch.float32, device=self.device)

        old_g, old_dt        = self.gp_model_.gamma.clone(), self.gp_model_.dt
        self.gp_model_.gamma = g_t
        self.gp_model_.dt    = dt

        # ── freeze training scalers so new input uses same normalisation ──
        self.gp_model_.phys._gamma_scale_frozen  = True
        self.gp_model_.phys._feature_stds_frozen = True

        self.gp_model_.eval(); self.likelihood_.eval()
        with torch.no_grad(), gpytorch.settings.cholesky_jitter(1e-6):
            x    = self.gp_model_.compute_x()
            dist = self.likelihood_(self.gp_model_(x))
            mu   = dist.mean.cpu().numpy()
            var  = dist.variance.cpu().numpy()

        # ── unfreeze so retraining works correctly ────────────────────────
        self.gp_model_.phys._gamma_scale_frozen  = False
        self.gp_model_.phys._feature_stds_frozen = False

        self.gp_model_.gamma, self.gp_model_.dt = old_g, old_dt

        sy, sm = self.scaler_y_.scale_[0], self.scaler_y_.mean_[0]
        pred   = mu * sy + sm
        std    = np.sqrt(np.maximum(var, 0.0)) * sy
        return (pred, std) if return_std else pred

    # ============================================================
    # summary
    # ============================================================
    def summary(self):
        self._check_fitted()
        p = self.gp_model_.phys.named_phys_params()
        lines = [
            "=" * 58,
            f"  RheoGP  —  {self.model_name}",
            "=" * 58,
            f"  GP kernel      : {self.kernel_name}",
            f"  Training mode  : {'two-phase (RBF → Linear)' if self._two_phase else 'single-phase'}",
            f"  Inducing pts   : {self.inducing_points}",
            "",
            "  Learned physics parameters",
            "  --------------------------",
        ]
        for k, v in p.items():
            lines.append(f"    {k:<10} = {v:.6f}")

        lines += ["", "  Prefactors  (mean  [95% CI])", "  ----------------------------"]
        for name, info in self.prefactors_.items():
            lines.append(
                f"    {name:<8} = {info['scalar_mean']:>10.4f}"
                f"  [{info['scalar_lower']:>10.4f},"
                f" {info['scalar_upper']:>10.4f}]"
            )

        try:
            derived = self._derived_params()
        except Exception:
            derived = {}
        if derived:
            lines += ["", "  Derived parameters", "  ------------------"]
            for k, v in derived.items():
                lines.append(f"    {k:<10} = {v:.6f}")

        lines += [
            "", "  Fit metrics", "  -----------",
            f"    RMSE  = {self.metrics_['rmse']:.6f} Pa",
            f"    R²    = {self.metrics_['r2']:.6f}",
            f"    BIC   = {self.metrics_['bic']:.2f}",
            "=" * 58,
        ]
        return "\n".join(lines)

    # ============================================================
    # Plotting
    # ============================================================
    def plot_fit(self, ax=None, **kw):
        self._check_fitted()
        return plots.plot_fit(self.t_, self.stress_, self.pred_,
                              self.model_name, self.metrics_, ax=ax, **kw)

    def plot_prefactors(self, ax=None, **kw):
        self._check_fitted()
        return plots.plot_prefactors(self.t_, self.prefactors_,
                                     self.model_name, ax=ax, **kw)

    def plot_Gstar(self, ax=None, add_fft=False, fft_kwargs=None, fft_eps=1e-12, **kw):
        if isinstance(self.gp_model_.phys, KernelFreeKernel):
            print("KernelFreeKernel has no G*(omega). "
                  "Use plot_prefactors() instead.")
            return
        self._check_fitted()
        omega = np.logspace(np.log10(self.omega_min),
                            np.log10(self.omega_max), self.n_omega)
        scalar_pf = {k: v["scalar_mean"] for k, v in self.prefactors_.items()}
        Gp, Gdp = self.gp_model_.phys.Gstar(omega, scalar_pf)

        ax_obj = plots.plot_Gstar(
            omega, Gp, Gdp,
            self.gp_model_.phys.named_phys_params(),
            scalar_pf, self.model_name, ax=ax, **kw
        )

        if add_fft:
            try:
                omega_f, Gp_f, Gdp_f = self._chirp_fft_Gstar(eps=fft_eps)
                mask = (omega_f >= self.omega_min) & (omega_f <= self.omega_max)
                omega_f, Gp_f, Gdp_f = omega_f[mask], Gp_f[mask], Gdp_f[mask]
                if omega_f.size:
                    kwargs = dict(s=20, alpha=0.9)
                    if fft_kwargs:
                        kwargs.update(fft_kwargs)

                    # Collect all Axes that plots.plot_Gstar created
                    def _is_axes(a):
                        return hasattr(a, "plot") and hasattr(a, "scatter")

                    axes = []
                    if _is_axes(ax_obj):
                        axes = [ax_obj]
                    elif isinstance(ax_obj, (list, tuple)):
                        axes = [a for a in ax_obj if _is_axes(a)]
                    else:
                        try:
                            import numpy as _np
                            if isinstance(ax_obj, _np.ndarray):
                                axes = [a for a in ax_obj.ravel() if _is_axes(a)]
                        except Exception:
                            pass
                        if not axes and hasattr(ax_obj, "axes"):
                            axes = [a for a in getattr(ax_obj, "axes") if _is_axes(a)]

                    # Identify storage/loss axes from labels/titles
                    def _role(a):
                        txt = f"{a.get_title()} {a.get_ylabel()} {a.get_xlabel()}".lower()
                        if any(k in txt for k in ["phase", "angle", "tan", "δ", "delta"]):
                            return "phase"
                        if any(k in txt for k in ["storage", "g′", "g'", "g prime"]):
                            return "gp"
                        if any(k in txt for k in ["loss", "g″", 'g"', "g''", "g double prime"]):
                            return "gdp"
                        return "unknown"

                    gp_ax = None
                    gdp_ax = None
                    roles = [_role(a) for a in axes]
                    for a, r in zip(axes, roles):
                        if r == "gp" and gp_ax is None:
                            gp_ax = a
                        if r == "gdp" and gdp_ax is None:
                            gdp_ax = a

                    # Fallback: first two non-phase axes in order
                    if gp_ax is None or gdp_ax is None:
                        non_phase = [a for a, r in zip(axes, roles) if r != "phase"]
                        if gp_ax is None and non_phase:
                            gp_ax = non_phase[0]
                        if gdp_ax is None and len(non_phase) >= 2:
                            gdp_ax = non_phase[1]
                        elif gdp_ax is None and non_phase:
                            gdp_ax = non_phase[0]  # plot both on same axis if only one

                    # Scatter on the identified axes (exactly where curves are)
                    if gp_ax is not None:
                        gp_ax.scatter(omega_f, Gp_f, marker='o', facecolors='none',
                                      edgecolors='#1f77b4', label="FFT G′", **kwargs)
                        try:
                            gp_ax.legend(loc='best')
                        except Exception:
                            pass
                    if gdp_ax is not None:
                        gdp_ax.scatter(omega_f, Gdp_f, marker='s', facecolors='none',
                                       edgecolors='#ff7f0e', label="FFT G″", **kwargs)
                        try:
                            gdp_ax.legend(loc='best')
                        except Exception:
                            pass
                    if gp_ax is None and gdp_ax is None:
                        import warnings
                        warnings.warn("FFT overlay could not find G′/G″ axes to draw on.")
            except Exception as e:
                import warnings
                warnings.warn(f"FFT overlay failed: {e}")

        return ax_obj


    def _chirp_fft_Gstar(self, eps=1e-12):
        t, e, s = self.t_, self.strain_, self.stress_
        N = len(t)
        if not (len(e) == N and len(s) == N):
            raise ValueError("time, strain, stress must be the same length.")

        # sampling (assume roughly uniform)
        dt = float(np.median(np.diff(t)))

        # rFFT (no mean removal, no windowing — assume data are preprocessed)
        E = np.fft.rfft(e)
        S = np.fft.rfft(s)
        f = np.fft.rfftfreq(N, d=dt)
        omega = 2 * np.pi * f

        # drop DC and bins with tiny |E|
        magE = np.abs(E)
        m = (f > 0) & (magE > eps * magE.max())
        omega = omega[m]
        Gstar = S[m] / (E[m] + 0j)
        Gp = np.real(Gstar)
        Gdp = np.imag(Gstar)
        return omega, Gp, Gdp

    def plot_convergence(self, **kw):
        self._check_fitted()
        return plots.plot_convergence(
            self.history_["loss"],
            dict(self.history_["params"]),   # copy — never mutate
            self.model_name, **kw
        )

    def plot_kernel(self, ax=None, **kw):
        self._check_fitted()
        return plots.plot_kernel(self.gp_model_.phys, self.dt_,
                                  self.model_name, ax=ax, **kw)

    # ============================================================
    # Save
    # ============================================================
    def save(self, path):
        """Save all results to a directory of .npy + .csv + metadata.txt."""
        self._check_fitted()
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)

        np.save(root / "t.npy",          self.t_)
        np.save(root / "strain.npy",     self.strain_)
        np.save(root / "stress.npy",     self.stress_)
        np.save(root / "sigma_pred.npy", self.pred_)

        try:
            self.gp_model_.eval(); self.likelihood_.eval()
            with torch.no_grad(), gpytorch.settings.cholesky_jitter(1e-6):
                xf   = self.gp_model_.compute_x()
                dist = self.likelihood_(self.gp_model_(xf))
                var  = dist.variance.cpu().numpy()
            std = np.sqrt(np.maximum(var, 0.0)) * self.scaler_y_.scale_[0]
        except Exception as e:
            import warnings
            warnings.warn(f"Could not compute posterior std: {e}")
            std = np.full_like(self.pred_, np.nan)
        np.save(root / "sigma_std.npy", std)

        pf_dir = root / "prefactors"
        pf_dir.mkdir(exist_ok=True)
        for name, info in self.prefactors_.items():
            np.save(pf_dir / f"{name}_mean.npy",    info["mean"])
            np.save(pf_dir / f"{name}_lower.npy",   info["lower"])
            np.save(pf_dir / f"{name}_upper.npy",   info["upper"])
            np.save(pf_dir / f"{name}_samples.npy", info["samples"])

        omega     = np.logspace(np.log10(self.omega_min),
                                np.log10(self.omega_max), self.n_omega)
        scalar_pf = {k: v["scalar_mean"] for k, v in self.prefactors_.items()}
        try:
            Gp, Gdp = self.gp_model_.phys.Gstar(omega, scalar_pf)
        except Exception:
            Gp = Gdp = np.full_like(omega, np.nan)
        gs_dir = root / "Gstar"
        gs_dir.mkdir(exist_ok=True)
        np.save(gs_dir / "omega.npy", omega)
        np.save(gs_dir / "Gp.npy",    Gp)
        np.save(gs_dir / "Gdp.npy",   Gdp)

        try:
            omega_f, Gp_f, Gdp_f = self._chirp_fft_Gstar()
            ff_dir = root / "fft_Gstar"
            ff_dir.mkdir(exist_ok=True)
            np.save(ff_dir / "omega.npy", omega_f)
            np.save(ff_dir / "Gp.npy",    Gp_f)
            np.save(ff_dir / "Gdp.npy",   Gdp_f)
        except Exception as e:
            import warnings
            warnings.warn(f"Could not compute FFT G*: {e}")

        cv_dir = root / "convergence"
        cv_dir.mkdir(exist_ok=True)
        np.save(cv_dir / "loss.npy", np.array(self.history_["loss"]))
        for k, v in self.history_["params"].items():
            np.save(cv_dir / f"{k}.npy", np.array(v))

        phys_p = self.gp_model_.phys.named_phys_params()
        rows   = []
        for k, v in phys_p.items():
            rows.append({"name": k, "value": float(v),
                         "lower": np.nan, "upper": np.nan})
        for name, info in self.prefactors_.items():
            rows.append({"name": name,
                         "value": float(info["scalar_mean"]),
                         "lower": float(info["scalar_lower"]),
                         "upper": float(info["scalar_upper"])})
        for k in ("rmse", "r2", "bic"):
            rows.append({"name": k, "value": float(self.metrics_[k]),
                         "lower": np.nan, "upper": np.nan})
        np.save(root / "params.npy", np.array(rows, dtype=object))

        with open(root / "params.csv", "w") as f:
            f.write("name,value,lower,upper\n")
            for r in rows:
                f.write(f"{r['name']},{r['value']},{r['lower']},{r['upper']}\n")

        try:
            summary_text = self.summary()
        except Exception:
            summary_text = f"Model: {self.model_name}"
        with open(root / "metadata.txt", "w") as f:
            f.write(f"model_name: {self.model_name}\n")
            f.write(summary_text)

        if self.verbose:
            print(f"  Results saved to {root.resolve()}")

    # ============================================================
    # Internals
    # ============================================================
    def _seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark     = False

    def _build_phys_kernel(self):
        key = _normalise_key(self.model_name)
        cls = KERNELS.get(key)
        if cls is None:
            raise ValueError(
                f"Unknown model '{self.model_name}'.\n"
                f"Available: {sorted(KERNELS.keys())}"
            )
        kwargs = dict(_DEFAULTS[cls])
        if self.alpha_init is not None: kwargs["alpha_init"] = self.alpha_init
        if self.beta_init  is not None: kwargs["beta_init"]  = self.beta_init
        if self.tau_c_init is not None: kwargs["tau_c_init"] = self.tau_c_init

        if cls is KernelFreeKernel:
            kwargs["feature"] = getattr(self, 'feature', 'strain')

        return cls(**{k: v for k, v in kwargs.items()
                      if k in cls.__init__.__code__.co_varnames})

    # ── training dispatcher ──────────────────────────────────────────────
    def _train(self, y_tensor, sigma_obs):
        if self._two_phase:
            self._train_phase1(y_tensor, sigma_obs)
            self._train_phase2(y_tensor, sigma_obs)
        else:
            self.history_ = {}
            self._train_loop(
                y_tensor, sigma_obs,
                lr_gp      = self.learning_rate,
                lr_phys    = self.learning_rate_physics,
                n_epochs   = self.n_epochs,
                patience   = self.patience,
                phase_tag  = "",
                freeze_phys= False,
                extra_params=None,
            )

    # ── core Adam loop (shared by single-phase and both two-phase steps) ─
    def _train_loop(self, y_tensor, sigma_obs,
                    lr_gp, lr_phys, n_epochs, patience,
                    phase_tag, freeze_phys, extra_params):
        """
        Parameters
        ----------
        freeze_phys  : bool
            If True, physics parameters are already frozen before entry;
            the optimiser only sees GP hypers + likelihood.
        extra_params : list[Parameter] or None
            When not None, use ONLY these params + likelihood (Phase 2).
            When None, use all model params with separate lr groups.
        """
        model = self.gp_model_; lik = self.likelihood_
        mll   = gpytorch.mlls.VariationalELBO(
                    lik, model, num_data=y_tensor.numel())

        if extra_params is not None:
            # Phase 2: physics frozen, only GP hypers + likelihood
            opt = torch.optim.Adam(
                extra_params + list(lik.parameters()),
                lr=lr_gp
            )
        else:
            # Single-phase or Phase 1: separate lr groups
            phys_params = list(model.phys.parameters())
            gp_params   = [p for n, p in model.named_parameters()
                           if not n.startswith("phys.")]
            opt = torch.optim.Adam([
                {"params": phys_params, "lr": lr_phys},
                {"params": gp_params,   "lr": lr_gp},
                {"params": list(lik.parameters()), "lr": lr_gp},
            ])

        model.train(); lik.train()
        best_loss = np.inf; wait = 0
        loss_h    = []
        param_h   = {k: [] for k in model.phys.named_phys_params()}
        tag       = f"[{self.model_name}{phase_tag}]"

        with gpytorch.settings.cholesky_jitter(1e-6):
            for i in range(n_epochs):
                opt.zero_grad()
                x    = model.compute_x()
                loss = -mll(model(x), y_tensor)

                # ---- Add physics prior if the kernel defines one ----
                #phys = getattr(model, "phys", None)
                #if phys is not None and hasattr(phys, "prior_penalty"):
                #    loss = loss + phys.prior_penalty()
                # ------------------------------------------------------
                
                loss.backward()
                opt.step()

                lv = loss.item()
                loss_h.append(lv)
                for k, v in model.phys.named_phys_params().items():
                    param_h[k].append(v)

                if lv < best_loss - 1e-4: best_loss = lv; wait = 0
                else:                      wait += 1

                if self.verbose and i % 200 == 0:
                    pstr = "  ".join(
                        f"{k}={v:.3f}"
                        for k, v in model.phys.named_phys_params().items()
                    )
                    print(f"  {tag} step {i:5d} | "
                          f"-ELBO {lv:.4f} | {pstr} | wait {wait}")
                if wait >= patience:
                    if self.verbose:
                        print(f"  {tag} early stopping at step {i}")
                    break

        # concatenate onto history (phase 2 appends to phase 1)
        if "loss" not in self.history_:
            self.history_ = {"loss": loss_h, "params": param_h}
        else:
            self.history_["loss"].extend(loss_h)
            for k in param_h:
                self.history_["params"][k].extend(param_h[k])

        # final eval + metrics
        model.eval(); lik.eval()
        with torch.no_grad(), gpytorch.settings.cholesky_jitter(1e-6):
            xf     = model.compute_x()
            pred_s = lik(model(xf)).mean.cpu().numpy()

        sy, sm     = self.scaler_y_.scale_[0], self.scaler_y_.mean_[0]
        self.pred_ = pred_s * sy + sm

        rmse = float(np.sqrt(np.mean((self.pred_ - sigma_obs)**2)))
        r2   = float(1 - np.sum((self.pred_ - sigma_obs)**2) /
                         np.sum((sigma_obs - sigma_obs.mean())**2))
        n_shape = len(model.phys.named_phys_params())
        #from .kernels import PREFACTOR_COLS
        #n_pref  = len(PREFACTOR_COLS[type(model.phys)])
        #kappa   = n_shape + n_pref
        bic     = float(2 * best_loss + n_shape * np.log(len(y_tensor)))
        self.metrics_ = dict(rmse=rmse, r2=r2, bic=bic, elbo=best_loss)

        if self.verbose:
            print(f"\n  {tag} RMSE={rmse:.4f}  R²={r2:.4f}  BIC={bic:.2f} ELBO={best_loss:.4f}")
            print(f"  params: {model.phys.named_phys_params()}")

        return best_loss

    # ── two-phase helpers ────────────────────────────────────────────────
    def _train_phase1(self, y_tensor, sigma_obs):
        """Phase 1: RBF, physics free — discovers α, β, τ_c."""
        if self.verbose:
            print(f"\n  ── Phase 1: RBF  (physics discovery) ──────────────")
        self.history_ = {}
        self._train_loop(
            y_tensor, sigma_obs,
            lr_gp      = self.learning_rate,
            lr_phys    = self.learning_rate_physics,
            n_epochs   = self.n_epochs,
            patience   = self.patience,
            phase_tag  = " Ph1-RBF",
            freeze_phys= False,
            extra_params=None,
        )
        if self.verbose:
            print(f"  Phase 1 done — locked: "
                  f"{self.gp_model_.phys.named_phys_params()}")

    def _train_phase2(self, y_tensor, sigma_obs):
        """
        Phase 2: LinearKernel, physics frozen — refines prefactors.

        Why LinearKernel is correct here
        ----------------------------------
        The constitutive law is  σ = Σⱼ θⱼ xⱼ — a strict linear map.
        Once the features xⱼ are fixed (α, β, τ_c locked from Phase 1),
        fitting σ is a convex linear problem.  The LinearKernel encodes
        exactly this: k(x,x') = s² xᵀx', so the GP mean is linear in x.
        This removes the curvature-amplitude coupling that inflates
        prefactors when using RBF with still-moving features.
        """
        if self.verbose:
            print(f"\n  ── Phase 2: Linear  (prefactor refinement) ─────────")

        trainable = self.gp_model_.swap_to_linear()
        self._train_loop(
            y_tensor, sigma_obs,
            lr_gp      = self.learning_rate_linear,
            lr_phys    = None,           # unused — physics are frozen
            n_epochs   = self.n_epochs_linear,
            patience   = self.patience_linear,
            phase_tag  = " Ph2-Linear",
            freeze_phys= True,
            extra_params=trainable,
        )
        if self.verbose:
            print(f"  Phase 2 done — physics (frozen): "
                  f"{self.gp_model_.phys.named_phys_params()}")

    '''
    # ── prefactor extraction ─────────────────────────────────────────────
    def _extract_prefactors(self):
        model  = self.gp_model_; lik = self.likelihood_
        sig_y  = self.scaler_y_.scale_[0]
        col_map = PREFACTOR_COLS[type(model.phys)]

        # ensure _feature_stds populated
        model.eval(); lik.eval()
        with torch.no_grad(), gpytorch.settings.cholesky_jitter(1e-4):
            _ = model.compute_x()
        stds = model.phys._feature_stds

        with gpytorch.settings.cholesky_jitter(1e-4), \
             gpytorch.settings.fast_pred_var():

            x_raw   = model.compute_x().detach().clone().requires_grad_(True)
            preds   = lik(model(x_raw))
            S       = self.num_posterior_samples
            samples = preds.rsample(torch.Size([S]))   # (S, N)

            grads = []
            for s in range(S):
                g = torch.autograd.grad(
                    samples[s], x_raw,
                    grad_outputs=torch.ones_like(samples[s]),
                    retain_graph=True
                )[0]
                grads.append(g.detach())
            grads = torch.stack(grads).cpu().numpy()   # (S, N, n_feat)

        gamma_factors = getattr(model.phys, "_gamma_factors", None)

        out = {}
        for name, col in col_map.items():
            phys = grads[:, :, col] * (sig_y / stds[col])   # (S, N)

            # reattach Γ(1-α) stripped from kernel (fixes α→1 blow-up)
            if gamma_factors is not None:
                phys = phys * gamma_factors[col]

            mean  = phys.mean(axis=0)
            lower = np.quantile(phys, 0.025, axis=0)
            upper = np.quantile(phys, 0.975, axis=0)

            # scalar summary: positive finite values only
            m = np.isfinite(phys) & (phys > 0)
            if m.any():
                pos          = phys[m]
                scalar_mean  = float(np.mean(pos))
                scalar_lower = float(np.quantile(pos, 0.025))
                scalar_upper = float(np.quantile(pos, 0.975))
            else:
                scalar_mean = scalar_lower = scalar_upper = float("nan")

            out[name] = dict(
                samples      = phys,
                mean         = mean,
                lower        = lower,
                upper        = upper,
                scalar_mean  = scalar_mean,
                scalar_lower = scalar_lower,
                scalar_upper = scalar_upper,
            )
            if self.verbose:
                print(f"  {name}: {scalar_mean:.4f}  "
                      f"95%CI [{scalar_lower:.4f}, {scalar_upper:.4f}]")
        return out
    '''

    '''

    def _extract_prefactors(self):
        model  = self.gp_model_; lik = self.likelihood_
        sig_y  = self.scaler_y_.scale_[0]
        col_map = PREFACTOR_COLS[type(model.phys)]
        
        # ensure _feature_stds populated
        model.eval(); lik.eval()
        with torch.no_grad(), gpytorch.settings.cholesky_jitter(1e-16):
            _ = model.compute_x()
        stds = model.phys._feature_stds

        model.eval(); lik.eval()
        with gpytorch.settings.cholesky_jitter(1e-16), \
             gpytorch.settings.fast_pred_var():

            x_raw  = model.compute_x().detach().clone().requires_grad_(True)
            preds  = lik(model(x_raw))
            S      = self.num_posterior_samples
            samples = preds.rsample(torch.Size([S]))        # (S, N)

            grads = []
            for s in range(S):
                g = torch.autograd.grad(
                    samples[s], x_raw,
                    grad_outputs=torch.ones_like(samples[s]),
                    #retain_grad=False, 
                    retain_graph=True
                )[0]
                grads.append(g.detach())
            grads = torch.stack(grads).cpu().numpy()        # (S, N, n_feat)

        out = {}
        for name, col in col_map.items():
            #phys = grads[:, :, col] * (sig_y / stds[col])  # (S, N)
            #out[name] = dict(
            #    samples      = phys,
            #    mean         = phys.mean(axis=0),
            #    lower        = np.quantile(phys, 0.025, axis=0),
            #    upper        = np.quantile(phys, 0.975, axis=0),
                #scalar_mean  = float(phys.mean()),
                
            #    scalar_lower = float(np.quantile(phys, 0.025)),
            #    scalar_upper = float(np.quantile(phys, 0.975)),
            #)
            phys = grads[:, :, col] * (sig_y / stds[col])  # (S, N)self.alpha if col == 0 else self.beta


            # Per-time stats (unchanged)
            mean  = phys.mean(axis=0)
            lower = np.quantile(phys, 0.025, axis=0)
            upper = np.quantile(phys, 0.975, axis=0)

            # Scalar stats using only positive and finite entries
            m = np.isfinite(phys) & (phys > 0)
            if m.any():
                pos = phys[m]
                scalar_mean  = float(np.mean(pos))
                scalar_lower = float(np.quantile(pos, 0.025))
                scalar_upper = float(np.quantile(pos, 0.975))
            else:
                scalar_mean = scalar_lower = scalar_upper = float("nan")

            out[name] = dict(
                samples      = phys,
                mean         = mean,
                lower        = lower,
                upper        = upper,
                scalar_mean  = scalar_mean,
                scalar_lower = scalar_lower,
                scalar_upper = scalar_upper,
            )
            if self.verbose:
                print(f"  {name}: {out[name]['scalar_mean']:.4f}  "
                      f"95%CI [{out[name]['scalar_lower']:.4f},"
                      f" {out[name]['scalar_upper']:.4f}]")
        return out
    '''

    def _extract_prefactors(self):
        model  = self.gp_model_; lik = self.likelihood_
        sig_y  = self.scaler_y_.scale_[0]
        col_map = PREFACTOR_COLS[type(model.phys)]

        # ensure _feature_stds populated
        model.eval(); lik.eval()
        with torch.no_grad(), gpytorch.settings.cholesky_jitter(1e-6):
            _ = model.compute_x()
        stds = model.phys._feature_stds

        with gpytorch.settings.cholesky_jitter(1e-6), \
             gpytorch.settings.fast_pred_var():

            x_raw   = model.compute_x().detach().clone().requires_grad_(True)
            preds   = lik(model(x_raw))
            S       = self.num_posterior_samples
            samples = preds.rsample(torch.Size([S]))   # (S, N)

            grads = []
            for s in range(S):
                g = torch.autograd.grad(
                    samples[s], x_raw,
                    grad_outputs=torch.ones_like(samples[s]),
                    retain_graph=True
                )[0]
                grads.append(g.detach())
            grads = torch.stack(grads).cpu().numpy()   # (S, N, n_feat)

        # --- ONLY for kernels that define a PCA/rotation matrix _P (i.e. FKV) ---
        P = getattr(model.phys, "_P", None)
        if P is not None:
            # P: (2,2) torch.Tensor; grads: (S, N, 2)
            P_np = P.detach().cpu().numpy()
            grads = grads @ P_np.T   # grads_x = grads_z @ Pᵀ
        # ------------------------------------------------------------------------

         
        gamma_factors = getattr(model.phys, "_gamma_factors", None)

        out = {}
        for name, col in col_map.items():
            #phys = grads[:, :, col] * (sig_y / stds[col])   # (S, N)
            gamma_scale = getattr(model.phys, "_gamma_scale", 1.0)

            if torch.is_tensor(gamma_scale):
                gamma_scale = gamma_scale.item()

            phys = grads[:, :, col] * (sig_y / stds[col]) / gamma_scale
            
            if gamma_factors is not None:
                phys = phys * gamma_factors[col]

            mean  = phys.mean(axis=0)
            lower = np.quantile(phys, 0.025, axis=0)
            upper = np.quantile(phys, 0.975, axis=0)

            # scalar summary: positive finite values only
            m = np.isfinite(phys) & (phys > 0)
            if m.any():
                pos          = phys[m]
                scalar_mean  = float(np.mean(pos))
                scalar_lower = float(np.quantile(pos, 0.025))
                scalar_upper = float(np.quantile(pos, 0.975))
            else:
                scalar_mean = scalar_lower = scalar_upper = float("nan")

            out[name] = dict(
                samples      = phys,
                mean         = mean,
                lower        = lower,
                upper        = upper,
                scalar_mean  = scalar_mean,
                scalar_lower = scalar_lower,
                scalar_upper = scalar_upper,
            )
            if self.verbose:
                print(f"  {name}: {scalar_mean:.4f}  "
                      f"95%CI [{scalar_lower:.4f}, {scalar_upper:.4f}]")
        return out


    def _derived_params(self):
        p  = self.gp_model_.phys.named_phys_params()
        pf = {k: v["scalar_mean"] for k, v in self.prefactors_.items()}
        out = {}
        cls = type(self.gp_model_.phys)

        if cls is MaxwellKernel:
            out["tau_c"] = p["tau_c"]
            out["Gc"]    = pf.get("Gc", np.nan)

        elif cls is FractionalMaxwellGelKernel:
            tc = p["tau_c"]; a = p["alpha"]; Gc = pf.get("Gc", np.nan)
            out["tau_c"] = tc
            out["Gc"]    = Gc
            out["V"]     = Gc * tc ** a
            out["G"]     = Gc

        elif cls is FractionalMaxwellLiquidKernel:
            out["tau_c"] = p["tau_c"]
            out["Gc"]    = pf.get("Gc", np.nan)

        elif cls is FractionalMaxwellKernel:
            tc = p["tau_c"]; a = p["alpha"]; b = p["beta"]
            Gc = pf.get("Gc", np.nan)
            out["tau_c"] = tc
            out["Gc"]    = Gc
            out["V"]     = Gc * tc ** a
            out["G_bb"]  = Gc * tc ** b

        elif cls in (FKVSKernel, FractionalKelvinVoigtKernel):
            V    = pf.get("V",    pf.get("Gc",  np.nan))
            G_bb = pf.get("G_bb", pf.get("G",   np.nan))
            a    = p.get("alpha", p.get("beta",  np.nan))
            b    = p.get("beta",  0.0)
            exp  = (a - b) if ("alpha" in p and "beta" in p) else a
            if all(np.isfinite([V, G_bb, exp])) and G_bb > 0 and exp > 0:
                out["tau_c"] = float((V / G_bb) ** (1.0 / exp))
                out["Gc"]    = float(V * out["tau_c"] ** (-a))


        #elif cls in FractionalKelvinVoigtKernel:
        #    tc = p["tau_c"]; a = p["alpha"]; b = p["beta"]
        #    Gc = pf.get("Gc", np.nan)
        #    out["tau_c"] = tc
        #    out["Gc"]    = Gc
        #    out["V"]     = Gc * tc ** a
        #    out["G_bb"]  = Gc * tc ** b

        return out

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Call fit() before using this method.")