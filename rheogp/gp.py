"""
rheogp.gp
=========
Sparse variational GP backbone.

Two-phase training strategy
----------------------------
Phase 1 — physics discovery (kernel="rbf" or any smooth kernel):
    All parameters free: α, β, τ_c + GP hyperparameters optimised
    jointly through the ELBO.  The flexible RBF kernel can fit the
    data even when the features are not yet correct, which gives the
    optimiser a broad basin to find the right physics.

Phase 2 — linear refinement (kernel="linear"):
    Physics parameters (α, β, τ_c) are FROZEN at their Phase-1 values.
    The GP covariance is swapped to a LinearKernel, enforcing the
    constitutively correct relationship  σ = Σⱼ θⱼ xⱼ  exactly.
    Only the GP hyperparameters (linear kernel variance, noise) are
    retrained.  The LinearKernel variance directly encodes the
    prefactor magnitude, making prefactor recovery more stable.

    Call model.swap_to_linear() between the two phases, or use
    kernel="rbf+linear" in SPGP to run both phases automatically.
"""

import torch
import gpytorch

__all__ = ["RheoGPModel"]


class RheoGPModel(gpytorch.models.ApproximateGP):
    """
    Sparse GP with a physics kernel embedded as a sub-module.

    Parameters
    ----------
    physics_kernel : SpringpotKernel | FKVKernel | FMMKernel
    gamma          : (N,) strain buffer
    dt             : float
    num_inducing   : int
    gp_kernel_name : "rbf" | "matern12" | "matern32" | "matern52" | "linear"
                     Use "rbf+linear" to trigger two-phase training via SPGP.
    """

    _GP_KERNELS = {
        "rbf":      lambda d: gpytorch.kernels.RBFKernel(ard_num_dims=None),
        "matern12": lambda d: gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=d),
        "matern32": lambda d: gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=d),
        "matern52": lambda d: gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=d),
        "linear":   lambda d: gpytorch.kernels.LinearKernel(ard_num_dims=None),
    }

    def __init__(self, physics_kernel, gamma, dt,
                 num_inducing=300, gp_kernel_name="matern32"):
        n_feat       = physics_kernel.n_features
        inducing_pts = torch.randn(num_inducing, n_feat,
                                   device=gamma.device) * 0.1
        var_dist  = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing
        )
        var_strat = gpytorch.variational.VariationalStrategy(
            self, inducing_pts, var_dist, learn_inducing_locations=True
        )
        super().__init__(var_strat)

        self.mean_module  = gpytorch.means.ConstantMean()

        # "rbf+linear" uses rbf for Phase 1; swap_to_linear() handles Phase 2
        _name = "rbf" if gp_kernel_name.lower() == "rbf+linear" else gp_kernel_name.lower()
        base_k = self._GP_KERNELS.get(_name, self._GP_KERNELS["matern32"])(n_feat)
        self.covar_module = gpytorch.kernels.ScaleKernel(base_k)

        self.phys = physics_kernel
        self.register_buffer("gamma", gamma)
        self.dt = dt

    def compute_x(self):
        """Recompute the convolution feature with current physics params."""
        return self.phys.compute_x(self.gamma, self.dt)

    def compute_x_phys(self):
        """Recompute the convolution feature with current physics params."""
        return self.phys.compute_x_phys(self.gamma, self.dt)

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )

    def swap_to_linear(self):
        """
        Phase 1 → Phase 2 transition.

        1. Freeze all physics parameters (α, β, τ_c) — they are now fixed.
        2. Replace the GP covariance with a LinearKernel.
        3. Re-initialise the variational distribution and inducing points so
           Phase-2 training starts fresh (the inducing locations from Phase 1
           were tuned for the RBF kernel and are not meaningful for a linear
           kernel).
        4. Return the new list of trainable parameters for the Phase-2
           optimiser (physics params excluded).

        Why freeze physics?
        -------------------
        The LinearKernel enforces  k(x,x') = s² xᵀx', which means the GP
        mean function is linear in the features.  Gradients through a linear
        kernel are well-conditioned only when the features are fixed — if α
        or β were still free, the interaction between the kernel curvature
        and the feature shape would create conflated gradients and the
        prefactors would be unreliable.  Freezing decouples the two tasks:
        Phase 1 finds the right feature shape; Phase 2 finds the right
        amplitude.

        Returns
        -------
        trainable_params : list of torch.nn.Parameter
            Parameters for the Phase-2 optimiser (GP hypers + likelihood).
        """
        # ── 1. freeze physics ────────────────────────────────
        for p in self.phys.parameters():
            p.requires_grad_(False)

        # ── 2. swap covariance to LinearKernel ───────────────
        n_feat = self.phys.n_features
        new_base = gpytorch.kernels.LinearKernel(num_dimensions=n_feat)
        self.covar_module = gpytorch.kernels.ScaleKernel(new_base)
        # move to same device as the model
        device = self.gamma.device
        self.covar_module = self.covar_module.to(device)

        # ── 3. re-initialise variational components ──────────
        #    Reuse the existing inducing point COUNT but re-randomise
        #    locations so they are appropriate for the (now fixed) features.
        with torch.no_grad():
            x_fixed = self.compute_x()          # (N, d) with frozen physics
            N, d    = x_fixed.shape
            M       = self.variational_strategy.inducing_points.shape[0]
            # pick M random data points as inducing locations (good init for linear)
            idx     = torch.randperm(N, device=device)[:M]
            new_ind = x_fixed[idx].clone()

        # Replace variational strategy wholesale
        var_dist  = gpytorch.variational.CholeskyVariationalDistribution(M).to(self.gamma.device)
        var_strat = gpytorch.variational.VariationalStrategy(
            self, new_ind, var_dist, learn_inducing_locations=False
        )   # learn_inducing_locations=False for linear kernel (inducing pts = data subset)
        self.variational_strategy = var_strat

        # ── 4. return only the trainable (non-physics) params ─
        trainable = [p for p in self.parameters() if p.requires_grad]
        return trainable