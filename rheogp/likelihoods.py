"""
rheogp.likelihoods -- observation-noise models
=============================================

`SPGP` fits one scalar observation variance for the whole pooled dataset
(`gpytorch.likelihoods.GaussianLikelihood`). For a single record that is
usually fine. For a multi-record chirp fit it is an approximation worth
being able to relax, for two reasons measured on the DGEBA-IPDA set:

* BETWEEN records, the residual standard deviation spans ~7.6x while the
  stress amplitude spans only ~2.2x, so the residual-to-signal ratio
  varies about 5x from record to record.
* WITHIN a single record it varies by a median factor of ~3.5, reaching
  ~44x in the warmest chirps, and the residual magnitude correlates with
  the predicted stress (Pearson 0.36-0.66 in those records).

That second point is the important one: a chirp sweeps frequency under a
tapered envelope, so its stress amplitude changes enormously along the
record, and the noise changes with it. A model that assigns one variance
per record cannot represent that; a model that ties the noise to the
signal can, with two parameters rather than a second Gaussian process.

`ProportionalGaussianLikelihood` implements

    sigma^2(t) = (c * |f(t)|)^2 + sigma_0^2

with `f(t)` the GP's own latent mean -- the predicted stress -- so the
noise floor `sigma_0` and the relative term `c` are learned jointly with
everything else in the same ELBO. This is the usual instrument noise
model in rheometry (a torque-resolution floor plus a term proportional
to the measured signal), stated rather than inferred nonparametrically.

Both parameters are ordinary likelihood parameters, so
`SPGP._count_parameters` picks them up automatically and the information
criteria stay honest: this model costs exactly one parameter more than
the shared one.

Not implemented here: a full heteroscedastic GP, in which log sigma^2 is
itself a latent function with its own GP prior (Goldberg 1998, Kersting
2007; `gpytorch.likelihoods.HeteroskedasticNoise`). That is the general
case, and it is usable under a variational ELBO, but GPyTorch's
implementation uses only the noise GP's predictive mean and expects that
GP to have been fitted separately against empirical log-variances, so it
reintroduces a two-pass scheme. The proportional form above encodes the
signal dependence the data actually show, at a far smaller cost.
"""
import math

import torch
import gpytorch
from gpytorch.constraints import Positive
from gpytorch.likelihoods import GaussianLikelihood
from linear_operator.operators import DiagLinearOperator
from torch.distributions import Normal

__all__ = ["ProportionalGaussianLikelihood", "build_likelihood", "NOISE_MODELS"]


class ProportionalGaussianLikelihood(GaussianLikelihood):
    r"""Gaussian likelihood whose noise scales with the predicted signal.

    .. math::
        \sigma^2(t) = \left(c\,|f(t)|\right)^2 + \sigma_0^2

    `sigma_0` is the inherited `GaussianLikelihood` noise (the floor) and
    `c` is a new non-negative scalar. Because the noise depends on the
    latent mean, gradients flow into the variational parameters through
    the noise term as well as through the fit, which is the intended
    behaviour: the model can explain a large residual either by moving
    the mean or by acknowledging the point is noisy, and the ELBO decides.

    Parameters
    ----------
    rel_init : float
        starting value of `c`, as a fraction. The default 0.05 says
        "expect noise of order 5% of the signal" before seeing data.
    """

    def __init__(self, rel_init=0.05, **kwargs):
        super().__init__(**kwargs)
        self.register_parameter(
            "raw_rel", torch.nn.Parameter(torch.tensor(float(rel_init))))
        self.register_constraint("raw_rel", Positive())

    @property
    def rel(self):
        """The relative noise coefficient `c`, constrained positive."""
        return self.raw_rel_constraint.transform(self.raw_rel)

    @rel.setter
    def rel(self, value):
        self.initialize(raw_rel=self.raw_rel_constraint.inverse_transform(
            torch.as_tensor(value).to(self.raw_rel)))

    def _signal_noise(self, f):
        """sigma^2 at each point, given the latent values `f`."""
        return (self.rel * f.abs()) ** 2 + self.noise

    # -- the three entry points a Gaussian likelihood has to cover -----
    def expected_log_prob(self, target, input, *params, **kwargs):
        """E_q(f)[log p(y|f)] -- what VariationalELBO calls.

        Same form as the homoscedastic case, with the constant variance
        replaced by the per-point one. The noise is evaluated at the
        variational mean rather than integrated over q(f): integrating
        (c|f|)^2 exactly would add a term in Var[f], which is negligible
        here (the posterior is tight relative to the signal) and would
        make the noise depend on the inducing-point parameterisation.
        """
        mean, variance = input.mean, input.variance
        noise = self._signal_noise(mean)
        res = ((target - mean) ** 2 + variance) / noise \
            + noise.log() + math.log(2 * math.pi)
        return res.mul(-0.5)

    def forward(self, function_samples, *params, **kwargs):
        return Normal(function_samples,
                      self._signal_noise(function_samples).sqrt())

    def marginal(self, function_dist, *params, **kwargs):
        mean, covar = function_dist.mean, function_dist.lazy_covariance_matrix
        return function_dist.__class__(
            mean, covar + DiagLinearOperator(self._signal_noise(mean)))


def build_likelihood(noise_model="shared", rel_init=0.05, **kwargs):
    """Construct the observation likelihood `SPGP` should use.

    "shared" reproduces the previous behaviour EXACTLY -- a plain
    `gpytorch.likelihoods.GaussianLikelihood` with no extra parameters --
    so switching noise models cannot alter any result computed before
    this option existed.
    """
    if noise_model in ("shared", "homoscedastic", None):
        return gpytorch.likelihoods.GaussianLikelihood(**kwargs)
    if noise_model in ("proportional", "signal"):
        return ProportionalGaussianLikelihood(rel_init=rel_init, **kwargs)
    raise ValueError(
        f"unknown noise_model {noise_model!r} -- use 'shared' (default) or "
        "'proportional'")


NOISE_MODELS = ("shared", "proportional")
