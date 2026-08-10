"""
RheoGP 0.3 — quickstart
=======================
Covers the four things that changed in v0.3:

  1. one-liner fitting API
  2. causal vs steady feature construction
  3. stress-controlled (creep) fitting
  4. the FitResult container: every quantity as plain numpy

Run:  python examples/quickstart.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rheogp
from rheogp.utils import (
    make_synthetic_springpot,          # strain-controlled synthetic chirp
    make_synthetic_springpot_stress,   # stress-controlled synthetic chirp
)

# ================================================================
# 1. Strain-controlled fit  (default: causal convolution from 0,
#    exact product-integration quadrature — no start-up transient)
# ================================================================
t, strain, stress = make_synthetic_springpot(
    V=100.0, alpha=0.5, n=1200, dt=2e-3, noise=0.02,
)

model = rheogp.fit(
    t, strain, stress,
    model="Springpot",
    control="strain",          # default
    convolution="causal",      # default; "steady" for the -inf convolution
    quadrature="exact",        # default; "left" reproduces rheogp <= 0.2
    n_epochs=800, patience=250, verbose=False,
)
print(model.summary())

# ================================================================
# 2. Everything you might want to plot, as numpy arrays
# ================================================================
res = model.results()

print("physical features x_i(t):", res.features.shape, res.feature_names)
print("shape params:", res.phys_params)
print("scalar prefactors:", res.scalar_prefactors())

fig, ax = plt.subplots(2, 2, figsize=(10, 7))
ax[0, 0].plot(res.t, res.target, "k.", ms=1, label="data")
ax[0, 0].plot(res.t, res.prediction, "r", lw=1, label="GP")
ax[0, 0].fill_between(res.t,
                      res.prediction - 2 * res.prediction_std,
                      res.prediction + 2 * res.prediction_std,
                      alpha=0.3, color="r")
ax[0, 0].set(xlabel="t [s]", ylabel="stress [Pa]"); ax[0, 0].legend()

ax[0, 1].plot(res.t, res.features[:, 0], lw=0.8)
ax[0, 1].set(xlabel="t [s]", ylabel=f"feature  {res.feature_names[0]}")

s = res.sensitivities["V"]
ax[1, 0].plot(res.t, s["mean"], "b", lw=0.8)
ax[1, 0].fill_between(res.t, s["lower"], s["upper"], alpha=0.3)
ax[1, 0].set(xlabel="t [s]", ylabel="V(t) [Pa s$^\\alpha$]")

ax[1, 1].loglog(res.omega, res.Gp, "b", label="G'")
ax[1, 1].loglog(res.omega, res.Gdp, "b--", label='G"')
ax[1, 1].loglog(res.fft_omega, res.fft_Gp, "k.", ms=2)
ax[1, 1].loglog(res.fft_omega, res.fft_Gdp, "kx", ms=2)
ax[1, 1].set(xlabel="ω [rad/s]", ylabel="G', G'' [Pa]"); ax[1, 1].legend()

fig.tight_layout()
fig.savefig("quickstart_strain.png", dpi=140)
print("wrote quickstart_strain.png")

# Save / reload everything (npz + json, no torch required to reload)
res.save("springpot_fit")
res2 = rheogp.FitResult.load("springpot_fit")
assert np.allclose(res2.prediction, res.prediction)

# ================================================================
# 3. Stress-controlled fit (creep representation)
#    Same call signature — just flip `control`.
#    Learned prefactors are compliances (1/V, ...); moduli are
#    reported automatically as derived parameters.
# ================================================================
t2, strain2, stress2 = make_synthetic_springpot_stress(
    V=100.0, alpha=0.5, n=1200, dt=2e-3, noise=0.02,
)

model_s = rheogp.fit(
    t2, strain2, stress2,
    model="Springpot",
    control="stress",
    n_epochs=800, patience=250, verbose=False,
)
print(model_s.summary())     # shows invV and derived V

# ================================================================
# 4. Steady (from -inf) features + causal prediction override
#    Steady mode assumes the record is one period of a periodic
#    protocol — exact for optimally-windowed chirps.  When you then
#    predict on a *non-periodic* protocol (e.g. step strain), pass
#    convolution="causal" to override.
# ================================================================
model_f = rheogp.fit(
    t, strain, stress,
    model="Springpot",
    convolution="steady",
    n_epochs=800, patience=250, verbose=False,
)

step = np.where(np.arange(1000) * 2e-3 > 0.4, 0.05, 0.0)
pred, std = model_f.predict(step, return_std=True, convolution="causal")
print("step-strain relaxation predicted:", pred.shape)

# Tip for FKV-family models under stress control (tau_c inside the
# Mittag-Leffler creep kernel converges slowly):
#   rheogp.fit(..., model="FKVS", control="stress",
#              learning_rate_physics=0.02, n_epochs=6000, patience=1500)
