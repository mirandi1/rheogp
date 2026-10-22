"""
End-to-end validation of the joint multi-record fit
===================================================
1. generate synthetic chirps for ONE thermorheologically simple FMM whose
   spectrum shifts with temperature (synthetic_tts.py)
2. fit each record independently  -- shows which exponents a single record
   can and cannot constrain
3. fit all records jointly        -- shared alpha, beta, Gc; one a_T each
4. compare the recovered a_T against the WLF law used to generate the data

Because the generator works in the frequency domain from the analytical
G*(w) and the estimator works in the time domain with rheogp's causal
convolution, agreement is a genuine test rather than a round trip.

Run:  python validate_multi.py     (or paste the blocks into a notebook)
"""

import numpy as np
import matplotlib.pyplot as plt

from synthetic_tts import make_dataset, describe, plot_dataset, wlf_aT
from rheogp import SPGP
from rheogp_multi import MultiRecordSPGP

MODEL = "FMM"
FITKW = dict(model=MODEL, kernel="linear", convolution="causal",
             quadrature="exact", inducing_points=200,
             learning_rate=0.01, learning_rate_physics=0.05,
             n_epochs=4000, patience=500,
             omega_min=1e-3, omega_max=1e3, verbose=False)


# ---------------------------------------------------------------- 1. data
truth, records = make_dataset()
TEMPS = [m["T"] for m in truth["_meta"]]
I_REF = TEMPS.index(int(truth["T_ref"]))
describe(truth)
plot_dataset(truth, records)


# ------------------------------------------------- 2. independent fits
print("\n" + "=" * 66)
print("  INDEPENDENT FITS  (what a single record can constrain)")
print("=" * 66)
print(f"{'T':>4} {'in band':>8} {'alpha':>10} {'beta':>10} {'tau_c':>12} "
      f"{'RMSE':>12}")
indep = {}
for T, (t, g, s), meta in zip(TEMPS, records, truth["_meta"]):
    m = SPGP(**FITKW).fit(t, g, s)
    indep[T] = m
    p = m.gp_model_.phys.named_phys_params()
    print(f"{T:>4} {str(meta['in_band']):>8} {p['alpha']:10.4f} "
          f"{p['beta']:10.4f} {p['tau_c']:12.5g} {m.metrics_['rmse']:12.5g}")
print(f"\n  truth: alpha = {truth['alpha']}, beta = {truth['beta']}")
print("  records whose crossover is outside the band should show a poorly")
print("  determined exponent -- that is the motivation for pooling.")


# ------------------------------------------------------- 3. joint fit
print("\n" + "=" * 66)
print("  JOINT FIT")
print("=" * 66)
mj = MultiRecordSPGP(records, i_ref=I_REF, **FITKW)
mj.verbose = True
mj.fit_all()
aT_fit = mj.report(labels=[f"{T} C" for T in TEMPS], truth=truth)


# ------------------------------------------------ 4. shift factors vs WLF
aT_true = np.array([wlf_aT(T, truth["C1"], truth["C2"], truth["T_ref"])
                    for T in TEMPS])
aT_true = aT_true / aT_true[I_REF]

print(f"\n{'T':>4} {'a_T fitted':>14} {'a_T truth':>14} {'error %':>10}")
for T, af, at in zip(TEMPS, aT_fit, aT_true):
    print(f"{T:>4} {af:14.6g} {at:14.6g} {100*(af-at)/at:10.2f}")

# WLF regression on the recovered shift factors
from scipy.optimize import curve_fit
Tv = np.array(TEMPS, float)
f = lambda TT, C1, C2: -C1 * (TT - truth["T_ref"]) / (C2 + TT - truth["T_ref"])
(C1f, C2f), cov = curve_fit(f, Tv, np.log10(aT_fit), p0=[8.0, 130.0],
                            maxfev=40000)
sd = np.sqrt(np.diag(cov))
print(f"\n  WLF from the joint fit : C1 = {C1f:.3f} +/- {sd[0]:.3f}, "
      f"C2 = {C2f:.2f} +/- {sd[1]:.2f} K")
print(f"  ground truth           : C1 = {truth['C1']}, C2 = {truth['C2']}")


# ------------------------------------------------------------- 5. figures
cmap = plt.get_cmap("viridis")
fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))

ax[0].semilogy(TEMPS, aT_true, "k-", lw=1, label="truth (WLF)")
ax[0].semilogy(TEMPS, aT_fit, "o", ms=4, color="tab:orange", label="joint fit")
ax[0].set_xlabel(r"$T$ [$^\circ$C]"); ax[0].set_ylabel(r"$a_T$")
ax[0].set_title("shift factors", fontsize=8)
ax[0].legend(frameon=False, fontsize=7)

ax[1].plot(TEMPS, [indep[T].gp_model_.phys.named_phys_params()["alpha"]
                   for T in TEMPS], "o-", ms=3, label=r"$\alpha$ indep.")
ax[1].plot(TEMPS, [indep[T].gp_model_.phys.named_phys_params()["beta"]
                   for T in TEMPS], "s-", ms=3, label=r"$\beta$ indep.")
pj = mj.gp_model_.phys.named_phys_params()
ax[1].axhline(truth["alpha"], color="k", lw=1, ls="--")
ax[1].axhline(truth["beta"],  color="k", lw=1, ls=":")
ax[1].axhline(pj["alpha"], color="tab:red", lw=1)
ax[1].axhline(pj["beta"],  color="tab:red", lw=1, ls="--")
ax[1].set_xlabel(r"$T$ [$^\circ$C]"); ax[1].set_ylabel("exponent")
ax[1].set_title("dashed black = truth, red = joint", fontsize=8)
ax[1].legend(frameon=False, fontsize=6)

res = mj.results()
tau_ref = pj["tau_c"]
for i, T in enumerate(TEMPS):
    lo, hi = mj._bounds[i]
    col = cmap(i / (len(TEMPS) - 1))
    ax[2].plot(res.t[lo:hi] - res.t[lo],
               res.stress[lo:hi] - res.prediction[lo:hi],
               "o", ms=.5, color=col, alpha=.4)
ax[2].axhline(0, color="k", lw=.8)
ax[2].set_xlabel("$t$ [s]"); ax[2].set_ylabel(r"$\sigma-\bar{\sigma}$ [Pa]")
ax[2].set_title("joint-fit residuals by record", fontsize=8)

for a_ in ax: a_.grid(False)
fig.tight_layout(); plt.show()
