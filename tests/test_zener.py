"""
tests/test_zener.py
====================
Tests for the Fractional Zener family (Zener, FractionalZenerSolidS,
FractionalZenerLiquidS, FractionalZenerLiquidD, FractionalZenerS,
FractionalZener):

  * registry / API smoke tests (every model builds and reports the
    right prefactor / shape-parameter names)
  * Gstar(ω) low- and high-frequency limits -- a fast, deterministic
    check of the analytic storage/loss-modulus formulas that does not
    depend on any GP fit, so it cannot be flaky
  * end-to-end fit quality (R²) on synthetic data for all six models,
    plus a tight parameter-recovery check for the simplest model
    (Zener has a single shape parameter, tau_c, and is well
    identified from one quick fit -- mirroring how the existing
    Springpot tests check alpha recovery).  The remaining, more
    heavily parametrised models (2-4 shape parameters) are only
    checked on fit quality here: a single quick gradient-descent fit
    is not expected to pin down every exponent of a fractional model
    with a band-limited chirp (the package's own docs recommend
    multiple random restarts for that) -- exactly why the existing
    test suite does the same for FMG/FML/FMM/FKV.
  * NotImplementedError is raised early and clearly for
    control="stress", which is intentionally not implemented for this
    family (see rheogp/kernels.py module docstring).
"""

import numpy as np
import pytest

from rheogp import SPGP
from rheogp.kernels import KERNELS
from rheogp.utils import (
    make_synthetic_zener, make_synthetic_fzss, make_synthetic_fzls,
    make_synthetic_fzld, make_synthetic_fzs, make_synthetic_fz,
)

N_SMALL = 1200


def _quick(model_name, **kw):
    return SPGP(model=model_name, inducing_points=80, n_epochs=1200,
                patience=250, verbose=False, **kw)


ZENER_MODELS = [
    "Zener", "FractionalZenerSolidS", "FractionalZenerLiquidS",
    "FractionalZenerLiquidD", "FractionalZenerS", "FractionalZener",
]


# ----------------------------------------------------------------
# registry / API
# ----------------------------------------------------------------

@pytest.mark.parametrize("name", ZENER_MODELS)
def test_registered_and_buildable(name):
    key = name.lower().replace(" ", "").replace("-", "").replace("_", "")
    assert key in KERNELS
    phys = KERNELS[key]()
    phys.configure(control="strain")
    assert len(phys.prefactor_names()) == 2
    assert "Gc" in phys.prefactor_names()


def test_short_aliases_match_full_names():
    assert KERNELS["fzss"] is KERNELS["fractionalzenersolids"]
    assert KERNELS["fzls"] is KERNELS["fractionalzenerliquids"]
    assert KERNELS["fzld"] is KERNELS["fractionalzenerliquidd"]
    assert KERNELS["fzs"]  is KERNELS["fractionalzeners"]
    assert KERNELS["fz"]   is KERNELS["fractionalzener"]


@pytest.mark.parametrize("name", ZENER_MODELS)
def test_stress_control_not_implemented(name):
    key = name.lower().replace(" ", "").replace("-", "").replace("_", "")
    phys = KERNELS[key]()
    phys.configure(control="stress")
    with pytest.raises(NotImplementedError):
        phys.prefactor_names()          # dispatches to _prefactors_stress()
    with pytest.raises(NotImplementedError):
        phys.n_features                 # what RheoGPModel.__init__ calls first
    # and the same failure surfaces immediately (at model construction,
    # before any training) when a user actually requests it via SPGP:
    t = np.arange(50) * 1e-3
    g = np.sin(t)
    s = np.cos(t)
    with pytest.raises(NotImplementedError):
        SPGP(model=name, control="stress", inducing_points=10,
            n_epochs=1, patience=1, verbose=False).fit(t, g, s)


# ----------------------------------------------------------------
# Gstar(ω) limits -- deterministic, no fitting
# ----------------------------------------------------------------

def _limits(name, kwargs, prefactors, omega=(1e-8, 1e8)):
    phys = KERNELS[name.lower()](**kwargs)
    phys.configure(control="strain")
    Gp, Gdp = phys.Gstar(np.asarray(omega, dtype=float), prefactors)
    return Gp, Gdp


def test_zener_gstar_limits():
    Gp, Gdp = _limits("zener", dict(tau_c_init=1.0),
                      {"Gc": 70.0, "G0": 30.0})
    assert Gp[0] == pytest.approx(30.0, abs=1e-2)          # ω→0 : G0
    assert Gp[1] == pytest.approx(100.0, rel=1e-2)          # ω→∞ : G0+Gc
    assert Gdp[0] < 1e-2 and Gdp[1] < 1e-2                  # loss -> 0 both ends


def test_fzss_gstar_limits():
    Gp, Gdp = _limits("fzss", dict(alpha_init=0.5, tau_c_init=1.0),
                      {"Gc": 70.0, "G0": 30.0})
    assert Gp[0] == pytest.approx(30.0, abs=1.0)
    assert Gp[1] == pytest.approx(100.0, rel=1e-2)
    assert Gdp[0] < 1.0 and Gdp[1] < 1.0


def test_fzs_gstar_low_freq_limit_is_G0():
    Gp, Gdp = _limits("fzs",
                      dict(alpha_init=0.7, beta_init=0.3, tau_c_init=1.0),
                      {"Gc": 70.0, "G0": 30.0})
    assert Gp[0] == pytest.approx(30.0, abs=1.0)
    assert Gdp[0] < 1.0


def test_fzld_dashpot_dominates_loss_at_high_freq():
    # dashpot(eta_p) in parallel -> Gdp ~ eta_p * omega at high freq,
    # and contributes NOTHING to storage (Gp) at any frequency.
    omega = np.array([1e-6, 1e6])
    phys = KERNELS["fzld"](beta_init=0.5, tau_c_init=1.0)
    phys.configure(control="strain")
    Gp, Gdp = phys.Gstar(omega, {"Gc": 70.0, "eta_p": 5.0})
    assert Gdp[1] == pytest.approx(5.0 * 1e6, rel=1e-2)
    assert Gp[0] == pytest.approx(0.0, abs=1e-2)


def test_fz_general_kappa_branch_contributes_to_both_moduli():
    omega = np.array([10.0])
    phys = KERNELS["fz"](alpha_init=0.7, beta_init=0.3, kappa_init=0.4,
                         tau_c_init=1.0)
    phys.configure(control="strain")
    Gp, Gdp = phys.Gstar(omega, {"Gc": 70.0, "K": 20.0})
    k = phys.kappa.item()
    springpot_Gp = 20.0 * 10.0**k * np.cos(np.pi * k / 2)
    springpot_Gdp = 20.0 * 10.0**k * np.sin(np.pi * k / 2)
    assert Gp[0] > springpot_Gp - 1e-6     # arm adds a strictly positive bit
    assert Gdp[0] > springpot_Gdp - 1e-6


# ----------------------------------------------------------------
# end-to-end fit quality
# ----------------------------------------------------------------

def test_zener_strain_causal_recovers_tau_c():
    t, g, s = make_synthetic_zener(G0=30.0, G1=70.0, eta=70.0,
                                   n=N_SMALL, dt=2e-3, noise=0.02)
    m = _quick("Zener").fit(t, g, s)
    assert m.metrics_["r2"] > 0.95
    assert abs(m.gp_model_.phys.tau_c.item() - 1.0) < 0.35


@pytest.mark.parametrize("name,gen,kw", [
    ("FractionalZenerSolidS", make_synthetic_fzss,
     dict(G0=30., V=100., G1=100., alpha=0.5)),
    ("FractionalZenerLiquidS", make_synthetic_fzls,
     dict(G0=30., G_bb=100., eta=100., beta=0.5)),
    ("FractionalZenerLiquidD", make_synthetic_fzld,
     dict(eta_p=10., G_bb=100., eta_s=100., beta=0.5)),
    ("FractionalZenerS", make_synthetic_fzs,
     dict(G0=30., V=100., G_bb=100., alpha=0.7, beta=0.3)),
    ("FractionalZener", make_synthetic_fz,
     dict(K=20., kappa=0.2, V=100., G_bb=100., alpha=0.7, beta=0.3)),
])
def test_zener_family_fits_synthetic_data_well(name, gen, kw):
    t, g, s = gen(n=N_SMALL, dt=2e-3, noise=0.02, **kw)
    m = _quick(name).fit(t, g, s)
    assert m.metrics_["r2"] > 0.95
    assert np.isfinite(m.metrics_["bic"])
    # every reported prefactor sensitivity should be a finite, positive
    # scalar -- catches sign errors / degenerate features outright,
    # without demanding exact recovery of every shape exponent from a
    # single quick fit (see module docstring).
    for pname, info in m.prefactors_.items():
        assert np.isfinite(info["scalar_mean"]), pname
        assert info["scalar_mean"] > 0, pname


def test_zener_results_and_summary_work():
    t, g, s = make_synthetic_zener(n=N_SMALL, dt=2e-3, noise=0.02)
    m = _quick("Zener").fit(t, g, s)
    res = m.results()
    assert res.feature_names == ["Gc", "G0"]
    assert res.omega is not None and res.Gp is not None
    txt = m.summary()
    assert "Zener" in txt and "tau_c" in txt
