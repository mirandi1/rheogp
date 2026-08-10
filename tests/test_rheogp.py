"""
tests/test_rheogp.py
====================
Smoke + numerical tests for rheogp 0.3:
  * feature engine (causal exact / steady) against analytic references
  * strain- and stress-controlled fitting
  * FitResult accessors + save/load roundtrip
"""

import numpy as np
import torch
import pytest

import rheogp
from rheogp import SPGP, FitResult
from rheogp.features import (
    rate, frac_deriv_causal, frac_deriv_steady,
)
from rheogp.utils import (
    make_synthetic_springpot,
    make_synthetic_springpot_stress,
    make_synthetic_maxwell_stress,
)

N_SMALL = 400


def _quick(model_name, **kw):
    return SPGP(model=model_name, inducing_points=50, n_epochs=250,
                patience=80, verbose=False, **kw)


# ----------------------------------------------------------------
# feature engine
# ----------------------------------------------------------------

def test_steady_feature_is_exact_on_periodic_sine():
    N, dt = 2048, 1e-3
    T = N * dt
    w = 2 * np.pi * 8 / T                 # exactly periodic
    t = np.arange(N) * dt
    alpha = 0.6
    g = torch.tensor(np.sin(w * t), dtype=torch.float32)
    x = frac_deriv_steady(g, dt, torch.tensor(alpha)).numpy()
    x_true = w**alpha * np.sin(w * t + np.pi * alpha / 2)
    assert np.max(np.abs(x - x_true)) / w**alpha < 1e-4


def test_causal_exact_beats_left_rule():
    N, dt = 2048, 1e-3
    T = N * dt
    w = 2 * np.pi * 8 / T
    t = np.arange(N) * dt
    alpha = 0.6
    g = torch.tensor(np.sin(w * t), dtype=torch.float32)
    x_true = w**alpha * np.sin(w * t + np.pi * alpha / 2)
    late = slice(N // 2, N)
    gd = rate(g, dt)
    e_exact = np.max(np.abs(
        frac_deriv_causal(gd, dt, torch.tensor(alpha), "exact"
                          ).numpy()[late] - x_true[late]))
    e_left = np.max(np.abs(
        frac_deriv_causal(gd, dt, torch.tensor(alpha), "left"
                          ).numpy()[late] - x_true[late]))
    assert e_exact < 0.1 * e_left


# ----------------------------------------------------------------
# strain-controlled fitting
# ----------------------------------------------------------------

def test_springpot_strain_causal():
    t, g, s = make_synthetic_springpot(n=N_SMALL, dt=2e-3, noise=0.02)
    m = _quick("Springpot").fit(t, g, s)
    assert m.metrics_["r2"] > 0.95
    assert abs(m.gp_model_.phys.alpha.item() - 0.5) < 0.15


def test_springpot_strain_steady_on_windowed_chirp():
    # steady mode assumes the record is one period of a periodic
    # protocol -> valid for a Tukey-tapered (OWCh-style) chirp
    t, g, s = make_synthetic_springpot(n=800, dt=2e-3, noise=0.02,
                                       taper=0.1)
    m = _quick("Springpot", convolution="steady").fit(t, g, s)
    assert m.metrics_["r2"] > 0.95
    assert abs(m.gp_model_.phys.alpha.item() - 0.5) < 0.1


# ----------------------------------------------------------------
# stress-controlled fitting
# ----------------------------------------------------------------

def test_springpot_stress_control():
    t, g, s = make_synthetic_springpot_stress(V=100., alpha=0.5,
                                              n=600, dt=2e-3, noise=0.02)
    m = SPGP(model="Springpot", control="stress", inducing_points=60,
             n_epochs=800, patience=250, verbose=False).fit(t, g, s)
    assert m.metrics_["r2"] > 0.98
    assert abs(m._derived_params()["V"] - 100.0) < 10.0


def test_maxwell_stress_control_has_no_shape_params():
    t, g, s = make_synthetic_maxwell_stress(n=600, dt=4e-3, noise=0.01)
    m = SPGP(model="Maxwell", control="stress", inducing_points=60,
             n_epochs=800, patience=250, verbose=False).fit(t, g, s)
    assert m.gp_model_.phys.named_phys_params() == {}
    d = m._derived_params()
    assert abs(d["Gc"] - 100.0) < 10.0
    assert abs(d["tau_c"] - 1.0) < 0.15


# ----------------------------------------------------------------
# results container
# ----------------------------------------------------------------

def test_results_and_roundtrip(tmp_path):
    t, g, s = make_synthetic_springpot(n=N_SMALL, dt=2e-3, noise=0.02)
    m = _quick("Springpot").fit(t, g, s)
    res = m.results()
    assert res.features.shape == (N_SMALL, 1)
    assert res.feature_names == ["V"]
    assert res.omega is not None and res.Gp is not None
    assert "loss" in res.history

    res.save(tmp_path / "out")
    res2 = FitResult.load(tmp_path / "out")
    assert np.allclose(res2.prediction, res.prediction)
    assert np.allclose(res2.features, res.features)


def test_predict_with_convolution_override():
    t, g, s = make_synthetic_springpot(n=N_SMALL, dt=2e-3, noise=0.02)
    m = _quick("Springpot", convolution="steady").fit(t, g, s)
    step = np.where(np.arange(200) * 2e-3 > 0.1, 0.05, 0.0)
    p = m.predict(step, convolution="causal")
    assert np.isfinite(p).all()


def test_information_criteria_complete():
    t, g, s = make_synthetic_springpot(n=N_SMALL, dt=2e-3, noise=0.02)
    m = _quick("Springpot").fit(t, g, s)
    met = m.metrics_
    for key in ("aic", "aicc", "bic", "nll", "n_params", "n_data"):
        assert key in met
    np_ = met["n_params"]
    # Springpot: 1 shape param (alpha) + 1 prefactor (V) + GP hypers
    assert np_["n_shape"] == 1
    assert np_["n_prefactors"] == 1
    assert np_["n_gp_hyper"] >= 2          # at least a scale + noise
    assert np_["k_total"] == (np_["n_shape"] + np_["n_prefactors"]
                              + np_["n_gp_hyper"])
    # definitions hold
    k, N = np_["k_total"], met["n_data"]
    assert abs(met["bic"] - (2 * met["nll"] + k * np.log(N))) < 1e-6
    assert abs(met["aic"] - (2 * met["nll"] + 2 * k)) < 1e-6
    assert met["aicc"] >= met["aic"]


def test_one_liner_api():
    t, g, s = make_synthetic_springpot(n=N_SMALL, dt=2e-3, noise=0.02)
    m = rheogp.fit(t, g, s, model="Springpot", inducing_points=50,
                   n_epochs=200, patience=80, verbose=False)
    assert m._is_fitted
