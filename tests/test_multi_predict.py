"""
tests/test_multi_predict.py
============================
Regression tests for MultiRecordSPGP.predict_at_shift() /
predict_at_temperature() -- the joint-fit "surrogate" prediction path
added to fix the inherited SPGP.predict(), which mis-slices any input
that isn't byte-identical to the training record layout (see
CHANGELOG). These use a small/cheap synthetic Maxwell fit purely to
exercise the *mechanics* (shape handling, shift-law dispatch, error
messages) -- not to re-validate physics recovery, which the slower
FMM/FZSS notebooks already cover.
"""

import numpy as np
import pytest
import torch

from rheogp.rheogp_multi import MultiRecordSPGP

TEMPS = [30.0, 50.0, 70.0]


def _toy_records(seed=0, n=300, temps=TEMPS):
    rng = np.random.default_rng(seed)
    records = []
    for T in temps:
        aT = 10.0 ** (-0.02 * (T - 50.0))     # arbitrary monotone shift law
        t = np.linspace(0, 3, n)
        g = 0.05 * np.sin(2 * np.pi * (0.5 + 2 * t) * t)
        tau = 0.5 * aT
        s = 4000.0 * aT * g + 50 * rng.standard_normal(n)  # crude toy signal
        records.append((t, g, s))
    return records


def _quick_fit(temps=TEMPS, **kw):
    records = _toy_records(temps=temps)
    m = MultiRecordSPGP(records, i_ref=1, temps=temps, model="Maxwell",
                        kernel="linear", inducing_points=25, n_epochs=40,
                        patience=100, verbose=False, **kw)
    m.fit_all()
    return m


def test_predict_at_shift_matches_input_shape():
    m = _quick_fit()
    t_new = np.linspace(0, 2, 150)
    g_new = 0.03 * np.sin(2 * np.pi * (0.4 + 1.5 * t_new) * t_new)
    pred, std, bounds = m.predict_at_shift([(t_new, g_new, 1.0)])
    assert bounds == [(0, 150)]
    assert pred.shape == (150,)
    assert std.shape == (150,)
    assert np.all(np.isfinite(pred))
    assert np.all(std >= 0)


def test_predict_at_shift_multi_record_different_bounds_than_training():
    m = _quick_fit()   # trained on 3 records of length 300 each
    t1 = np.linspace(0, 1.5, 90)
    g1 = 0.02 * np.sin(2 * np.pi * (0.6 + 3 * t1) * t1)
    t2 = np.linspace(0, 2.2, 140)
    g2 = 0.04 * np.sin(2 * np.pi * (0.3 + 1.0 * t2) * t2)
    pred, std, bounds = m.predict_at_shift([(t1, g1, 2.0), (t2, g2, 0.5)])
    assert bounds == [(0, 90), (90, 230)]
    assert pred.shape == (230,)
    # training bounds must be restored untouched afterward
    assert m.gp_model_.phys.bounds == m._bounds


def test_predict_at_temperature_exact_matches_predict_at_shift():
    m = _quick_fit()
    t_new = np.linspace(0, 2, 120)
    g_new = 0.02 * np.sin(2 * np.pi * (0.5 + 2 * t_new) * t_new)
    aT_fit = m.shift_factors()
    i70 = TEMPS.index(70.0)

    pred_shift, _, _ = m.predict_at_shift([(t_new, g_new, aT_fit[i70])])
    pred_temp, _, _ = m.predict_at_temperature([(t_new, g_new, 70.0)])
    np.testing.assert_allclose(pred_shift, pred_temp, rtol=1e-5, atol=1e-8)


def test_predict_at_temperature_interpolates_without_a_law():
    m = _quick_fit()
    t_new = np.linspace(0, 2, 100)
    g_new = 0.02 * np.sin(2 * np.pi * (0.5 + 2 * t_new) * t_new)
    # strictly between two fitted temperatures -> must NOT raise
    pred, std, bounds = m.predict_at_temperature([(t_new, g_new, 42.0)])
    assert pred.shape == (100,)


def test_predict_at_temperature_refuses_to_extrapolate_silently():
    m = _quick_fit()
    t_new = np.linspace(0, 2, 100)
    g_new = 0.02 * np.sin(2 * np.pi * (0.5 + 2 * t_new) * t_new)
    with pytest.raises(ValueError, match="outside the fitted temperature range"):
        m.predict_at_temperature([(t_new, g_new, 200.0)])


def test_predict_at_temperature_accepts_caller_supplied_shift_law():
    m = _quick_fit()
    t_new = np.linspace(0, 2, 100)
    g_new = 0.02 * np.sin(2 * np.pi * (0.5 + 2 * t_new) * t_new)
    # any callable is accepted -- library imposes no functional form
    my_law = lambda T: 10.0 ** (-0.02 * (T - 50.0))
    pred, std, bounds = m.predict_at_temperature(
        [(t_new, g_new, 200.0)], shift_law=my_law)
    assert pred.shape == (100,)

    # a completely different (even nonsensical) callable is equally accepted --
    # confirms no law is privileged over any other
    pred2, _, _ = m.predict_at_temperature(
        [(t_new, g_new, 200.0)], shift_law=lambda T: 1.0)
    assert pred2.shape == (100,)
    assert not np.allclose(pred, pred2)   # different laws -> different answers


def test_predict_at_temperature_requires_temps_at_construction():
    records = _toy_records()
    m = MultiRecordSPGP(records, i_ref=1, model="Maxwell", kernel="linear",
                        inducing_points=25, n_epochs=40, patience=100,
                        verbose=False)   # no temps=
    m.fit_all()
    t_new = np.linspace(0, 2, 100)
    g_new = 0.02 * np.sin(2 * np.pi * (0.5 + 2 * t_new) * t_new)
    with pytest.raises(ValueError, match="needs to know"):
        m.predict_at_temperature([(t_new, g_new, 50.0)])


def test_temps_length_mismatch_raises_at_construction():
    records = _toy_records()
    with pytest.raises(ValueError, match="same length"):
        MultiRecordSPGP(records, i_ref=1, temps=[30.0, 50.0],  # wrong length
                        model="Maxwell", kernel="linear")


def test_tau_ref_property():
    m = _quick_fit()
    tr = m.tau_ref
    assert tr > 0 and np.isfinite(tr)
