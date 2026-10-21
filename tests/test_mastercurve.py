"""
tests/test_mastercurve.py
==========================
End-to-end tests for the multi-temperature master-curve pipeline
(`rheogp.mastercurve`) and the reusable chirp-cleaning utilities
(`rheogp.preprocessing`).

The core correctness check is `test_compare_master_curve_models_picks_true_model`:
generate synthetic chirps at 3 temperatures from a KNOWN model
(FractionalMaxwell) with a WLF shift, fit both the true model and a
deliberately mis-specified one (Maxwell) jointly across all chirps,
and assert the correctly-specified model wins by BIC -- this is far
more robust than checking exact shift-factor / parameter recovery
(which needs multiple random restarts to pin down reliably, as noted
in tests/test_zener.py and the package's own docs) and is exactly the
"compare models with BIC/AIC for the master curve" workflow this
module exists for.

Fitting is expensive, so each *fitted* object (a plain `MasterCurve`
fit and the two-model comparison) is built ONCE per test session via a
module-scoped fixture and reused read-only across the assertion tests
below, rather than re-fit per test.
"""

import numpy as np
import pytest

from rheogp.utils import make_tts_dataset
from rheogp.mastercurve import MasterCurve, compare_master_curve_models, fft_transfer
from rheogp.preprocessing import clean_chirp, rolling_rms, parse_temperature

_FITKW = dict(kernel="rbf+linear", inducing_points=100,
             n_epochs=1500, patience=250,
             n_epochs_linear=700, patience_linear=150,
             learning_rate=0.01, learning_rate_physics=0.02,
             verbose=False)


@pytest.fixture(scope="module")
def tts_data():
    truth, records = make_tts_dataset(
        model="fmm", temps=[60, 70, 80], T_ref=70,
        C1=4.0, C2=150.0, tau_c_ref=1.0,
        n=700, dt=2e-3, noise=0.01)
    return truth, records


@pytest.fixture(scope="module")
def fitted_mc(tts_data):
    """One FractionalMaxwell MasterCurve, fit once and reused read-only."""
    truth, records = tts_data
    mc = MasterCurve(records, truth["temperatures"], T_ref=70.0)
    mc.fit(model="FractionalMaxwell", **_FITKW)
    mc.fit_wlf()
    return mc


@pytest.fixture(scope="module")
def comparison_result(tts_data):
    """compare_master_curve_models(true model vs. mis-specified model),
    fit once and reused read-only."""
    truth, records = tts_data
    return compare_master_curve_models(
        records, truth["temperatures"], ["Maxwell", "FractionalMaxwell"],
        T_ref=70.0, criterion="bic", **_FITKW)


# ----------------------------------------------------------------
# construction / validation
# ----------------------------------------------------------------

def test_mastercurve_rejects_mismatched_or_too_few_records():
    with pytest.raises(ValueError):
        MasterCurve([(np.arange(10), np.zeros(10), np.zeros(10))], [50.0])
    with pytest.raises(ValueError):
        MasterCurve([(np.arange(10), np.zeros(10), np.zeros(10))] * 2,
                    [50.0, 60.0, 70.0])  # length mismatch


def test_mastercurve_sorts_by_temperature(tts_data):
    truth, records = tts_data
    scrambled_T = [truth["temperatures"][2], truth["temperatures"][0],
                  truth["temperatures"][1]]
    scrambled_rec = [records[2], records[0], records[1]]
    mc = MasterCurve(scrambled_rec, scrambled_T)
    assert mc.temperatures == sorted(scrambled_T)


def test_fit_wlf_before_fit_raises():
    mc = MasterCurve([(np.arange(10), np.zeros(10), np.zeros(10))] * 3,
                     [50.0, 60.0, 70.0])
    with pytest.raises(RuntimeError):
        mc.fit_wlf()


# ----------------------------------------------------------------
# one shared fitted MasterCurve -> several read-only assertions
# ----------------------------------------------------------------

def test_mastercurve_fit_quality_and_shift_factor_ordering(fitted_mc):
    assert fitted_mc.metrics_["r2"] > 0.99
    assert fitted_mc.aT_[fitted_mc.i_ref] == pytest.approx(1.0, abs=1e-6)
    # thermorheologically simple WLF shift -> a_T strictly decreases
    # with increasing temperature (below T_ref a_T>1, above T_ref a_T<1)
    assert fitted_mc.aT_[0] > fitted_mc.aT_[1] > fitted_mc.aT_[2]


def test_mastercurve_summary_mentions_model_and_shift_factors(fitted_mc):
    summary = fitted_mc.summary()
    assert "FractionalMaxwell" in summary
    assert "a_T" in summary


def test_fit_wlf_returns_finite_constants_and_predicts_1_at_Tref(fitted_mc):
    assert np.isfinite(fitted_mc.wlf_["C1"])
    assert np.isfinite(fitted_mc.wlf_["C2"])
    assert fitted_mc.predict_aT(70.0) == pytest.approx(1.0, abs=0.05)


def test_mastercurve_predict_at_shift_and_temperature(fitted_mc, tts_data):
    """MasterCurve.predict_at_shift/predict_at_temperature delegate to
    the underlying MultiRecordSPGP (fit() now passes temps=), and
    predict_at_temperature never assumes WLF unless explicitly given
    fitted_mc.predict_aT as the shift_law."""
    truth, records = tts_data
    t_new, g_new, _ = records[0]
    t_new, g_new = t_new[:200], g_new[:200]   # a shorter, different-length probe

    pred, std, bounds = fitted_mc.predict_at_shift([(t_new, g_new, 1.0)])
    assert pred.shape == (200,) == std.shape

    # exact fitted temperature
    T0 = fitted_mc.temperatures[0]
    pred2, _, _ = fitted_mc.predict_at_temperature([(t_new, g_new, T0)])
    assert pred2.shape == (200,)

    # outside the fitted range: refuses without a law
    with pytest.raises(ValueError, match="outside the fitted temperature range"):
        fitted_mc.predict_at_temperature([(t_new, g_new, 500.0)])

    # ... but accepts this MasterCurve's own WLF fit if you explicitly ask for it
    pred3, _, _ = fitted_mc.predict_at_temperature(
        [(t_new, g_new, 500.0)], shift_law=fitted_mc.predict_aT)
    assert pred3.shape == (200,)


def test_model_curve_shapes_and_credible_band(fitted_mc):
    omega, Gp, Gdp, Gp_ci, Gdp_ci = fitted_mc.model_curve(n_omega=30)
    assert omega.shape == (30,) == Gp.shape == Gdp.shape
    assert Gp_ci.shape == (2, 30) and Gdp_ci.shape == (2, 30)
    assert np.all(Gp_ci[0] <= Gp_ci[1] + 1e-9)     # lower <= upper
    _, _, _, none1, none2 = fitted_mc.model_curve(n_omega=10,
                                                  credible_interval=False)
    assert none1 is None and none2 is None


def test_reduced_data_returns_points_on_reduced_axis(fitted_mc, tts_data):
    _, records = tts_data
    omega_ref, Gstar, rec_idx = fitted_mc.reduced_data()
    assert omega_ref.shape == Gstar.shape == rec_idx.shape
    assert omega_ref.size > 0
    assert set(np.unique(rec_idx)) <= set(range(len(records)))


def test_fft_transfer_matches_known_analytic_ratio():
    # y = k*x (a trivial "transfer function" H(w)=k for all w)
    n, dt = 2048, 1e-3
    t = np.arange(n) * dt
    x = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 11 * t)
    y = 3.0 * x
    omega, H = fft_transfer(t, x, y, min_snr_db=40.0)
    assert omega.size > 0
    assert np.allclose(H.real, 3.0, atol=1e-6)
    assert np.allclose(H.imag, 0.0, atol=1e-6)


# ----------------------------------------------------------------
# AIC/BIC model comparison for master curves -- the core ask
# ----------------------------------------------------------------

def test_compare_master_curve_models_picks_true_model(comparison_result):
    best, table = comparison_result
    assert best.model_name == "FractionalMaxwell"
    assert set(table) == {"Maxwell", "FractionalMaxwell"}
    assert (table["FractionalMaxwell"].metrics_["bic"]
           < table["Maxwell"].metrics_["bic"])
    assert (table["FractionalMaxwell"].metrics_["r2"]
           > table["Maxwell"].metrics_["r2"])


# ----------------------------------------------------------------
# preprocessing
# ----------------------------------------------------------------

def test_clean_chirp_discards_startup_and_detrends():
    # static pads around a chirp, plus a linear stress drift; the
    # pads are what clean_chirp fits the drift on
    n, dt = 4000, 1e-3
    t = np.arange(n) * dt
    strain = np.zeros(n)
    i0, i1 = 1500, 3500                   # chirp between the pads
    tail = t[i0:i1] - t[i0]
    strain[i0:i1] = 0.05 * np.sin(2 * np.pi * (0.5 + 5 * tail / tail[-1]) * tail)
    drift = 2.0 * t + 3.0                 # linear stress drift to be removed
    rng = np.random.default_rng(0)
    stress = 1000.0 * strain + drift + 0.05 * rng.normal(size=n)

    tc, gc, sc = clean_chirp(t, strain, stress, discard_s=0.5, pad_s=0.5)
    assert tc[0] == 0.0
    assert len(tc) == len(gc) == len(sc)
    # the first discard_s seconds are dropped
    assert len(tc) == pytest.approx(n - 0.5 / dt, abs=2)
    # drift is removed: what is left is the material response plus noise
    assert np.allclose(sc, 1000.0 * gc, atol=0.5)

    # detrend=False only truncates and re-references time
    tn, gn, sn = clean_chirp(t, strain, stress, discard_s=0.5, detrend=False)
    assert np.array_equal(gn, strain[len(t) - len(tn):])
    assert np.array_equal(sn, stress[len(t) - len(tn):])


def test_clean_chirp_rejects_degenerate_record():
    t = np.array([0.0, 1e-3])
    with pytest.raises(ValueError):
        clean_chirp(t, np.array([0.0, np.nan]), np.array([0.0, 1.0]))


def test_rolling_rms_matches_constant_signal():
    x = np.full(50, 2.0)
    r = rolling_rms(x, 5)
    assert np.allclose(r, 2.0)


@pytest.mark.parametrize("fname,expected", [
    ("chirp_02_T90C.csv", 90.0),
    ("foo_t-15c.csv", -15.0),
    ("no_temp_here.csv", None),
])
def test_parse_temperature(fname, expected):
    assert parse_temperature(fname) == expected
