"""Unit tests for multiregime synthetic generators."""

from __future__ import annotations

import numpy as np
import pytest

from tsg.generators.chaotic import (
    HenonMap,
    LogisticMap,
    LogisticWeak,
    LorenzX,
    Rossler,
    validate_logistic_x0,
)
from tsg.generators.config import (
    GeneratorSuiteConfig,
    HenonParams,
    LogisticParams,
    LorenzParams,
    RegimeSwitchingParams,
    RosslerParams,
    SineParams,
)
from tsg.generators.nonstationary import RegimeSwitching
from tsg.generators.periodic import GOLDEN_RATIO, Quasiperiodic, SineWave
from tsg.generators.registry import (
    CANONICAL_REGIMES,
    DATASET_IDS,
    build_default_generators,
    build_generator,
    generate_suite,
)
from tsg.generators.stochastic import (
    ARMACorrelated,
    BrownianMotion,
    FractionalBrownianMotion,
    WhiteNoise,
)


N_STEPS = 512
SEED = 123


@pytest.fixture
def suite() -> list:
    from dataclasses import replace

    cfg = GeneratorSuiteConfig()
    cfg = replace(
        cfg,
        global_params=replace(cfg.global_params, n_steps=N_STEPS, seed=SEED),
        lorenz=replace(cfg.lorenz, burn_in=100),
        logistic=replace(cfg.logistic, burn_in=50),
    )
    return generate_suite(cfg)


def test_registry_has_six_dataset_ids() -> None:
    assert list(DATASET_IDS) == [
        "white_noise",
        "brownian",
        "sine",
        "quasiperiodic",
        "lorenz",
        "logistic",
    ]
    assert len(build_default_generators()) == 6


def test_suite_lengths_and_finite(suite: list) -> None:
    assert len(suite) == 6
    for series in suite:
        assert series.length == N_STEPS
        assert series.values.dtype == float or np.issubdtype(series.values.dtype, np.floating)
        assert np.isfinite(series.values).all()
        assert series.dataset_id in DATASET_IDS


def test_reproducibility_same_seed() -> None:
    cfg = GeneratorSuiteConfig()
    from dataclasses import replace

    cfg = replace(cfg, global_params=replace(cfg.global_params, n_steps=256, seed=7))
    a = generate_suite(cfg)
    b = generate_suite(cfg)
    for sa, sb in zip(a, b, strict=True):
        np.testing.assert_allclose(sa.values, sb.values)


def test_different_seed_changes_stochastic() -> None:
    a = WhiteNoise().generate(200, seed=1)
    b = WhiteNoise().generate(200, seed=2)
    assert not np.allclose(a.values, b.values)


def test_sine_jitter_toggle() -> None:
    with_jitter = SineWave(params=SineParams(jitter=True, jitter_sigma=1e-3))
    without = SineWave(params=SineParams(jitter=False))
    y_j = with_jitter.generate(300, seed=0).values
    y_c = without.generate(300, seed=0).values
    assert not np.allclose(y_j, y_c)
    # clean sine must be exactly periodic (no seed dependency)
    y_c2 = without.generate(300, seed=99).values
    np.testing.assert_allclose(y_c, y_c2)


def test_quasiperiodic_uses_golden_ratio() -> None:
    gen = Quasiperiodic()
    params = gen.params_dict()
    assert params["f1"] == pytest.approx(0.02)
    assert params["f2"] == pytest.approx(0.02 / GOLDEN_RATIO)


def test_lorenz_rk4_dt_and_variable() -> None:
    gen = LorenzX(params=LorenzParams(dt=0.02, burn_in=50))
    series = gen.generate(100, seed=0, burn_in=50)
    assert series.variable == "x-value"
    assert series.params["dt"] == 0.02
    assert series.burn_in == 50
    assert series.length == 100
    assert np.isfinite(series.values).all()


def test_logistic_rejects_degenerate_x0() -> None:
    assert validate_logistic_x0(0.0) == 0.1
    assert validate_logistic_x0(0.5) == 0.1
    assert validate_logistic_x0(1.0) == 0.1
    assert validate_logistic_x0(1.5) == 0.1
    assert validate_logistic_x0(0.3) == 0.3

    # Realization x0 is sampled from x0_range (params.x0 is only a catalog default).
    series = LogisticMap(params=LogisticParams(x0=0.5, burn_in=10)).generate(
        50, seed=0, burn_in=10
    )
    assert np.isfinite(series.values).all()
    assert 0.05 <= series.params["x0"] <= 0.95
    assert series.params["x0"] not in {0.0, 0.5, 1.0}


def test_brownian_is_cumulative() -> None:
    series = BrownianMotion().generate(100, seed=5)
    # Random walk variance grows; |last| typically >> typical single step
    assert series.values.std() > 1.0


def test_seed_variability_quasi_lorenz_logistic() -> None:
    """Ensemble realizations must differ for quasiperiodic / lorenz / logistic."""
    q1 = Quasiperiodic().generate(300, seed=1).values
    q2 = Quasiperiodic().generate(300, seed=2).values
    assert not np.allclose(q1, q2)

    lo1 = LorenzX(params=LorenzParams(burn_in=50)).generate(200, seed=1).values
    lo2 = LorenzX(params=LorenzParams(burn_in=50)).generate(200, seed=2).values
    assert not np.allclose(lo1, lo2)

    lg1 = LogisticMap(params=LogisticParams(burn_in=20)).generate(200, seed=1).values
    lg2 = LogisticMap(params=LogisticParams(burn_in=20)).generate(200, seed=2).values
    assert not np.allclose(lg1, lg2)


def test_lorenz_samples_rho_sigma_in_safe_window() -> None:
    gen = LorenzX(params=LorenzParams(burn_in=40))
    series = gen.generate(80, seed=11, burn_in=40)
    assert 27.0 <= series.params["rho"] <= 29.0
    assert 9.5 <= series.params["sigma"] <= 10.5
    assert series.params["beta"] == pytest.approx(8.0 / 3.0)


def test_logistic_keeps_r_fixed_and_random_burn_in() -> None:
    gen = LogisticMap()
    a = gen.generate(64, seed=3)
    b = gen.generate(64, seed=4)
    assert a.params["r"] == 4.0
    assert b.params["r"] == 4.0
    assert 500 <= a.params["burn_in"] < 1500
    assert 500 <= b.params["burn_in"] < 1500
    assert a.params["burn_in"] != b.params["burn_in"] or not np.allclose(
        a.values, b.values
    )


def test_quasiperiodic_perturbs_f1_and_amplitudes() -> None:
    a = Quasiperiodic().generate(200, seed=1)
    b = Quasiperiodic().generate(200, seed=2)
    assert 0.018 <= a.params["f1"] <= 0.022
    assert 0.85 <= a.params["amp2_ratio"] <= 1.15
    assert a.params["f1"] != b.params["f1"] or a.params["amp2_ratio"] != b.params[
        "amp2_ratio"
    ]


def _frontier_generators() -> list:
    return [
        FractionalBrownianMotion(variant="antipersistent"),
        FractionalBrownianMotion(variant="persistent"),
        ARMACorrelated(),
        Rossler(params=RosslerParams(burn_in=200)),
        HenonMap(params=HenonParams(burn_in=200)),
        LogisticWeak(),
        RegimeSwitching(
            params=RegimeSwitchingParams(lorenz_burn_in=150, rossler_burn_in=150)
        ),
    ]


@pytest.mark.parametrize(
    "gen",
    _frontier_generators(),
    ids=lambda g: g.dataset_id,
)
def test_frontier_generators_finite_reproducible_diverse(gen) -> None:
    """New frontier generators: finite, seeded, distinct, non-degenerate."""
    a = gen.generate(256, seed=11)
    b = gen.generate(256, seed=11)
    c = gen.generate(256, seed=12)
    assert a.length == 256
    assert a.dataset_id == gen.dataset_id
    assert np.isfinite(a.values).all()
    assert float(a.values.std()) > 0.0
    np.testing.assert_allclose(a.values, b.values)
    assert not np.allclose(a.values, c.values)


def test_fbm_hurst_window_and_increment_correlation() -> None:
    anti = FractionalBrownianMotion(variant="antipersistent").generate(2048, seed=4)
    pers = FractionalBrownianMotion(variant="persistent").generate(2048, seed=4)
    assert 0.15 <= anti.params["hurst"] <= 0.25
    assert 0.75 <= pers.params["hurst"] <= 0.85

    def lag1(path: np.ndarray) -> float:
        inc = np.diff(path)
        inc = inc - inc.mean()
        return float(np.dot(inc[:-1], inc[1:]) / np.dot(inc, inc))

    assert lag1(anti.values) < 0.0
    assert lag1(pers.values) > 0.0


def test_arma_is_strongly_positively_autocorrelated() -> None:
    series = ARMACorrelated().generate(2000, seed=9)
    assert 0.96 <= series.params["rho"] <= 0.99
    x = series.values - series.values.mean()
    lag1 = float(np.dot(x[:-1], x[1:]) / np.dot(x, x))
    assert lag1 > 0.5


def test_henon_stays_bounded_and_samples_a() -> None:
    series = HenonMap(params=HenonParams(burn_in=200)).generate(400, seed=3)
    assert 1.38 <= series.params["a"] <= 1.41
    assert series.params["b"] == pytest.approx(0.3)
    assert np.max(np.abs(series.values)) < 5.0


def test_logistic_weak_stays_in_band_and_unit_interval() -> None:
    series = LogisticWeak().generate(300, seed=6)
    r = series.params["r"]
    assert series.params["r_range"][0] <= r <= series.params["r_range"][1] or r == pytest.approx(
        series.params["r_fallback"]
    )
    assert np.all(series.values > 0.0)
    assert np.all(series.values < 1.0)


def test_rossler_samples_chaotic_window() -> None:
    series = Rossler(params=RosslerParams(burn_in=100)).generate(200, seed=2)
    assert 0.18 <= series.params["a"] <= 0.22
    assert 0.18 <= series.params["b"] <= 0.22
    assert 5.5 <= series.params["c"] <= 5.9
    assert series.variable == "x-value"


def test_regime_switching_change_point_and_source() -> None:
    gen = RegimeSwitching(
        params=RegimeSwitchingParams(lorenz_burn_in=80, rossler_burn_in=80)
    )
    series = gen.generate(400, seed=15)
    n = series.length
    lo = int(np.floor(0.35 * n))
    hi = int(np.floor(0.65 * n))
    assert lo <= series.params["change_index"] <= hi
    assert series.params["transition"] in {"abrupt", "smooth"}
    assert series.params["chaotic_source"] in {"lorenz", "rossler"}
    assert series.params["standardized"] is True
    assert series.regime == "nonstationary"
    other = gen.generate(400, seed=16)
    assert series.params["change_index"] != other.params["change_index"] or not np.allclose(
        series.values, other.values
    )


def test_factory_registers_frontier_regimes_without_shifting_phase1_seeds() -> None:
    assert CANONICAL_REGIMES[:6] == DATASET_IDS
    assert CANONICAL_REGIMES.index("logistic") == 5
    cfg = GeneratorSuiteConfig()
    for dataset_id in CANONICAL_REGIMES:
        gen = build_generator(dataset_id, cfg)
        assert gen.dataset_id == dataset_id
    with pytest.raises(KeyError):
        build_generator("not_a_regime", cfg)
