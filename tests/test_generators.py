"""Unit tests for multiregime synthetic generators."""

from __future__ import annotations

import numpy as np
import pytest

from tsg.generators.chaotic import LogisticMap, LorenzX, validate_logistic_x0
from tsg.generators.config import (
    GeneratorSuiteConfig,
    LogisticParams,
    LorenzParams,
    SineParams,
)
from tsg.generators.periodic import GOLDEN_RATIO, Quasiperiodic, SineWave
from tsg.generators.registry import DATASET_IDS, build_default_generators, generate_suite
from tsg.generators.stochastic import BrownianMotion, WhiteNoise


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

    series = LogisticMap(params=LogisticParams(x0=0.5, burn_in=10)).generate(50)
    assert np.isfinite(series.values).all()
    assert series.params["x0"] == 0.1


def test_brownian_is_cumulative() -> None:
    series = BrownianMotion().generate(100, seed=5)
    # Random walk variance grows; |last| typically >> typical single step
    assert series.values.std() > 1.0
