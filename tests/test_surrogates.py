"""Unit tests for surrogate controls and directed dHVG."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from tsg.audit.surrogates import (
    SurrogateAuditConfig,
    SurrogateAuditResult,
    _format_markdown,
    _plot_summary_figure,
)
from tsg.descriptors.directional import temporal_irreversibility_kl
from tsg.descriptors.pipeline import DescriptorPipeline
from tsg.generators.surrogates import (
    IAAFT_CONVERGENCE_CRITERION,
    ft_surrogate,
    iaaft_surrogate,
    make_surrogate,
    shuffled_surrogate,
)
from tsg.graphs.config import TS2GSuiteConfig
from tsg.graphs.geometry import DHVGTransformer, HVGTransformer
from tsg.graphs.registry import ALL_METHOD_IDS, METHOD_IDS, build_transformers

N = 512
SEED = 11


@pytest.fixture
def chaotic_series() -> np.ndarray:
    """Asymmetric logistic-map trajectory (temporal irreversibility)."""
    rng = np.random.default_rng(SEED)
    x = 0.1 + 0.01 * rng.normal()
    out = np.empty(N, dtype=float)
    for i in range(N):
        x = 4.0 * x * (1.0 - x)
        out[i] = x
    return out


@pytest.fixture
def noisy_series() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    t = np.arange(N, dtype=float)
    return np.sin(2 * np.pi * 0.05 * t) + 0.3 * rng.normal(size=N)


def test_shuffled_preserves_sorted_values(noisy_series: np.ndarray) -> None:
    rng = np.random.default_rng(0)
    surr = shuffled_surrogate(noisy_series, rng)
    assert surr.shape == noisy_series.shape
    assert np.allclose(np.sort(surr), np.sort(noisy_series))
    # Extremely unlikely to keep the same order for N=512.
    assert not np.allclose(surr, noisy_series)


def test_ft_preserves_power_spectrum(noisy_series: np.ndarray) -> None:
    rng = np.random.default_rng(1)
    surr = ft_surrogate(noisy_series, rng)
    amp_orig = np.abs(np.fft.rfft(noisy_series))
    amp_surr = np.abs(np.fft.rfft(surr))
    # Phase randomization keeps magnitudes exactly (up to numerical noise).
    assert np.allclose(amp_surr, amp_orig, rtol=1e-8, atol=1e-8)
    assert surr.shape == noisy_series.shape
    assert np.isfinite(surr).all()


def test_iaaft_preserves_amplitudes_and_spectrum(noisy_series: np.ndarray) -> None:
    rng = np.random.default_rng(2)
    surr = iaaft_surrogate(noisy_series, rng, max_iter=250, tol=1e-8, min_iter=10)
    assert np.allclose(np.sort(surr), np.sort(noisy_series))
    amp_orig = np.abs(np.fft.rfft(noisy_series))
    amp_surr = np.abs(np.fft.rfft(surr))
    # Relative spectrum error must be tight after proper spectral convergence.
    rel = np.linalg.norm(amp_surr - amp_orig) / (np.linalg.norm(amp_orig) + 1e-15)
    assert rel < 0.05
    assert np.isfinite(surr).all()


def test_iaaft_does_not_stop_on_invariant_norm(noisy_series: np.ndarray) -> None:
    """Regression: ||y|| is invariant under rank-order and must not be the stop rule."""
    rng = np.random.default_rng(5)
    # One iteration leaves a larger spectral residual than a fully iterated IAAFT.
    one = iaaft_surrogate(noisy_series, rng, max_iter=1, tol=0.0, min_iter=1)
    rng2 = np.random.default_rng(5)
    full = iaaft_surrogate(noisy_series, rng2, max_iter=250, tol=1e-8, min_iter=10)
    amp = np.abs(np.fft.rfft(noisy_series))
    err_one = np.linalg.norm(np.abs(np.fft.rfft(one)) - amp) / (np.linalg.norm(amp) + 1e-15)
    err_full = np.linalg.norm(np.abs(np.fft.rfft(full)) - amp) / (np.linalg.norm(amp) + 1e-15)
    assert err_full < err_one
    assert err_full < 0.05


def test_iaaft_arma_spectrum_matches_better_than_one_iter() -> None:
    """Linear ARMA must not keep a large spectral residual after IAAFT."""
    from tsg.generators.config import GeneratorSuiteConfig
    from tsg.generators.registry import build_generator

    x = build_generator("arma_correlated", GeneratorSuiteConfig()).generate(1000, seed=42).values
    amp = np.abs(np.fft.rfft(x))
    broken = iaaft_surrogate(x, np.random.default_rng(0), max_iter=1, tol=0.0, min_iter=1)
    fixed = iaaft_surrogate(x, np.random.default_rng(0), max_iter=250, tol=1e-6, min_iter=10)
    err_broken = np.linalg.norm(np.abs(np.fft.rfft(broken)) - amp) / (np.linalg.norm(amp) + 1e-15)
    err_fixed = np.linalg.norm(np.abs(np.fft.rfft(fixed)) - amp) / (np.linalg.norm(amp) + 1e-15)
    assert err_broken > 0.02
    assert err_fixed < 0.02
    assert np.allclose(np.sort(fixed), np.sort(x))


def test_make_surrogate_forwards_iaaft_iteration_bounds(noisy_series: np.ndarray) -> None:
    rng = np.random.default_rng(7)
    direct = iaaft_surrogate(
        noisy_series, rng, max_iter=250, tol=1e-8, min_iter=10
    )
    via = make_surrogate(
        noisy_series,
        "iaaft",
        np.random.default_rng(7),
        max_iter=250,
        tol=1e-8,
        min_iter=10,
    )
    assert np.allclose(direct, via)


def test_surrogate_config_records_corrected_iaaft() -> None:
    """JSON and Markdown record the spectral-error IAAFT parameters."""
    root = Path(__file__).resolve().parents[1]
    cfg = SurrogateAuditConfig.from_yaml(root / "configs" / "phase2_surrogates.yaml")
    assert cfg.iaaft_max_iter == 250
    assert cfg.iaaft_min_iter == 10
    assert IAAFT_CONVERGENCE_CRITERION == "relative_spectral_error"
    result = SurrogateAuditResult(
        config=cfg,
        records=[],
        temporal_sensitivity={"per_family": {}},
        nonlinearity_binary={},
        nonlinearity_multiclass={},
        regime_multiclass_by_surrogate={},
    )
    block = result.metrics_dict()["config"]
    assert block["iaaft_max_iter"] == 250
    assert block["iaaft_min_iter"] == 10
    assert block["iaaft_convergence_criterion"] == "relative_spectral_error"
    md = _format_markdown(result)
    assert "min_iter=10" in md
    assert "max_iter=250" in md
    assert "relative_spectral_error" in md


def test_summary_panel_title_uses_mathtext(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[str] = []
    import matplotlib.pyplot as plt

    real_subplots = plt.subplots

    def _spy_subplots(*args: object, **kwargs: object):
        fig, axes = real_subplots(*args, **kwargs)
        if isinstance(axes, np.ndarray):
            original = axes[0].set_title

            def _capture(title: str, *a: object, **k: object):
                captured.append(title)
                return original(title, *a, **k)

            axes[0].set_title = _capture  # type: ignore[method-assign]
        return fig, axes

    monkeypatch.setattr("tsg.audit.surrogates.plt.subplots", _spy_subplots)
    result = SurrogateAuditResult(
        config=SurrogateAuditConfig(),
        records=[],
        temporal_sensitivity={"per_family": {}},
        nonlinearity_binary={},
        nonlinearity_multiclass={},
        regime_multiclass_by_surrogate={},
    )
    _plot_summary_figure(result, tmp_path)
    assert captured == [
        r"$\Delta_T = \mathrm{F1}(\mathrm{original}) - \mathrm{F1}(\mathrm{shuffled})$ [RF]"
    ]


def test_make_surrogate_dispatch(noisy_series: np.ndarray) -> None:
    rng = np.random.default_rng(3)
    orig = make_surrogate(noisy_series, "original", rng)
    assert np.allclose(orig, noisy_series)
    with pytest.raises(ValueError):
        make_surrogate(noisy_series, "nope", rng)  # type: ignore[arg-type]


def test_surrogate_reproducibility(noisy_series: np.ndarray) -> None:
    a = shuffled_surrogate(noisy_series, np.random.default_rng(99))
    b = shuffled_surrogate(noisy_series, np.random.default_rng(99))
    c = shuffled_surrogate(noisy_series, np.random.default_rng(100))
    assert np.allclose(a, b)
    assert not np.allclose(a, c)


def test_dhvg_is_digraph_with_time_arrow(chaotic_series: np.ndarray) -> None:
    result = DHVGTransformer().transform(chaotic_series, "logistic", regime="chaotic")
    assert result.method == "dHVG"
    assert result.directed is True
    assert isinstance(result.graph, nx.DiGraph)
    assert result.n_nodes == chaotic_series.size
    # All edges follow the time arrow i → j with i < j.
    for u, v in result.graph.edges():
        assert int(u) < int(v)


def test_dhvg_idir_nonzero_on_chaotic(chaotic_series: np.ndarray) -> None:
    dhvg = DHVGTransformer().transform(chaotic_series, "logistic")
    hvg = HVGTransformer().transform(chaotic_series, "logistic")
    pipe = DescriptorPipeline()
    d_dir = pipe.compute(dhvg.graph, dataset_id="logistic", method="dHVG")
    d_und = pipe.compute(hvg.graph, dataset_id="logistic", method="HVG")
    assert d_dir.values["i_dir"] > 0.0
    assert d_und.values["i_dir"] == 0.0
    assert d_dir.extras.get("kl_inout", 0.0) >= 0.0
    # Chaotic logistic should show nonzero irreversibility via KL.
    assert temporal_irreversibility_kl(dhvg.graph) > 0.0


def test_dhvg_registered_opt_in_not_default() -> None:
    assert "dHVG" not in METHOD_IDS
    assert "dHVG" in ALL_METHOD_IDS
    default = build_transformers()
    assert all(t.method != "dHVG" for t in default)
    with_dhvg = build_transformers(TS2GSuiteConfig(methods=("HVG", "dHVG")))
    assert [t.method for t in with_dhvg] == ["HVG", "dHVG"]
