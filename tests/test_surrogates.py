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
    _surrogate_job,
    collect_dhvg_irreversibility,
    plot_dhvg_irreversibility,
    summarize_dhvg_irreversibility,
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


def test_summarize_dhvg_irreversibility_paired_wilcoxon() -> None:
    """Original dHVG KL separates from surrogates; saturated i_dir does not."""
    from tsg.audit.surrogates import summarize_dhvg_irreversibility

    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    for realization in range(16):
        rows.append(
            {
                "dataset_id": "logistic",
                "realization": realization,
                "surrogate_kind": "original",
                "i_dir": 1.0,
                "kl_inout": 0.30 + 0.004 * float(rng.normal()),
            }
        )
        for kind, center in (("shuffled", 0.02), ("ft", 0.05), ("iaaft", 0.04)):
            rows.append(
                {
                    "dataset_id": "logistic",
                    "realization": realization,
                    "surrogate_kind": kind,
                    "i_dir": 1.0,
                    "kl_inout": center + 0.004 * float(rng.normal()),
                }
            )
    import pandas as pd

    table = summarize_dhvg_irreversibility(pd.DataFrame(rows))
    assert set(table["metric"]) == {"dHVG__kl_inout", "dHVG__i_dir"}
    assert set(table["contrast"]) == {"shuffled", "ft", "iaaft"}
    assert (table["test"] == "wilcoxon_signed_rank").all()
    assert (table["effect_size_name"] == "rank_biserial").all()

    kl_shuf = table[
        (table["metric"] == "dHVG__kl_inout") & (table["contrast"] == "shuffled")
    ].iloc[0]
    assert kl_shuf["n_pairs"] == 16
    assert kl_shuf["mean_original"] > kl_shuf["mean_surrogate"]
    assert kl_shuf["median_original"] > kl_shuf["median_surrogate"]
    assert kl_shuf["p_value"] < 0.01
    assert kl_shuf["effect_size"] > 0.9

    idir = table[
        (table["metric"] == "dHVG__i_dir") & (table["contrast"] == "iaaft")
    ].iloc[0]
    assert idir["mean_original"] == pytest.approx(1.0)
    assert idir["mean_surrogate"] == pytest.approx(1.0)
    assert idir["std_original"] == pytest.approx(0.0)
    assert idir["p_value"] == pytest.approx(1.0)
    assert idir["effect_size"] == pytest.approx(0.0)


def test_collect_dhvg_irreversibility_uses_stored_kl(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stored ``kl_inout`` is read from the matrix and graphs are not rebuilt."""
    import pandas as pd

    from tsg.audit.surrogates import SurrogateAuditConfig, collect_dhvg_irreversibility
    from tsg.descriptors.base import DESCRIPTOR_IDS

    index = list(DESCRIPTOR_IDS) + ["kl_inout"]
    matrix = pd.DataFrame(0.0, index=index, columns=["HVG", "dHVG"])
    matrix.loc["i_dir", "dHVG"] = 0.37
    matrix.loc["kl_inout", "dHVG"] = 0.21

    def _boom(*_args: object, **_kwargs: object) -> dict[tuple[str, int, str], float]:
        raise AssertionError("dHVG should not be recomputed when kl_inout is stored")

    monkeypatch.setattr("tsg.audit.surrogates._recompute_kl_map", _boom)
    frame = collect_dhvg_irreversibility(
        [
            {
                "dataset_id": "logistic",
                "regime": "chaotic",
                "realization": 0,
                "seed": 1,
                "surrogate_kind": "original",
                "series_length": 8,
                "matrix": matrix,
            }
        ],
        SurrogateAuditConfig(),
    )
    assert frame.loc[0, "kl_source"] == "records"
    assert frame.loc[0, "kl_inout"] == pytest.approx(0.21)
    assert frame.loc[0, "i_dir"] == pytest.approx(0.37)


def test_plot_dhvg_irreversibility_writes_trio(tmp_path: Path) -> None:
    import pandas as pd

    from tsg.audit.surrogates import plot_dhvg_irreversibility

    frame = pd.DataFrame(
        [
            {
                "dataset_id": "logistic",
                "realization": r,
                "surrogate_kind": kind,
                "i_dir": 1.0,
                "kl_inout": 0.2 if kind == "original" else 0.02,
            }
            for r in range(6)
            for kind in ("original", "shuffled", "ft", "iaaft")
        ]
    )
    paths = plot_dhvg_irreversibility(frame, tmp_path, lang="en", regimes=("logistic",))
    stems = {p.stem for p in paths}
    suffixes = {p.suffix for p in paths}
    assert stems == {"surrogate_dhvg_irreversibility"}
    assert suffixes == {".png", ".pdf", ".svg"}
    assert all(p.stat().st_size > 0 for p in paths)


def test_collect_dhvg_kl_recomputes_only_directed_hvg() -> None:
    """``kl_inout`` is absent from the 16-vector and is rebuilt from dHVG."""
    import pandas as pd
    from dataclasses import replace

    from tsg.audit.config import AuditConfig
    from tsg.descriptors.config import DescriptorConfig
    from tsg.descriptors.directional import temporal_irreversibility_kl
    from tsg.generators.config import GeneratorSuiteConfig
    from tsg.generators.registry import build_generator
    from tsg.graphs.geometry import DHVGTransformer

    cfg = SurrogateAuditConfig(
        audit=replace(
            AuditConfig(),
            n_steps=160,
            n_realizations=1,
            seed=42,
            n_folds=2,
            profile="dev",
            regimes=("white_noise",),
            methods=("HVG", "dHVG"),
            families={"visibility": ("HVG", "dHVG")},
        ),
        iaaft_max_iter=15,
        iaaft_min_iter=2,
        iaaft_tol=1e-4,
    )
    generator_config = GeneratorSuiteConfig()
    descriptor_config = DescriptorConfig()
    records = [
        _surrogate_job(
            "white_noise",
            0,
            kind,
            config=cfg,
            generator_config=generator_config,
            descriptor_config=descriptor_config,
        )
        for kind in ("original", "shuffled", "ft", "iaaft")
    ]
    assert "kl_inout" not in records[0]["matrix"].index
    frame = collect_dhvg_irreversibility(
        records,
        cfg,
        generator_config=generator_config,
        n_workers=1,
    )
    assert set(frame["kl_source"]) == {"recomputed_dhvg"}
    assert np.allclose(frame["i_dir"], 1.0)
    assert (frame["kl_inout"] >= 0.0).all()
    assert frame["kl_inout"].nunique() >= 2

    original = next(rec for rec in records if rec["surrogate_kind"] == "original")
    series = build_generator("white_noise", generator_config).generate(
        cfg.audit.n_steps,
        seed=int(original["seed"]),
    )
    direct = temporal_irreversibility_kl(
        DHVGTransformer().transform(np.asarray(series.values, dtype=float), "white_noise").graph
    )
    stored = float(frame.loc[frame["surrogate_kind"] == "original", "kl_inout"].iloc[0])
    assert stored == pytest.approx(direct)

    again = collect_dhvg_irreversibility(
        records,
        cfg,
        generator_config=generator_config,
        n_workers=2,
    )
    pd.testing.assert_frame_equal(frame, again)

    summary = summarize_dhvg_irreversibility(frame)
    assert len(summary) == 6
    assert (summary.loc[summary["metric"] == "dHVG__i_dir", "p_value"] == 1.0).all()


def test_dhvg_registered_opt_in_not_default() -> None:
    assert "dHVG" not in METHOD_IDS
    assert "dHVG" in ALL_METHOD_IDS
    default = build_transformers()
    assert all(t.method != "dHVG" for t in default)
    with_dhvg = build_transformers(TS2GSuiteConfig(methods=("HVG", "dHVG")))
    assert [t.method for t in with_dhvg] == ["HVG", "dHVG"]


def test_temporal_row_additive_decomposition() -> None:
    from tsg.audit.surrogates import _temporal_row

    row = _temporal_row(f1_orig=1.0, f1_shuf=0.2, f1_ft=0.5, f1_iaaft=0.8)
    assert row["delta_T"] == pytest.approx(0.8)
    assert row["delta_marginal"] == pytest.approx(0.2)
    assert row["delta_linear"] == pytest.approx(0.3)
    assert row["delta_iaaft_minus_ft"] == pytest.approx(0.3)
    assert row["delta_nonlinear"] == pytest.approx(0.2)
    assert (
        row["delta_marginal"]
        + row["delta_linear"]
        + row["delta_iaaft_minus_ft"]
        + row["delta_nonlinear"]
    ) == pytest.approx(row["macro_f1_original"])


def test_end_match_series_closes_periodic_seam() -> None:
    from scripts.diagnose_arma_iaaft import end_match_series

    raw = np.linspace(0.0, 1.0, 64) + 0.05 * np.sin(np.linspace(0.0, 8.0, 64))
    matched = end_match_series(raw)
    assert matched[0] == pytest.approx(matched[-1])
    assert abs(raw[0] - raw[-1]) > 0.5
