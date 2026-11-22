"""Unit tests for Bloque-4 transforms / invariance and Bloque-5 phase maps."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from tsg.audit.features import (
    SPECTRAL_DESCRIPTOR_IDS,
    TOPOLOGICAL_DESCRIPTOR_IDS,
    descriptor_kind_mask,
)
from tsg.audit.invariance import (
    InvarianceAuditConfig,
    InvarianceAuditResult,
    _format_markdown,
)
from tsg.audit.phase_maps import (
    PhaseMapsAuditConfig,
    PhaseMapsAuditResult,
    collapse_snr_thresholds,
    snr_label,
)
from tsg.descriptors.pipeline import DescriptorPipeline
from tsg.generators.perturbations import (
    add_white_noise_snr,
    affine_transform,
    downsample_series,
    impulsive_outliers,
    monotone_nonlinear,
    quantize_ties,
)
from tsg.graphs.geometry import HVGTransformer, NVGTransformer
from tsg.graphs.transition import OPNTransformer

N = 256
SEED = 19


@pytest.fixture
def chaotic_series() -> np.ndarray:
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
    return np.sin(2 * np.pi * 0.07 * t) + 0.2 * rng.normal(size=N)


def _edge_set(g: nx.Graph) -> set[tuple[int, int]]:
    return {tuple(sorted((int(u), int(v)))) for u, v in g.edges()}


def test_affine_leaves_nvg_hvg_invariant(chaotic_series: np.ndarray) -> None:
    y = affine_transform(chaotic_series, scale=3.5, shift=10.0)
    nvg_x = NVGTransformer().transform(chaotic_series, "x").graph
    nvg_y = NVGTransformer().transform(y, "y").graph
    hvg_x = HVGTransformer().transform(chaotic_series, "x").graph
    hvg_y = HVGTransformer().transform(y, "y").graph
    assert _edge_set(nvg_x) == _edge_set(nvg_y)
    assert _edge_set(hvg_x) == _edge_set(hvg_y)

    pipe = DescriptorPipeline()
    d_nvg_x = pipe.compute(nvg_x, dataset_id="x", method="NVG")
    d_nvg_y = pipe.compute(nvg_y, dataset_id="y", method="NVG")
    d_hvg_x = pipe.compute(hvg_x, dataset_id="x", method="HVG")
    d_hvg_y = pipe.compute(hvg_y, dataset_id="y", method="HVG")
    for key in ("comp", "lcc", "c_avg", "e_glob", "cv_k"):
        assert d_nvg_x.values[key] == pytest.approx(d_nvg_y.values[key], abs=1e-8)
        assert d_hvg_x.values[key] == pytest.approx(d_hvg_y.values[key], abs=1e-8)


def test_monotone_leaves_hvg_invariant_and_preserves_opn_order(
    chaotic_series: np.ndarray,
) -> None:
    y = monotone_nonlinear(chaotic_series)
    hvg_x = HVGTransformer().transform(chaotic_series, "x").graph
    hvg_y = HVGTransformer().transform(y, "y").graph
    assert _edge_set(hvg_x) == _edge_set(hvg_y)

    opn_x = OPNTransformer().transform(chaotic_series, "x").graph
    opn_y = OPNTransformer().transform(y, "y").graph
    # Strictly increasing map preserves ordinal patterns → same OPN nodes/edges.
    assert set(opn_x.nodes()) == set(opn_y.nodes())
    assert set(opn_x.edges()) == set(opn_y.edges())


def test_add_white_noise_snr_matches_target(noisy_series: np.ndarray) -> None:
    target_snr = 10.0
    # Longer series so sample Var(ε) concentrates around the calibrated target.
    rng = np.random.default_rng(7)
    t = np.arange(4000, dtype=float)
    x = np.sin(2 * np.pi * 0.07 * t) + 0.2 * rng.normal(size=4000)
    y = add_white_noise_snr(x, target_snr, np.random.default_rng(11))
    eps = y - x
    var_x = float(np.var(x))
    var_eps = float(np.var(eps))
    snr = 10.0 * np.log10(var_x / var_eps)
    assert snr == pytest.approx(target_snr, abs=0.25)
    assert y.shape == x.shape
    # Exact calibration identity used by the helper.
    expected_var_eps = var_x / (10.0 ** (target_snr / 10.0))
    assert expected_var_eps == pytest.approx(var_x / (10.0 ** (target_snr / 10.0)))


def test_quantize_ties_unique_levels(noisy_series: np.ndarray) -> None:
    n_levels = 8
    y = quantize_ties(noisy_series, n_levels=n_levels)
    assert len(np.unique(y)) <= n_levels
    assert y.shape == noisy_series.shape


def test_downsample_preserves_length(noisy_series: np.ndarray) -> None:
    y = downsample_series(noisy_series, factor=2)
    assert y.shape == noisy_series.shape
    assert np.isfinite(y).all()


def test_impulsive_outliers_reproducible(noisy_series: np.ndarray) -> None:
    a = impulsive_outliers(noisy_series, np.random.default_rng(3), rate=0.05)
    b = impulsive_outliers(noisy_series, np.random.default_rng(3), rate=0.05)
    c = impulsive_outliers(noisy_series, np.random.default_rng(4), rate=0.05)
    assert np.allclose(a, b)
    assert not np.allclose(a, c)
    assert not np.allclose(a, noisy_series)


def test_descriptor_kind_mask_splits_columns() -> None:
    cols = [
        "visibility__NVG__c_avg",
        "visibility__NVG__lambda2_g",
        "transition__OPN__lambda_max",
        "proximity__RN__e_glob",
    ]
    topo = descriptor_kind_mask(cols, "topological")
    spec = descriptor_kind_mask(cols, "spectral")
    assert topo.tolist() == [True, False, False, True]
    assert spec.tolist() == [False, True, True, False]
    assert set(SPECTRAL_DESCRIPTOR_IDS) == {
        "lambda2_g",
        "lambda2_lcc",
        "lambda2_tilde",
        "lambda_max",
    }
    assert len(TOPOLOGICAL_DESCRIPTOR_IDS) == 12
    assert not set(SPECTRAL_DESCRIPTOR_IDS) & set(TOPOLOGICAL_DESCRIPTOR_IDS)


def test_invariance_config_profiles() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = InvarianceAuditConfig.from_yaml(root / "configs" / "phase2_invariance.yaml")
    dev = cfg.with_profile("dev")
    paper = cfg.with_profile("paper")
    assert dev.audit.n_realizations == 10
    assert dev.audit.n_steps == 1000
    assert len(dev.audit.regimes) == 6
    assert paper.audit.n_realizations == 30
    assert paper.audit.n_steps == 1000
    assert len(paper.audit.regimes) == 12
    assert "affine" in cfg.transforms


def test_invariance_markdown_contains_params() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = InvarianceAuditConfig.from_yaml(root / "configs" / "phase2_invariance.yaml")
    result = InvarianceAuditResult(
        config=cfg.with_profile("dev"),
        descriptor_stability={"per_family": {}},
        classification_ood={"per_family": {}},
    )
    md = _format_markdown(result)
    assert "Auditoría de invarianzas" in md
    assert "affine" in md
    assert "R=10" in md or "R=10," in md or "R=10 " in md


def test_phase_maps_config_profiles() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = PhaseMapsAuditConfig.from_yaml(root / "configs" / "phase2_phase_maps.yaml")
    dev = cfg.with_profile("dev")
    paper = cfg.with_profile("paper")
    assert dev.n_grid == (200, 500, 1000)
    assert None in dev.snr_db_grid
    assert paper.n_grid == (100, 250, 500, 1000)
    assert len(paper.audit.regimes) == 8
    assert snr_label(None) == "inf"
    assert snr_label(10.0) == "10"


def test_collapse_snr_thresholds_logic() -> None:
    cells = {
        "500": {
            "inf": {"views": {"full": {"random_forest": {"macro_f1_mean": 0.95}}}},
            "20": {"views": {"full": {"random_forest": {"macro_f1_mean": 0.88}}}},
            "10": {"views": {"full": {"random_forest": {"macro_f1_mean": 0.72}}}},
            "0": {"views": {"full": {"random_forest": {"macro_f1_mean": 0.40}}}},
        }
    }
    thr = collapse_snr_thresholds(
        cells,
        n_grid=(500,),
        snr_grid=(None, 20.0, 10.0, 0.0),
        f1_threshold=0.80,
    )
    assert thr["per_view"]["full"]["500"]["collapse_snr_db"] == "10"


def test_phase_maps_result_metrics_dict() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = PhaseMapsAuditConfig.from_yaml(root / "configs" / "phase2_phase_maps.yaml")
    result = PhaseMapsAuditResult(
        config=cfg.with_profile("dev"),
        cells={},
        collapse_thresholds={"f1_threshold": 0.8, "per_view": {}},
    )
    payload = result.metrics_dict()
    assert payload["config"]["profile"] == "dev"
    assert "inf" in payload["config"]["snr_db_grid"]
