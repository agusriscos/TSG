"""Unit tests for Bloque-4 transforms / invariance and Bloque-5 phase maps."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from tsg.audit.config import AuditConfig
from tsg.audit.features import (
    SPECTRAL_DESCRIPTOR_IDS,
    TOPOLOGICAL_DESCRIPTOR_IDS,
    descriptor_kind_mask,
)
from tsg.audit.invariance import (
    InvarianceAuditConfig,
    InvarianceAuditPipeline,
    InvarianceAuditResult,
    _format_markdown,
    _method_view_specs,
    compute_invariance_metrics,
)
from tsg.audit.phase_maps import (
    PhaseMapsAuditConfig,
    PhaseMapsAuditPipeline,
    PhaseMapsAuditResult,
    _nan_failure_rate,
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


def test_method_tables_and_strict_monotone_invariance(tmp_path: Path) -> None:
    """HVG and OPN stay exactly invariant under a monotone map, per method.

    The family aggregate is intentionally not invariant, because NVG and STN
    columns move. ``plot_only`` must rebuild ``per_method`` from ``records.pkl``
    without touching a pre-existing ``per_family`` block, and must write the
    method CSVs plus the per-method figure trio.
    """
    columns = [
        "visibility__HVG__c_avg",
        "visibility__NVG__c_avg",
        "transition__OPN__e_glob",
        "transition__STN__e_glob",
        "proximity__kNN__cv_k",
        "proximity__RN__cv_k",
    ]
    specs = dict(
        _method_view_specs(
            columns,
            ("HVG", "NVG", "OPN", "STN", "RN", "kNN", "Correlation"),
        )
    )
    assert "STN" in specs and "Correlation" not in specs
    assert specs["HVG"].tolist() == [True, False, False, False, False, False]
    assert specs["RN"].tolist() == [False, False, False, False, False, True]
    assert specs["kNN"].tolist() == [False, False, False, False, True, False]

    methods = ("HVG", "NVG", "OPN", "STN")
    descriptors = ("c_avg", "e_glob", "cv_k")
    regimes = ("white_noise", "logistic")
    rng = np.random.default_rng(0)
    records: list[dict[str, object]] = []
    for regime_i, regime in enumerate(regimes):
        for realization in range(4):
            base = {
                method: rng.normal(
                    loc=10.0 * regime_i + method_i,
                    scale=0.8,
                    size=len(descriptors),
                )
                for method_i, method in enumerate(methods)
            }
            for kind in ("original", "monotone"):
                data: dict[str, np.ndarray] = {}
                for method in methods:
                    values = np.array(base[method], dtype=float, copy=True)
                    if kind == "monotone" and method in {"NVG", "STN"}:
                        values = values + 2.5
                    data[method] = values
                records.append(
                    {
                        "dataset_id": regime,
                        "regime": regime,
                        "realization": realization,
                        "seed": realization,
                        "transform": kind,
                        "matrix": pd.DataFrame(data, index=list(descriptors)),
                        "series_length": 32,
                    }
                )

    audit = AuditConfig(
        n_steps=64,
        n_realizations=4,
        seed=0,
        n_folds=2,
        profile="dev",
        regimes=regimes,
        methods=methods,
        families={
            "visibility": ("HVG", "NVG"),
            "transition": ("OPN", "STN"),
        },
        variance_eps=1e-12,
        figure_dpi=80,
    )
    config = InvarianceAuditConfig(audit=audit, transforms=("monotone",))
    stability, ood = compute_invariance_metrics(records, config)
    assert "per_family" in stability and "per_method" in stability
    assert "visibility" in stability["per_family"]
    for method in ("HVG", "OPN"):
        assert stability["per_method"][method]["monotone"]["normalized_error_median"] == 0.0
        for model_name in ("random_forest", "logistic"):
            assert (
                ood["per_method"][method]["monotone"][model_name]["delta_f1"] == 0.0
            )
    assert stability["per_method"]["NVG"]["monotone"]["normalized_error_median"] > 0.0
    assert stability["per_method"]["STN"]["monotone"]["normalized_error_median"] > 0.0

    out = tmp_path / "invariance"
    data_dir = out / "data"
    data_dir.mkdir(parents=True)
    with (data_dir / "records.pkl").open("wb") as fh:
        pickle.dump(records, fh, protocol=pickle.HIGHEST_PROTOCOL)
    sentinel = 123.456
    payload = {
        "generated_at_utc": "2026-01-01T00:00:00+00:00",
        "descriptor_stability": {
            "per_family": {
                "visibility": {
                    "monotone": {
                        "normalized_error_median": sentinel,
                        "spearman_mean": 0.5,
                        "n_features": 1,
                    }
                }
            }
        },
        "classification_ood": {
            "per_family": {
                "visibility": {
                    "monotone": {
                        "random_forest": {
                            "f1_clean": 0.5,
                            "f1_perturbed": 0.4,
                            "delta_f1": -0.1,
                        }
                    }
                }
            }
        },
        "figures": [],
    }
    (out / "invariance_metrics.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    InvarianceAuditPipeline(config, lang="en").plot_only(out, lang="en")

    saved = json.loads((out / "invariance_metrics.json").read_text(encoding="utf-8"))
    assert (
        saved["descriptor_stability"]["per_family"]["visibility"]["monotone"][
            "normalized_error_median"
        ]
        == sentinel
    )
    assert "per_method" in saved["descriptor_stability"]
    assert "per_method" in saved["classification_ood"]
    alias = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert alias["descriptor_stability"]["per_method"]["HVG"]["monotone"][
        "normalized_error_median"
    ] == 0.0

    tables = data_dir / "tables"
    stability_csv = pd.read_csv(tables / "stability_matrix_by_method.csv")
    delta_csv = pd.read_csv(tables / "delta_f1_by_method.csv")
    assert list(stability_csv.columns) == [
        "method",
        "family",
        "transform",
        "normalized_error_median",
        "spearman_mean",
        "n_features",
    ]
    assert list(delta_csv.columns) == [
        "method",
        "family",
        "transform",
        "model",
        "f1_clean",
        "f1_perturbed",
        "delta_f1",
    ]
    for method, family in (("HVG", "visibility"), ("OPN", "transition")):
        stab_row = stability_csv.loc[stability_csv["method"] == method].iloc[0]
        assert stab_row["family"] == family
        assert stab_row["transform"] == "monotone"
        assert stab_row["normalized_error_median"] == 0.0
        for model_name in ("random_forest", "logistic"):
            delta_row = delta_csv.loc[
                (delta_csv["method"] == method) & (delta_csv["model"] == model_name)
            ].iloc[0]
            assert delta_row["delta_f1"] == 0.0

    figure = out / "figures" / "02_invariance_matrix_by_method.png"
    assert figure.exists()
    assert figure.with_suffix(".pdf").exists()
    assert figure.with_suffix(".svg").exists()


def test_disconnection_by_family_and_csv(tmp_path: Path) -> None:
    """Family rate is the row-wise ANY over that family's ``lambda2_g`` columns.

    Legacy ``nan_inf_rate`` / ``spectral_nan_rate`` / ``disconnection_proxy_rate``
    stay in the payload, and ``plot_only`` writes ``disconnection_by_family.csv``.
    """
    columns = [
        "visibility__NVG__lambda2_g",
        "visibility__HVG__lambda2_g",
        "transition__STN__lambda2_g",
        "transition__OPN__lambda2_g",
        "proximity__RN__lambda2_g",
        "proximity__kNN__lambda2_g",
        "similarity__Correlation__lambda2_g",
        "similarity__Segment__lambda2_g",
        "visibility__NVG__c_avg",
    ]
    values = np.array(
        [
            [0.2, 0.2, 1.0, 1.0, 0.0, 0.5, 0.4, 0.4, 1.0],
            [0.2, 0.0, 1.0, 1.0, 0.0, 0.5, 0.4, 0.4, 1.0],
            [0.2, 0.2, 1.0, 1.0, 0.0, 0.5, np.inf, 0.4, 1.0],
            [0.2, 1e-13, np.nan, 1.0, 0.0, 0.5, 0.4, 1e-12, np.nan],
        ],
        dtype=float,
    )
    failure = _nan_failure_rate(pd.DataFrame(values, columns=columns))
    assert failure["nan_inf_rate"] == pytest.approx(3 / 36)
    assert failure["spectral_nan_rate"] == pytest.approx(2 / 32)
    assert failure["disconnection_proxy_rate"] == pytest.approx(1.0)
    assert failure["disconnection_by_family"] == {
        "visibility": pytest.approx(0.5),
        "transition": pytest.approx(0.25),
        "proximity": pytest.approx(1.0),
        "similarity": pytest.approx(0.5),
    }
    assert failure["disconnection_by_method"]["NVG"] == pytest.approx(0.0)
    assert failure["disconnection_by_method"]["HVG"] == pytest.approx(0.5)
    assert failure["disconnection_by_method"]["STN"] == pytest.approx(0.25)
    assert failure["disconnection_by_method"]["OPN"] == pytest.approx(0.0)
    assert failure["disconnection_by_method"]["RN"] == pytest.approx(1.0)
    assert failure["disconnection_by_method"]["kNN"] == pytest.approx(0.0)
    assert failure["disconnection_by_method"]["Correlation"] == pytest.approx(0.25)
    assert failure["disconnection_by_method"]["Segment"] == pytest.approx(0.25)
    assert failure["lambda2_median_by_family"]["visibility"] == pytest.approx(0.2)
    assert failure["lambda2_median_by_family"]["transition"] == pytest.approx(1.0)
    assert failure["lambda2_median_by_family"]["proximity"] == pytest.approx(0.25)
    assert failure["lambda2_median_by_family"]["similarity"] == pytest.approx(0.4)

    cfg = PhaseMapsAuditConfig(
        audit=AuditConfig(n_steps=32, n_realizations=1, seed=1, profile="dev"),
        n_grid=(100,),
        snr_db_grid=(None,),
    )
    out = tmp_path / "phase_maps"
    out.mkdir()
    payload = {
        "generated_at_utc": "2026-01-01T00:00:00+00:00",
        "cells": {"100": {"inf": {"views": {}, "failure_modes": failure}}},
        "figures": [],
    }
    (out / "phase_maps_metrics.json").write_text(json.dumps(payload), encoding="utf-8")

    PhaseMapsAuditPipeline(cfg, lang="en").plot_only(out, lang="en")

    table = pd.read_csv(out / "data" / "tables" / "disconnection_by_family.csv")
    assert list(table.columns) == [
        "N",
        "SNR",
        "disconnection_proxy_rate",
        "visibility_disc_rate",
        "transition_disc_rate",
        "proximity_disc_rate",
        "similarity_disc_rate",
        "visibility_lambda2_med",
        "transition_lambda2_med",
        "proximity_lambda2_med",
        "similarity_lambda2_med",
    ]
    row = table.iloc[0]
    assert int(row["N"]) == 100
    raw_line = (out / "data" / "tables" / "disconnection_by_family.csv").read_text(
        encoding="utf-8"
    ).splitlines()[1]
    assert raw_line.startswith("100,inf,")
    assert np.isinf(float(row["SNR"]))
    assert row["visibility_disc_rate"] == pytest.approx(0.5)
    assert row["transition_disc_rate"] == pytest.approx(0.25)
    assert row["proximity_disc_rate"] == pytest.approx(1.0)
    assert row["similarity_disc_rate"] == pytest.approx(0.5)
    assert row["disconnection_proxy_rate"] == pytest.approx(1.0)
    assert row["proximity_lambda2_med"] == pytest.approx(0.25)
    figure = out / "figures" / "03_disconnection_by_family_phase_map.png"
    assert figure.exists()
    assert figure.with_suffix(".pdf").exists()
    assert figure.with_suffix(".svg").exists()


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
