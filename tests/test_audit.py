"""Tests for Phase-1 discriminability audit."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsg.audit.classification import evaluate_classification
from tsg.audit.config import AuditConfig, apply_profile
from tsg.audit.ensemble import (
    ENSEMBLE_CACHE_VERSION,
    ensemble_cache_fingerprint,
    load_generator_suite_config,
)
from tsg.audit.features import build_feature_table, feature_name, matrix_to_feature_row
from tsg.audit.geometry import analyze_geometry
from tsg.audit.health import diagnose_and_clean
from tsg.audit.pipeline import AuditPipeline
from tsg.audit.verdict import Verdict, decide_verdict
from tsg.descriptors.base import DESCRIPTOR_IDS
from tsg.generators.config import GeneratorSuiteConfig
from tsg.graphs.registry import METHOD_IDS


def _fake_matrix(dataset_id: str, noise: float = 0.01) -> pd.DataFrame:
    rng = np.random.default_rng(abs(hash(dataset_id)) % (2**32))
    # Regime-specific offset so classes are linearly separable in tests.
    offsets = {
        "white_noise": 0.0,
        "brownian": 1.0,
        "sine": 2.0,
        "quasiperiodic": 3.0,
        "lorenz": 4.0,
        "logistic": 5.0,
    }
    base = offsets[dataset_id]
    data = base + noise * rng.normal(size=(len(DESCRIPTOR_IDS), len(METHOD_IDS)))
    return pd.DataFrame(data, index=list(DESCRIPTOR_IDS), columns=list(METHOD_IDS))


def test_feature_naming_and_table() -> None:
    assert feature_name("NVG", "c_avg") == "visibility__NVG__c_avg"
    assert feature_name("STN", "i_dir") == "transition__STN__i_dir"
    records = [
        {"dataset_id": "sine", "realization": 0, "matrix": _fake_matrix("sine")},
        {"dataset_id": "lorenz", "realization": 0, "matrix": _fake_matrix("lorenz")},
    ]
    cfg = AuditConfig(n_realizations=1)
    x, y = build_feature_table(records, cfg)
    assert x.shape == (2, len(DESCRIPTOR_IDS) * len(METHOD_IDS))
    assert list(y) == ["sine", "lorenz"]


def test_health_drops_nan_and_constant() -> None:
    cfg = AuditConfig()
    x = pd.DataFrame(
        {
            "visibility__NVG__c_avg": [1.0, 2.0, 3.0],
            "bad_nan": [1.0, np.nan, 2.0],
            "visibility__NVG__i_dir": [0.0, 0.0, 0.0],
            "transition__STN__assortativity_r": [np.nan, np.nan, np.nan],
            "bad_const": [1.0, 1.0, 1.0],
            "bad_inf": [1.0, np.inf, 3.0],
            "transition__OPN__d_lcc": [0.4, 0.4, 0.4],
        }
    )
    report = diagnose_and_clean(x, cfg)
    assert "bad_nan" in report.dropped_features
    assert "bad_const" in report.dropped_features
    assert "visibility__NVG__c_avg" in report.x_clean.columns
    assert "visibility__NVG__i_dir" in report.structural_constant_features
    assert "transition__STN__assortativity_r" in report.numerical_nan_features
    assert "bad_inf" in report.numerical_inf_features
    assert "transition__OPN__d_lcc" in report.low_variance_non_structural_features
    assert "visibility__NVG__i_dir" not in report.low_variance_non_structural_features
    accounted = (
        len(report.numerical_nan_features)
        + len(report.numerical_inf_features)
        + len(report.structural_constant_features)
        + len(report.low_variance_non_structural_features)
    )
    assert accounted == len(report.dropped_features)


def test_verdict_green_on_perfect_separation() -> None:
    cfg = AuditConfig(n_realizations=8, n_folds=4, n_steps=100)
    records = []
    for dataset_id in cfg.regimes:
        for r in range(cfg.n_realizations):
            records.append(
                {
                    "dataset_id": dataset_id,
                    "realization": r,
                    "matrix": _fake_matrix(dataset_id, noise=0.01),
                }
            )
    x, y = build_feature_table(records, cfg)
    health = diagnose_and_clean(x, cfg)
    geometry = analyze_geometry(health.x_clean, y, cfg)
    assert geometry.silhouette_global > 0.5
    classification = evaluate_classification(health.x_clean, y, cfg)
    verdict = decide_verdict(classification, cfg)
    assert verdict.verdict in {Verdict.GREEN, Verdict.YELLOW}
    assert classification.full_rf() is not None
    assert classification.full_rf().macro_f1_mean >= 0.9
    assert classification.rf_gini_importance is not None
    assert classification.logistic_l2_importance is not None
    assert float(classification.rf_gini_importance["importance_mean"].sum()) > 0.0
    assert float(classification.logistic_l2_importance["importance_mean"].max()) > 0.0

    view_names = {v.view for v in classification.views}
    assert view_names == {
        "visibility",
        "proximity",
        "transition",
        "similarity",
        "full",
    }

    # Confusion JSON orientation: outer key = true label; each row sums to 1.
    payload = classification.to_dict()
    for view in payload["views"]:
        cmn = view["confusion_normalized"]
        for true_label, row in cmn.items():
            assert abs(sum(row.values()) - 1.0) < 1e-6, true_label

    # Loadings include feature names.
    geo_payload = geometry.to_dict()
    for entry in geo_payload["top_loadings_pc1_pc2"]:
        assert "feature" in entry
        assert isinstance(entry["feature"], str) and entry["feature"]


def test_default_brownian_mode_is_levels() -> None:
    assert AuditConfig().brownian_mode == "levels"


def test_cache_fingerprint_changes_with_mode() -> None:
    gen = GeneratorSuiteConfig()
    a = AuditConfig(brownian_mode="levels", n_steps=100)
    b = AuditConfig(brownian_mode="increments", n_steps=100)
    fa = ensemble_cache_fingerprint(a, gen)
    fb = ensemble_cache_fingerprint(b, gen)
    assert fa != fb
    assert fa.startswith(ENSEMBLE_CACHE_VERSION)


def test_cache_fingerprint_includes_regimes_and_profile() -> None:
    gen = GeneratorSuiteConfig()
    dev = AuditConfig(
        n_steps=1000,
        n_realizations=10,
        regimes=("white_noise", "lorenz"),
        profile="dev",
    )
    paper = AuditConfig(
        n_steps=1000,
        n_realizations=30,
        regimes=("white_noise", "lorenz"),
        profile="paper",
    )
    other = AuditConfig(
        n_steps=1000,
        n_realizations=10,
        regimes=("white_noise", "rossler"),
        profile="dev",
    )
    lengths = AuditConfig(
        n_steps=2500,
        n_realizations=10,
        regimes=("white_noise", "lorenz"),
        profile="dev",
    )
    fingerprints = {
        ensemble_cache_fingerprint(cfg, gen) for cfg in (dev, paper, other, lengths)
    }
    assert len(fingerprints) == 4


def test_apply_profile_dev_and_paper() -> None:
    cfg = AuditConfig.from_yaml(
        Path(__file__).resolve().parents[1] / "configs" / "phase2_hard_frontiers.yaml"
    )
    dev = apply_profile(cfg, "dev")
    paper = apply_profile(cfg, "paper")
    assert dev.n_realizations == 10 and dev.n_steps == 1000 and dev.profile == "dev"
    assert paper.n_realizations == 30 and paper.n_steps == 1500 and paper.profile == "paper"
    assert cfg.brownian_mode == "levels"
    assert len(cfg.regimes) == 12
    assert ("arma_correlated", "lorenz") in cfg.critical_pairs
    assert ("fbm_persistent", "brownian") in cfg.critical_pairs


def test_generator_suite_yaml_loads() -> None:
    cfg = load_generator_suite_config()
    assert cfg.lorenz.rho_range == (27.0, 29.0)
    assert cfg.logistic.burn_in_range == (500, 1500)
    assert cfg.logistic.r == 4.0


def test_audit_pipeline_quick(tmp_path: Path) -> None:
    cfg = AuditConfig(
        n_steps=120,
        n_realizations=2,
        seed=1,
        n_folds=2,
        brownian_mode="levels",
        methods=("HVG", "STN", "OPN", "Correlation"),
        families={
            "visibility": ("HVG",),
            "transition": ("STN", "OPN"),
            "proximity": (),
            "similarity": ("Correlation",),
        },
    )
    result = AuditPipeline(cfg).run(output_dir=tmp_path)
    assert (tmp_path / "metrics_summary.json").exists()
    assert (tmp_path / "report_phase1.md").exists()
    assert (tmp_path / "report_phase1.html").exists()
    figures = tmp_path / "figures"
    assert (figures / "01_pca_projection.png").exists()
    assert (figures / "01_pca_projection.svg").exists()
    assert (figures / "02_confusion_matrices_grid.png").exists()
    assert (figures / "02_confusion_matrices_grid.svg").exists()
    assert (figures / "04_family_ablation_scores.png").exists()
    assert (figures / "04_family_ablation_scores.svg").exists()
    assert result.verdict.verdict in {Verdict.GREEN, Verdict.YELLOW, Verdict.RED}

    metrics = json.loads((tmp_path / "metrics_summary.json").read_text(encoding="utf-8"))
    md = (tmp_path / "report_phase1.md").read_text(encoding="utf-8")
    assert metrics["n_features_raw"] == result.health.n_features
    assert metrics["n_features_clean"] == result.health.x_clean.shape[1]
    assert f"**{metrics['n_features_raw']}** / **{metrics['n_features_clean']}**" in md
    assert metrics["generated_at_utc"] in md
    assert "trayectoria browniana integrada" in md.lower() or "levels" in md
    health = result.health
    accounted = (
        len(health.numerical_nan_features)
        + len(health.numerical_inf_features)
        + len(health.structural_constant_features)
        + len(health.low_variance_non_structural_features)
    )
    assert accounted == len(health.dropped_features)
    drop_line = (
        f"- dropped: {len(health.dropped_features)} = numerical_nan "
        f"({len(health.numerical_nan_features)}) "
        f"+ structural_constant ({len(health.structural_constant_features)}) "
        f"+ low_variance_non_structural "
        f"({len(health.low_variance_non_structural_features)}) "
        f"+ numerical_inf ({len(health.numerical_inf_features)})"
    )
    assert "low_variance_features (no estructurales)" in md
    assert drop_line in md

    # Versioned cache subdirectory exists.
    cache_root = tmp_path / "ensemble_cache"
    assert cache_root.exists()
    children = [p for p in cache_root.iterdir() if p.is_dir()]
    assert children
    assert children[0].name.startswith(ENSEMBLE_CACHE_VERSION)

    view_names = {v.view for v in result.classification.views}
    # proximity empty → skipped; similarity present via Correlation.
    assert "similarity" in view_names
    assert "full" in view_names


def test_publication_plotting_includes_similarity_and_svg(tmp_path: Path) -> None:
    """Confusion + ablation figures cover five views and write PNG/SVG pairs."""
    from tsg.audit.classification import ClassificationResult, ViewMetrics
    from tsg.audit.geometry import GeometryResult
    from tsg.audit.plotting import (
        plot_confusion_grid,
        plot_family_ablation,
        plot_pca_projection,
    )

    regimes = (
        "white_noise",
        "fbm_antipersistent",
        "brownian",
        "fbm_persistent",
        "arma_correlated",
        "quasiperiodic",
        "lorenz",
        "rossler",
        "henon",
        "logistic_weak",
        "logistic",
        "regime_switching",
    )
    cfg = AuditConfig(regimes=regimes, n_realizations=2, n_steps=64, figure_dpi=80)
    labels = list(regimes)
    identity = pd.DataFrame(
        np.eye(len(labels), dtype=float), index=labels, columns=labels
    )
    views = []
    for view_name, score in [
        ("visibility", 0.91),
        ("transition", 0.88),
        ("proximity", 0.84),
        ("similarity", 0.79),
        ("full", 0.97),
    ]:
        views.append(
            ViewMetrics(
                view=view_name,
                model="random_forest",
                accuracy_mean=score,
                accuracy_std=0.01,
                macro_f1_mean=score,
                macro_f1_std=0.02,
                f1_by_class={r: score for r in labels},
                confusion=identity.copy(),
                confusion_normalized=identity.copy(),
            )
        )
    classification = ClassificationResult(views=views)
    features = [
        "visibility__NVG__c_avg",
        "transition__STN__lambda2_g",
        "proximity__RN__e_glob",
        "similarity__Correlation__c_avg",
    ]
    loadings = pd.DataFrame(
        {
            "PC1": [0.4, -0.3, 0.2, 0.1],
            "PC2": [0.1, 0.35, -0.25, 0.15],
        },
        index=features,
    )
    emb_rows = []
    for i, regime in enumerate(regimes):
        emb_rows.append({"regime": regime, "PC1": float(i), "PC2": float(i % 3)})
        emb_rows.append({"regime": regime, "PC1": float(i) + 0.1, "PC2": float(i % 3) + 0.1})
        emb_rows.append({"regime": regime, "PC1": float(i) - 0.1, "PC2": float(i % 3) - 0.1})
    geometry = GeometryResult(
        explained_variance_ratio=[0.4, 0.2],
        cumulative_variance=[0.4, 0.6],
        loadings=loadings,
        embedding=pd.DataFrame(emb_rows),
        silhouette_global=0.3,
        silhouette_by_regime={r: 0.3 for r in regimes},
        centroid_euclidean=pd.DataFrame(),
        centroid_cosine=pd.DataFrame(),
        critical_pair_distances={},
    )

    pca_path = plot_pca_projection(geometry, cfg, tmp_path)
    cm_path = plot_confusion_grid(classification, cfg, tmp_path)
    abl_path = plot_family_ablation(classification, geometry, cfg, tmp_path)

    for path in (pca_path, cm_path, abl_path):
        assert path.exists()
        assert path.with_suffix(".svg").exists()
        assert path.stat().st_size > 0
        assert path.with_suffix(".svg").stat().st_size > 0

    # Source-level contract: the five-view order is hard-coded for publication.
    import inspect

    from tsg.audit import plotting as plotting_mod

    assert "similarity" in inspect.getsource(plotting_mod.plot_confusion_grid)
    assert "similarity" in inspect.getsource(plotting_mod.plot_family_ablation)
    pca_src = inspect.getsource(plotting_mod.plot_pca_projection)
    assert "fig.legend" in pca_src
    assert "pca_family_legend" in pca_src or "familia" in pca_src
    abl_src = inspect.getsource(plotting_mod.plot_family_ablation)
    assert "rotation=45" in abl_src
    assert 'ha="right"' in abl_src
    assert "rotation_mode" in abl_src
