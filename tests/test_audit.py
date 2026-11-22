"""Tests for Phase-1 discriminability audit."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsg.audit.classification import evaluate_classification
from tsg.audit.config import AuditConfig
from tsg.audit.features import build_feature_table, feature_name, matrix_to_feature_row
from tsg.audit.geometry import analyze_geometry
from tsg.audit.health import diagnose_and_clean
from tsg.audit.pipeline import AuditPipeline
from tsg.audit.verdict import Verdict, decide_verdict
from tsg.descriptors.base import DESCRIPTOR_IDS
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
            "bad_const": [1.0, 1.0, 1.0],
        }
    )
    report = diagnose_and_clean(x, cfg)
    assert "bad_nan" in report.dropped_features
    assert "bad_const" in report.dropped_features
    assert "visibility__NVG__c_avg" in report.x_clean.columns


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


def test_audit_pipeline_quick(tmp_path: Path) -> None:
    cfg = AuditConfig(
        n_steps=120,
        n_realizations=2,
        seed=1,
        n_folds=2,
        brownian_mode="increments",
        methods=("HVG", "STN", "OPN", "Correlation"),
        families={
            "visibility": ("HVG",),
            "transition": ("STN", "OPN"),
            "proximity": (),
        },
    )
    # Empty proximity family should be skipped gracefully in ablation.
    result = AuditPipeline(cfg).run(output_dir=tmp_path)
    assert (tmp_path / "metrics_summary.json").exists()
    assert (tmp_path / "report_phase1.md").exists()
    assert (tmp_path / "report_phase1.html").exists()
    assert (tmp_path / "figures" / "01_pca_projection.png").exists()
    assert result.verdict.verdict in {Verdict.GREEN, Verdict.YELLOW, Verdict.RED}
