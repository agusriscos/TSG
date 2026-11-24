"""Tests for the B3 non-saturated difficulty protocol."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsg.audit.classification import ViewMetrics, _make_models
from tsg.audit.config import AuditConfig
from tsg.audit.ensemble import ensemble_cache_fingerprint
from tsg.audit.hard_frontiers_difficulty import (
    DifficultyConfig,
    GridPoint,
    _SnrTransform,
    difficulty_noise_seed,
    evaluate_views,
    format_snr,
    proper_scores_from_oof,
    refresh_paper_bundle_manifest,
    resolve_headline,
    scoped_rng,
    summarize_predictions,
)
from tsg.generators.config import GeneratorSuiteConfig


_YAML = Path(__file__).resolve().parents[1] / "configs" / "phase2_hard_frontiers.yaml"


def _point(
    n_steps: int,
    snr: float | None,
    mean: float,
    low: float,
    high: float,
) -> GridPoint:
    return GridPoint(
        n_steps=n_steps,
        snr_db=snr,
        macro_f1_mean=mean,
        macro_f1_std=0.01,
        ci_low=low,
        ci_high=high,
    )


def _grid() -> list[GridPoint]:
    """Nine cells. Several sit inside the target band; one is saturated."""
    return [
        _point(250, None, 0.90, 0.84, 0.95),
        _point(250, 20.0, 0.84, 0.78, 0.90),
        _point(250, 10.0, 0.76, 0.70, 0.82),
        _point(500, None, 0.96, 0.93, 0.99),
        _point(500, 20.0, 0.88, 0.82, 0.93),
        _point(500, 10.0, 0.81, 0.74, 0.87),
        _point(1000, None, 0.995, 0.99, 1.0),
        _point(1000, 20.0, 0.94, 0.90, 0.97),
        _point(1000, 10.0, 0.91, 0.86, 0.95),
    ]


def test_difficulty_block_loads_prescribed_headline() -> None:
    cfg = DifficultyConfig.from_yaml(_YAML)
    audit = AuditConfig.from_yaml(_YAML)
    assert cfg.n_grid == (250, 500, 750, 1000, 1500)
    assert cfg.snr_db == (None, 20.0, 10.0)
    assert cfg.n_realizations == 20
    assert cfg.n_bootstrap == 1000
    assert cfg.bootstrap_seed == 42
    assert cfg.headline_n_steps == 500
    assert cfg.headline_snr_db == 20.0
    assert cfg.clean_n_steps == 1500
    assert audit.n_steps == 1500
    assert len(audit.regimes) == 12


def test_clean_cache_fingerprint_ignores_omitted_snr() -> None:
    gen = GeneratorSuiteConfig()
    cfg = AuditConfig(n_steps=1500, n_realizations=30, seed=42, profile="paper")
    plain = ensemble_cache_fingerprint(cfg, gen)
    again = ensemble_cache_fingerprint(cfg, gen)
    explicit_clean = ensemble_cache_fingerprint(cfg, gen, snr_db=None)
    noisy = ensemble_cache_fingerprint(cfg, gen, snr_db=20)
    assert plain == again
    assert plain.startswith("fase2_v1_")
    assert explicit_clean != plain
    assert noisy != plain
    assert noisy != explicit_clean


def test_noise_seed_depends_on_length_and_snr() -> None:
    base = difficulty_noise_seed(42, "lorenz", 3, 500, 20.0)
    assert base == difficulty_noise_seed(42, "lorenz", 3, 500, 20.0)
    assert base != difficulty_noise_seed(42, "lorenz", 3, 250, 20.0)
    assert base != difficulty_noise_seed(42, "lorenz", 3, 500, 10.0)
    assert base != difficulty_noise_seed(42, "lorenz", 4, 500, 20.0)
    assert base != difficulty_noise_seed(42, "rossler", 3, 500, 20.0)


def test_snr_transform_matches_requested_decibels() -> None:
    rng = np.random.default_rng(1)
    signal = rng.normal(size=4000)
    noisy = _SnrTransform(20.0, seed=7)(signal)
    again = _SnrTransform(20.0, seed=7)(signal)
    other = _SnrTransform(10.0, seed=7)(signal)
    assert np.allclose(noisy, again)
    assert not np.allclose(noisy, other)
    noise = noisy - signal
    measured = 10.0 * np.log10(np.var(signal) / np.var(noise))
    assert measured == pytest.approx(20.0, abs=0.4)


def test_resolve_keeps_prescribed_point_inside_the_band() -> None:
    point, fallback, reason, _detail = resolve_headline(
        _grid(), prescribed_n=500, prescribed_snr=20.0
    )
    assert point.key == (500, "20")
    assert fallback is False
    assert reason == "prescribed"


def test_resolve_saturation_moves_to_n250_snr20() -> None:
    grid = _grid()
    saturated = [
        _point(500, 20.0, 1.0, 1.0, 1.0) if p.key == (500, "20") else p for p in grid
    ]
    point, fallback, reason, detail = resolve_headline(
        saturated, prescribed_n=500, prescribed_snr=20.0
    )
    assert point.key == (250, "20")
    assert fallback is True
    assert reason == "saturation_to_n250_snr20"
    assert "CV Macro-F1=1.0000" in detail


def test_resolve_undershoot_picks_closest_to_center() -> None:
    grid = [
        _point(250, None, 0.97, 0.94, 0.99),  # above the band; not eligible
        _point(250, 20.0, 0.72, 0.64, 0.80),
        _point(250, 10.0, 0.60, 0.50, 0.70),
        _point(500, None, 0.965, 0.94, 0.99),
        _point(500, 20.0, 0.70, 0.62, 0.78),
        _point(500, 10.0, 0.66, 0.58, 0.74),
        _point(1000, None, 0.99, 0.97, 1.0),
        _point(1000, 20.0, 0.93, 0.88, 0.97),
        _point(1000, 10.0, 0.90, 0.84, 0.95),
    ]
    point, fallback, reason, _detail = resolve_headline(
        grid, prescribed_n=500, prescribed_snr=20.0
    )
    # Centre of [0.80, 0.95] is 0.875. 0.90 is closer than 0.93.
    assert point.key == (1000, "10")
    assert fallback is True
    assert "undershoot_closest_to_center" in reason


def test_resolve_does_not_select_nonpositive_snr() -> None:
    grid = [
        _point(250, 20.0, 0.86, 0.80, 0.91),
        _point(500, 20.0, 0.70, 0.60, 0.78),
        _point(500, -5.0, 0.875, 0.80, 0.93),
        _point(1000, 10.0, 0.84, 0.78, 0.90),
    ]
    point, _fallback, _reason, _detail = resolve_headline(
        grid, prescribed_n=500, prescribed_snr=20.0
    )
    assert point.snr_db is None or point.snr_db > 0
    assert point.key != (500, "-5")


def test_perfect_oof_proper_scores_and_ci_includes_one() -> None:
    classes = ["a", "b", "c"]
    y = ["a", "b", "c"] * 12
    proba = np.zeros((len(y), 3), dtype=float)
    for i, label in enumerate(y):
        proba[i, classes.index(label)] = 1.0
    fold = np.array([i % 4 for i in range(len(y))], dtype=int)
    stats = summarize_predictions(
        y_true=y,
        y_pred=y,
        y_proba=proba,
        class_names=classes,
        fold_id=fold,
        pairs=(("a", "b"), ("b", "c")),
        rng=scoped_rng(42, "unit"),
        n_bootstrap=80,
    )
    assert stats["macro_f1_oof"] == pytest.approx(1.0)
    assert stats["macro_f1_mean"] == pytest.approx(1.0)
    assert stats["log_loss"] < 0.02
    assert stats["brier"] < 0.02
    assert stats["ece"] < 0.02
    assert stats["mean_prob_margin"] == pytest.approx(1.0, abs=1e-6)
    assert stats["macro_f1_ci_low"] == pytest.approx(1.0)
    assert stats["macro_f1_ci_high"] == pytest.approx(1.0)


def test_log_loss_follows_class_column_order_not_lexicographic() -> None:
    classes = ["z_regime", "a_regime"]
    y = ["z_regime", "a_regime", "z_regime", "a_regime"]
    proba = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=float,
    )
    stats = summarize_predictions(
        y_true=y,
        y_pred=y,
        y_proba=proba,
        class_names=classes,
        fold_id=np.array([0, 0, 1, 1], dtype=int),
        pairs=(("z_regime", "a_regime"),),
        rng=scoped_rng(42, "order"),
        n_bootstrap=20,
    )
    assert stats["log_loss"] < 0.02


def test_layer0_table_from_oof_bundles() -> None:
    classes = ["a", "b"]
    # Each fold contains both classes, so per-fold Macro-F1 is 1.
    y = ["a" if i % 2 == 0 else "b" for i in range(16)]
    proba = np.zeros((len(y), 2), dtype=float)
    for i, label in enumerate(y):
        proba[i, classes.index(label)] = 1.0
    bundles = {
        "full__random_forest": {
            "y_true": y,
            "y_pred": y,
            "y_proba": proba,
            "fold_id": np.array([i // 2 % 4 for i in range(len(y))]),
            "class_names": classes,
        }
    }
    frame = proper_scores_from_oof(
        bundles,
        pairs=(("a", "b"),),
        n_steps=1500,
        n_bootstrap=30,
        bootstrap_seed=42,
    )
    assert list(frame["view"]) == ["full"]
    assert frame.loc[0, "snr_db"] == "inf"
    assert frame.loc[0, "source"] == "oof_predictions"
    assert frame.loc[0, "macro_f1_mean"] == pytest.approx(1.0)


def test_scaler_lives_inside_the_pipeline_and_x_is_not_rewritten() -> None:
    models = _make_models(0, rf_n_jobs=1)
    assert list(models["logistic"].named_steps) == ["scaler", "clf"]
    assert list(models["random_forest"].named_steps) == ["scaler", "clf"]
    rng = np.random.default_rng(0)
    rows = []
    labels = []
    columns = [
        "visibility__NVG__c_avg",
        "transition__STN__c_avg",
        "proximity__kNN__c_avg",
        "similarity__Correlation__c_avg",
    ]
    for class_id, name in enumerate(("white_noise", "lorenz")):
        for _ in range(8):
            rows.append(class_id + 0.05 * rng.normal(size=len(columns)))
            labels.append(name)
    x = pd.DataFrame(rows, columns=columns)
    y = pd.Series(labels, name="regime")
    before = x.to_numpy().copy()
    cfg = AuditConfig(
        regimes=("white_noise", "lorenz"),
        n_folds=4,
        seed=42,
        critical_pairs=(("white_noise", "lorenz"),),
    )
    views = evaluate_views(x, y, cfg, rf_n_jobs=1)
    np.testing.assert_allclose(x.to_numpy(), before)
    assert {view.view for view in views} == {
        "visibility",
        "transition",
        "proximity",
        "similarity",
        "full",
    }
    full = next(view for view in views if view.view == "full" and view.model == "random_forest")
    assert full.macro_f1_mean > 0.9


def test_difficulty_curve_plot_roundtrips_float_snr_from_csv(tmp_path: Path) -> None:
    """Pandas reads ``20`` / ``inf`` as floats; the plot must still match labels."""
    rows = []
    for n in (250, 500, 1000):
        for snr in ("inf", "20", "10"):
            for view in ("visibility", "transition", "proximity", "similarity", "full"):
                for model in ("random_forest", "logistic"):
                    base = 0.7 + 0.02 * (n / 1000.0) + (0.05 if view == "full" else 0.0)
                    rows.append(
                        {
                            "n_steps": n,
                            "snr_db": snr,
                            "view": view,
                            "model": model,
                            "macro_f1_oof": base,
                            "macro_f1_ci_low": base - 0.03,
                            "macro_f1_ci_high": base + 0.03,
                        }
                    )
    written = pd.DataFrame(rows)
    path = tmp_path / "curve.csv"
    written.to_csv(path, index=False)
    loaded = pd.read_csv(path)  # snr_db becomes float / inf
    assert set(map(type, loaded["snr_db"].tolist())) == {float}
    from tsg.audit.hard_frontiers_difficulty import (
        GridPoint,
        _normalize_snr_column,
        plot_difficulty_curve,
    )

    normalized = _normalize_snr_column(loaded)
    assert list(dict.fromkeys(normalized["snr_db"].tolist())) == ["inf", "20", "10"]
    operating = GridPoint(500, 20.0, 0.88, 0.04, 0.84, 0.92)
    paths = plot_difficulty_curve(
        loaded,
        operating,
        fallback_reason="prescribed",
        output_dir=tmp_path / "figs",
        dpi=100,
    )
    assert any(path.name.startswith("08_difficulty_curve") for path in paths)


def test_manifest_refresh_counts_new_files(tmp_path: Path) -> None:
    tables = tmp_path / "tables"
    figures = tmp_path / "figures"
    notes = tmp_path / "notes"
    tables.mkdir()
    figures.mkdir()
    notes.mkdir()
    (tables / "b3_difficulty_curve.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (figures / "b3_hard_08_difficulty_curve.png").write_bytes(b"\x89PNG")
    (notes / "b2_arma_iaaft_diagnosis.md").write_text("note", encoding="utf-8")
    refresh_paper_bundle_manifest(tmp_path, tmp_path)
    text = (tmp_path / "MANIFEST.md").read_text(encoding="utf-8")
    assert "b3_difficulty_curve.csv" in text
    assert "b3_hard_08_difficulty_curve" in text
    assert "Figures: **1** stems / **1** files" in text
    assert "Tables: **1** CSV" in text
    assert format_snr(None) == "inf"
