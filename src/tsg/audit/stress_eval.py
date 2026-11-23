"""Predictive stress tests that break the Macro-F1 ceiling (RQ1 task 4).

Noise (SNR) and short-window truncation are applied to the **raw** time
series *before* TS2G graph extraction. Both Macro-F1 and log-loss
(cross-entropy) are reported; log-loss is the primary metric because F1
saturates near 1.0 on the clean N=1500 panel.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.metrics import log_loss

from tsg.audit.classification import ViewMetrics, _evaluate_view, _make_models
from tsg.audit.config import AuditConfig
from tsg.audit.ensemble import (
    generate_one_matrix,
    load_generator_suite_config,
    realization_seed,
)
from tsg.audit.features import build_feature_table
from tsg.audit.health import diagnose_and_clean
from tsg.audit.information import FAMILY_ORDER

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StressCondition:
    """One artificial-difficulty setting applied to raw series."""

    name: str
    snr_db: float | None = None
    n_keep: int | None = None  # truncate to this many leading samples


@dataclass(frozen=True)
class StressConfig:
    """Artificial difficulty knobs for the secondary evaluation pipeline."""

    snr_db_levels: tuple[float, ...] = (10.0, 5.0)
    short_n_steps: tuple[int, ...] = (200,)
    seed: int = 0
    primary_metric: str = "log_loss"
    n_realizations: int | None = None  # override AuditConfig; None → keep
    include_baseline: bool = True
    model_names: tuple[str, ...] = ("random_forest", "logistic")
    rf_estimators: int | None = None  # shrink forest for smoke tests
    views: tuple[str, ...] = ("full",)  # family-subset labels to score
    # Optional regime subset (smoke / ablation). None → full audit panel.
    regimes: tuple[str, ...] | None = None


def add_white_noise_snr(
    series: NDArray[np.floating],
    snr_db: float,
    *,
    seed: int,
) -> NDArray[np.floating]:
    """Add Gaussian noise so that SNR equals ``snr_db`` (power ratio).

    SNR is defined as
    :math:`10 \\log_{10}(\\mathrm{Var}(x) / \\mathrm{Var}(\\varepsilon))`.
    """
    x = np.asarray(series, dtype=float).ravel()
    signal_var = float(np.var(x))
    if signal_var <= 0.0 or not np.isfinite(snr_db):
        return x.copy()
    noise_var = signal_var / (10.0 ** (float(snr_db) / 10.0))
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, np.sqrt(noise_var), size=x.size)
    return x + noise


def truncate_series(
    series: NDArray[np.floating],
    n_steps: int,
) -> NDArray[np.floating]:
    """Keep the leading ``n_steps`` samples (stress via short windows)."""
    x = np.asarray(series, dtype=float).ravel()
    n = int(max(1, n_steps))
    if x.size <= n:
        return x.copy()
    return x[:n].copy()


def multiclass_log_loss(
    y_true: Sequence[str],
    y_proba: NDArray[np.floating],
    *,
    labels: Sequence[str],
) -> float:
    """Cross-entropy (log-loss) with a fixed label order."""
    proba = np.asarray(y_proba, dtype=float)
    if proba.size == 0:
        return float("nan")
    # Folds may leave missing-class columns as NaN; replace then renormalise.
    proba = np.nan_to_num(proba, nan=1e-15, posinf=1.0, neginf=1e-15)
    proba = np.clip(proba, 1e-15, 1.0)
    row_sums = proba.sum(axis=1, keepdims=True)
    proba = proba / np.maximum(row_sums, 1e-15)
    return float(log_loss(list(y_true), proba, labels=list(labels)))


def log_loss_from_view(view: ViewMetrics) -> float:
    """OOF log-loss of a :class:`ViewMetrics` bundle.

    Columns of ``view.y_proba`` follow ``view.class_names`` (audit regime
    order). We re-order them lexicographically before calling
    :func:`sklearn.metrics.log_loss`, which assumes that ordering when
    ``labels`` is passed explicitly.
    """
    labels = list(view.class_names) if view.class_names else sorted(set(view.y_true))
    proba = np.asarray(view.y_proba, dtype=float)
    if proba.size == 0:
        return float("nan")
    order = sorted(range(len(labels)), key=lambda i: str(labels[i]))
    labels_sorted = [labels[i] for i in order]
    proba_sorted = proba[:, order]
    return multiclass_log_loss(view.y_true, proba_sorted, labels=labels_sorted)


def build_stress_conditions(config: StressConfig) -> list[StressCondition]:
    """Baseline + SNR levels + short-N windows."""
    conditions: list[StressCondition] = []
    if config.include_baseline:
        conditions.append(StressCondition(name="baseline_clean"))
    for snr in config.snr_db_levels:
        conditions.append(
            StressCondition(name=f"snr_{int(snr)}db", snr_db=float(snr))
        )
    for n_keep in config.short_n_steps:
        conditions.append(
            StressCondition(name=f"short_n{int(n_keep)}", n_keep=int(n_keep))
        )
    return conditions


def _make_transform(
    condition: StressCondition,
    *,
    seed: int,
) -> Callable[[NDArray[np.floating]], NDArray[np.floating]] | None:
    """Raw-series transform applied before any TS2G method."""
    if condition.snr_db is None and condition.n_keep is None:
        return None

    def _transform(values: NDArray[np.floating]) -> NDArray[np.floating]:
        x = np.asarray(values, dtype=float).ravel()
        if condition.n_keep is not None:
            x = truncate_series(x, condition.n_keep)
        if condition.snr_db is not None:
            x = add_white_noise_snr(x, condition.snr_db, seed=seed)
        return x

    return _transform


def generate_stressed_records(
    audit_config: AuditConfig,
    condition: StressCondition,
    *,
    n_realizations: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Regenerate the panel: raw series → stress transform → TS2G descriptors."""
    gen_cfg = load_generator_suite_config()
    records: list[dict[str, Any]] = []
    for dataset_id in audit_config.regimes:
        for realization in range(n_realizations):
            unit_seed = realization_seed(seed, dataset_id, realization)
            transform = _make_transform(condition, seed=unit_seed + 17_389)
            logger.info(
                "Stress %-14s regime=%-20s r=%02d seed=%d",
                condition.name,
                dataset_id,
                realization,
                unit_seed,
            )
            record = generate_one_matrix(
                dataset_id,
                seed=unit_seed,
                audit_config=audit_config,
                generator_config=gen_cfg,
                values_transform=transform,
            )
            record["realization"] = realization
            record["stress_condition"] = condition.name
            record["stress_snr_db"] = condition.snr_db
            record["stress_n_keep"] = condition.n_keep
            records.append(record)
    return records


def _family_mask_for_view(
    columns: pd.Index,
    view: str,
) -> NDArray[np.bool_]:
    from tsg.audit.features import family_feature_mask

    if view == "full":
        mask = np.zeros(len(columns), dtype=bool)
        for family in FAMILY_ORDER:
            mask |= family_feature_mask(columns, family)
        return mask
    if view == "empty":
        return np.zeros(len(columns), dtype=bool)
    # Accept short labels like "vis+prox".
    from tsg.audit.information import FAMILY_SHORT

    short_to_family = {v: k for k, v in FAMILY_SHORT.items()}
    parts = view.split("+")
    mask = np.zeros(len(columns), dtype=bool)
    for part in parts:
        family = short_to_family.get(part, part)
        mask |= family_feature_mask(columns, family)
    return mask


def evaluate_stressed_panel(
    records: list[dict[str, Any]],
    audit_config: AuditConfig,
    *,
    condition: StressCondition,
    model_names: Sequence[str],
    views: Sequence[str],
    rf_estimators: int | None = None,
) -> list[dict[str, Any]]:
    """Clean descriptors and score Macro-F1 + log-loss under one condition."""
    features, labels = build_feature_table(records, audit_config)
    health = diagnose_and_clean(features, audit_config)
    clean = health.x_clean
    clean.attrs.clear()
    models = _make_models(audit_config.seed)
    if rf_estimators is not None:
        models["random_forest"].set_params(clf__n_estimators=int(rf_estimators))

    rows: list[dict[str, Any]] = []
    for view in views:
        mask = _family_mask_for_view(clean.columns, view)
        if not bool(mask.any()):
            logger.warning("Stress view %s has no columns; skipping", view)
            continue
        block = clean.loc[:, mask]
        for model_name in model_names:
            metrics = _evaluate_view(
                block,
                labels,
                view=view,
                model_name=model_name,
                model=clone(models[model_name]),
                config=audit_config,
            )
            ll = log_loss_from_view(metrics)
            row = {
                "condition": condition.name,
                "snr_db": condition.snr_db,
                "n_keep": condition.n_keep,
                "view": view,
                "model": model_name,
                "n_samples": int(len(labels)),
                "n_features": int(block.shape[1]),
                "macro_f1_mean": float(metrics.macro_f1_mean),
                "macro_f1_std": float(metrics.macro_f1_std),
                "log_loss": float(ll),
                "accuracy_mean": float(metrics.accuracy_mean),
            }
            rows.append(row)
            logger.info(
                "Stress %-14s view=%-8s model=%-14s F1=%.4f ± %.4f  logloss=%.4f",
                condition.name,
                view,
                model_name,
                metrics.macro_f1_mean,
                metrics.macro_f1_std,
                ll,
            )
    return rows


@dataclass
class StressSuiteResult:
    """All stress-condition score rows."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def as_dict(self) -> dict[str, Any]:
        return {
            "conditions": list(self.conditions),
            "config": self.config,
            "rows": self.rows,
        }


def run_stress_suite(
    audit_config: AuditConfig,
    *,
    stress_config: StressConfig | None = None,
) -> StressSuiteResult:
    """Full stress pipeline: raw series → TS2G → Macro-F1 + log-loss."""
    cfg = stress_config or StressConfig()
    n_realizations = (
        int(cfg.n_realizations)
        if cfg.n_realizations is not None
        else int(audit_config.n_realizations)
    )
    # Keep generation length at the audit N; truncation happens in the transform.
    regime_override = list(cfg.regimes) if cfg.regimes is not None else None
    run_config = replace(
        audit_config,
        n_realizations=n_realizations,
        **({"regimes": regime_override} if regime_override is not None else {}),
    )
    conditions = build_stress_conditions(cfg)
    all_rows: list[dict[str, Any]] = []
    for condition in conditions:
        logger.info(
            "=== Stress condition %s (R=%d, N_gen=%d) ===",
            condition.name,
            n_realizations,
            run_config.n_steps,
        )
        records = generate_stressed_records(
            run_config,
            condition,
            n_realizations=n_realizations,
            seed=cfg.seed if cfg.seed else run_config.seed,
        )
        rows = evaluate_stressed_panel(
            records,
            run_config,
            condition=condition,
            model_names=cfg.model_names,
            views=cfg.views,
            rf_estimators=cfg.rf_estimators,
        )
        all_rows.extend(rows)
    return StressSuiteResult(
        rows=all_rows,
        conditions=[c.name for c in conditions],
        config={
            **asdict(cfg),
            "audit_n_steps": run_config.n_steps,
            "audit_n_realizations": n_realizations,
            "regimes": list(run_config.regimes),
        },
    )
