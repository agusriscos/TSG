"""Non-saturated classification protocol for B3 hard frontiers.

The clean N=1500, R=30 ensemble (geometry, figures 01–07, B1 cache) is left
untouched. This module adds two layers:

* Layer 0 — proper scores (log-loss, Brier, ECE, critical-pair probability
  margin) on an existing clean out-of-fold file, or by refitting classifiers
  on ``features_clean.npz`` only.
* Layer 1 — an N × SNR difficulty curve on the 12 B3 regimes, plus binary
  critical-pair tasks at a pre-specified headline operating point.

``StandardScaler`` stays inside the logistic and random-forest pipelines and
is fit on each training fold only. Observational noise uses
:func:`tsg.generators.perturbations.add_white_noise_snr`.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import subprocess
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from tsg.audit.artifacts import (
    atomic_write_json,
    atomic_write_text,
    load_features_npz,
    load_oof_predictions,
    save_features_npz,
    save_figure_trio,
    save_table_csv,
)
from tsg.audit.classification import (
    ClassificationResult,
    ViewMetrics,
    _evaluate_view,
    _make_models,
    _mean_prob_margin,
)
from tsg.audit.config import AuditConfig
from tsg.audit.ensemble import (
    generate_one_matrix,
    load_generator_suite_config,
    load_record,
    realization_seed,
    save_record,
    versioned_cache_dir,
)
from tsg.audit.features import build_feature_table, family_feature_mask
from tsg.audit.geometry import analyze_geometry
from tsg.audit.health import diagnose_and_clean
from tsg.audit.plotting import FAMILY_COLORS, plot_family_ablation
from tsg.descriptors.config import DescriptorConfig
from tsg.generators.config import GeneratorSuiteConfig
from tsg.generators.perturbations import add_white_noise_snr

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]

VIEW_ORDER: tuple[str, ...] = (
    "visibility",
    "transition",
    "proximity",
    "similarity",
    "full",
)
MODEL_ORDER: tuple[str, ...] = ("random_forest", "logistic")
VIEW_COLORS: dict[str, str] = {**FAMILY_COLORS, "full": "#222222"}

# A priori operating-point rule. Not tuned after seeing the curve.
F1_FLOOR = 0.80
F1_CEILING = 0.95
F1_SATURATION = 0.98
F1_TARGET_CENTER = 0.5 * (F1_FLOOR + F1_CEILING)  # 0.875
FALLBACK_SATURATED_N = 250
FALLBACK_SATURATED_SNR = 20.0
N_ECE_BINS = 15

CURVE_COLUMNS: tuple[str, ...] = (
    "n_steps",
    "snr_db",
    "view",
    "model",
    "n_samples",
    "macro_f1_mean",
    "macro_f1_std",
    "accuracy_mean",
    "accuracy_std",
    "macro_f1_oof",
    "macro_f1_ci_low",
    "macro_f1_ci_high",
    "log_loss",
    "log_loss_ci_low",
    "log_loss_ci_high",
    "brier",
    "brier_ci_low",
    "brier_ci_high",
    "ece",
    "ece_ci_low",
    "ece_ci_high",
    "mean_prob_margin",
    "mean_prob_margin_ci_low",
    "mean_prob_margin_ci_high",
    "n_bootstrap",
)

BINARY_COLUMNS: tuple[str, ...] = (
    "pair",
    "view",
    "model",
    "n_steps",
    "snr_db",
    "n_samples",
    "f1",
    "f1_std",
    "f1_oof",
    "f1_ci_low",
    "f1_ci_high",
    "auroc",
    "auroc_ci_low",
    "auroc_ci_high",
    "log_loss",
    "log_loss_ci_low",
    "log_loss_ci_high",
    "brier",
    "brier_ci_low",
    "brier_ci_high",
    "mean_prob_margin",
    "mean_prob_margin_ci_low",
    "mean_prob_margin_ci_high",
    "n_bootstrap",
)

PROPER_COLUMNS: tuple[str, ...] = (
    "n_steps",
    "snr_db",
    "source",
    "view",
    "model",
    "n_samples",
    "macro_f1_mean",
    "macro_f1_std",
    "accuracy_mean",
    "macro_f1_oof",
    "macro_f1_ci_low",
    "macro_f1_ci_high",
    "log_loss",
    "log_loss_ci_low",
    "log_loss_ci_high",
    "brier",
    "brier_ci_low",
    "brier_ci_high",
    "ece",
    "ece_ci_low",
    "ece_ci_high",
    "mean_prob_margin",
    "mean_prob_margin_ci_low",
    "mean_prob_margin_ci_high",
    "n_bootstrap",
)

_TABLE_BUNDLE_NAMES: dict[str, str] = {
    "difficulty_curve.csv": "b3_difficulty_curve.csv",
    "critical_pairs_binary_hard.csv": "b3_critical_pairs_binary_hard.csv",
    "proper_scores_n1500_clean.csv": "b3_proper_scores_n1500_clean.csv",
    "headline_operating_point.csv": "b3_headline_operating_point.csv",
    # Headline ablation replaces the saturated clean-panel F1 table in the bundle.
    "ablation.csv": "b3_ablation.csv",
    "ablation_n1500_clean.csv": "b3_ablation_n1500_clean.csv",
}
_FIGURE_BUNDLE_STEMS: dict[str, str] = {
    "04_family_ablation_scores": "b3_hard_04_family_ablation_scores",
    "08_difficulty_curve": "b3_hard_08_difficulty_curve",
    "09_critical_pairs_binary_hard": "b3_hard_09_critical_pairs_binary_hard",
}


@dataclass(frozen=True)
class DifficultyConfig:
    """N × SNR grid and the pre-specified headline cell."""

    n_grid: tuple[int, ...]
    snr_db: tuple[float | None, ...]
    n_realizations: int
    headline_n_steps: int
    headline_snr_db: float | None
    n_bootstrap: int = 1000
    bootstrap_seed: int = 42
    clean_n_steps: int = 1500

    @classmethod
    def from_yaml(cls, path: Path | str) -> DifficultyConfig:
        """Load the ``difficulty:`` block from the B3 YAML."""
        import yaml

        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict) or "difficulty" not in raw:
            raise ValueError(f"YAML {path} has no top-level 'difficulty' block")
        block = raw["difficulty"]
        headline = block.get("headline") or {}
        audit = raw.get("audit") or {}
        snr_values = tuple(_parse_snr(v) for v in block["snr_db"])
        return cls(
            n_grid=tuple(int(v) for v in block["n_grid"]),
            snr_db=snr_values,
            n_realizations=int(block.get("n_realizations", 20)),
            headline_n_steps=int(headline["n_steps"]),
            headline_snr_db=_parse_snr(headline.get("snr_db", 20)),
            n_bootstrap=int(block.get("n_bootstrap", 1000)),
            bootstrap_seed=int(block.get("bootstrap_seed", 42)),
            clean_n_steps=int(audit.get("n_steps", 1500)),
        )


@dataclass(frozen=True)
class GridPoint:
    """Full random-forest Macro-F1 at one (N, SNR) cell."""

    n_steps: int
    snr_db: float | None
    macro_f1_mean: float
    macro_f1_std: float
    ci_low: float
    ci_high: float

    @property
    def key(self) -> tuple[int, str]:
        return (int(self.n_steps), format_snr(self.snr_db))


@dataclass
class DifficultyResult:
    """Tables, figure paths and the resolved headline."""

    curve: pd.DataFrame
    binary: pd.DataFrame
    proper_scores: pd.DataFrame | None
    headline: pd.DataFrame
    operating_point: GridPoint
    fallback_applied: bool
    fallback_reason: str
    figure_paths: list[Path]
    layer0_status: str
    acceptance: list[str]


class _SnrTransform:
    """Picklable raw-series map: white noise at a fixed SNR and seed."""

    def __init__(self, snr_db: float, seed: int) -> None:
        self.snr_db = float(snr_db)
        self.seed = int(seed)

    def __call__(self, values: NDArray[np.floating]) -> NDArray[np.floating]:
        rng = np.random.default_rng(self.seed)
        return add_white_noise_snr(np.asarray(values, dtype=float), self.snr_db, rng)


def _parse_snr(value: Any) -> float | None:
    """Map YAML ``null`` / ``inf`` to a clean series; otherwise a finite SNR."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"inf", "infty", "none", "null", "∞"}:
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    return number


def format_snr(snr_db: float | None) -> str:
    """CSV / axis token. Clean series are ``inf`` (same convention as B5)."""
    if snr_db is None or not np.isfinite(snr_db):
        return "inf"
    value = float(snr_db)
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


def _snr_title(label: str) -> str:
    if label in {"inf", "None"}:
        return "SNR = ∞ (clean)"
    return f"SNR = {label} dB"


def difficulty_noise_seed(
    base_seed: int,
    dataset_id: str,
    realization: int,
    n_steps: int,
    snr_db: float | None,
) -> int:
    """Noise-stream seed: ``realization_seed`` plus an offset that depends on N and SNR.

    The generator keeps :func:`realization_seed`. Noise is drawn from a second
    generator so the clean series at the same ``(regime, realization)`` stays
    identical and the streams do not collide across the difficulty grid.
    """
    unit = realization_seed(int(base_seed), dataset_id, int(realization))
    if snr_db is None:
        snr_tag = 0
    else:
        snr_tag = int(round(float(snr_db) * 1000.0)) + 50_000
    return int(unit + 1_000_003 * int(n_steps) + 97 * snr_tag + 913_373)


def scoped_rng(seed: int, *tokens: object) -> np.random.Generator:
    """Independent Generator whose seed depends on ``seed`` and the tokens."""
    payload = json.dumps([int(seed), *[str(tok) for tok in tokens]], separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _limit_blas_threads() -> None:
    """One BLAS thread per worker so graph eigensolves do not oversubscribe."""
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(var, "1")


def _prepare_proba(proba: NDArray[np.floating] | None, n_rows: int, n_classes: int) -> NDArray[np.floating]:
    if proba is None:
        return np.full((n_rows, n_classes), np.nan, dtype=float)
    out = np.asarray(proba, dtype=float)
    if out.ndim != 2 or out.shape[0] != n_rows:
        return np.full((n_rows, n_classes), np.nan, dtype=float)
    out = np.nan_to_num(out, nan=1e-15, posinf=1.0, neginf=1e-15)
    out = np.clip(out, 1e-15, 1.0)
    totals = out.sum(axis=1, keepdims=True)
    return out / np.maximum(totals, 1e-15)


def _fold_scores(
    y_true: NDArray[np.object_],
    y_pred: NDArray[np.object_],
    fold_id: NDArray[np.integer] | None,
    class_names: Sequence[str],
) -> tuple[float, float, float, float]:
    """Mean and population std of per-fold Macro-F1 and accuracy."""
    if fold_id is None or len(fold_id) != len(y_true):
        f1 = float(
            f1_score(y_true, y_pred, average="macro", labels=list(class_names), zero_division=0)
        )
        acc = float(accuracy_score(y_true, y_pred))
        return f1, 0.0, acc, 0.0
    f1s: list[float] = []
    accs: list[float] = []
    for fold in np.unique(fold_id):
        mask = fold_id == fold
        f1s.append(
            float(
                f1_score(
                    y_true[mask],
                    y_pred[mask],
                    average="macro",
                    labels=list(class_names),
                    zero_division=0,
                )
            )
        )
        accs.append(float(accuracy_score(y_true[mask], y_pred[mask])))
    return (
        float(np.mean(f1s)),
        float(np.std(f1s)),
        float(np.mean(accs)),
        float(np.std(accs)),
    )


def aligned_log_loss(
    y_true: Sequence[str] | NDArray[np.object_],
    proba: NDArray[np.floating],
    class_names: Sequence[str],
) -> float:
    """Cross-entropy using column ``class_names`` order.

    This sklearn build one-hots ``labels`` in lexicographic order and does not
    reorder ``y_proba``. Scoring the true-class column directly keeps the
    regime order used by :class:`ViewMetrics`.
    """
    labels = [str(v) for v in y_true]
    if not labels:
        return float("nan")
    index = {name: i for i, name in enumerate(class_names)}
    columns = np.fromiter((index[label] for label in labels), dtype=int, count=len(labels))
    chosen = np.clip(proba[np.arange(len(labels)), columns], 1e-15, 1.0)
    return float(-np.mean(np.log(chosen)))


def multiclass_brier(
    y_true: Sequence[str] | NDArray[np.object_],
    proba: NDArray[np.floating],
    class_names: Sequence[str],
) -> float:
    """Mean squared error of the predictive distribution, summed over classes."""
    labels = [str(v) for v in y_true]
    if len(labels) == 0:
        return float("nan")
    index = {name: i for i, name in enumerate(class_names)}
    onehot = np.zeros_like(proba)
    for i, label in enumerate(labels):
        onehot[i, index[label]] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def expected_calibration_error(
    y_true: Sequence[str] | NDArray[np.object_],
    proba: NDArray[np.floating],
    class_names: Sequence[str],
    *,
    n_bins: int = N_ECE_BINS,
) -> float:
    """Top-label expected calibration error with equal-width bins on ``[0, 1]``."""
    labels = np.asarray([str(v) for v in y_true], dtype=object)
    if labels.size == 0:
        return float("nan")
    pred_idx = np.argmax(proba, axis=1)
    names = list(class_names)
    predicted = np.asarray([names[int(i)] for i in pred_idx], dtype=object)
    correct = predicted == labels
    confidence = proba[np.arange(labels.size), pred_idx]
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    n = int(labels.size)
    for bin_id in range(int(n_bins)):
        lo, hi = float(edges[bin_id]), float(edges[bin_id + 1])
        if bin_id == 0:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence > lo) & (confidence <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        ece += (count / n) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(ece)


def pair_probability_margin(
    y_true: Sequence[str] | NDArray[np.object_],
    proba: NDArray[np.floating],
    class_names: Sequence[str],
    a: str,
    b: str,
) -> float:
    """Mean ``|P(a) − P(b)|`` on samples whose true label is in ``{a, b}``.

    Same functional as :func:`tsg.audit.classification._mean_prob_margin`.
    """
    names = list(class_names)
    if a not in names or b not in names:
        return float("nan")
    labels = np.asarray([str(v) for v in y_true], dtype=object)
    mask = (labels == a) | (labels == b)
    if not np.any(mask):
        return float("nan")
    ia = names.index(a)
    ib = names.index(b)
    return float(np.nanmean(np.abs(proba[mask, ia] - proba[mask, ib])))


def critical_pair_mean_margin(
    y_true: Sequence[str] | NDArray[np.object_],
    proba: NDArray[np.floating],
    class_names: Sequence[str],
    pairs: Sequence[tuple[str, str]],
) -> float:
    """Unweighted mean of :func:`pair_probability_margin` over critical pairs."""
    margins = [
        pair_probability_margin(y_true, proba, class_names, a, b) for a, b in pairs
    ]
    finite = [m for m in margins if np.isfinite(m)]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def _bootstrap_ci(
    stat_fn: Any,
    n: int,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float, float]:
    """Percentile 95% interval. Returns ``(bootstrap mean, low, high)``."""
    if n <= 0 or n_bootstrap <= 0:
        return float("nan"), float("nan"), float("nan")
    stats = np.empty(int(n_bootstrap), dtype=float)
    for i in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        try:
            stats[i] = float(stat_fn(idx))
        except (ValueError, IndexError):
            stats[i] = float("nan")
    finite = stats[np.isfinite(stats)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(np.mean(finite)), float(low), float(high)


def _macro_f1_on(
    y_true: NDArray[np.object_],
    y_pred: NDArray[np.object_],
    class_names: Sequence[str],
    idx: NDArray[np.integer],
) -> float:
    return float(
        f1_score(
            y_true[idx],
            y_pred[idx],
            average="macro",
            labels=list(class_names),
            zero_division=0,
        )
    )


def _ci_includes_one(low: float, high: float) -> bool:
    if not np.isfinite(low) or not np.isfinite(high):
        return False
    return low <= 1.0 + 1e-12 and high >= 1.0 - 1e-12


def _ci_excludes_zero(low: float, high: float) -> bool:
    if not np.isfinite(low) or not np.isfinite(high):
        return False
    return high < 0.0 or low > 0.0


def _snr_allowed(snr_db: float | None) -> bool:
    """B3 difficulty stays out of the B5 algebraic-disconnection regime."""
    if snr_db is None:
        return True
    return float(snr_db) > 0.0


def summarize_predictions(
    *,
    y_true: Sequence[str],
    y_pred: Sequence[str],
    y_proba: NDArray[np.floating] | None,
    class_names: Sequence[str],
    fold_id: NDArray[np.integer] | None,
    pairs: Sequence[tuple[str, str]],
    rng: np.random.Generator,
    n_bootstrap: int,
    macro_f1_mean: float | None = None,
    macro_f1_std: float | None = None,
    accuracy_mean: float | None = None,
    accuracy_std: float | None = None,
) -> dict[str, float]:
    """Point estimates and OOF bootstrap intervals for one ``(view, model)``."""
    y_t = np.asarray([str(v) for v in y_true], dtype=object)
    y_p = np.asarray([str(v) for v in y_pred], dtype=object)
    names = [str(v) for v in class_names]
    proba = _prepare_proba(y_proba, len(y_t), len(names))
    folds = None if fold_id is None else np.asarray(fold_id)
    f1_mean, f1_std, acc_mean, acc_std = _fold_scores(y_t, y_p, folds, names)
    if macro_f1_mean is not None:
        f1_mean = float(macro_f1_mean)
    if macro_f1_std is not None:
        f1_std = float(macro_f1_std)
    if accuracy_mean is not None:
        acc_mean = float(accuracy_mean)
    if accuracy_std is not None:
        acc_std = float(accuracy_std)

    oof_f1 = float(
        f1_score(y_t, y_p, average="macro", labels=names, zero_division=0)
    )
    logloss = aligned_log_loss(y_t, proba, names)
    brier = multiclass_brier(y_t, proba, names)
    ece = expected_calibration_error(y_t, proba, names)
    margin = critical_pair_mean_margin(y_t, proba, names, pairs)

    def f1_stat(idx: NDArray[np.integer]) -> float:
        return _macro_f1_on(y_t, y_p, names, idx)

    def loss_stat(idx: NDArray[np.integer]) -> float:
        return aligned_log_loss(y_t[idx], proba[idx], names)

    def brier_stat(idx: NDArray[np.integer]) -> float:
        return multiclass_brier(y_t[idx], proba[idx], names)

    def ece_stat(idx: NDArray[np.integer]) -> float:
        return expected_calibration_error(y_t[idx], proba[idx], names)

    def margin_stat(idx: NDArray[np.integer]) -> float:
        return critical_pair_mean_margin(y_t[idx], proba[idx], names, pairs)

    _, f1_lo, f1_hi = _bootstrap_ci(f1_stat, len(y_t), rng, n_bootstrap)
    # Separate generators would change the draw. One stream, metrics in a fixed
    # order, so the Macro-F1 interval does not depend on later statistics.
    _, loss_lo, loss_hi = _bootstrap_ci(loss_stat, len(y_t), rng, n_bootstrap)
    _, brier_lo, brier_hi = _bootstrap_ci(brier_stat, len(y_t), rng, n_bootstrap)
    _, ece_lo, ece_hi = _bootstrap_ci(ece_stat, len(y_t), rng, n_bootstrap)
    _, margin_lo, margin_hi = _bootstrap_ci(margin_stat, len(y_t), rng, n_bootstrap)
    return {
        "n_samples": float(len(y_t)),
        "macro_f1_mean": f1_mean,
        "macro_f1_std": f1_std,
        "accuracy_mean": acc_mean,
        "accuracy_std": acc_std,
        "macro_f1_oof": oof_f1,
        "macro_f1_ci_low": f1_lo,
        "macro_f1_ci_high": f1_hi,
        "log_loss": logloss,
        "log_loss_ci_low": loss_lo,
        "log_loss_ci_high": loss_hi,
        "brier": brier,
        "brier_ci_low": brier_lo,
        "brier_ci_high": brier_hi,
        "ece": ece,
        "ece_ci_low": ece_lo,
        "ece_ci_high": ece_hi,
        "mean_prob_margin": margin,
        "mean_prob_margin_ci_low": margin_lo,
        "mean_prob_margin_ci_high": margin_hi,
        "n_bootstrap": float(n_bootstrap),
    }


def binary_auroc(
    y_true: Sequence[str] | NDArray[np.object_],
    proba: NDArray[np.floating],
    class_names: Sequence[str],
    positive: str,
) -> float:
    """AUROC with ``positive`` scored by its predicted probability."""
    labels = np.asarray([str(v) for v in y_true], dtype=object)
    y_bin = (labels == positive).astype(int)
    if np.unique(y_bin).size < 2:
        return float("nan")
    score = proba[:, list(class_names).index(positive)]
    return float(roc_auc_score(y_bin, score))


def summarize_binary_view(
    view: ViewMetrics,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, float]:
    """Binary Macro-F1, AUROC, log-loss, Brier and ``_mean_prob_margin``."""
    names = [str(v) for v in view.class_names]
    if len(names) != 2:
        raise ValueError(f"Binary summary expects 2 classes, got {names}")
    a, b = names[0], names[1]
    y_t = np.asarray([str(v) for v in view.y_true], dtype=object)
    y_p = np.asarray([str(v) for v in view.y_pred], dtype=object)
    proba = _prepare_proba(view.y_proba, len(y_t), len(names))
    oof_f1 = float(f1_score(y_t, y_p, average="macro", labels=names, zero_division=0))
    # Point estimate goes through the existing helper, not a reimplementation.
    margin = float(_mean_prob_margin(view, a, b))
    auroc = binary_auroc(y_t, proba, names, b)
    logloss = aligned_log_loss(y_t, proba, names)
    brier = multiclass_brier(y_t, proba, names)

    def f1_stat(idx: NDArray[np.integer]) -> float:
        return _macro_f1_on(y_t, y_p, names, idx)

    def auroc_stat(idx: NDArray[np.integer]) -> float:
        return binary_auroc(y_t[idx], proba[idx], names, b)

    def loss_stat(idx: NDArray[np.integer]) -> float:
        return aligned_log_loss(y_t[idx], proba[idx], names)

    def brier_stat(idx: NDArray[np.integer]) -> float:
        return multiclass_brier(y_t[idx], proba[idx], names)

    def margin_stat(idx: NDArray[np.integer]) -> float:
        return pair_probability_margin(y_t[idx], proba[idx], names, a, b)

    _, f1_lo, f1_hi = _bootstrap_ci(f1_stat, len(y_t), rng, n_bootstrap)
    _, auroc_lo, auroc_hi = _bootstrap_ci(auroc_stat, len(y_t), rng, n_bootstrap)
    _, loss_lo, loss_hi = _bootstrap_ci(loss_stat, len(y_t), rng, n_bootstrap)
    _, brier_lo, brier_hi = _bootstrap_ci(brier_stat, len(y_t), rng, n_bootstrap)
    _, margin_lo, margin_hi = _bootstrap_ci(margin_stat, len(y_t), rng, n_bootstrap)
    return {
        "n_samples": float(len(y_t)),
        "f1": float(view.macro_f1_mean),
        "f1_std": float(view.macro_f1_std),
        "f1_oof": oof_f1,
        "f1_ci_low": f1_lo,
        "f1_ci_high": f1_hi,
        "auroc": auroc,
        "auroc_ci_low": auroc_lo,
        "auroc_ci_high": auroc_hi,
        "log_loss": logloss,
        "log_loss_ci_low": loss_lo,
        "log_loss_ci_high": loss_hi,
        "brier": brier,
        "brier_ci_low": brier_lo,
        "brier_ci_high": brier_hi,
        "mean_prob_margin": margin,
        "mean_prob_margin_ci_low": margin_lo,
        "mean_prob_margin_ci_high": margin_hi,
        "n_bootstrap": float(n_bootstrap),
    }


def iter_view_frames(x: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Family views plus the full matrix. Columns are not scaled."""
    frames: list[tuple[str, pd.DataFrame]] = []
    for name in VIEW_ORDER:
        if name == "full":
            mask = np.ones(x.shape[1], dtype=bool)
        else:
            mask = family_feature_mask(x.columns, name)
        if bool(np.any(mask)):
            frames.append((name, x.loc[:, mask]))
    return frames


def evaluate_views(
    x: pd.DataFrame,
    y: pd.Series,
    config: AuditConfig,
    *,
    rf_n_jobs: int = 1,
) -> list[ViewMetrics]:
    """Cross-validated views. Scaling happens inside each fold's pipeline."""
    models = _make_models(config.seed, rf_n_jobs=rf_n_jobs)
    scored: list[ViewMetrics] = []
    for view_name, x_view in iter_view_frames(x):
        for model_name in MODEL_ORDER:
            scored.append(
                _evaluate_view(
                    x_view,
                    y,
                    view=view_name,
                    model_name=model_name,
                    model=models[model_name],
                    config=config,
                )
            )
    return scored


def _views_to_curve_rows(
    views: Sequence[ViewMetrics],
    pairs: Sequence[tuple[str, str]],
    *,
    n_steps: int,
    snr_db: float | None,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    label = format_snr(snr_db)
    for view in views:
        rng = scoped_rng(bootstrap_seed, "curve", int(n_steps), label, view.view, view.model)
        stats = summarize_predictions(
            y_true=view.y_true,
            y_pred=view.y_pred,
            y_proba=view.y_proba,
            class_names=view.class_names,
            fold_id=view.fold_id,
            pairs=pairs,
            rng=rng,
            n_bootstrap=n_bootstrap,
            macro_f1_mean=view.macro_f1_mean,
            macro_f1_std=view.macro_f1_std,
            accuracy_mean=view.accuracy_mean,
            accuracy_std=view.accuracy_std,
        )
        row: dict[str, Any] = {
            "n_steps": int(n_steps),
            "snr_db": label,
            "view": view.view,
            "model": view.model,
        }
        row.update(stats)
        rows.append(row)
    return rows


def _saturated(point: GridPoint) -> bool:
    return point.macro_f1_mean >= F1_SATURATION - 1e-12 or _ci_includes_one(
        point.ci_low, point.ci_high
    )


def _in_target_band(point: GridPoint) -> bool:
    return F1_FLOOR - 1e-12 <= point.macro_f1_mean <= F1_CEILING + 1e-12


def _acceptable(point: GridPoint) -> bool:
    return (
        _in_target_band(point)
        and not _ci_includes_one(point.ci_low, point.ci_high)
        and _snr_allowed(point.snr_db)
        and np.isfinite(point.macro_f1_mean)
    )


def _closest(points: Sequence[GridPoint]) -> GridPoint:
    def sort_key(point: GridPoint) -> tuple[float, int, str]:
        return (abs(point.macro_f1_mean - F1_TARGET_CENTER), point.n_steps, format_snr(point.snr_db))

    return min(points, key=sort_key)


def resolve_headline(
    points: Sequence[GridPoint],
    *,
    prescribed_n: int,
    prescribed_snr: float | None,
) -> tuple[GridPoint, bool, str, str]:
    """Apply the pre-specified headline rule.

    Returns
    -------
    tuple
        ``(point, fallback_applied, reason_code, detail)``.

    The rule, fixed before looking at the curve, is:

    1. Start at the prescribed cell (paper: N=500, SNR=20 dB).
    2. If full RF CV Macro-F1 ≥ 0.98 or the OOF 95% CI includes 1, move to
       N=250, SNR=20 dB.
    3. If full RF CV Macro-F1 < 0.80, choose the grid point closest to 0.875
       (the centre of ``[0.80, 0.95]``), preferring points that already sit in
       that interval and whose CI excludes 1. The written candidates N=500
       clean and N=1000 at 10 dB are inside that search, not a second menu.
    4. If the chosen cell is still outside ``[0.80, 0.95]`` or its CI includes
       1, move to the in-range grid point closest to 0.875 whose CI excludes 1.
       SNR ≤ 0 dB is never eligible.
    """
    by_key = {point.key: point for point in points}
    prescribed_key = (int(prescribed_n), format_snr(prescribed_snr))
    saturated_key = (int(FALLBACK_SATURATED_N), format_snr(FALLBACK_SATURATED_SNR))
    if prescribed_key not in by_key:
        raise ValueError(f"Prescribed headline {prescribed_key} is not in the difficulty grid")
    if saturated_key not in by_key:
        raise ValueError(f"Saturation fallback {saturated_key} is not in the difficulty grid")

    prescribed = by_key[prescribed_key]
    chosen = prescribed
    fallback = False
    reason = "prescribed"
    allowed = [point for point in points if _snr_allowed(point.snr_db)]
    acceptable = [point for point in allowed if _acceptable(point)]

    if _saturated(prescribed):
        chosen = by_key[saturated_key]
        fallback = True
        reason = "saturation_to_n250_snr20"
    elif prescribed.macro_f1_mean < F1_FLOOR:
        pool = acceptable or allowed
        chosen = _closest(pool)
        fallback = True
        reason = "undershoot_closest_to_center"

    if not _acceptable(chosen):
        if not acceptable:
            detail = (
                f"prescribed N={prescribed.n_steps} SNR={format_snr(prescribed.snr_db)} "
                f"CV Macro-F1={prescribed.macro_f1_mean:.4f} "
                f"CI=[{prescribed.ci_low:.4f}, {prescribed.ci_high:.4f}]; "
                "no grid point lies in [0.80, 0.95] with a CI that excludes 1"
            )
            return chosen, True, reason + "+no_acceptable_cell", detail
        guarded = _closest(acceptable)
        if guarded.key != chosen.key:
            fallback = True
            reason = reason + "+interval_guard_closest_to_center"
            chosen = guarded

    detail = (
        f"prescribed N={prescribed.n_steps} SNR={format_snr(prescribed.snr_db)} "
        f"CV Macro-F1={prescribed.macro_f1_mean:.4f} "
        f"CI=[{prescribed.ci_low:.4f}, {prescribed.ci_high:.4f}]; "
        f"selected N={chosen.n_steps} SNR={format_snr(chosen.snr_db)} "
        f"CV Macro-F1={chosen.macro_f1_mean:.4f} "
        f"CI=[{chosen.ci_low:.4f}, {chosen.ci_high:.4f}]"
    )
    return chosen, fallback, reason, detail


def evaluate_binary_pairs(
    x: pd.DataFrame,
    y: pd.Series,
    config: AuditConfig,
    *,
    n_steps: int,
    snr_db: float | None,
    n_bootstrap: int,
    bootstrap_seed: int,
    rf_n_jobs: int = 1,
) -> pd.DataFrame:
    """Dedicated binary tasks for each critical pair, same N, SNR and CV seed."""
    rows: list[dict[str, Any]] = []
    label = format_snr(snr_db)
    y_text = y.astype(str)
    for a, b in config.critical_pairs:
        mask = y_text.isin([a, b]).to_numpy()
        if int(mask.sum()) < 4:
            logger.warning("Skipping binary pair %s vs %s: only %d rows", a, b, int(mask.sum()))
            continue
        x_sub = x.loc[mask].reset_index(drop=True)
        y_sub = y_text.loc[mask].reset_index(drop=True)
        binary_cfg = replace(config, regimes=(a, b))
        views = evaluate_views(x_sub, y_sub, binary_cfg, rf_n_jobs=rf_n_jobs)
        for view in views:
            rng = scoped_rng(
                bootstrap_seed, "binary", int(n_steps), label, a, b, view.view, view.model
            )
            stats = summarize_binary_view(view, rng=rng, n_bootstrap=n_bootstrap)
            row: dict[str, Any] = {
                "pair": f"{a}__{b}",
                "view": view.view,
                "model": view.model,
                "n_steps": int(n_steps),
                "snr_db": label,
            }
            row.update(stats)
            rows.append(row)
    frame = pd.DataFrame(rows)
    return _ordered_frame(frame, BINARY_COLUMNS)


def pairwise_f1_differences(
    views: Sequence[ViewMetrics],
    *,
    n_steps: int,
    snr_db: float | None,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    """Paired OOF bootstrap of Macro-F1 differences between random-forest views."""
    rf = {view.view: view for view in views if view.model == "random_forest"}
    names = [name for name in VIEW_ORDER if name in rf]
    rows: list[dict[str, Any]] = []
    label = format_snr(snr_db)
    for i, left_name in enumerate(names):
        for right_name in names[i + 1 :]:
            left = rf[left_name]
            right = rf[right_name]
            if len(left.y_true) != len(right.y_true) or any(
                a != b for a, b in zip(left.y_true, right.y_true)
            ):
                raise RuntimeError(
                    f"OOF labels for {left_name} and {right_name} are not aligned"
                )
            y_t = np.asarray([str(v) for v in left.y_true], dtype=object)
            pred_l = np.asarray([str(v) for v in left.y_pred], dtype=object)
            pred_r = np.asarray([str(v) for v in right.y_pred], dtype=object)
            class_names = list(left.class_names)
            delta_oof = _macro_f1_on(
                y_t, pred_l, class_names, np.arange(len(y_t))
            ) - _macro_f1_on(y_t, pred_r, class_names, np.arange(len(y_t)))
            rng = scoped_rng(
                bootstrap_seed,
                "delta",
                int(n_steps),
                label,
                left_name,
                right_name,
            )

            def _delta(idx: NDArray[np.integer]) -> float:
                return _macro_f1_on(y_t, pred_l, class_names, idx) - _macro_f1_on(
                    y_t, pred_r, class_names, idx
                )

            _, low, high = _bootstrap_ci(_delta, len(y_t), rng, n_bootstrap)
            rows.append(
                {
                    "record_type": "f1_difference",
                    "n_steps": int(n_steps),
                    "snr_db": label,
                    "view": left_name,
                    "model": "random_forest",
                    "compare_view": right_name,
                    "delta_macro_f1": float(delta_oof),
                    "delta_macro_f1_cv": float(left.macro_f1_mean - right.macro_f1_mean),
                    "delta_ci_low": low,
                    "delta_ci_high": high,
                    "ci_excludes_zero": bool(_ci_excludes_zero(low, high)),
                }
            )
    return rows


def _ordered_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=list(columns))
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = np.nan
    return out.loc[:, list(columns)]


def _json_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for row in rows:
        clean: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, (np.floating, float)):
                number = float(value)
                clean[key] = number if np.isfinite(number) else None
            elif isinstance(value, (np.integer,)):
                clean[key] = int(value)
            elif isinstance(value, (np.bool_,)):
                clean[key] = bool(value)
            else:
                clean[key] = value
        payload.append(clean)
    return payload


def _generate_cell_records(
    audit: AuditConfig,
    *,
    n_steps: int,
    snr_db: float | None,
    n_realizations: int,
    cache_root: Path,
    generator_config: GeneratorSuiteConfig,
    descriptor_config: DescriptorConfig,
    force: bool,
) -> list[dict[str, Any]]:
    cell_audit = replace(audit, n_steps=int(n_steps), n_realizations=int(n_realizations))
    cache = versioned_cache_dir(
        cache_root, cell_audit, generator_config, snr_db=snr_db
    )
    cache.mkdir(parents=True, exist_ok=True)
    logger.info("Difficulty cell cache %s", cache)
    records: list[dict[str, Any]] = []
    total = len(cell_audit.regimes) * int(n_realizations)
    done = 0
    for dataset_id in cell_audit.regimes:
        for realization in range(int(n_realizations)):
            done += 1
            unit_seed = realization_seed(cell_audit.seed, dataset_id, realization)
            cached = None if force else load_record(cache, dataset_id, realization, unit_seed)
            if cached is not None:
                cached.setdefault("realization", realization)
                records.append(cached)
                continue
            transform = None
            noise_seed: int | None = None
            if snr_db is not None:
                noise_seed = difficulty_noise_seed(
                    cell_audit.seed, dataset_id, realization, int(n_steps), snr_db
                )
                transform = _SnrTransform(float(snr_db), noise_seed)
            if done == 1 or done % 20 == 0 or done == total:
                logger.info(
                    "Difficulty series %d/%d N=%d SNR=%s regime=%s r=%02d",
                    done,
                    total,
                    int(n_steps),
                    format_snr(snr_db),
                    dataset_id,
                    realization,
                )
            record = generate_one_matrix(
                dataset_id,
                seed=unit_seed,
                audit_config=cell_audit,
                generator_config=generator_config,
                descriptor_config=descriptor_config,
                values_transform=transform,
            )
            record["realization"] = realization
            record["n_steps"] = int(n_steps)
            record["snr_db"] = None if snr_db is None else float(snr_db)
            record["noise_seed"] = noise_seed
            save_record(record, cache)
            records.append(record)
    return records


def _score_clean_table(
    x: pd.DataFrame,
    y: pd.Series,
    audit: AuditConfig,
    *,
    n_steps: int,
    snr_db: float | None,
    n_bootstrap: int,
    bootstrap_seed: int,
    rf_n_jobs: int,
) -> tuple[list[dict[str, Any]], list[ViewMetrics]]:
    health = diagnose_and_clean(x, audit)
    logger.info(
        "Difficulty features N=%d SNR=%s raw=%d clean=%d",
        int(n_steps),
        format_snr(snr_db),
        health.n_features,
        health.x_clean.shape[1],
    )
    views = evaluate_views(health.x_clean, y, audit, rf_n_jobs=rf_n_jobs)
    rows = _views_to_curve_rows(
        views,
        audit.critical_pairs,
        n_steps=int(n_steps),
        snr_db=snr_db,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
    )
    return rows, views


def _cell_checkpoint_paths(difficulty_dir: Path, n_steps: int, snr_db: float | None) -> tuple[Path, Path]:
    tag = f"n{int(n_steps)}_snr{format_snr(snr_db)}"
    return (
        difficulty_dir / f"features_{tag}.npz",
        difficulty_dir / f"metrics_{tag}.json",
    )


def _difficulty_cell_job(
    n_steps: int,
    snr_db: float | None,
    *,
    audit: AuditConfig,
    n_realizations: int,
    n_bootstrap: int,
    bootstrap_seed: int,
    cache_root: str,
    difficulty_dir: str,
    force: bool,
    generator_config: GeneratorSuiteConfig,
    descriptor_config: DescriptorConfig,
) -> list[dict[str, Any]]:
    """One (N, SNR) cell. Top-level so joblib can pickle it."""
    _limit_blas_threads()
    features_path, metrics_path = _cell_checkpoint_paths(
        Path(difficulty_dir), int(n_steps), snr_db
    )
    if features_path.exists() and metrics_path.exists() and not force:
        logger.info("Reusing difficulty checkpoint %s", metrics_path.name)
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        return list(payload["rows"])

    cell_audit = replace(
        audit, n_steps=int(n_steps), n_realizations=int(n_realizations)
    )
    records = _generate_cell_records(
        audit,
        n_steps=int(n_steps),
        snr_db=snr_db,
        n_realizations=int(n_realizations),
        cache_root=Path(cache_root),
        generator_config=generator_config,
        descriptor_config=descriptor_config,
        force=force,
    )
    x, y = build_feature_table(records, cell_audit)
    rows, _views = _score_clean_table(
        x,
        y,
        cell_audit,
        n_steps=int(n_steps),
        snr_db=snr_db,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
        rf_n_jobs=1,
    )
    health = diagnose_and_clean(x, cell_audit)
    save_features_npz(
        features_path,
        x=health.x_clean,
        y=y,
        regime_names=list(cell_audit.regimes),
    )
    atomic_write_json(metrics_path, {"rows": _json_rows(rows)})
    full = next(row for row in rows if row["view"] == "full" and row["model"] == "random_forest")
    logger.info(
        "Difficulty cell N=%d SNR=%s full RF Macro-F1=%.4f CI=[%.4f, %.4f]",
        int(n_steps),
        format_snr(snr_db),
        float(full["macro_f1_mean"]),
        float(full["macro_f1_ci_low"]),
        float(full["macro_f1_ci_high"]),
    )
    return rows


def _load_checkpoint_frame(features_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    loaded = load_features_npz(features_path)
    x = pd.DataFrame(np.asarray(loaded["X"], dtype=float), columns=loaded["feature_names"])
    y = pd.Series(loaded["y"], name="regime")
    return x, y


def _full_rf_points(curve: pd.DataFrame) -> list[GridPoint]:
    points: list[GridPoint] = []
    block = curve[(curve["view"] == "full") & (curve["model"] == "random_forest")]
    for row in block.itertuples(index=False):
        points.append(
            GridPoint(
                n_steps=int(row.n_steps),
                snr_db=_parse_snr(row.snr_db),
                macro_f1_mean=float(row.macro_f1_mean),
                macro_f1_std=float(row.macro_f1_std),
                ci_low=float(row.macro_f1_ci_low),
                ci_high=float(row.macro_f1_ci_high),
            )
        )
    return points


def proper_scores_from_oof(
    bundles: Mapping[str, Mapping[str, Any]],
    pairs: Sequence[tuple[str, str]],
    *,
    n_steps: int,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    """Layer 0 from an existing ``oof_predictions.npz`` payload."""
    rows: list[dict[str, Any]] = []
    for key in sorted(bundles):
        if "__" not in key:
            continue
        view_name, model_name = key.split("__", 1)
        bundle = bundles[key]
        class_names = [str(v) for v in bundle.get("class_names", [])]
        if not class_names:
            class_names = sorted(set(bundle["y_true"]))
        rng = scoped_rng(bootstrap_seed, "layer0", view_name, model_name)
        stats = summarize_predictions(
            y_true=bundle["y_true"],
            y_pred=bundle["y_pred"],
            y_proba=bundle.get("y_proba"),
            class_names=class_names,
            fold_id=np.asarray(bundle["fold_id"], dtype=int),
            pairs=pairs,
            rng=rng,
            n_bootstrap=n_bootstrap,
        )
        row: dict[str, Any] = {
            "n_steps": int(n_steps),
            "snr_db": "inf",
            "source": "oof_predictions",
            "view": view_name,
            "model": model_name,
        }
        row.update(stats)
        rows.append(row)
    return _ordered_frame(pd.DataFrame(rows), PROPER_COLUMNS)


def proper_scores_from_features(
    x: pd.DataFrame,
    y: pd.Series,
    config: AuditConfig,
    *,
    n_steps: int,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    """Layer 0 by refitting classifiers on a saved clean feature matrix."""
    rows, _views = _score_clean_table(
        x,
        y,
        config,
        n_steps=n_steps,
        snr_db=None,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
        rf_n_jobs=1,
    )
    for row in rows:
        row["source"] = "features_clean_refit"
        row["snr_db"] = "inf"
    return _ordered_frame(pd.DataFrame(rows), PROPER_COLUMNS)


def try_layer0(
    output_dir: Path,
    config: AuditConfig,
    difficulty: DifficultyConfig,
) -> tuple[pd.DataFrame | None, str]:
    """Proper scores if the clean panel is already on disk. Never rebuilds graphs."""
    data_dir = Path(output_dir) / "data"
    oof_path = data_dir / "oof_predictions.npz"
    features_path = data_dir / "features_clean.npz"
    if oof_path.exists():
        logger.info("Layer 0 proper scores from %s", oof_path)
        bundles = load_oof_predictions(oof_path)
        frame = proper_scores_from_oof(
            bundles,
            config.critical_pairs,
            n_steps=difficulty.clean_n_steps,
            n_bootstrap=difficulty.n_bootstrap,
            bootstrap_seed=difficulty.bootstrap_seed,
        )
        return frame, f"oof:{oof_path}"
    if features_path.exists():
        logger.info("Layer 0 refitting classifiers on %s (no graph rebuild)", features_path)
        loaded = load_features_npz(features_path)
        x = pd.DataFrame(np.asarray(loaded["X"], dtype=float), columns=loaded["feature_names"])
        y = pd.Series(loaded["y"], name="regime")
        frame = proper_scores_from_features(
            x,
            y,
            config,
            n_steps=difficulty.clean_n_steps,
            n_bootstrap=difficulty.n_bootstrap,
            bootstrap_seed=difficulty.bootstrap_seed,
        )
        return frame, f"features:{features_path}"
    message = (
        "Clean-panel proper scores were not computed: "
        f"{oof_path} and {features_path} are absent. "
        "The N=1500 ensemble was not rebuilt."
    )
    logger.warning(message)
    return None, message


def _normalize_snr_column(frame: pd.DataFrame) -> pd.DataFrame:
    """Map CSV floats (``20.0``, ``inf``) back to canonical tokens (``20``, ``inf``)."""
    out = frame.copy()
    if "snr_db" in out.columns:
        out["snr_db"] = out["snr_db"].map(lambda value: format_snr(_parse_snr(value)))
    return out


def _sort_curve(frame: pd.DataFrame, difficulty: DifficultyConfig) -> pd.DataFrame:
    snr_rank = {format_snr(snr): i for i, snr in enumerate(difficulty.snr_db)}
    view_rank = {name: i for i, name in enumerate(VIEW_ORDER)}
    model_rank = {name: i for i, name in enumerate(MODEL_ORDER)}
    out = _normalize_snr_column(frame)
    out["_snr"] = out["snr_db"].map(lambda value: snr_rank.get(str(value), 99))
    out["_view"] = out["view"].map(lambda value: view_rank.get(str(value), 99))
    out["_model"] = out["model"].map(lambda value: model_rank.get(str(value), 99))
    out = out.sort_values(["n_steps", "_snr", "_view", "_model"]).drop(
        columns=["_snr", "_view", "_model"]
    )
    return _ordered_frame(out.reset_index(drop=True), CURVE_COLUMNS)


def _preserve_clean_ablation(tables_dir: Path, bundle_dir: Path | None = None) -> None:
    """Keep a copy of the saturated N=1500 ablation before overwriting it."""
    tables_dir = Path(tables_dir)
    dest = tables_dir / "ablation_n1500_clean.csv"
    if dest.exists():
        return
    candidates = [tables_dir / "ablation.csv"]
    if bundle_dir is not None:
        candidates.append(Path(bundle_dir) / "tables" / "b3_ablation.csv")
        candidates.append(Path(bundle_dir) / "tables" / "b3_ablation_n1500_clean.csv")
    for src in candidates:
        if not src.exists():
            continue
        frame = pd.read_csv(src)
        full_rf = frame[
            (frame["view"] == "full") & (frame["model"] == "random_forest")
        ]
        if full_rf.empty:
            continue
        if float(full_rf.iloc[0]["macro_f1_mean"]) < 0.999:
            continue
        save_table_csv(dest, frame)
        logger.info("Preserved clean N=1500 ablation at %s", dest)
        return


def write_headline_ablation_artifacts(
    *,
    x: pd.DataFrame,
    y: pd.Series,
    views: Sequence[ViewMetrics],
    audit: AuditConfig,
    operating: GridPoint,
    tables_dir: Path,
    figures_dir: Path,
    bundle_dir: Path | None = None,
) -> Path:
    """Rewrite family-ablation table/figure at the non-saturated headline cell.

    The clean N=1500 ablation is archived as ``ablation_n1500_clean.csv`` when
    still present. Geometry (global silhouette) is recomputed on the same
    headline feature matrix so both panels share one operating point.
    """
    tables_dir = Path(tables_dir)
    figures_dir = Path(figures_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    _preserve_clean_ablation(tables_dir, bundle_dir)

    classification = ClassificationResult(views=list(views))
    geometry = analyze_geometry(x, y, audit)
    abl_rows = [
        {
            "view": view.view,
            "model": view.model,
            "macro_f1_mean": view.macro_f1_mean,
            "macro_f1_std": view.macro_f1_std,
            "accuracy_mean": view.accuracy_mean,
            "accuracy_std": view.accuracy_std,
            "n_steps": int(operating.n_steps),
            "snr_db": format_snr(operating.snr_db),
        }
        for view in views
    ]
    save_table_csv(tables_dir / "ablation.csv", pd.DataFrame(abl_rows))
    note = (
        f"Headline operating point: N={operating.n_steps}, "
        f"{_snr_title(format_snr(operating.snr_db))}, "
        f"R={audit.n_realizations} (non-saturated B3 protocol)"
    )
    path = plot_family_ablation(
        classification,
        geometry,
        audit,
        figures_dir,
        lang="en",
        note=note,
    )
    logger.info(
        "Headline ablation full RF Macro-F1=%.4f silhouette=%.4f figure=%s",
        next(
            v.macro_f1_mean
            for v in views
            if v.view == "full" and v.model == "random_forest"
        ),
        geometry.silhouette_global,
        path,
    )
    return path


def plot_difficulty_curve(
    curve: pd.DataFrame,
    operating: GridPoint,
    *,
    fallback_reason: str,
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    """Macro-F1 versus N, one panel per SNR, with bootstrap ribbons."""
    output_dir.mkdir(parents=True, exist_ok=True)
    curve = _normalize_snr_column(curve)
    snr_levels = list(dict.fromkeys(str(v) for v in curve["snr_db"].tolist()))
    fig, axes = plt.subplots(
        1,
        len(snr_levels),
        figsize=(4.15 * len(snr_levels) + 1.6, 4.6),
        sharey=True,
        squeeze=False,
    )
    axes_flat = list(axes[0])
    y_min = float(np.nanmin(curve["macro_f1_ci_low"].to_numpy(dtype=float)))
    y_max = float(np.nanmax(curve["macro_f1_ci_high"].to_numpy(dtype=float)))
    for ax, snr in zip(axes_flat, snr_levels):
        block = curve[curve["snr_db"].astype(str) == snr]
        ax.axhspan(F1_FLOOR, F1_CEILING, color="#c8c8c8", alpha=0.35, zorder=0)
        for view in VIEW_ORDER:
            color = VIEW_COLORS.get(view, "#777777")
            rf = block[(block["view"] == view) & (block["model"] == "random_forest")].sort_values(
                "n_steps"
            )
            if rf.empty:
                continue
            x = rf["n_steps"].to_numpy(dtype=float)
            y = rf["macro_f1_oof"].to_numpy(dtype=float)
            low = rf["macro_f1_ci_low"].to_numpy(dtype=float)
            high = rf["macro_f1_ci_high"].to_numpy(dtype=float)
            ax.plot(
                x,
                y,
                color=color,
                marker="o",
                lw=1.7,
                ms=4.5,
                label=view,
                zorder=3,
            )
            ax.fill_between(x, low, high, color=color, alpha=0.18, linewidth=0, zorder=2)
            lr = block[(block["view"] == view) & (block["model"] == "logistic")].sort_values(
                "n_steps"
            )
            if not lr.empty:
                ax.plot(
                    lr["n_steps"].to_numpy(dtype=float),
                    lr["macro_f1_oof"].to_numpy(dtype=float),
                    color=color,
                    ls="--",
                    lw=1.0,
                    marker=".",
                    ms=3.5,
                    zorder=2,
                )
        ax.set_title(_snr_title(snr))
        ax.set_xlabel("Series length N")
        ax.set_xticks(sorted(int(v) for v in block["n_steps"].unique()))
        ax.grid(True, axis="y", lw=0.4, alpha=0.45)
        if format_snr(operating.snr_db) == snr:
            ax.axvline(operating.n_steps, color="black", ls=":", lw=0.9, zorder=1)
            full = block[
                (block["view"] == "full")
                & (block["model"] == "random_forest")
                & (block["n_steps"] == operating.n_steps)
            ]
            if not full.empty:
                ax.scatter(
                    [operating.n_steps],
                    [float(full.iloc[0]["macro_f1_oof"])],
                    marker="*",
                    s=160,
                    color="black",
                    zorder=5,
                    label="headline",
                )
    axes_flat[0].set_ylabel("Macro-F1 (pooled out-of-fold)")
    axes_flat[0].set_ylim(max(0.0, y_min - 0.04), min(1.05, y_max + 0.04))
    handles, labels = axes_flat[0].get_legend_handles_labels()
    # Logistic is drawn in every panel; one proxy keeps the legend readable.
    from matplotlib.lines import Line2D

    handles.append(Line2D([0], [0], color="#444444", ls="--", lw=1.1))
    labels.append("logistic (dashed)")
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.01, 0.55),
        frameon=False,
        fontsize=8,
    )
    fig.suptitle("Hard-frontier difficulty curve (12 regimes)", fontsize=12)
    fig.text(
        0.01,
        0.01,
        (
            "Solid lines: random forest. Ribbons: 95% bootstrap CI of pooled OOF Macro-F1. "
            f"Grey band: target interval [{F1_FLOOR:.2f}, {F1_CEILING:.2f}]. "
            f"Star: headline N={operating.n_steps}, SNR={format_snr(operating.snr_db)} "
            f"(CV Macro-F1={operating.macro_f1_mean:.3f}, {fallback_reason})."
        ),
        fontsize=7.5,
    )
    fig.tight_layout(rect=(0.0, 0.06, 0.86, 0.92))
    paths = save_figure_trio(fig, output_dir / "08_difficulty_curve", dpi=dpi)
    plt.close(fig)
    return paths


def _cell_text_color(value: float, vmin: float, vmax: float) -> str:
    if not np.isfinite(value):
        return "black"
    span = max(vmax - vmin, 1e-6)
    # viridis is dark on the low end and yellow on the high end.
    return "white" if (value - vmin) / span < 0.62 else "black"


def plot_binary_critical_pairs(
    binary: pd.DataFrame,
    operating: GridPoint,
    *,
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    """Binary Macro-F1 and AUROC by critical pair and family at the headline."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rf = binary[binary["model"] == "random_forest"].copy()
    pairs = list(dict.fromkeys(rf["pair"].astype(str).tolist()))
    views = [view for view in VIEW_ORDER if view in set(rf["view"].astype(str))]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, max(4.8, 0.62 * len(pairs) + 2.2)))
    specs = (
        (axes[0], "f1", "Binary Macro-F1", 0.45, 1.0),
        (axes[1], "auroc", "AUROC", 0.50, 1.0),
    )
    for ax, column, title, vmin, vmax in specs:
        matrix = np.full((len(pairs), len(views)), np.nan, dtype=float)
        for i, pair in enumerate(pairs):
            for j, view in enumerate(views):
                hit = rf[(rf["pair"] == pair) & (rf["view"] == view)]
                if not hit.empty:
                    matrix[i, j] = float(hit.iloc[0][column])
        image = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(views)))
        ax.set_xticklabels(views, rotation=30, ha="right")
        ax.set_yticks(range(len(pairs)))
        ax.set_yticklabels([pair.replace("__", " vs ") for pair in pairs], fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Family view")
        for i in range(len(pairs)):
            for j in range(len(views)):
                value = matrix[i, j]
                text = "—" if not np.isfinite(value) else f"{value:.2f}"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=_cell_text_color(value, vmin, vmax),
                )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("Critical pair")
    fig.suptitle(
        (
            "Binary critical-pair tasks at "
            f"N={operating.n_steps}, {_snr_title(format_snr(operating.snr_db))} "
            "(random forest)"
        ),
        fontsize=12,
    )
    fig.tight_layout()
    paths = save_figure_trio(fig, output_dir / "09_critical_pairs_binary_hard", dpi=dpi)
    plt.close(fig)
    return paths


def _curve_ci(
    curve: pd.DataFrame | None,
    *,
    n_steps: int,
    snr_label: str,
    view: str,
    model: str,
) -> tuple[float, float, float]:
    """Return ``(oof Macro-F1, CI low, CI high)`` from the difficulty curve."""
    if curve is None or curve.empty:
        return float("nan"), float("nan"), float("nan")
    frame = _normalize_snr_column(curve)
    hit = frame[
        (frame["n_steps"] == int(n_steps))
        & (frame["snr_db"].astype(str) == snr_label)
        & (frame["view"] == view)
        & (frame["model"] == model)
    ]
    if hit.empty:
        return float("nan"), float("nan"), float("nan")
    row = hit.iloc[0]
    return float(row["macro_f1_oof"]), float(row["macro_f1_ci_low"]), float(row["macro_f1_ci_high"])


def _build_headline_frame(
    operating: GridPoint,
    *,
    prescribed: GridPoint,
    fallback_applied: bool,
    fallback_reason: str,
    fallback_detail: str,
    views: Sequence[ViewMetrics],
    differences: Sequence[Mapping[str, Any]],
    curve: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "record_type": "operating_point",
            "n_steps": operating.n_steps,
            "snr_db": format_snr(operating.snr_db),
            "prescribed_n_steps": prescribed.n_steps,
            "prescribed_snr_db": format_snr(prescribed.snr_db),
            "prescribed_macro_f1_mean": prescribed.macro_f1_mean,
            "prescribed_macro_f1_ci_low": prescribed.ci_low,
            "prescribed_macro_f1_ci_high": prescribed.ci_high,
            "fallback_applied": bool(fallback_applied),
            "fallback_reason": fallback_reason,
            "fallback_detail": fallback_detail,
            "view": "full",
            "model": "random_forest",
            "compare_view": "",
            "macro_f1_mean": operating.macro_f1_mean,
            "macro_f1_std": operating.macro_f1_std,
            "macro_f1_ci_low": operating.ci_low,
            "macro_f1_ci_high": operating.ci_high,
            "ci_excludes_one": (not _ci_includes_one(operating.ci_low, operating.ci_high)),
            "delta_macro_f1": np.nan,
            "delta_macro_f1_cv": np.nan,
            "delta_ci_low": np.nan,
            "delta_ci_high": np.nan,
            "ci_excludes_zero": False,
        }
    ]
    for view in views:
        _, ci_low, ci_high = _curve_ci(
            curve,
            n_steps=operating.n_steps,
            snr_label=format_snr(operating.snr_db),
            view=view.view,
            model=view.model,
        )
        rows.append(
            {
                "record_type": "view_score",
                "n_steps": operating.n_steps,
                "snr_db": format_snr(operating.snr_db),
                "prescribed_n_steps": prescribed.n_steps,
                "prescribed_snr_db": format_snr(prescribed.snr_db),
                "prescribed_macro_f1_mean": prescribed.macro_f1_mean,
                "prescribed_macro_f1_ci_low": prescribed.ci_low,
                "prescribed_macro_f1_ci_high": prescribed.ci_high,
                "fallback_applied": bool(fallback_applied),
                "fallback_reason": fallback_reason,
                "fallback_detail": fallback_detail,
                "view": view.view,
                "model": view.model,
                "compare_view": "",
                "macro_f1_mean": view.macro_f1_mean,
                "macro_f1_std": view.macro_f1_std,
                "macro_f1_ci_low": ci_low,
                "macro_f1_ci_high": ci_high,
                "ci_excludes_one": bool(
                    np.isfinite(ci_low)
                    and np.isfinite(ci_high)
                    and not _ci_includes_one(ci_low, ci_high)
                ),
                "delta_macro_f1": np.nan,
                "delta_macro_f1_cv": np.nan,
                "delta_ci_low": np.nan,
                "delta_ci_high": np.nan,
                "ci_excludes_zero": False,
            }
        )
    for diff in differences:
        rows.append(
            {
                "record_type": "f1_difference",
                "n_steps": operating.n_steps,
                "snr_db": format_snr(operating.snr_db),
                "prescribed_n_steps": prescribed.n_steps,
                "prescribed_snr_db": format_snr(prescribed.snr_db),
                "prescribed_macro_f1_mean": prescribed.macro_f1_mean,
                "prescribed_macro_f1_ci_low": prescribed.ci_low,
                "prescribed_macro_f1_ci_high": prescribed.ci_high,
                "fallback_applied": bool(fallback_applied),
                "fallback_reason": fallback_reason,
                "fallback_detail": fallback_detail,
                "view": diff["view"],
                "model": diff["model"],
                "compare_view": diff["compare_view"],
                "macro_f1_mean": np.nan,
                "macro_f1_std": np.nan,
                "macro_f1_ci_low": np.nan,
                "macro_f1_ci_high": np.nan,
                "ci_excludes_one": False,
                "delta_macro_f1": diff["delta_macro_f1"],
                "delta_macro_f1_cv": diff["delta_macro_f1_cv"],
                "delta_ci_low": diff["delta_ci_low"],
                "delta_ci_high": diff["delta_ci_high"],
                "ci_excludes_zero": bool(diff["ci_excludes_zero"]),
            }
        )
    return pd.DataFrame(rows)


def _acceptance_lines(
    operating: GridPoint,
    differences: Sequence[Mapping[str, Any]],
    binary: pd.DataFrame,
    n_pairs: int,
) -> list[str]:
    lines: list[str] = []
    if _acceptable(operating):
        lines.append(
            f"PASS full RF CV Macro-F1 {operating.macro_f1_mean:.4f} is inside "
            f"[{F1_FLOOR:.2f}, {F1_CEILING:.2f}] and the 95% CI "
            f"[{operating.ci_low:.4f}, {operating.ci_high:.4f}] excludes 1."
        )
    else:
        lines.append(
            f"FAIL full RF CV Macro-F1 {operating.macro_f1_mean:.4f} "
            f"CI [{operating.ci_low:.4f}, {operating.ci_high:.4f}] "
            "is outside the target band or the CI includes 1."
        )
    n_sig = sum(1 for row in differences if bool(row["ci_excludes_zero"]))
    if n_sig >= 2:
        lines.append(
            f"PASS {n_sig} random-forest view pairs have a Macro-F1 difference CI that excludes 0."
        )
    else:
        lines.append(
            f"FAIL only {n_sig} random-forest view pairs have a Macro-F1 difference CI that excludes 0."
        )
    present = sorted(binary["pair"].astype(str).unique()) if not binary.empty else []
    finite_ci = (
        binary["f1_ci_low"].notna().all() and binary["f1_ci_high"].notna().all()
        if not binary.empty
        else False
    )
    if len(present) == n_pairs and finite_ci:
        lines.append(f"PASS all {n_pairs} binary critical-pair tasks are present with F1 intervals.")
    else:
        lines.append(
            f"FAIL binary tasks: found {present}, expected {n_pairs} pairs with finite F1 intervals."
        )
    return lines


def _write_html_report(
    path: Path,
    *,
    difficulty: DifficultyConfig,
    audit: AuditConfig,
    curve: pd.DataFrame,
    headline: pd.DataFrame,
    binary: pd.DataFrame,
    proper: pd.DataFrame | None,
    figure_paths: Sequence[Path],
    layer0_status: str,
    acceptance: Sequence[str],
    cache_root: Path,
) -> None:
    def _table(frame: pd.DataFrame | None, max_rows: int = 40) -> str:
        if frame is None or frame.empty:
            return "<p>No rows.</p>"
        shown = frame.head(max_rows)
        return shown.to_html(index=False, float_format=lambda value: f"{value:.4f}")

    images = []
    for figure in figure_paths:
        if figure.suffix.lower() != ".png" or not figure.exists():
            continue
        encoded = base64.b64encode(figure.read_bytes()).decode("ascii")
        images.append(
            f'<figure><img alt="{figure.name}" src="data:image/png;base64,{encoded}"/>'
            f"<figcaption>{figure.name}</figcaption></figure>"
        )
    full_rf = curve[(curve["view"] == "full") & (curve["model"] == "random_forest")]
    proper_html = "<p>Not computed. " + layer0_status + "</p>"
    if proper is not None:
        proper_html = _table(proper, max_rows=20)
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>B3 difficulty protocol</title>
<style>
body{{font-family:IBM Plex Sans,Segoe UI,sans-serif;max-width:1100px;margin:1.5rem auto;padding:0 1rem;background:#f7f5f1;color:#1c1a16}}
table{{border-collapse:collapse;background:#fff;font-size:12px;margin:0.6rem 0 1.2rem}}
th,td{{border:1px solid #ddd6c8;padding:0.25rem 0.4rem;text-align:right}}
th{{background:#efeae2;text-align:center}}
img{{width:100%;margin:0.4rem 0;border:1px solid #ddd6c8;border-radius:8px}}
code{{background:#fff;padding:0.05rem 0.25rem}}
</style></head><body>
<h1>B3 hard frontiers — non-saturated classification</h1>
<p>Clean reference panel N={difficulty.clean_n_steps}, R={audit.n_realizations},
seed={audit.seed}, folds={audit.n_folds}, brownian_mode={audit.brownian_mode}.
Difficulty grid N={list(difficulty.n_grid)}, SNR={ [format_snr(s) for s in difficulty.snr_db] },
R={difficulty.n_realizations}, bootstrap={difficulty.n_bootstrap} (seed {difficulty.bootstrap_seed}).
Scaler is fit inside each training fold. Cache: <code>{cache_root}</code>.</p>
<p>{layer0_status}</p>
<h2>Acceptance</h2>
<ul>{''.join(f'<li>{line}</li>' for line in acceptance)}</ul>
<h2>Full random-forest curve</h2>
{_table(full_rf, max_rows=20)}
<h2>Headline operating point</h2>
{_table(headline, max_rows=30)}
<h2>Binary critical pairs (headline)</h2>
{_table(binary, max_rows=60)}
<h2>Clean-panel proper scores</h2>
{proper_html}
<h2>Figures</h2>
{''.join(images)}
</body></html>
"""
    atomic_write_text(path, html)


def _publish_bundle(report_dir: Path, bundle_dir: Path) -> list[Path]:
    written: list[Path] = []
    tables_src = report_dir / "data" / "tables"
    figures_src = report_dir / "figures"
    tables_dst = bundle_dir / "tables"
    figures_dst = bundle_dir / "figures"
    tables_dst.mkdir(parents=True, exist_ok=True)
    figures_dst.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in _TABLE_BUNDLE_NAMES.items():
        src = tables_src / src_name
        if not src.exists():
            continue
        dst = tables_dst / dst_name
        dst.write_bytes(src.read_bytes())
        written.append(dst)
    for src_stem, dst_stem in _FIGURE_BUNDLE_STEMS.items():
        # Paper bundle tracks PNG only (PDF/SVG stay in report dirs / gitignored).
        src = figures_src / f"{src_stem}.png"
        if not src.exists():
            continue
        dst = figures_dst / f"{dst_stem}.png"
        dst.write_bytes(src.read_bytes())
        written.append(dst)
    return written


def refresh_paper_bundle_manifest(bundle_dir: Path, repo_root: Path | None = None) -> None:
    """Rewrite MANIFEST.md and bundle_manifest.json from the files on disk."""
    bundle_dir = Path(bundle_dir)
    root = Path(repo_root) if repo_root is not None else _REPO_ROOT
    tables = sorted((bundle_dir / "tables").glob("*.csv"))
    figure_files = sorted(
        p
        for p in (bundle_dir / "figures").iterdir()
        if p.is_file() and p.suffix.lower() == ".png"
    ) if (bundle_dir / "figures").exists() else []
    stems: dict[str, set[str]] = {}
    for path in figure_files:
        stems.setdefault(path.stem, set()).add(path.suffix.lower().lstrip("."))
    note_dir = bundle_dir / "notes"
    notes = sorted(note_dir.glob("*.md")) if note_dir.exists() else []

    def _git(args: list[str]) -> str:
        try:
            return subprocess.check_output(args, cwd=root, text=True).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    commit = _git(["git", "rev-parse", "HEAD"])
    branch = _git(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _rows(path: Path) -> int:
        with path.open(encoding="utf-8") as fh:
            return max(0, sum(1 for _ in fh) - 1)

    table_lines = [
        f"- `{path.name}` ({path.stat().st_size} B, ~{_rows(path)} rows)" for path in tables
    ]
    stem_lines = [
        f"- `{stem}` (.png)" for stem in sorted(stems)
    ]
    note_lines = [f"- `{path.name}` ({path.stat().st_size} B)" for path in notes]
    markdown = "\n".join(
        [
            "# Paper bundle MANIFEST",
            "",
            f"- Generated (UTC): `{stamp}`",
            f"- Git commit: `{commit}`",
            f"- Git branch: `{branch}`",
            "",
            "## Counts",
            "",
            f"- Figures: **{len(stems)}** stems / **{len(figure_files)}** files",
            f"- Tables: **{len(tables)}** CSV",
            f"- Notes: **{len(notes)}**",
            "",
            "## Tables",
            "",
            *table_lines,
            "",
            "## Figure stems",
            "",
            *stem_lines,
            "",
            "## Notes",
            "",
            *note_lines,
            "",
        ]
    )
    atomic_write_text(bundle_dir / "MANIFEST.md", markdown)
    payload = {
        "generated_at_utc": stamp,
        "git_commit": commit,
        "git_branch": branch,
        "counts": {
            "figure_stems": len(stems),
            "figure_files": len(figure_files),
            "tables_csv": len(tables),
            "notes": len(notes),
        },
        "tables": [path.name for path in tables],
        "figure_stems": sorted(stems),
        "notes": [path.name for path in notes],
    }
    atomic_write_json(bundle_dir / "bundle_manifest.json", payload)


def run_difficulty_protocol(
    audit: AuditConfig,
    *,
    config_path: Path,
    output_dir: Path,
    n_workers: int = 1,
    lang: str = "en",
    force: bool = False,
    bundle_dir: Path | None = None,
    publish: bool = True,
) -> DifficultyResult:
    """Run layer 0 (if inputs exist) and the N × SNR difficulty sweep.

    Parameters
    ----------
    lang
        Retained for CLI compatibility. Figure text is English.
    publish
        Copy new tables and figures into the paper bundle and refresh the manifest.
    """
    del lang  # figures are English by protocol
    _limit_blas_threads()
    difficulty = DifficultyConfig.from_yaml(config_path)
    output_dir = Path(output_dir)
    tables_dir = output_dir / "data" / "tables"
    figures_dir = output_dir / "figures"
    difficulty_dir = output_dir / "data" / "difficulty"
    cache_root = output_dir / "ensemble_cache_difficulty"
    for path in (tables_dir, figures_dir, difficulty_dir, cache_root):
        path.mkdir(parents=True, exist_ok=True)

    logger.info(
        "B3 difficulty grid N=%s SNR=%s R=%d bootstrap=%d workers=%d cache=%s",
        list(difficulty.n_grid),
        [format_snr(s) for s in difficulty.snr_db],
        difficulty.n_realizations,
        difficulty.n_bootstrap,
        n_workers,
        cache_root,
    )
    proper, layer0_status = try_layer0(output_dir, audit, difficulty)
    if proper is not None:
        save_table_csv(tables_dir / "proper_scores_n1500_clean.csv", proper)

    generator_config = load_generator_suite_config()
    descriptor_config = DescriptorConfig()
    jobs = [(int(n_steps), snr) for n_steps in difficulty.n_grid for snr in difficulty.snr_db]
    workers = max(1, int(n_workers))
    if workers == 1:
        raw_rows: list[list[dict[str, Any]]] = []
        for n_steps, snr in jobs:
            raw_rows.append(
                _difficulty_cell_job(
                    n_steps,
                    snr,
                    audit=audit,
                    n_realizations=difficulty.n_realizations,
                    n_bootstrap=difficulty.n_bootstrap,
                    bootstrap_seed=difficulty.bootstrap_seed,
                    cache_root=str(cache_root),
                    difficulty_dir=str(difficulty_dir),
                    force=force,
                    generator_config=generator_config,
                    descriptor_config=descriptor_config,
                )
            )
    else:
        from joblib import Parallel, delayed

        raw_rows = Parallel(n_jobs=workers, prefer="processes")(
            delayed(_difficulty_cell_job)(
                n_steps,
                snr,
                audit=audit,
                n_realizations=difficulty.n_realizations,
                n_bootstrap=difficulty.n_bootstrap,
                bootstrap_seed=difficulty.bootstrap_seed,
                cache_root=str(cache_root),
                difficulty_dir=str(difficulty_dir),
                force=force,
                generator_config=generator_config,
                descriptor_config=descriptor_config,
            )
            for n_steps, snr in jobs
        )
    curve = _sort_curve(pd.DataFrame([row for block in raw_rows for row in block]), difficulty)
    expected = {
        (int(n_steps), format_snr(snr), view, model)
        for n_steps in difficulty.n_grid
        for snr in difficulty.snr_db
        for view in VIEW_ORDER
        for model in MODEL_ORDER
    }
    present = {
        (int(row.n_steps), str(row.snr_db), str(row.view), str(row.model))
        for row in curve.itertuples(index=False)
    }
    missing = sorted(expected - present)
    if missing:
        raise RuntimeError(
            f"Difficulty curve is missing {len(missing)} view/model cells; first={missing[:5]}"
        )
    save_table_csv(tables_dir / "difficulty_curve.csv", curve)

    points = _full_rf_points(curve)
    operating, fallback, reason, detail = resolve_headline(
        points,
        prescribed_n=difficulty.headline_n_steps,
        prescribed_snr=difficulty.headline_snr_db,
    )
    prescribed = next(point for point in points if point.key == (difficulty.headline_n_steps, format_snr(difficulty.headline_snr_db)))
    logger.info("Headline %s (%s)", detail, reason)

    features_path, _metrics_path = _cell_checkpoint_paths(
        difficulty_dir, operating.n_steps, operating.snr_db
    )
    x_head, y_head = _load_checkpoint_frame(features_path)
    cell_audit = replace(
        audit,
        n_steps=operating.n_steps,
        n_realizations=difficulty.n_realizations,
    )
    # The checkpoint matrix is already health-filtered. A second pass is
    # idempotent and keeps the same columns the curve was scored on.
    # Bootstrap intervals for the curve were saved with the cell; here we only
    # need out-of-fold labels for paired differences and the binary tasks.
    health = diagnose_and_clean(x_head, cell_audit)
    views = evaluate_views(health.x_clean, y_head, cell_audit, rf_n_jobs=1)
    x_head = health.x_clean
    differences = pairwise_f1_differences(
        views,
        n_steps=operating.n_steps,
        snr_db=operating.snr_db,
        n_bootstrap=difficulty.n_bootstrap,
        bootstrap_seed=difficulty.bootstrap_seed,
    )
    binary = evaluate_binary_pairs(
        x_head,
        y_head,
        cell_audit,
        n_steps=operating.n_steps,
        snr_db=operating.snr_db,
        n_bootstrap=difficulty.n_bootstrap,
        bootstrap_seed=difficulty.bootstrap_seed,
        rf_n_jobs=1,
    )
    save_table_csv(tables_dir / "critical_pairs_binary_hard.csv", binary)
    headline = _build_headline_frame(
        operating,
        prescribed=prescribed,
        fallback_applied=fallback,
        fallback_reason=reason,
        fallback_detail=detail,
        views=views,
        differences=differences,
        curve=curve,
    )
    save_table_csv(tables_dir / "headline_operating_point.csv", headline)

    destination = (
        Path(bundle_dir) if bundle_dir is not None else _REPO_ROOT / "outputs" / "paper_bundle"
    )
    ablation_path = write_headline_ablation_artifacts(
        x=x_head,
        y=y_head,
        views=views,
        audit=cell_audit,
        operating=operating,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        bundle_dir=destination,
    )

    figure_paths = plot_difficulty_curve(
        curve,
        operating,
        fallback_reason=reason,
        output_dir=figures_dir,
        dpi=audit.figure_dpi,
    )
    figure_paths.extend(
        plot_binary_critical_pairs(
            binary,
            operating,
            output_dir=figures_dir,
            dpi=audit.figure_dpi,
        )
    )
    figure_paths.append(ablation_path)
    acceptance = _acceptance_lines(
        operating, differences, binary, n_pairs=len(audit.critical_pairs)
    )
    for line in acceptance:
        logger.info("%s", line)
    _write_html_report(
        output_dir / "difficulty_report.html",
        difficulty=difficulty,
        audit=audit,
        curve=curve,
        headline=headline,
        binary=binary,
        proper=proper,
        figure_paths=figure_paths,
        layer0_status=layer0_status,
        acceptance=acceptance,
        cache_root=cache_root,
    )
    atomic_write_json(
        output_dir / "data" / "difficulty_summary.json",
        {
            "fallback_applied": fallback,
            "fallback_reason": reason,
            "fallback_detail": detail,
            "n_steps": operating.n_steps,
            "snr_db": format_snr(operating.snr_db),
            "full_rf_macro_f1_mean": operating.macro_f1_mean,
            "full_rf_macro_f1_std": operating.macro_f1_std,
            "full_rf_macro_f1_ci_low": operating.ci_low,
            "full_rf_macro_f1_ci_high": operating.ci_high,
            "acceptance": list(acceptance),
            "layer0_status": layer0_status,
            "cache_root": str(cache_root),
        },
    )
    if publish:
        published = _publish_bundle(output_dir, destination)
        refresh_paper_bundle_manifest(destination, _REPO_ROOT)
        logger.info("Published %d bundle files to %s", len(published), destination)

    return DifficultyResult(
        curve=curve,
        binary=binary,
        proper_scores=proper,
        headline=headline,
        operating_point=operating,
        fallback_applied=fallback,
        fallback_reason=reason,
        figure_paths=figure_paths,
        layer0_status=layer0_status,
        acceptance=acceptance,
    )
