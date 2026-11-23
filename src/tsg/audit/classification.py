"""Supervised discriminability evaluation and family ablation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tsg.audit.config import AuditConfig
from tsg.audit.features import family_feature_mask


@dataclass
class ViewMetrics:
    view: str
    model: str
    accuracy_mean: float
    accuracy_std: float
    macro_f1_mean: float
    macro_f1_std: float
    f1_by_class: dict[str, float]
    confusion: pd.DataFrame
    confusion_normalized: pd.DataFrame
    y_true: list[str] = field(default_factory=list)
    y_pred: list[str] = field(default_factory=list)
    y_proba: np.ndarray | None = None
    fold_id: np.ndarray | None = None
    class_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "view": self.view,
            "model": self.model,
            "accuracy_mean": self.accuracy_mean,
            "accuracy_std": self.accuracy_std,
            "macro_f1_mean": self.macro_f1_mean,
            "macro_f1_std": self.macro_f1_std,
            "f1_by_class": self.f1_by_class,
            # orient="index" → outer key is true label (row); each group sums to 1.0
            "confusion": self.confusion.to_dict(orient="index"),
            "confusion_normalized": self.confusion_normalized.to_dict(orient="index"),
        }

    def oof_bundle(self) -> dict[str, Any]:
        """Out-of-fold arrays for NPZ persistence."""
        return {
            "y_true": list(self.y_true),
            "y_pred": list(self.y_pred),
            "y_proba": self.y_proba,
            "fold_id": (
                np.asarray(self.fold_id, dtype=int)
                if self.fold_id is not None
                else np.zeros(len(self.y_true), dtype=int)
            ),
            "class_names": list(self.class_names),
        }


@dataclass
class ClassificationResult:
    views: list[ViewMetrics] = field(default_factory=list)
    permutation_importance: pd.DataFrame | None = None
    rf_gini_importance: pd.DataFrame | None = None
    logistic_l2_importance: pd.DataFrame | None = None
    critical_pair_confusion_rate: dict[str, float] = field(default_factory=dict)
    # pair → view → model → metrics (all views × both models).
    critical_pair_details: dict[str, dict[str, dict[str, dict[str, float]]]] = field(
        default_factory=dict
    )

    def full_rf(self) -> ViewMetrics | None:
        for v in self.views:
            if v.view == "full" and v.model == "random_forest":
                return v
        return None

    def full_linear(self) -> ViewMetrics | None:
        for v in self.views:
            if v.view == "full" and v.model == "logistic":
                return v
        return None

    def to_dict(self) -> dict[str, Any]:
        def _records(frame: pd.DataFrame | None) -> list[dict[str, Any]] | None:
            if frame is None:
                return None
            return frame.to_dict(orient="records")

        return {
            "views": [v.to_dict() for v in self.views],
            "permutation_importance": _records(self.permutation_importance),
            "rf_gini_importance": _records(self.rf_gini_importance),
            "logistic_l2_importance": _records(self.logistic_l2_importance),
            "critical_pair_confusion_rate": self.critical_pair_confusion_rate,
            "critical_pair_details": self.critical_pair_details,
        }

    def oof_bundles(self) -> dict[str, dict[str, Any]]:
        """Map ``view__model`` → OOF arrays for disk persistence."""
        return {f"{v.view}__{v.model}": v.oof_bundle() for v in self.views}


def _make_models(seed: int, *, rf_n_jobs: int = -1) -> dict[str, Pipeline]:
    return {
        "logistic": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=4000,
                        solver="lbfgs",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=seed,
                        n_jobs=rf_n_jobs,
                        class_weight="balanced_subsample",
                    ),
                ),
            ]
        ),
    }


def _evaluate_view(
    x: pd.DataFrame,
    y: pd.Series,
    *,
    view: str,
    model_name: str,
    model: Pipeline,
    config: AuditConfig,
) -> ViewMetrics:
    labels = list(config.regimes)
    n_splits = min(config.n_folds, int(y.value_counts().min()))
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=config.seed,
    )
    accs: list[float] = []
    f1s: list[float] = []
    y_true_all: list[str] = []
    y_pred_all: list[str] = []
    xa = x.to_numpy(dtype=float)
    ya = y.to_numpy()
    # OOF stored in fold-visit order; fold_id stays aligned with y_true/y_pred/y_proba.
    y_true_ordered: list[str] = []
    y_pred_ordered: list[str] = []
    fold_ordered: list[int] = []
    proba_rows: list[np.ndarray] = []

    label_to_col = {lab: i for i, lab in enumerate(labels)}
    for fold_i, (train_idx, test_idx) in enumerate(skf.split(xa, ya)):
        fitted = clone(model)
        fitted.fit(xa[train_idx], ya[train_idx])
        pred = fitted.predict(xa[test_idx])
        accs.append(float(accuracy_score(ya[test_idx], pred)))
        f1s.append(
            float(
                f1_score(
                    ya[test_idx],
                    pred,
                    average="macro",
                    labels=labels,
                    zero_division=0,
                )
            )
        )
        y_true_all.extend(ya[test_idx].tolist())
        y_pred_all.extend(pred.tolist())
        y_true_ordered.extend(str(v) for v in ya[test_idx].tolist())
        y_pred_ordered.extend(str(v) for v in pred.tolist())
        fold_ordered.extend([fold_i] * len(test_idx))
        if hasattr(fitted, "predict_proba"):
            proba = fitted.predict_proba(xa[test_idx])
            # Align columns to ``labels`` order (sklearn uses sorted unique train labels).
            class_order = list(fitted.classes_)
            aligned = np.full((len(test_idx), len(labels)), np.nan, dtype=float)
            for src_j, cls in enumerate(class_order):
                dst_j = label_to_col.get(str(cls))
                if dst_j is not None:
                    aligned[:, dst_j] = proba[:, src_j]
            proba_rows.append(aligned)

    cm = confusion_matrix(y_true_all, y_pred_all, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    cm_norm_df = pd.DataFrame(cm_norm, index=labels, columns=labels)

    f1_per = f1_score(
        y_true_all,
        y_pred_all,
        average=None,
        labels=labels,
        zero_division=0,
    )
    f1_by = {lab: float(v) for lab, v in zip(labels, f1_per)}
    y_proba_oof = (
        np.vstack(proba_rows) if proba_rows else np.zeros((0, len(labels)), dtype=float)
    )

    return ViewMetrics(
        view=view,
        model=model_name,
        accuracy_mean=float(np.mean(accs)),
        accuracy_std=float(np.std(accs)),
        macro_f1_mean=float(np.mean(f1s)),
        macro_f1_std=float(np.std(f1s)),
        f1_by_class=f1_by,
        confusion=cm_df,
        confusion_normalized=cm_norm_df,
        y_true=y_true_ordered,
        y_pred=y_pred_ordered,
        y_proba=y_proba_oof,
        fold_id=np.asarray(fold_ordered, dtype=int),
        class_names=list(labels),
    )


def _critical_confusion_rates(
    cm_norm: pd.DataFrame,
    critical_pairs: tuple[tuple[str, str], ...],
) -> dict[str, float]:
    rates: dict[str, float] = {}
    for a, b in critical_pairs:
        if a not in cm_norm.index or b not in cm_norm.columns:
            continue
        # Symmetric confusion: mean of both directions.
        rate = 0.5 * (float(cm_norm.loc[a, b]) + float(cm_norm.loc[b, a]))
        rates[f"{a}__{b}"] = rate
    return rates


def _mean_prob_margin(
    view: ViewMetrics,
    a: str,
    b: str,
) -> float:
    """Mean |P(a) − P(b)| over OOF samples whose true label is in ``{a, b}``."""
    if view.y_proba is None or view.y_proba.size == 0 or not view.y_true:
        return float("nan")
    class_names = list(view.class_names) if view.class_names else []
    if a not in class_names or b not in class_names:
        return float("nan")
    idx_a = class_names.index(a)
    idx_b = class_names.index(b)
    mask = np.asarray([(yt == a or yt == b) for yt in view.y_true], dtype=bool)
    if not np.any(mask):
        return float("nan")
    proba = np.asarray(view.y_proba, dtype=float)
    margins = np.abs(proba[mask, idx_a] - proba[mask, idx_b])
    return float(np.nanmean(margins))


def _critical_pair_details_for_view(
    view: ViewMetrics,
    critical_pairs: tuple[tuple[str, str], ...],
) -> dict[str, dict[str, float]]:
    """Per-pair continuous metrics for a single ``(view, model)`` block."""
    cm_norm = view.confusion_normalized
    out: dict[str, dict[str, float]] = {}
    for a, b in critical_pairs:
        if a not in cm_norm.index or b not in cm_norm.columns:
            continue
        key = f"{a}__{b}"
        conf_ab = float(cm_norm.loc[a, b])
        conf_ba = float(cm_norm.loc[b, a])
        f1_a = float(view.f1_by_class.get(a, float("nan")))
        f1_b = float(view.f1_by_class.get(b, float("nan")))
        out[key] = {
            "confusion_rate": 0.5 * (conf_ab + conf_ba),
            "confusion_a_to_b": conf_ab,
            "confusion_b_to_a": conf_ba,
            "pair_macro_f1": 0.5 * (f1_a + f1_b),
            "mean_prob_margin": _mean_prob_margin(view, a, b),
        }
    return out


def compute_critical_pair_details(
    views: Sequence[ViewMetrics],
    critical_pairs: tuple[tuple[str, str], ...],
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Aggregate critical-pair metrics over all views and models.

    Returns
    -------
    dict
        Nested map ``pair → view → model → metrics``.
    """
    nested: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for view in views:
        per_pair = _critical_pair_details_for_view(view, critical_pairs)
        for pair_key, metrics in per_pair.items():
            nested.setdefault(pair_key, {}).setdefault(view.view, {})[view.model] = (
                metrics
            )
    return nested


def _importance_dataframe(
    feature_names: Sequence[str],
    fold_importances: np.ndarray,
) -> pd.DataFrame:
    """Aggregate per-fold importances into a ranked frame."""
    frame = pd.DataFrame(
        {
            "feature": list(feature_names),
            "importance_mean": fold_importances.mean(axis=0),
            "importance_std": fold_importances.std(axis=0),
        }
    )
    frame["family"] = frame["feature"].str.split("__").str[0]
    return frame.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def _compute_full_view_importances(
    x: pd.DataFrame,
    y: pd.Series,
    config: AuditConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """RF Gini impurity and logistic L2-coef norms on the full feature view.

    Permutation importance collapses to ~0 under Macro-F1=1 with collinear
    descriptors; these intrinsic scores remain informative.
    """
    n_folds = min(config.n_folds, int(y.value_counts().min()))
    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=config.seed,
    )
    models = _make_models(config.seed)
    xa = x.to_numpy(dtype=float)
    ya = y.to_numpy()
    n_features = xa.shape[1]
    rf_folds = np.zeros((n_folds, n_features), dtype=float)
    lr_folds = np.zeros((n_folds, n_features), dtype=float)

    for fold_i, (train_idx, _test_idx) in enumerate(skf.split(xa, ya)):
        rf = clone(models["random_forest"])
        rf.fit(xa[train_idx], ya[train_idx])
        rf_folds[fold_i] = rf.named_steps["clf"].feature_importances_

        lr = clone(models["logistic"])
        lr.fit(xa[train_idx], ya[train_idx])
        coef = lr.named_steps["clf"].coef_
        lr_folds[fold_i] = np.linalg.norm(coef, axis=0)

    return (
        _importance_dataframe(x.columns, rf_folds),
        _importance_dataframe(x.columns, lr_folds),
    )


def evaluate_classification(
    x: pd.DataFrame,
    y: pd.Series,
    config: AuditConfig,
) -> ClassificationResult:
    """Run linear/nonlinear classifiers on family views + full matrix."""
    models = _make_models(config.seed)
    views_spec: list[tuple[str, np.ndarray]] = [
        ("visibility", family_feature_mask(x.columns, "visibility")),
        ("transition", family_feature_mask(x.columns, "transition")),
        ("proximity", family_feature_mask(x.columns, "proximity")),
        ("similarity", family_feature_mask(x.columns, "similarity")),
        ("full", np.ones(len(x.columns), dtype=bool)),
    ]

    results = ClassificationResult()
    for view_name, mask in views_spec:
        if not mask.any():
            continue
        x_view = x.loc[:, mask]
        for model_name, model in models.items():
            metrics = _evaluate_view(
                x_view,
                y,
                view=view_name,
                model_name=model_name,
                model=clone(model),
                config=config,
            )
            results.views.append(metrics)

    rf_gini, logistic_l2 = _compute_full_view_importances(x, y, config)
    results.rf_gini_importance = rf_gini
    results.logistic_l2_importance = logistic_l2
    # Kept for backward-compatible JSON schema; no longer computed (always None).
    results.permutation_importance = None

    full_rf = results.full_rf()
    if full_rf is not None:
        results.critical_pair_confusion_rate = _critical_confusion_rates(
            full_rf.confusion_normalized,
            config.critical_pairs,
        )
    results.critical_pair_details = compute_critical_pair_details(
        results.views,
        config.critical_pairs,
    )
    return results
