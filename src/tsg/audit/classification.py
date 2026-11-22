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

    def to_dict(self) -> dict[str, Any]:
        return {
            "view": self.view,
            "model": self.model,
            "accuracy_mean": self.accuracy_mean,
            "accuracy_std": self.accuracy_std,
            "macro_f1_mean": self.macro_f1_mean,
            "macro_f1_std": self.macro_f1_std,
            "f1_by_class": self.f1_by_class,
            "confusion": self.confusion.to_dict(),
            "confusion_normalized": self.confusion_normalized.to_dict(),
        }


@dataclass
class ClassificationResult:
    views: list[ViewMetrics] = field(default_factory=list)
    permutation_importance: pd.DataFrame | None = None
    rf_gini_importance: pd.DataFrame | None = None
    logistic_l2_importance: pd.DataFrame | None = None
    critical_pair_confusion_rate: dict[str, float] = field(default_factory=dict)

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
        }


def _make_models(seed: int) -> dict[str, Pipeline]:
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
                        n_jobs=-1,
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
    skf = StratifiedKFold(
        n_splits=min(config.n_folds, int(y.value_counts().min())),
        shuffle=True,
        random_state=config.seed,
    )
    accs: list[float] = []
    f1s: list[float] = []
    y_true_all: list[str] = []
    y_pred_all: list[str] = []
    xa = x.to_numpy(dtype=float)
    ya = y.to_numpy()

    for train_idx, test_idx in skf.split(xa, ya):
        model.fit(xa[train_idx], ya[train_idx])
        pred = model.predict(xa[test_idx])
        accs.append(float(accuracy_score(ya[test_idx], pred)))
        f1s.append(float(f1_score(ya[test_idx], pred, average="macro", labels=labels, zero_division=0)))
        y_true_all.extend(ya[test_idx].tolist())
        y_pred_all.extend(pred.tolist())

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
    return results
