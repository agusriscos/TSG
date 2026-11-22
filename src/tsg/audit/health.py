"""Descriptor health diagnostics: NaN/Inf/near-zero variance."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tsg.audit.config import AuditConfig

# Descriptor suffixes that are constant by graph construction (not numeric failure).
_STRUCTURAL_CONSTANT_SUFFIXES: frozenset[str] = frozenset(
    {
        "i_dir",  # 0 on undirected graphs
        "self_loop_ratio",  # 0 on undirected graphs
        "comp",  # 1 on connected-by-construction graphs (NVG/HVG/STN/OPN)
        "lcc",  # 1 on connected-by-construction graphs
    }
)


def _descriptor_suffix(feature: str) -> str:
    """Return the trailing descriptor id from ``family__method__descriptor``."""
    parts = feature.split("__")
    return parts[-1] if parts else feature


def classify_dropped_feature(
    feature: str,
    *,
    is_nan: bool,
    is_inf: bool,
    is_low_variance: bool,
) -> tuple[str, str]:
    """Return ``(category, reason)`` for a dropped feature.

    Categories
    ----------
    numerical_nan
        Undefined numeric result (e.g. assortativity with zero degree variance).
    numerical_inf
        Non-finite Inf values.
    structural_constant
        Constant by graph construction (directional zeros, connected LCC, …).
    low_variance
        Empirically near-zero variance without a known structural cause.
    """
    suffix = _descriptor_suffix(feature)
    if is_nan:
        if suffix == "assortativity_r":
            return (
                "numerical_nan",
                "assortativity undefined when degree variance is zero (regular/complete graph)",
            )
        return ("numerical_nan", "feature contains NaN across the ensemble")
    if is_inf:
        return ("numerical_inf", "feature contains Inf across the ensemble")
    if is_low_variance and suffix in _STRUCTURAL_CONSTANT_SUFFIXES:
        reasons = {
            "i_dir": "always 0 on undirected TS2G graphs",
            "self_loop_ratio": "always 0 on undirected TS2G graphs",
            "comp": "always 1 on graphs connected by construction",
            "lcc": "always 1 on graphs connected by construction",
        }
        return ("structural_constant", reasons.get(suffix, "structurally constant"))
    return ("low_variance", "near-zero empirical variance across realizations")


@dataclass
class HealthReport:
    n_features: int
    n_samples: int
    nan_features: list[str]
    inf_features: list[str]
    low_variance_features: list[str]
    dropped_features: list[str]
    x_clean: pd.DataFrame
    structural_constant_features: list[str] = field(default_factory=list)
    numerical_nan_features: list[str] = field(default_factory=list)
    numerical_inf_features: list[str] = field(default_factory=list)
    low_variance_non_structural_features: list[str] = field(default_factory=list)
    drop_reasons: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_features": self.n_features,
            "n_features_raw": self.n_features,
            "n_samples": self.n_samples,
            "nan_features": self.nan_features,
            "inf_features": self.inf_features,
            "low_variance_features": self.low_variance_features,
            "structural_constant_features": self.structural_constant_features,
            "numerical_nan_features": self.numerical_nan_features,
            "numerical_inf_features": self.numerical_inf_features,
            "low_variance_non_structural_features": (
                self.low_variance_non_structural_features
            ),
            "dropped_features": self.dropped_features,
            "drop_reasons": self.drop_reasons,
            "n_features_clean": int(self.x_clean.shape[1]),
        }


def diagnose_and_clean(x: pd.DataFrame, config: AuditConfig) -> HealthReport:
    """Drop features with NaN/Inf or near-zero variance across the ensemble."""
    arr = x.to_numpy(dtype=float)
    nan_mask = np.isnan(arr).any(axis=0)
    inf_mask = np.isinf(arr).any(axis=0)
    # Replace inf for variance calc after marking.
    safe = np.nan_to_num(arr, nan=np.nan, posinf=np.nan, neginf=np.nan)
    with np.errstate(all="ignore"):
        variances = np.nanvar(safe, axis=0)
    low_var = np.isnan(variances) | (variances <= config.variance_eps)

    nan_features = x.columns[nan_mask].tolist()
    inf_features = x.columns[inf_mask].tolist()
    low_variance_features = x.columns[low_var].tolist()
    drop_mask = nan_mask | inf_mask | low_var
    dropped = x.columns[drop_mask].tolist()

    structural_constant_features: list[str] = []
    numerical_nan_features: list[str] = []
    numerical_inf_features: list[str] = []
    low_variance_non_structural_features: list[str] = []
    drop_reasons: dict[str, str] = {}
    for idx, name in enumerate(x.columns):
        if not drop_mask[idx]:
            continue
        category, reason = classify_dropped_feature(
            name,
            is_nan=bool(nan_mask[idx]),
            is_inf=bool(inf_mask[idx]),
            is_low_variance=bool(low_var[idx]),
        )
        drop_reasons[name] = f"{category}: {reason}"
        if category == "structural_constant":
            structural_constant_features.append(name)
        elif category == "numerical_nan":
            numerical_nan_features.append(name)
        elif category == "numerical_inf":
            numerical_inf_features.append(name)
        elif category == "low_variance":
            low_variance_non_structural_features.append(name)

    x_clean = x.loc[:, ~drop_mask].copy()
    # Remaining NaNs (if any row-wise) → column median.
    if x_clean.isna().any().any():
        x_clean = x_clean.fillna(x_clean.median(numeric_only=True))

    return HealthReport(
        n_features=int(x.shape[1]),
        n_samples=int(x.shape[0]),
        nan_features=nan_features,
        inf_features=inf_features,
        low_variance_features=low_variance_features,
        dropped_features=dropped,
        x_clean=x_clean,
        structural_constant_features=structural_constant_features,
        numerical_nan_features=numerical_nan_features,
        numerical_inf_features=numerical_inf_features,
        low_variance_non_structural_features=low_variance_non_structural_features,
        drop_reasons=drop_reasons,
    )
