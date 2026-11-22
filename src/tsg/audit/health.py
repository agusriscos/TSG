"""Descriptor health diagnostics: NaN/Inf/near-zero variance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tsg.audit.config import AuditConfig


@dataclass
class HealthReport:
    n_features: int
    n_samples: int
    nan_features: list[str]
    inf_features: list[str]
    low_variance_features: list[str]
    dropped_features: list[str]
    x_clean: pd.DataFrame

    def to_dict(self) -> dict:
        return {
            "n_features": self.n_features,
            "n_samples": self.n_samples,
            "nan_features": self.nan_features,
            "inf_features": self.inf_features,
            "low_variance_features": self.low_variance_features,
            "dropped_features": self.dropped_features,
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
    )
