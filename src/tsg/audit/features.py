"""Vectorize descriptor matrices D into supervised feature tables."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from tsg.audit.config import AuditConfig
from tsg.descriptors.base import DESCRIPTOR_IDS
from tsg.graphs.registry import METHOD_IDS

METHOD_TO_FAMILY: dict[str, str] = {
    "NVG": "visibility",
    "HVG": "visibility",
    "STN": "transition",
    "OPN": "transition",
    "RN": "proximity",
    "kNN": "proximity",
    "Correlation": "similarity",
    "Segment": "similarity",
}


def feature_name(method: str, descriptor: str) -> str:
    """Canonical column name ``{family}__{method}__{descriptor}``."""
    family = METHOD_TO_FAMILY.get(method, "other")
    return f"{family}__{method}__{descriptor}"


def all_feature_names(
    methods: Iterable[str] = METHOD_IDS,
    descriptors: Iterable[str] = DESCRIPTOR_IDS,
) -> list[str]:
    return [feature_name(m, d) for m in methods for d in descriptors]


def matrix_to_feature_row(
    matrix: pd.DataFrame,
    *,
    methods: Iterable[str] | None = None,
    descriptors: Iterable[str] | None = None,
) -> dict[str, float]:
    """Flatten one ``p × M`` descriptor matrix into a named feature dict."""
    methods = list(methods) if methods is not None else list(matrix.columns)
    descriptors = list(descriptors) if descriptors is not None else list(matrix.index)
    row: dict[str, float] = {}
    for method in methods:
        for desc in descriptors:
            row[feature_name(method, desc)] = float(matrix.loc[desc, method])
    return row


def build_feature_table(
    records: list[dict],
    config: AuditConfig,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build ``X`` (n_samples × p·M) and label vector ``y`` from ensemble records.

    Each record must contain ``dataset_id``, ``realization``, ``matrix``.
    """
    rows: list[dict[str, float]] = []
    labels: list[str] = []
    meta_rows: list[dict[str, object]] = []
    for rec in records:
        flat = matrix_to_feature_row(rec["matrix"], methods=config.methods)
        rows.append(flat)
        labels.append(str(rec["dataset_id"]))
        meta_rows.append(
            {
                "dataset_id": rec["dataset_id"],
                "realization": rec["realization"],
                "seed": rec.get("seed"),
            }
        )
    x = pd.DataFrame(rows, columns=all_feature_names(config.methods, DESCRIPTOR_IDS))
    y = pd.Series(labels, name="regime")
    meta = pd.DataFrame(meta_rows)
    x.attrs["meta"] = meta
    return x, y


def family_feature_mask(columns: Iterable[str], family: str) -> np.ndarray:
    """Boolean mask selecting columns that belong to a TS2G family prefix."""
    prefix = f"{family}__"
    return np.asarray([c.startswith(prefix) for c in columns], dtype=bool)
