"""Unsupervised geometry: PCA, silhouette and centroid distances."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, silhouette_samples, silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from tsg.audit.config import AuditConfig
from tsg.audit.features import family_feature_mask


@dataclass
class GeometryResult:
    explained_variance_ratio: list[float]
    cumulative_variance: list[float]
    loadings: pd.DataFrame
    embedding: pd.DataFrame
    silhouette_global: float
    silhouette_by_regime: dict[str, float]
    centroid_euclidean: pd.DataFrame
    centroid_cosine: pd.DataFrame
    critical_pair_distances: dict[str, dict[str, float]]
    # view → pair → metrics (continuous separability on that feature slice).
    critical_pair_geometry_by_view: dict[str, dict[str, dict[str, float]]] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "explained_variance_ratio": self.explained_variance_ratio,
            "cumulative_variance": self.cumulative_variance,
            "silhouette_global": self.silhouette_global,
            "silhouette_by_regime": self.silhouette_by_regime,
            "centroid_euclidean": self.centroid_euclidean.to_dict(),
            "centroid_cosine": self.centroid_cosine.to_dict(),
            "critical_pair_distances": self.critical_pair_distances,
            "critical_pair_geometry_by_view": self.critical_pair_geometry_by_view,
            "top_loadings_pc1_pc2": _top_loadings_records(self.loadings),
        }


def _top_loadings_records(loadings: pd.DataFrame) -> list[dict[str, Any]]:
    """Serialize top PC1/PC2 loadings with an explicit ``feature`` key."""
    frame = loadings.rename_axis("feature").reset_index()
    pc1 = frame["PC1"] if "PC1" in frame.columns else 0.0
    pc2 = frame["PC2"] if "PC2" in frame.columns else 0.0
    frame = frame.assign(score=np.sqrt(np.asarray(pc1) ** 2 + np.asarray(pc2) ** 2))
    return frame.sort_values("score", ascending=False).head(15).to_dict(orient="records")


def _pair_geometry_metrics(
    xs: np.ndarray,
    y: np.ndarray,
    a: str,
    b: str,
) -> dict[str, float]:
    """Continuous separability metrics for classes ``{a, b}`` in feature space ``xs``."""
    mask_a = y == a
    mask_b = y == b
    n_a = int(np.sum(mask_a))
    n_b = int(np.sum(mask_b))
    d = int(xs.shape[1]) if xs.ndim == 2 else 0
    nan_block = {
        "pair_silhouette": float("nan"),
        "centroid_euclidean_norm": float("nan"),
        "centroid_cosine": float("nan"),
        "fisher_ratio": float("nan"),
        "n_features": float(d),
    }
    if d == 0 or n_a < 1 or n_b < 1:
        return nan_block

    xa = xs[mask_a]
    xb = xs[mask_b]
    mu_a = xa.mean(axis=0)
    mu_b = xb.mean(axis=0)
    delta = mu_a - mu_b
    euc_norm = float(np.linalg.norm(delta) / np.sqrt(d))
    # Cosine distance between centroids (1 − cosine similarity).
    na = float(np.linalg.norm(mu_a))
    nb = float(np.linalg.norm(mu_b))
    if na < 1e-15 or nb < 1e-15:
        cos_dist = float("nan")
    else:
        cos_sim = float(np.dot(mu_a, mu_b) / (na * nb))
        cos_dist = float(1.0 - np.clip(cos_sim, -1.0, 1.0))

    # Fisher ratio: between-centroid energy vs mean per-dim within-class variance.
    def _mean_var_per_dim(block: np.ndarray) -> float:
        if block.shape[0] < 2:
            return 0.0
        # Sum of per-feature sample variances ≡ trace(Cov) (handles d=1 scalars).
        return float(np.sum(np.var(block, axis=0, ddof=1)))

    cov_a_trace = _mean_var_per_dim(xa)
    cov_b_trace = _mean_var_per_dim(xb)
    denom = (cov_a_trace / d) + (cov_b_trace / d)
    num = float(np.dot(delta, delta))
    if denom > 0.0:
        fisher = float(num / denom)
    elif num > 0.0:
        # Numerically zero within-class variance with separated centroids.
        fisher = float("inf")
    else:
        fisher = float("nan")

    pair_sil = float("nan")
    if n_a >= 2 and n_b >= 2:
        x_pair = np.vstack([xa, xb])
        y_pair = np.asarray([0] * n_a + [1] * n_b, dtype=int)
        try:
            pair_sil = float(silhouette_score(x_pair, y_pair, metric="euclidean"))
        except ValueError:
            pair_sil = float("nan")

    return {
        "pair_silhouette": pair_sil,
        "centroid_euclidean_norm": euc_norm,
        "centroid_cosine": cos_dist,
        "fisher_ratio": fisher,
        "n_features": float(d),
    }


def compute_critical_pair_geometry_by_view(
    x: pd.DataFrame,
    y: pd.Series,
    critical_pairs: tuple[tuple[str, str], ...],
) -> dict[str, dict[str, dict[str, float]]]:
    """Per-view critical-pair geometry on standardized feature slices.

    Returns
    -------
    dict
        Nested map ``view → pair → metrics``.
    """
    y_arr = y.to_numpy()
    views_spec: list[tuple[str, np.ndarray]] = [
        ("visibility", family_feature_mask(x.columns, "visibility")),
        ("transition", family_feature_mask(x.columns, "transition")),
        ("proximity", family_feature_mask(x.columns, "proximity")),
        ("similarity", family_feature_mask(x.columns, "similarity")),
        ("full", np.ones(len(x.columns), dtype=bool)),
    ]
    out: dict[str, dict[str, dict[str, float]]] = {}
    for view_name, mask in views_spec:
        if not np.any(mask):
            continue
        x_view = x.loc[:, mask].to_numpy(dtype=float)
        xs = StandardScaler().fit_transform(x_view)
        pair_map: dict[str, dict[str, float]] = {}
        for a, b in critical_pairs:
            key = f"{a}__{b}"
            pair_map[key] = _pair_geometry_metrics(xs, y_arr, a, b)
        out[view_name] = pair_map
    return out


def analyze_geometry(
    x: pd.DataFrame,
    y: pd.Series,
    config: AuditConfig,
) -> GeometryResult:
    """PCA + silhouette + centroid distances on standardized features."""
    scaler = StandardScaler()
    xs = scaler.fit_transform(x.to_numpy(dtype=float))
    n_components = min(10, xs.shape[0], xs.shape[1])
    pca = PCA(n_components=n_components, random_state=config.seed)
    z = pca.fit_transform(xs)
    explained = pca.explained_variance_ratio_.tolist()
    cumulative = np.cumsum(explained).tolist()

    loadings = pd.DataFrame(
        pca.components_[: min(3, n_components)].T,
        index=x.columns,
        columns=[f"PC{i+1}" for i in range(min(3, n_components))],
    )
    embedding = pd.DataFrame(
        {
            "PC1": z[:, 0],
            "PC2": z[:, 1] if z.shape[1] > 1 else 0.0,
            "PC3": z[:, 2] if z.shape[1] > 2 else 0.0,
            "regime": y.to_numpy(),
        }
    )

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)
    sil_global = float(silhouette_score(xs, y_enc, metric="euclidean"))
    sil_samples = silhouette_samples(xs, y_enc, metric="euclidean")
    sil_by: dict[str, float] = {}
    for label in encoder.classes_:
        mask = y.to_numpy() == label
        sil_by[str(label)] = float(np.mean(sil_samples[mask]))

    centroids = []
    labels = []
    for label in config.regimes:
        mask = y.to_numpy() == label
        if not np.any(mask):
            continue
        centroids.append(xs[mask].mean(axis=0))
        labels.append(label)
    cmat = np.vstack(centroids)
    euc = pairwise_distances(cmat, metric="euclidean")
    cos = pairwise_distances(cmat, metric="cosine")
    euc_df = pd.DataFrame(euc, index=labels, columns=labels)
    cos_df = pd.DataFrame(cos, index=labels, columns=labels)

    critical: dict[str, dict[str, float]] = {}
    for a, b in config.critical_pairs:
        key = f"{a}__{b}"
        if a in euc_df.index and b in euc_df.columns:
            critical[key] = {
                "euclidean": float(euc_df.loc[a, b]),
                "cosine": float(cos_df.loc[a, b]),
            }

    geometry_by_view = compute_critical_pair_geometry_by_view(
        x, y, config.critical_pairs
    )

    return GeometryResult(
        explained_variance_ratio=explained,
        cumulative_variance=cumulative,
        loadings=loadings,
        embedding=embedding,
        silhouette_global=sil_global,
        silhouette_by_regime=sil_by,
        centroid_euclidean=euc_df,
        centroid_cosine=cos_df,
        critical_pair_distances=critical,
        critical_pair_geometry_by_view=geometry_by_view,
    )
