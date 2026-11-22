"""Unsupervised geometry: PCA, silhouette and centroid distances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, silhouette_samples, silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from tsg.audit.config import AuditConfig


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "explained_variance_ratio": self.explained_variance_ratio,
            "cumulative_variance": self.cumulative_variance,
            "silhouette_global": self.silhouette_global,
            "silhouette_by_regime": self.silhouette_by_regime,
            "centroid_euclidean": self.centroid_euclidean.to_dict(),
            "centroid_cosine": self.centroid_cosine.to_dict(),
            "critical_pair_distances": self.critical_pair_distances,
            "top_loadings_pc1_pc2": self.loadings.assign(
                score=np.sqrt(self.loadings["PC1"] ** 2 + self.loadings["PC2"] ** 2)
            )
            .sort_values("score", ascending=False)
            .head(15)
            .to_dict(orient="records"),
        }


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
    )
