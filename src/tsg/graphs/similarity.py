"""Similarity and correlation networks on contiguous segments."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform
from scipy.stats import t as student_t

from tsg.graphs.base import BaseGraphTransformer, Family
from tsg.graphs.config import CorrelationParams, SegmentParams


def contiguous_windows(values: NDArray[np.floating], L: int) -> NDArray[np.floating]:
    """Split a series into ``floor(N/L)`` non-overlapping windows of length ``L``."""
    x = np.asarray(values, dtype=float).ravel()
    n_win = x.size // L
    if n_win < 2:
        raise ValueError(f"Need at least 2 windows of length L={L}, got n={x.size}")
    return x[: n_win * L].reshape(n_win, L)


def pairwise_pearson(windows: NDArray[np.floating]) -> NDArray[np.floating]:
    """Pearson correlation matrix with zero-variance rows mapped to 0 (no NaN)."""
    x = np.asarray(windows, dtype=float)
    mu = x.mean(axis=1, keepdims=True)
    xc = x - mu
    norms = np.sqrt(np.sum(xc * xc, axis=1))
    safe = norms > 1e-12
    corr = np.zeros((x.shape[0], x.shape[0]), dtype=float)
    if not np.any(safe):
        return corr
    xs = xc[safe] / norms[safe][:, None]
    block = xs @ xs.T
    idx = np.where(safe)[0]
    corr[np.ix_(idx, idx)] = block
    np.fill_diagonal(corr, 0.0)
    return corr


def bonferroni_corr_threshold(n_windows: int, L: int, alpha: float = 0.05) -> float:
    """Absolute Pearson threshold with Bonferroni correction over pair tests."""
    n_pairs = comb(n_windows, 2)
    if n_pairs < 1 or L <= 2:
        return 1.0
    alpha_star = alpha / n_pairs
    df = L - 2
    # Two-sided critical t, then convert to |r|.
    t_crit = float(student_t.ppf(1.0 - alpha_star / 2.0, df))
    r = t_crit / np.sqrt(t_crit**2 + df)
    return float(min(max(r, 0.0), 1.0))


@dataclass(frozen=True)
class CorrelationNetworkTransformer(BaseGraphTransformer):
    """Segment correlation network with absolute Pearson threshold."""

    params: CorrelationParams = CorrelationParams()

    @property
    def method(self) -> str:
        return "Correlation"

    @property
    def family(self) -> Family:
        return "Similarity & Corr."

    @property
    def directed(self) -> bool:
        return False

    @property
    def weighted(self) -> bool:
        return False

    def params_dict(self) -> dict[str, Any]:
        p = self.params
        return {
            "L": p.L,
            "theta_corr": p.theta_corr,
            "use_bonferroni": p.use_bonferroni,
            "alpha": p.alpha,
        }

    def expected_n_nodes(self, series_length: int) -> int | None:
        return series_length // self.params.L

    def _build(self, series: NDArray[np.floating]) -> nx.Graph:
        p = self.params
        windows = contiguous_windows(series, p.L)
        corr = pairwise_pearson(windows)
        theta = p.theta_corr
        if p.use_bonferroni:
            theta = bonferroni_corr_threshold(windows.shape[0], p.L, p.alpha)
        adj = np.abs(corr) >= theta
        np.fill_diagonal(adj, False)
        g = nx.from_numpy_array(adj.astype(float))
        g.graph["theta_effective"] = float(theta)
        g.graph["n_windows"] = int(windows.shape[0])
        return g

    def transform(
        self,
        series: NDArray[np.floating],
        dataset_id: str,
        *,
        regime: str = "",
    ):
        result = super().transform(series, dataset_id, regime=regime)
        theta_eff = result.graph.graph.pop("theta_effective", None)
        n_windows = result.graph.graph.pop("n_windows", None)
        extra: dict[str, Any] = {}
        if theta_eff is not None:
            extra["theta_effective"] = theta_eff
        if n_windows is not None:
            extra["n_windows"] = n_windows
        if extra:
            result.params = {**result.params, **extra}
        return result


@dataclass(frozen=True)
class SegmentNetworkTransformer(BaseGraphTransformer):
    """Segment Euclidean-distance network with percentile threshold."""

    params: SegmentParams = SegmentParams()

    @property
    def method(self) -> str:
        return "Segment"

    @property
    def family(self) -> Family:
        return "Similarity & Corr."

    @property
    def directed(self) -> bool:
        return False

    @property
    def weighted(self) -> bool:
        return False

    def params_dict(self) -> dict[str, Any]:
        p = self.params
        return {"L": p.L, "eps_percentile": p.eps_percentile}

    def expected_n_nodes(self, series_length: int) -> int | None:
        return series_length // self.params.L

    def _build(self, series: NDArray[np.floating]) -> nx.Graph:
        p = self.params
        windows = contiguous_windows(series, p.L)
        dists = squareform(pdist(windows, metric="euclidean"))
        iu = np.triu_indices_from(dists, k=1)
        thresh = float(np.percentile(dists[iu], p.eps_percentile))
        adj = (dists <= thresh) & (dists > 0)
        np.fill_diagonal(adj, False)
        g = nx.from_numpy_array(adj.astype(float))
        g.graph["threshold"] = thresh
        g.graph["n_windows"] = int(windows.shape[0])
        return g

    def transform(
        self,
        series: NDArray[np.floating],
        dataset_id: str,
        *,
        regime: str = "",
    ):
        result = super().transform(series, dataset_id, regime=regime)
        thresh = result.graph.graph.pop("threshold", None)
        n_windows = result.graph.graph.pop("n_windows", None)
        extra: dict[str, Any] = {}
        if thresh is not None:
            extra["threshold"] = thresh
        if n_windows is not None:
            extra["n_windows"] = n_windows
        if extra:
            result.params = {**result.params, **extra}
        return result
