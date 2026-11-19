"""Phase-space networks: recurrence (RN) and k-NN with Theiler exclusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform

from tsg.graphs.base import BaseGraphTransformer, Family
from tsg.graphs.config import KNNParams, RNParams
from tsg.graphs.embedding import delay_embedding, theiler_mask


@dataclass(frozen=True)
class RecurrenceNetworkTransformer(BaseGraphTransformer):
    """Recurrence network with global ε = percentile of non-Theiler distances."""

    params: RNParams = RNParams()

    @property
    def method(self) -> str:
        return "RN"

    @property
    def family(self) -> Family:
        return "Phase Space"

    @property
    def directed(self) -> bool:
        return False

    @property
    def weighted(self) -> bool:
        return False

    def params_dict(self) -> dict[str, Any]:
        p = self.params
        return {
            "m": p.m,
            "tau": p.tau,
            "w": p.w,
            "eps_percentile": p.eps_percentile,
        }

    def expected_n_nodes(self, series_length: int) -> int | None:
        return series_length - (self.params.m - 1) * self.params.tau

    def _build(self, series: NDArray[np.floating]) -> nx.Graph:
        p = self.params
        emb = delay_embedding(series, m=p.m, tau=p.tau)
        n = emb.shape[0]
        dists = squareform(pdist(emb, metric="euclidean"))
        forbidden = theiler_mask(n, p.w)
        valid = ~forbidden
        if not np.any(valid):
            raise ValueError("No valid pairs outside Theiler window for RN")
        # Percentile over unique unordered pairs outside Theiler (upper triangle).
        iu = np.triu_indices(n, k=1)
        pair_ok = valid[iu]
        if not np.any(pair_ok):
            raise ValueError("No upper-triangle pairs outside Theiler window for RN")
        eps = float(np.percentile(dists[iu][pair_ok], p.eps_percentile))
        adj = (dists <= eps) & valid
        np.fill_diagonal(adj, False)
        g = nx.from_numpy_array(adj.astype(float))
        # Store effective ε used for this series in params snapshot via graph attrs.
        g.graph["eps"] = eps
        return g

    def transform(
        self,
        series: NDArray[np.floating],
        dataset_id: str,
        *,
        regime: str = "",
    ):
        result = super().transform(series, dataset_id, regime=regime)
        eps = result.graph.graph.pop("eps", None)
        if eps is not None:
            result.params = {**result.params, "eps": eps}
        return result


@dataclass(frozen=True)
class KNNTransformer(BaseGraphTransformer):
    """k-NN graph with Theiler exclusion and union symmetrization."""

    params: KNNParams = KNNParams()

    @property
    def method(self) -> str:
        return "kNN"

    @property
    def family(self) -> Family:
        return "Phase Space"

    @property
    def directed(self) -> bool:
        return False

    @property
    def weighted(self) -> bool:
        return False

    def params_dict(self) -> dict[str, Any]:
        p = self.params
        return {"m": p.m, "tau": p.tau, "w": p.w, "k": p.k}

    def expected_n_nodes(self, series_length: int) -> int | None:
        return series_length - (self.params.m - 1) * self.params.tau

    def _build(self, series: NDArray[np.floating]) -> nx.Graph:
        p = self.params
        emb = delay_embedding(series, m=p.m, tau=p.tau)
        n = emb.shape[0]
        if n <= p.w + p.k:
            raise ValueError(
                f"Series too short for kNN: n={n}, w={p.w}, k={p.k}"
            )
        dists = squareform(pdist(emb, metric="euclidean"))
        dists[theiler_mask(n, p.w)] = np.inf

        dig = nx.DiGraph()
        dig.add_nodes_from(range(n))
        for i in range(n):
            row = dists[i]
            finite = np.isfinite(row)
            n_valid = int(finite.sum())
            if n_valid == 0:
                continue
            kk = min(p.k, n_valid)
            # argpartition over finite candidates: replace inf temporarily handled
            nn = np.argpartition(row, kk - 1)[:kk]
            for j in nn:
                if np.isfinite(row[j]):
                    dig.add_edge(i, int(j))

        und = nx.Graph()
        und.add_nodes_from(dig.nodes())
        und.add_edges_from(dig.edges())  # union symmetrization
        return und
