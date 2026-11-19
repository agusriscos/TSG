"""Discretization / ordinal transition networks (STN, OPN)."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from tsg.graphs.base import BaseGraphTransformer, Family
from tsg.graphs.config import OPNParams, STNParams
from tsg.graphs.embedding import delay_embedding

logger = logging.getLogger(__name__)


def _add_weighted_transition(dig: nx.DiGraph, u: Any, v: Any) -> None:
    if dig.has_edge(u, v):
        dig[u][v]["weight"] += 1
    else:
        dig.add_edge(u, v, weight=1)


def _normalize_out_probabilities(dig: nx.DiGraph) -> None:
    """Attach ``prob`` = weight / out-strength for each directed edge."""
    for u in dig.nodes():
        out_w = sum(data.get("weight", 0) for _, _, data in dig.out_edges(u, data=True))
        if out_w <= 0:
            continue
        for _, v, data in dig.out_edges(u, data=True):
            data["prob"] = float(data["weight"]) / float(out_w)


@dataclass(frozen=True)
class STNTransformer(BaseGraphTransformer):
    """State Transition Network on M empirical quantiles."""

    params: STNParams = STNParams()

    @property
    def method(self) -> str:
        return "STN"

    @property
    def family(self) -> Family:
        return "Discr. & Transition"

    @property
    def directed(self) -> bool:
        return True

    @property
    def weighted(self) -> bool:
        return True

    def params_dict(self) -> dict[str, Any]:
        return {"M": self.params.M}

    def expected_n_nodes(self, series_length: int) -> int | None:
        del series_length
        return self.params.M

    def _build(self, series: NDArray[np.floating]) -> nx.DiGraph:
        m_bins = self.params.M
        x = np.asarray(series, dtype=float).ravel()
        quantiles = np.linspace(0.0, 1.0, m_bins + 1)[1:-1]
        edges = np.unique(np.quantile(x, quantiles))
        if edges.size < m_bins - 1:
            logger.warning(
                "STN quantile edges collapsed from %d to %d unique cuts",
                m_bins - 1,
                edges.size,
            )
        states = np.digitize(x, edges)  # 0 .. <= m_bins-1
        dig = nx.DiGraph()
        dig.add_nodes_from(range(m_bins))
        for a, b in zip(states[:-1], states[1:]):
            _add_weighted_transition(dig, int(a), int(b))
        _normalize_out_probabilities(dig)
        return dig


@dataclass(frozen=True)
class OPNTransformer(BaseGraphTransformer):
    """Ordinal Partition Network (Bandt–Pompe patterns)."""

    params: OPNParams = OPNParams()

    @property
    def method(self) -> str:
        return "OPN"

    @property
    def family(self) -> Family:
        return "OPN"

    @property
    def directed(self) -> bool:
        return True

    @property
    def weighted(self) -> bool:
        return True

    def params_dict(self) -> dict[str, Any]:
        p = self.params
        return {"m": p.m, "tau": p.tau, "max_patterns": int(math.factorial(p.m))}

    def expected_n_nodes(self, series_length: int) -> int | None:
        """Upper bound only (≤ m!); actual count depends on observed patterns."""
        del series_length
        return None

    def _build(self, series: NDArray[np.floating]) -> nx.DiGraph:
        p = self.params
        emb = delay_embedding(series, m=p.m, tau=p.tau)
        # Stable argsort breaks ties by temporal index order.
        # String node ids keep GraphML / GEXF serialization lossless and portable.
        patterns = [
            "-".join(str(i) for i in np.argsort(row, kind="mergesort").tolist())
            for row in emb
        ]
        dig = nx.DiGraph()
        for a, b in zip(patterns[:-1], patterns[1:]):
            dig.add_node(a, pattern=a)
            dig.add_node(b, pattern=b)
            _add_weighted_transition(dig, a, b)
        _normalize_out_probabilities(dig)
        return dig
