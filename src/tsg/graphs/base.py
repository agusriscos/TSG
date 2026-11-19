"""Abstract TS2G transformer interface and graph result container."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

import networkx as nx
import numpy as np
from numpy.typing import NDArray

Family = Literal[
    "Direct Geometry",
    "Phase Space",
    "Discr. & Transition",
    "OPN",
    "Similarity & Corr.",
]


@dataclass
class GraphResult:
    """Standardized graph artifact with construction provenance."""

    dataset_id: str
    regime: str
    method: str
    family: Family
    directed: bool
    weighted: bool
    graph: nx.Graph | nx.DiGraph
    params: dict[str, Any] = field(default_factory=dict)
    build_time_sec: float = 0.0
    series_length: int = 0

    @property
    def n_nodes(self) -> int:
        return int(self.graph.number_of_nodes())

    @property
    def n_edges(self) -> int:
        return int(self.graph.number_of_edges())

    @property
    def density(self) -> float:
        n = self.n_nodes
        if n < 2:
            return 0.0
        return float(nx.density(self.graph))

    @property
    def mean_degree(self) -> float:
        n = self.n_nodes
        if n == 0:
            return 0.0
        degrees = [d for _, d in self.graph.degree()]
        return float(np.mean(degrees)) if degrees else 0.0

    def to_metadata(self, path: str | None = None) -> dict[str, Any]:
        """Serializable index row for ``metadata.json``."""
        payload: dict[str, Any] = {
            "dataset_id": self.dataset_id,
            "regime": self.regime,
            "method": self.method,
            "family": self.family,
            "directed": self.directed,
            "weighted": self.weighted,
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "density": self.density,
            "mean_degree": self.mean_degree,
            "series_length": self.series_length,
            "build_time_sec": self.build_time_sec,
            "params": dict(self.params),
        }
        if path is not None:
            payload["path"] = path
        return payload


class BaseGraphTransformer(ABC):
    """Abstract factory mapping a univariate series to a NetworkX graph."""

    @property
    @abstractmethod
    def method(self) -> str:
        """Canonical method id from Table 3.2."""

    @property
    @abstractmethod
    def family(self) -> Family:
        """Algorithm family label."""

    @property
    @abstractmethod
    def directed(self) -> bool:
        """Whether the output graph is directed."""

    @property
    @abstractmethod
    def weighted(self) -> bool:
        """Whether edges carry a ``weight`` attribute."""

    @abstractmethod
    def params_dict(self) -> dict[str, Any]:
        """Exact construction parameters."""

    @abstractmethod
    def expected_n_nodes(self, series_length: int) -> int | None:
        """Theoretical node count from Table 3.2, or ``None`` if only an upper bound."""

    @abstractmethod
    def _build(self, series: NDArray[np.floating]) -> nx.Graph | nx.DiGraph:
        """Core mapping without timing / metadata wrapping."""

    def transform(
        self,
        series: NDArray[np.floating],
        dataset_id: str,
        *,
        regime: str = "",
    ) -> GraphResult:
        """Transform a series into a ``GraphResult`` with wall-clock timing."""
        values = np.asarray(series, dtype=float).ravel()
        t0 = time.perf_counter()
        graph = self._build(values)
        elapsed = time.perf_counter() - t0
        if nx.number_of_selfloops(graph) > 0:
            graph.remove_edges_from(nx.selfloop_edges(graph))
        return GraphResult(
            dataset_id=dataset_id,
            regime=regime,
            method=self.method,
            family=self.family,
            directed=self.directed,
            weighted=self.weighted,
            graph=graph,
            params=self.params_dict(),
            build_time_sec=elapsed,
            series_length=int(values.size),
        )
