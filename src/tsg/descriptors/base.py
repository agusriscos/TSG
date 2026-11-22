"""Descriptor contracts, IDs and graph preparation helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from tsg.descriptors.config import DescriptorConfig

DESCRIPTOR_IDS: tuple[str, ...] = (
    "comp",
    "lcc",
    "c_avg",
    "e_glob",
    "assortativity_r",
    "cv_k",
    "h_pk",
    "skew_k",
    "l_lcc",
    "d_lcc",
    "lambda2_g",
    "lambda2_lcc",
    "lambda2_tilde",
    "lambda_max",
    "i_dir",
    "self_loop_ratio",
)

DESCRIPTOR_CATEGORIES: dict[str, str] = {
    "comp": "Macro Connectivity",
    "lcc": "Macro Connectivity",
    "c_avg": "Integration & Cohesion",
    "e_glob": "Integration & Cohesion",
    "assortativity_r": "Degree Mixing",
    "cv_k": "Heterogeneity",
    "h_pk": "Heterogeneity",
    "skew_k": "Heterogeneity",
    "l_lcc": "Path Geometry",
    "d_lcc": "Path Geometry",
    "lambda2_g": "Laplacian Spectrum",
    "lambda2_lcc": "Laplacian Spectrum",
    "lambda2_tilde": "Laplacian Spectrum",
    "lambda_max": "Laplacian Spectrum",
    "i_dir": "Directional Dynamics",
    "self_loop_ratio": "Directional Dynamics",
}


@dataclass
class PreparedGraph:
    """Canonical views used by all descriptor families."""

    original: nx.Graph | nx.DiGraph
    undirected: nx.Graph
    lcc: nx.Graph
    w_directed: NDArray[np.floating] | None
    w_sym: sparse.csr_matrix
    node_order: list[Any]
    n_nodes: int
    n_edges_undirected: int
    is_directed_weighted: bool


@dataclass
class DescriptorResult:
    """Flat 16-descriptor vector for one graph."""

    dataset_id: str
    regime: str
    method: str
    family: str
    values: dict[str, float]
    build_time_sec: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)

    def vector(self) -> NDArray[np.floating]:
        return np.asarray([self.values[k] for k in DESCRIPTOR_IDS], dtype=float)


class BaseDescriptorExtractor(ABC):
    """Abstract extractor contributing one or more descriptor IDs."""

    @property
    @abstractmethod
    def descriptor_ids(self) -> tuple[str, ...]:
        """Subset of ``DESCRIPTOR_IDS`` produced by this extractor."""

    @abstractmethod
    def extract(self, prepared: PreparedGraph, config: DescriptorConfig) -> dict[str, float]:
        """Return a mapping id → value for this extractor's descriptors."""


def prepare_graph(graph: nx.Graph | nx.DiGraph) -> PreparedGraph:
    """Build undirected support, LCC and symmetrized weight matrix."""
    g = graph.copy()
    directed_weighted = bool(g.is_directed())
    node_order = list(g.nodes())
    n = len(node_order)

    w_directed: NDArray[np.floating] | None = None
    if directed_weighted:
        w_directed = nx.to_numpy_array(
            g,
            nodelist=node_order,
            weight="weight",
            dtype=float,
        )
        w_no_loops = w_directed.copy()
        np.fill_diagonal(w_no_loops, 0.0)
        w_sym_dense = 0.5 * (w_no_loops + w_no_loops.T)
        undirected = nx.Graph()
        undirected.add_nodes_from(node_order)
        for i, u in enumerate(node_order):
            for j in range(i + 1, n):
                if w_sym_dense[i, j] > 0.0:
                    undirected.add_edge(u, node_order[j])
        w_sym = sparse.csr_matrix(w_sym_dense)
    else:
        # Binary undirected: drop residual self-loops if any.
        if nx.number_of_selfloops(g) > 0:
            g.remove_edges_from(nx.selfloop_edges(g))
        undirected = nx.Graph(g)
        undirected.add_nodes_from(node_order)
        a = nx.to_scipy_sparse_array(
            undirected,
            nodelist=node_order,
            weight=None,
            dtype=float,
            format="csr",
        )
        # Ensure symmetry numerically.
        w_sym = sparse.csr_matrix(0.5 * (a + a.T))

    if undirected.number_of_nodes() == 0:
        lcc = nx.Graph()
    else:
        components = list(nx.connected_components(undirected))
        if not components:
            lcc = nx.Graph()
        else:
            largest = max(components, key=len)
            lcc = undirected.subgraph(largest).copy()

    return PreparedGraph(
        original=g,
        undirected=undirected,
        lcc=lcc,
        w_directed=w_directed,
        w_sym=w_sym,
        node_order=node_order,
        n_nodes=n,
        n_edges_undirected=int(undirected.number_of_edges()),
        is_directed_weighted=directed_weighted,
    )
