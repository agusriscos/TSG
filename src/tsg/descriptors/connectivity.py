"""Macro connectivity and cohesion descriptors."""

from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.sparse.csgraph import shortest_path

from tsg.descriptors.base import BaseDescriptorExtractor, PreparedGraph
from tsg.descriptors.config import DescriptorConfig


class ConnectivityExtractor(BaseDescriptorExtractor):
    """``comp``, ``lcc``, ``c_avg``, ``e_glob``."""

    @property
    def descriptor_ids(self) -> tuple[str, ...]:
        return ("comp", "lcc", "c_avg", "e_glob")

    def extract(self, prepared: PreparedGraph, config: DescriptorConfig) -> dict[str, float]:
        skip = set(config.skip_descriptors)
        g = prepared.undirected
        n = prepared.n_nodes
        if n == 0:
            return {"comp": 0.0, "lcc": 0.0, "c_avg": 0.0, "e_glob": 0.0}

        components = list(nx.connected_components(g))
        comp = float(len(components))
        lcc_size = prepared.lcc.number_of_nodes()
        lcc = float(lcc_size) / float(n)

        if prepared.n_edges_undirected == 0:
            c_avg = 0.0
            e_glob = 0.0
        else:
            c_avg = float(nx.average_clustering(g))
            if "e_glob" in skip:
                e_glob = 0.0
            else:
                e_glob = _global_efficiency_sparse(g, prepared.node_order)
        return {"comp": comp, "lcc": lcc, "c_avg": c_avg, "e_glob": e_glob}


def _global_efficiency_sparse(g: nx.Graph, node_order: list) -> float:
    """Global efficiency via sparse shortest paths (unweighted)."""
    n = len(node_order)
    if n < 2:
        return 0.0
    adj = nx.to_scipy_sparse_array(
        g, nodelist=node_order, weight=None, dtype=float, format="csr"
    )
    dist = shortest_path(adj, directed=False, unweighted=True)
    inv = np.zeros_like(dist, dtype=float)
    finite = np.isfinite(dist) & (dist > 0.0)
    inv[finite] = 1.0 / dist[finite]
    # Exclude diagonal (already zero); average over i != j.
    return float(inv.sum() / (n * (n - 1)))
