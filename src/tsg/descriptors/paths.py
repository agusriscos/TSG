"""Path-geometry descriptors on the largest connected component."""

from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.sparse.csgraph import shortest_path

from tsg.descriptors.base import BaseDescriptorExtractor, PreparedGraph
from tsg.descriptors.config import DescriptorConfig


class PathExtractor(BaseDescriptorExtractor):
    """``l_lcc``, ``d_lcc``."""

    @property
    def descriptor_ids(self) -> tuple[str, ...]:
        return ("l_lcc", "d_lcc")

    def extract(self, prepared: PreparedGraph, config: DescriptorConfig) -> dict[str, float]:
        skip = set(config.skip_descriptors)
        if "l_lcc" in skip and "d_lcc" in skip:
            return {"l_lcc": 0.0, "d_lcc": 0.0}
        lcc = prepared.lcc
        n = lcc.number_of_nodes()
        if n <= 1 or lcc.number_of_edges() == 0:
            return {"l_lcc": 0.0, "d_lcc": 0.0}

        nodes = list(lcc.nodes())
        adj = nx.to_scipy_sparse_array(
            lcc, nodelist=nodes, weight=None, dtype=float, format="csr"
        )
        dist = shortest_path(adj, directed=False, unweighted=True)
        # Upper triangle of finite positive distances.
        iu = np.triu_indices(n, k=1)
        vals = dist[iu]
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            return {"l_lcc": 0.0, "d_lcc": 0.0}
        return {
            "l_lcc": float(finite.mean()),
            "d_lcc": float(finite.max()),
        }
