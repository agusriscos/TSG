"""Degree-mixing and heterogeneity descriptors."""

from __future__ import annotations

import warnings

import networkx as nx
import numpy as np
from scipy.stats import entropy, skew

from tsg.descriptors.base import BaseDescriptorExtractor, PreparedGraph
from tsg.descriptors.config import DescriptorConfig


class DegreeExtractor(BaseDescriptorExtractor):
    """``assortativity_r``, ``cv_k``, ``h_pk``, ``skew_k``."""

    @property
    def descriptor_ids(self) -> tuple[str, ...]:
        return ("assortativity_r", "cv_k", "h_pk", "skew_k")

    def extract(self, prepared: PreparedGraph, config: DescriptorConfig) -> dict[str, float]:
        g = prepared.undirected
        if prepared.n_nodes == 0 or prepared.n_edges_undirected == 0:
            return {
                "assortativity_r": float("nan"),
                "cv_k": 0.0,
                "h_pk": 0.0,
                "skew_k": 0.0,
            }

        degrees = np.asarray([d for _, d in g.degree()], dtype=float)
        mu = float(degrees.mean())
        sigma = float(degrees.std(ddof=0))
        if mu == 0.0:
            cv_k = 0.0
        else:
            cv_k = sigma / mu

        if sigma == 0.0:
            assort = float("nan")
            skew_k = 0.0
        else:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    assort = float(nx.degree_assortativity_coefficient(g))
                if not np.isfinite(assort):
                    assort = float("nan")
            except (ValueError, ZeroDivisionError):
                assort = float("nan")
            skew_k = float(skew(degrees, bias=False, nan_policy="omit"))
            if not np.isfinite(skew_k):
                skew_k = 0.0

        # Shannon entropy of empirical degree distribution.
        _, counts = np.unique(degrees.astype(int), return_counts=True)
        probs = counts.astype(float) / counts.sum()
        base = np.e if config.entropy_base == "e" else 2.0
        h_pk = float(entropy(probs, base=base))

        return {
            "assortativity_r": assort,
            "cv_k": float(cv_k),
            "h_pk": h_pk,
            "skew_k": float(skew_k),
        }
