"""Directional-dynamics descriptors for transition and directed visibility graphs."""

from __future__ import annotations

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from tsg.descriptors.base import BaseDescriptorExtractor, PreparedGraph
from tsg.descriptors.config import DescriptorConfig


def smoothed_degree_distribution(
    degrees: NDArray[np.integer] | NDArray[np.floating],
    *,
    k_max: int | None = None,
    eps: float = 1e-12,
) -> NDArray[np.floating]:
    """Build a smoothed discrete degree pmf on ``{0, …, k_max}``.

    Parameters
    ----------
    degrees
        Per-node degree values (in or out).
    k_max
        Support upper bound. Defaults to ``max(degrees)``.
    eps
        Additive Laplace smoothing before renormalization.
    """
    deg = np.asarray(degrees, dtype=int).ravel()
    if deg.size == 0:
        support = 1 if k_max is None else max(1, k_max + 1)
        p = np.full(support, eps, dtype=float)
        return p / p.sum()
    vmax = int(deg.max()) if k_max is None else int(k_max)
    support = max(vmax, 0) + 1
    counts = np.bincount(np.clip(deg, 0, support - 1), minlength=support).astype(float)
    counts = counts + eps
    return counts / counts.sum()


def kl_divergence(
    p: NDArray[np.floating],
    q: NDArray[np.floating],
) -> float:
    """Kullback–Leibler divergence ``KL(p ‖ q)`` for strictly positive pmfs."""
    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    return float(np.sum(p_arr * np.log(p_arr / q_arr)))


def temporal_irreversibility_kl(
    graph: nx.DiGraph,
    *,
    eps: float = 1e-12,
) -> float:
    """Smoothed KL irreversibility between in- and out-degree distributions.

    For a directed graph (typically ``dHVG``),

    .. math::

        I = \\mathrm{KL}(P(k^{\\mathrm{in}}) \\,\\|\\, P(k^{\\mathrm{out}}))

    with Laplace-smoothed empirical degree pmfs sharing a common support.
    Returns ``0.0`` for undirected or empty graphs.
    """
    if not isinstance(graph, nx.DiGraph) or graph.number_of_nodes() == 0:
        return 0.0
    nodes = list(graph.nodes())
    k_in = np.asarray([graph.in_degree(n) for n in nodes], dtype=int)
    k_out = np.asarray([graph.out_degree(n) for n in nodes], dtype=int)
    k_max = int(max(int(k_in.max()), int(k_out.max()), 0))
    p_in = smoothed_degree_distribution(k_in, k_max=k_max, eps=eps)
    p_out = smoothed_degree_distribution(k_out, k_max=k_max, eps=eps)
    return kl_divergence(p_in, p_out)


class DirectionalExtractor(BaseDescriptorExtractor):
    """``i_dir``, ``self_loop_ratio``, plus ``kl_inout`` irreversibility (extras)."""

    @property
    def descriptor_ids(self) -> tuple[str, ...]:
        return ("i_dir", "self_loop_ratio")

    def extract(self, prepared: PreparedGraph, config: DescriptorConfig) -> dict[str, float]:
        del config
        if not prepared.is_directed_weighted or prepared.w_directed is None:
            return {
                "i_dir": 0.0,
                "self_loop_ratio": 0.0,
                "kl_inout": 0.0,
            }

        w = np.asarray(prepared.w_directed, dtype=float)
        diff = w - w.T
        sym = w + w.T
        denom = float(np.linalg.norm(sym, ord="fro"))
        if denom == 0.0:
            i_dir = 0.0
        else:
            i_dir = float(np.linalg.norm(diff, ord="fro") / denom)

        total = float(w.sum())
        if total == 0.0:
            self_loop_ratio = 0.0
        else:
            self_loop_ratio = float(np.trace(w) / total)

        kl_inout = temporal_irreversibility_kl(prepared.original)
        return {
            "i_dir": i_dir,
            "self_loop_ratio": self_loop_ratio,
            "kl_inout": kl_inout,
        }
