"""Laplacian spectral descriptors."""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

from tsg.descriptors.base import BaseDescriptorExtractor, PreparedGraph
from tsg.descriptors.config import DescriptorConfig

logger = logging.getLogger(__name__)


class SpectralExtractor(BaseDescriptorExtractor):
    """``lambda2_g``, ``lambda2_lcc``, ``lambda2_tilde``, ``lambda_max``."""

    @property
    def descriptor_ids(self) -> tuple[str, ...]:
        return ("lambda2_g", "lambda2_lcc", "lambda2_tilde", "lambda_max")

    def extract(self, prepared: PreparedGraph, config: DescriptorConfig) -> dict[str, float]:
        skip = set(config.skip_descriptors)
        spectral_ids = ("lambda2_g", "lambda2_lcc", "lambda2_tilde", "lambda_max")
        if all(s in skip for s in spectral_ids):
            return {k: 0.0 for k in spectral_ids}
        n = prepared.n_nodes
        if n == 0 or prepared.n_edges_undirected == 0:
            return {
                "lambda2_g": 0.0,
                "lambda2_lcc": 0.0,
                "lambda2_tilde": 0.0,
                "lambda_max": 0.0,
            }

        n_comp = nx.number_connected_components(prepared.undirected)
        lcc_frac = prepared.lcc.number_of_nodes() / float(n)
        lcc_n = prepared.lcc.number_of_nodes()

        if lcc_n <= 1 or prepared.lcc.number_of_edges() == 0:
            return {
                "lambda2_g": 0.0,
                "lambda2_lcc": 0.0,
                "lambda2_tilde": 0.0,
                "lambda_max": _laplacian_lambda_max(prepared.w_sym, config) if n_comp == 1 else 0.0,
            }

        index = {u: i for i, u in enumerate(prepared.node_order)}
        idx = [index[u] for u in prepared.lcc.nodes()]
        w_lcc = prepared.w_sym[np.ix_(idx, idx)].tocsr()

        if n_comp == 1:
            # Full graph == LCC: compute spectrum once.
            lambda2_lcc, lambda_max = _laplacian_lambda2_and_max(w_lcc, config)
            lambda2_g = lambda2_lcc
        else:
            lambda2_g = 0.0
            lambda2_lcc = _laplacian_lambda2(w_lcc, config)
            lambda_max = _laplacian_lambda_max(prepared.w_sym, config)

        lambda2_tilde = float((lcc_frac**2) * lambda2_lcc)
        return {
            "lambda2_g": float(lambda2_g),
            "lambda2_lcc": float(lambda2_lcc),
            "lambda2_tilde": float(lambda2_tilde),
            "lambda_max": float(lambda_max),
        }


def _combinatorial_laplacian(w_sym: sparse.spmatrix) -> sparse.csr_matrix:
    w = sparse.csr_matrix(w_sym, dtype=float)
    degrees = np.asarray(w.sum(axis=1)).ravel()
    return sparse.diags(degrees) - w


def _normalized_laplacian(w_sym: sparse.spmatrix) -> sparse.csr_matrix:
    w = sparse.csr_matrix(w_sym, dtype=float)
    degrees = np.asarray(w.sum(axis=1)).ravel()
    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.where(degrees > 0.0, 1.0 / np.sqrt(degrees), 0.0)
    d_left = sparse.diags(d_inv_sqrt)
    eye = sparse.eye(w.shape[0], format="csr")
    return eye - (d_left @ w @ d_left)


def _build_laplacian(w_sym: sparse.spmatrix, config: DescriptorConfig) -> sparse.csr_matrix:
    if config.laplacian == "normalized":
        return _normalized_laplacian(w_sym)
    return _combinatorial_laplacian(w_sym)


def _laplacian_eigs_dense(lap: sparse.spmatrix) -> np.ndarray:
    dense = lap.toarray()
    dense = 0.5 * (dense + dense.T)
    return eigh(dense, eigvals_only=True)


def _eigsh_smallest(lap: sparse.csr_matrix, k: int, config: DescriptorConfig) -> np.ndarray:
    """Smallest algebraic eigenvalues without dense fallback on large graphs."""
    n = lap.shape[0]
    k = min(k, max(n - 2, 1))
    ncv = min(n, max(4 * k + 1, 40))
    lap64 = lap.asfptype()

    # 1) Shift-invert near 0 (robust for near-singular Laplacians).
    try:
        vals = eigsh(
            lap64,
            k=k,
            sigma=1e-8,
            which="LM",
            tol=config.eigsh_tol,
            maxiter=max(config.eigsh_maxiter, 10 * n),
            ncv=ncv,
            return_eigenvectors=False,
        )
        return np.sort(np.real(vals))
    except Exception as exc1:  # noqa: BLE001
        logger.info("eigsh shift-invert failed (%s); trying which='SA'", exc1)

    # 2) Smallest algebraic without shift-invert.
    try:
        vals = eigsh(
            lap64,
            k=k,
            which="SA",
            tol=max(config.eigsh_tol, 1e-6),
            maxiter=max(config.eigsh_maxiter, 20 * n),
            ncv=ncv,
            return_eigenvectors=False,
        )
        return np.sort(np.real(vals))
    except Exception as exc2:  # noqa: BLE001
        logger.warning("eigsh SA failed (%s); using dense eigh only if n<=%d", exc2, config.large_graph_threshold)
        if n <= config.large_graph_threshold:
            return _laplacian_eigs_dense(lap)[:k]
        # Last resort for huge graphs: return zeros rather than O(n^3) hang.
        return np.zeros(k, dtype=float)


def _eigsh_largest(lap: sparse.csr_matrix, config: DescriptorConfig) -> float:
    n = lap.shape[0]
    lap64 = lap.asfptype()
    ncv = min(n, 40)
    try:
        vals = eigsh(
            lap64,
            k=1,
            which="LA",
            tol=config.eigsh_tol,
            maxiter=max(config.eigsh_maxiter, 10 * n),
            ncv=ncv,
            return_eigenvectors=False,
        )
        return float(np.real(vals[0]))
    except Exception as exc:  # noqa: BLE001
        logger.warning("eigsh LA failed (%s)", exc)
        if n <= config.large_graph_threshold:
            return float(_laplacian_eigs_dense(lap)[-1])
        return 0.0


def _laplacian_lambda2(w_sym: sparse.spmatrix, config: DescriptorConfig) -> float:
    n = w_sym.shape[0]
    if n <= 1:
        return 0.0
    lap = _build_laplacian(w_sym, config)
    if n <= config.large_graph_threshold:
        evals = _laplacian_eigs_dense(lap)
        return float(max(evals[1], 0.0))
    vals = _eigsh_smallest(lap, k=2, config=config)
    if vals.size < 2:
        return 0.0
    return float(max(vals[1], 0.0))


def _laplacian_lambda_max(w_sym: sparse.spmatrix, config: DescriptorConfig) -> float:
    n = w_sym.shape[0]
    if n == 0:
        return 0.0
    lap = _build_laplacian(w_sym, config)
    if n <= config.large_graph_threshold:
        return float(_laplacian_eigs_dense(lap)[-1])
    return _eigsh_largest(lap, config)


def _laplacian_lambda2_and_max(
    w_sym: sparse.spmatrix, config: DescriptorConfig
) -> tuple[float, float]:
    n = w_sym.shape[0]
    if n <= 1:
        return 0.0, 0.0
    if n <= config.large_graph_threshold:
        lap = _build_laplacian(w_sym, config)
        evals = _laplacian_eigs_dense(lap)
        return float(max(evals[1], 0.0)), float(evals[-1])
    return _laplacian_lambda2(w_sym, config), _laplacian_lambda_max(w_sym, config)
