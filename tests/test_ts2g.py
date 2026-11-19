"""Unit tests for TS2G transformers (Table 3.2)."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from tsg.graphs.config import (
    CorrelationParams,
    KNNParams,
    OPNParams,
    RNParams,
    STNParams,
    SegmentParams,
    TS2GSuiteConfig,
)
from tsg.graphs.geometry import HVGTransformer, NVGTransformer
from tsg.graphs.io import save_graph_suite
from tsg.graphs.phase_space import KNNTransformer, RecurrenceNetworkTransformer
from tsg.graphs.registry import METHOD_IDS, build_transformers, transform_suite
from tsg.graphs.similarity import CorrelationNetworkTransformer, SegmentNetworkTransformer
from tsg.graphs.transition import OPNTransformer, STNTransformer
from tsg.reporting.ts2g_report import TS2GReportBuilder

N = 240
SEED = 7


@pytest.fixture
def series() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    t = np.arange(N, dtype=float)
    return np.sin(2 * np.pi * 0.03 * t) + 0.05 * rng.normal(size=N)


def test_method_registry_has_eight() -> None:
    assert list(METHOD_IDS) == [
        "NVG",
        "HVG",
        "RN",
        "kNN",
        "STN",
        "OPN",
        "Correlation",
        "Segment",
    ]
    assert len(build_transformers()) == 8


def test_visibility_node_counts(series: np.ndarray) -> None:
    for tr in (NVGTransformer(), HVGTransformer()):
        result = tr.transform(series, "sine", regime="periodic")
        assert result.n_nodes == N
        assert isinstance(result.graph, nx.Graph)
        assert not result.directed
        assert nx.number_of_selfloops(result.graph) == 0


def test_phase_space_nodes_and_theiler(series: np.ndarray) -> None:
    rn = RecurrenceNetworkTransformer(params=RNParams(m=3, tau=1, w=5))
    knn = KNNTransformer(params=KNNParams(m=3, tau=1, w=5, k=5))
    expected = N - (3 - 1) * 1
    for tr in (rn, knn):
        result = tr.transform(series, "sine", regime="periodic")
        assert result.n_nodes == expected
        assert isinstance(result.graph, nx.Graph)
        for u, v in result.graph.edges():
            assert abs(int(u) - int(v)) > 5


def test_stn_and_opn_directed_weighted(series: np.ndarray) -> None:
    stn = STNTransformer(params=STNParams(M=10)).transform(series, "sine")
    assert stn.n_nodes == 10
    assert isinstance(stn.graph, nx.DiGraph)
    assert stn.directed and stn.weighted
    for _, _, data in stn.graph.edges(data=True):
        assert "weight" in data
        assert np.isfinite(data["weight"])

    opn = OPNTransformer(params=OPNParams(m=4, tau=1)).transform(series, "sine")
    assert isinstance(opn.graph, nx.DiGraph)
    assert 0 < opn.n_nodes <= 24
    for _, _, data in opn.graph.edges(data=True):
        assert "weight" in data
        assert np.isfinite(data["weight"])


def test_similarity_node_counts(series: np.ndarray) -> None:
    expected = N // 20
    corr = CorrelationNetworkTransformer(
        params=CorrelationParams(L=20, theta_corr=0.75)
    ).transform(series, "sine")
    seg = SegmentNetworkTransformer(
        params=SegmentParams(L=20, eps_percentile=20.0)
    ).transform(series, "sine")
    assert corr.n_nodes == expected
    assert seg.n_nodes == expected
    assert isinstance(corr.graph, nx.Graph)
    assert isinstance(seg.graph, nx.Graph)


def test_determinism(series: np.ndarray) -> None:
    tr = RecurrenceNetworkTransformer()
    a = tr.transform(series, "sine")
    b = tr.transform(series, "sine")
    assert nx.is_isomorphic(a.graph, b.graph) or a.graph.edges() == b.graph.edges()
    assert set(a.graph.edges()) == set(b.graph.edges())


def test_constant_series_robustness() -> None:
    const = np.ones(N, dtype=float)
    # Visibility / STN / OPN / Correlation should not crash.
    STNTransformer().transform(const, "const")
    OPNTransformer().transform(const, "const")
    CorrelationNetworkTransformer().transform(const, "const")
    SegmentNetworkTransformer().transform(const, "const")


def test_suite_and_report(tmp_path: Path, series: np.ndarray) -> None:
    items = [
        ("sine", "periodic", series),
        ("noise", "stochastic", np.random.default_rng(0).normal(size=N)),
    ]
    # Restrict methods for speed while covering families.
    cfg = TS2GSuiteConfig(
        methods=("HVG", "RN", "STN", "OPN", "Correlation", "Segment"),
        rn=RNParams(m=3, tau=1, w=5),
        knn=KNNParams(m=3, tau=1, w=5, k=3),
    )
    transformers = build_transformers(cfg)
    results = transform_suite(items, config=cfg, transformers=transformers)
    assert len(results) == 2 * 6
    save_graph_suite(results, tmp_path / "graphs")
    assert (tmp_path / "graphs" / "metadata.json").exists()
    report = TS2GReportBuilder().write(
        results,
        tmp_path / "report.html",
        transformers=transformers,
        also_latest=False,
    )
    html = report.read_text(encoding="utf-8")
    assert "HVG" in html
    assert "STN" in html
