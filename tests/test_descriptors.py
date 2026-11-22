"""Unit tests for Table 3.3 descriptor extraction."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from tsg.descriptors.base import DESCRIPTOR_IDS, prepare_graph
from tsg.descriptors.config import DescriptorConfig
from tsg.descriptors.directional import DirectionalExtractor
from tsg.descriptors.pipeline import (
    DescriptorPipeline,
    extract_descriptor_matrices,
    results_to_matrix,
    save_descriptor_outputs,
)
from tsg.graphs.registry import METHOD_IDS
from tsg.reporting.descriptors_report import DescriptorsReportBuilder


def test_descriptor_id_order() -> None:
    assert len(DESCRIPTOR_IDS) == 16
    assert DESCRIPTOR_IDS[0] == "comp"
    assert DESCRIPTOR_IDS[-1] == "self_loop_ratio"


def test_complete_graph_kn() -> None:
    n = 6
    g = nx.complete_graph(n)
    result = DescriptorPipeline().compute(g, dataset_id="K6", method="NVG")
    v = result.values
    assert v["comp"] == 1.0
    assert v["lcc"] == pytest.approx(1.0)
    assert v["c_avg"] == pytest.approx(1.0)
    assert v["e_glob"] == pytest.approx(1.0)
    assert v["l_lcc"] == pytest.approx(1.0)
    assert v["d_lcc"] == pytest.approx(1.0)
    # Combinatorial Laplacian of K_n: λ2 = n
    assert v["lambda2_g"] == pytest.approx(n, rel=1e-6)
    assert v["lambda2_lcc"] == pytest.approx(n, rel=1e-6)
    assert v["i_dir"] == 0.0
    assert v["self_loop_ratio"] == 0.0


def test_two_components() -> None:
    g = nx.disjoint_union(nx.complete_graph(4), nx.complete_graph(4))
    result = DescriptorPipeline().compute(g, dataset_id="2K4", method="HVG")
    v = result.values
    assert v["comp"] == 2.0
    assert v["lcc"] == pytest.approx(0.5)
    assert v["lambda2_g"] == pytest.approx(0.0, abs=1e-10)
    # λ2(LCC) for K4 combinatorial = 4; tilde = 0.25 * 4 = 1
    assert v["lambda2_lcc"] == pytest.approx(4.0, rel=1e-6)
    assert v["lambda2_tilde"] == pytest.approx(0.25 * v["lambda2_lcc"], rel=1e-6)


def test_directed_self_loops_and_idir() -> None:
    dig = nx.DiGraph()
    dig.add_weighted_edges_from(
        [
            (0, 0, 2.0),
            (0, 1, 3.0),
            (1, 0, 1.0),
            (1, 1, 4.0),
        ]
    )
    prepared = prepare_graph(dig)
    direction = DirectionalExtractor().extract(prepared, DescriptorConfig())
    total = 2 + 3 + 1 + 4
    assert direction["self_loop_ratio"] == pytest.approx((2 + 4) / total)
    w = nx.to_numpy_array(dig, nodelist=[0, 1], weight="weight")
    expected_idir = np.linalg.norm(w - w.T, ord="fro") / np.linalg.norm(w + w.T, ord="fro")
    assert direction["i_dir"] == pytest.approx(expected_idir)

    result = DescriptorPipeline().compute(dig, dataset_id="dir", method="STN")
    # Self-loops must not contaminate undirected clustering support.
    assert result.values["c_avg"] >= 0.0
    assert np.isfinite(result.values["lambda_max"])


def test_empty_and_regular_edge_cases() -> None:
    empty = nx.empty_graph(5)
    v = DescriptorPipeline().compute(empty, dataset_id="empty", method="RN").values
    assert v["c_avg"] == 0.0
    assert v["e_glob"] == 0.0
    assert np.isnan(v["assortativity_r"])
    assert v["cv_k"] == 0.0
    assert v["lambda2_g"] == 0.0

    cycle = nx.cycle_graph(8)  # 2-regular
    v2 = DescriptorPipeline().compute(cycle, dataset_id="C8", method="kNN").values
    assert np.isnan(v2["assortativity_r"])
    assert v2["cv_k"] == 0.0
    assert v2["skew_k"] == 0.0


def test_matrix_shape_and_report(tmp_path: Path) -> None:
    records = []
    for dataset_id in ("sine", "noise"):
        for method in METHOD_IDS:
            if method in {"STN", "OPN"}:
                g = nx.DiGraph()
                g.add_weighted_edges_from([(0, 1, 1.0), (1, 1, 2.0), (1, 0, 1.0)])
            else:
                g = nx.path_graph(12)
            records.append(
                {
                    "dataset_id": dataset_id,
                    "method": method,
                    "regime": "periodic" if dataset_id == "sine" else "stochastic",
                    "family": "test",
                    "graph": g,
                }
            )
    matrices, results = extract_descriptor_matrices(records)
    assert set(matrices) == {"sine", "noise"}
    for matrix in matrices.values():
        assert matrix.shape == (16, 8)
        assert list(matrix.index) == list(DESCRIPTOR_IDS)
        assert list(matrix.columns) == list(METHOD_IDS)

    save_descriptor_outputs(
        matrices, results, tmp_path / "desc", config=DescriptorConfig()
    )
    assert (tmp_path / "desc" / "sine_matrix.csv").exists()
    assert (tmp_path / "desc" / "all_descriptors_long.csv").exists()

    report = DescriptorsReportBuilder().write(
        matrices,
        results,
        tmp_path / "report.html",
        regimes={"sine": "periodic", "noise": "stochastic"},
        also_latest=True,
    )
    html = report.read_text(encoding="utf-8")
    assert "sine" in html
    assert (tmp_path / "latest.html").exists()
