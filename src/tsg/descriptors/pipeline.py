"""Orchestrate the 16×8 descriptor matrices for the synthetic suite."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Sequence

import networkx as nx
import numpy as np
import pandas as pd

from tsg.descriptors.base import (
    DESCRIPTOR_IDS,
    BaseDescriptorExtractor,
    DescriptorResult,
    prepare_graph,
)
from tsg.descriptors.config import DescriptorConfig
from tsg.descriptors.connectivity import ConnectivityExtractor
from tsg.descriptors.degree import DegreeExtractor
from tsg.descriptors.directional import DirectionalExtractor
from tsg.descriptors.paths import PathExtractor
from tsg.descriptors.spectral import SpectralExtractor
from tsg.graphs.registry import METHOD_IDS

logger = logging.getLogger(__name__)


class DescriptorPipeline:
    """Compute the full Table 3.3 descriptor vector for one graph."""

    def __init__(
        self,
        config: DescriptorConfig | None = None,
        extractors: Sequence[BaseDescriptorExtractor] | None = None,
    ) -> None:
        self.config = config or DescriptorConfig()
        self.extractors: list[BaseDescriptorExtractor] = list(
            extractors
            if extractors is not None
            else [
                ConnectivityExtractor(),
                DegreeExtractor(),
                PathExtractor(),
                SpectralExtractor(),
                DirectionalExtractor(),
            ]
        )

    def compute(
        self,
        graph: nx.Graph | nx.DiGraph,
        *,
        dataset_id: str,
        method: str,
        regime: str = "",
        family: str = "",
    ) -> DescriptorResult:
        """Extract all 16 descriptors for a single graph."""
        t0 = time.perf_counter()
        prepared = prepare_graph(graph)
        values: dict[str, float] = {}
        for extractor in self.extractors:
            values.update(extractor.extract(prepared, self.config))
        missing = [k for k in DESCRIPTOR_IDS if k not in values]
        if missing:
            raise RuntimeError(f"Missing descriptors: {missing}")
        elapsed = time.perf_counter() - t0
        return DescriptorResult(
            dataset_id=dataset_id,
            regime=regime,
            method=method,
            family=family,
            values={k: float(values[k]) for k in DESCRIPTOR_IDS},
            build_time_sec=elapsed,
            extras={
                "n_nodes": prepared.n_nodes,
                "n_edges_undirected": prepared.n_edges_undirected,
                "laplacian": self.config.laplacian,
                "entropy_base": self.config.entropy_base,
            },
        )


def results_to_matrix(
    results: Sequence[DescriptorResult],
    *,
    method_order: Sequence[str] = METHOD_IDS,
) -> pd.DataFrame:
    """Build one ``16 × 8`` descriptor matrix for a single dataset."""
    by_method = {r.method: r for r in results}
    missing = [m for m in method_order if m not in by_method]
    if missing:
        raise KeyError(f"Missing methods for matrix: {missing}")
    data = {
        method: [by_method[method].values[d] for d in DESCRIPTOR_IDS]
        for method in method_order
    }
    return pd.DataFrame(data, index=list(DESCRIPTOR_IDS))


def results_to_long_table(
    results: Sequence[DescriptorResult],
) -> pd.DataFrame:
    """Long-format table with metadata + 16 descriptor columns."""
    rows: list[dict[str, Any]] = []
    for r in results:
        row: dict[str, Any] = {
            "dataset_id": r.dataset_id,
            "regime": r.regime,
            "method": r.method,
            "family": r.family,
            "build_time_sec": r.build_time_sec,
        }
        row.update(r.values)
        rows.append(row)
    return pd.DataFrame(rows)


def extract_descriptor_matrices(
    graph_records: Sequence[dict[str, Any]],
    config: DescriptorConfig | None = None,
) -> tuple[dict[str, pd.DataFrame], list[DescriptorResult]]:
    """Compute descriptors for all graph records and group matrices by dataset.

    Each record must contain: ``dataset_id``, ``method``, ``graph``, and
    optionally ``regime`` / ``family``.
    """
    pipeline = DescriptorPipeline(config=config)
    results: list[DescriptorResult] = []
    for rec in graph_records:
        logger.info("Descriptors %s ← %s", rec["method"], rec["dataset_id"])
        results.append(
            pipeline.compute(
                rec["graph"],
                dataset_id=rec["dataset_id"],
                method=rec["method"],
                regime=rec.get("regime", ""),
                family=rec.get("family", ""),
            )
        )

    matrices: dict[str, pd.DataFrame] = {}
    dataset_ids = list(dict.fromkeys(r.dataset_id for r in results))
    for dataset_id in dataset_ids:
        subset = [r for r in results if r.dataset_id == dataset_id]
        matrices[dataset_id] = results_to_matrix(subset)
    return matrices, results


def save_descriptor_outputs(
    matrices: dict[str, pd.DataFrame],
    results: Sequence[DescriptorResult],
    output_dir: Path | str,
    *,
    config: DescriptorConfig,
) -> Path:
    """Persist 6 matrices, long CSV and metadata.json."""
    import json

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for dataset_id, matrix in matrices.items():
        path = root / f"{dataset_id}_matrix.csv"
        matrix.to_csv(path, float_format="%.10g")
        logger.info("Wrote %s shape=%s", path, matrix.shape)

    long_df = results_to_long_table(results)
    long_path = root / "all_descriptors_long.csv"
    long_df.to_csv(long_path, index=False, float_format="%.10g")

    nan_count = int(np.isnan(long_df[list(DESCRIPTOR_IDS)].to_numpy(dtype=float)).sum())
    meta = {
        "n_graphs": len(results),
        "n_matrices": len(matrices),
        "descriptor_ids": list(DESCRIPTOR_IDS),
        "methods": list(METHOD_IDS),
        "laplacian": config.laplacian,
        "entropy_base": config.entropy_base,
        "nan_count": nan_count,
        "total_build_time_sec": float(sum(r.build_time_sec for r in results)),
        "per_graph": [
            {
                "dataset_id": r.dataset_id,
                "method": r.method,
                "build_time_sec": r.build_time_sec,
                "n_nodes": r.extras.get("n_nodes"),
            }
            for r in results
        ],
    }
    meta_path = root / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Wrote descriptor suite to %s", root)
    return root
