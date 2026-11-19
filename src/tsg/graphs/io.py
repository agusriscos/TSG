"""Persist and reload TS2G graph artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from tsg.graphs.base import GraphResult

logger = logging.getLogger(__name__)


def load_series_suite(
    data_dir: Path | str,
) -> list[tuple[str, str, np.ndarray]]:
    """Load synthetic Phase-1 CSVs using ``metadata.json`` when available."""
    root = Path(data_dir)
    meta_path = root / "metadata.json"
    items: list[tuple[str, str, np.ndarray]] = []
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for row in meta:
            csv_name = row.get("csv", f"{row['dataset_id']}.csv")
            frame = pd.read_csv(root / csv_name)
            values = frame["value"].to_numpy(dtype=float)
            items.append((row["dataset_id"], row.get("regime", ""), values))
        return items

    for csv_path in sorted(root.glob("*.csv")):
        frame = pd.read_csv(csv_path)
        values = frame["value"].to_numpy(dtype=float)
        items.append((csv_path.stem, "", values))
    return items


def save_graph_result(result: GraphResult, output_dir: Path | str) -> Path:
    """Write one graph as GraphML under ``<dataset_id>/<method>.graphml``."""
    root = Path(output_dir)
    dataset_dir = root / result.dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    path = dataset_dir / f"{result.method}.graphml"
    # GraphML requires hashable scalar node/edge attrs; convert numpy scalars.
    g = result.graph.copy()
    nx.write_graphml(g, path)
    logger.info(
        "Wrote %s (n=%d, e=%d, %.3fs)",
        path,
        result.n_nodes,
        result.n_edges,
        result.build_time_sec,
    )
    return path


def save_graph_suite(
    results: list[GraphResult],
    output_dir: Path | str,
) -> Path:
    """Persist all graphs and a global ``metadata.json`` index."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    index: list[dict] = []
    for result in results:
        path = save_graph_result(result, root)
        rel = str(path.relative_to(root))
        index.append(result.to_metadata(path=rel))
    meta_path = root / "metadata.json"
    meta_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    logger.info("Wrote graph suite metadata (%d graphs) to %s", len(index), meta_path)
    return root
