#!/usr/bin/env python3
"""CLI: extract 16×8 descriptor matrices from the TS2G graph suite."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx

from tsg.descriptors.config import DescriptorConfig
from tsg.descriptors.pipeline import extract_descriptor_matrices, save_descriptor_outputs
from tsg.graphs.config import TS2GSuiteConfig
from tsg.graphs.io import load_series_suite
from tsg.graphs.registry import build_transformers
from tsg.reporting.descriptors_report import DescriptorsReportBuilder
from tsg.utils.logging import setup_logging

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "descriptors_phase1.yaml"
DEFAULT_GRAPHS = ROOT / "outputs" / "graphs" / "synthetic_phase1"
DEFAULT_DATA = ROOT / "outputs" / "data" / "synthetic_phase1"
TRANSITION_METHODS = {"STN", "OPN"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Table 3.3 descriptor matrices (16×8) per synthetic series."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--graphs-dir", type=Path, default=DEFAULT_GRAPHS)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "descriptors" / "synthetic_phase1",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=ROOT / "outputs" / "reports" / "descriptors",
    )
    parser.add_argument(
        "--refresh-transitions",
        action="store_true",
        default=True,
        help="Rebuild STN/OPN from series to restore self-loops (default: on).",
    )
    parser.add_argument(
        "--no-refresh-transitions",
        action="store_false",
        dest="refresh_transitions",
    )
    return parser.parse_args()


def _load_graph_records(
    graphs_dir: Path,
    data_dir: Path,
    *,
    refresh_transitions: bool,
) -> tuple[list[dict], dict[str, str]]:
    meta_path = graphs_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing graph metadata: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    regimes = {row["dataset_id"]: row.get("regime", "") for row in meta}

    transition_graphs: dict[tuple[str, str], nx.Graph | nx.DiGraph] = {}
    if refresh_transitions:
        logger.info("Refreshing STN/OPN graphs with self-loops from series")
        series_items = {ds: (reg, vals) for ds, reg, vals in load_series_suite(data_dir)}
        ts2g_cfg = TS2GSuiteConfig(methods=("STN", "OPN"))
        transformers = {t.method: t for t in build_transformers(ts2g_cfg)}
        for dataset_id, (regime, values) in series_items.items():
            regimes.setdefault(dataset_id, regime)
            for method, tr in transformers.items():
                result = tr.transform(values, dataset_id, regime=regime)
                transition_graphs[(dataset_id, method)] = result.graph
                out = graphs_dir / dataset_id / f"{method}.graphml"
                out.parent.mkdir(parents=True, exist_ok=True)
                nx.write_graphml(result.graph, out)

    records: list[dict] = []
    for row in meta:
        dataset_id = row["dataset_id"]
        method = row["method"]
        key = (dataset_id, method)
        if key in transition_graphs:
            graph = transition_graphs[key]
        else:
            graph = nx.read_graphml(graphs_dir / row["path"])
            # GraphML stores node ids as strings; cast numeric ids back when possible.
            if not graph.is_directed():
                mapping = {}
                for n in list(graph.nodes()):
                    try:
                        mapping[n] = int(n)
                    except (TypeError, ValueError):
                        mapping[n] = n
                graph = nx.relabel_nodes(graph, mapping)
        records.append(
            {
                "dataset_id": dataset_id,
                "method": method,
                "regime": row.get("regime", regimes.get(dataset_id, "")),
                "family": row.get("family", ""),
                "graph": graph,
            }
        )
    return records, regimes


def main() -> None:
    setup_logging()
    args = parse_args()
    config = DescriptorConfig.from_yaml(args.config)
    records, regimes = _load_graph_records(
        args.graphs_dir,
        args.data_dir,
        refresh_transitions=args.refresh_transitions,
    )
    matrices, results = extract_descriptor_matrices(records, config=config)
    save_descriptor_outputs(matrices, results, args.output_dir, config=config)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = args.report_dir / f"{stamp}_summary.html"
    DescriptorsReportBuilder().write(
        matrices,
        results,
        report_path,
        regimes=regimes,
    )
    logger.info(
        "Done. matrices=%d graphs=%d report=%s",
        len(matrices),
        len(results),
        report_path,
    )


if __name__ == "__main__":
    main()
