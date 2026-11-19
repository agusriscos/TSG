#!/usr/bin/env python3
"""CLI: build the 6×8 TS2G graph suite from synthetic Phase-1 series."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

from tsg.graphs.config import TS2GSuiteConfig
from tsg.graphs.io import load_series_suite, save_graph_suite
from tsg.graphs.registry import build_transformers, transform_suite
from tsg.reporting.ts2g_report import TS2GReportBuilder
from tsg.utils.logging import setup_logging

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "ts2g_mapping_phase1.yaml"
DEFAULT_DATA = ROOT / "outputs" / "data" / "synthetic_phase1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map synthetic series to graphs with the 8 TS2G methods."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "graphs" / "synthetic_phase1",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=ROOT / "outputs" / "reports" / "ts2g_mapping",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    config = TS2GSuiteConfig.from_yaml(args.config)
    transformers = build_transformers(config)

    series_items = load_series_suite(args.data_dir)
    if len(series_items) != 6:
        logger.warning("Expected 6 series, found %d in %s", len(series_items), args.data_dir)

    results = transform_suite(series_items, config=config, transformers=transformers)
    save_graph_suite(results, args.output_dir)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = args.report_dir / f"{stamp}_summary.html"
    TS2GReportBuilder().write(results, report_path, transformers=transformers)
    logger.info(
        "Done. graphs=%d output=%s report=%s",
        len(results),
        args.output_dir,
        report_path,
    )


if __name__ == "__main__":
    main()
