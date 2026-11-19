#!/usr/bin/env python3
"""CLI: generate Phase-1 synthetic multiregime suite + HTML report."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

from tsg.generators.config import GeneratorSuiteConfig
from tsg.generators.registry import generate_suite
from tsg.reporting.html_report import HTMLReportBuilder
from tsg.utils.io import save_suite
from tsg.utils.logging import setup_logging

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "synthetic_suite_phase1.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 1 controlled series per Dataset ID (Table 3.1)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="YAML suite configuration path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "data" / "synthetic_phase1",
        help="Directory for CSV series and metadata.json",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=ROOT / "outputs" / "reports" / "synthetic_suite",
        help="Directory for HTML validation report",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    config = GeneratorSuiteConfig.from_yaml(args.config)
    g = config.global_params

    logger.info(
        "Phase-1 suite: n_steps=%d seed=%d n_samples=%d",
        g.n_steps,
        g.seed,
        g.n_samples,
    )
    series_list = generate_suite(config)
    save_suite(series_list, args.output_dir)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = args.report_dir / f"{stamp}_summary.html"
    HTMLReportBuilder().write(
        series_list,
        report_path,
        seed=g.seed,
        n_steps=g.n_steps,
        config_summary={"n_datasets": len(series_list), "config": str(args.config)},
    )
    logger.info("Done. Data=%s Report=%s", args.output_dir, report_path)


if __name__ == "__main__":
    main()
