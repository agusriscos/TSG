#!/usr/bin/env python3
"""CLI: Bloque 1 — PID, O-information, exact Shapley and the Pareto front."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from tsg.audit.artifacts import default_workers, resolve_lang
from tsg.audit.config import AuditConfig, apply_profile
from tsg.audit.information import InformationAuditPipeline
from tsg.reporting.information_report import summary_text
from tsg.utils.logging import setup_logging

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "phase2_hard_frontiers.yaml"
DEFAULT_CACHE = ROOT / "outputs" / "reports" / "phase2_hard_frontiers" / "ensemble_cache"
DEFAULT_OUTPUT = ROOT / "outputs" / "reports" / "phase2_information"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Partial information, O-information and the cost-accuracy Pareto front "
            "on the phase-2 hard-frontier panel. Reuses the ensemble cache."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--profile",
        choices=("dev", "paper"),
        default="dev",
        help=(
            "dev: R=10, N=1000, cost at N=1000. "
            "paper: R=30, N=1500, cost at N=500, 1000 and 1500."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE,
        help="Phase-2 ensemble cache (default: phase2_hard_frontiers/ensemble_cache).",
    )
    parser.add_argument("--n-realizations", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel workers (default: max(1, cpu_count-1)).",
    )
    parser.add_argument(
        "--lang",
        choices=("en", "es"),
        default=None,
        help="Figure language (default: en when --profile paper).",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate figures from saved JSON/NPZ without recomputing graphs.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    config = apply_profile(AuditConfig.from_yaml(args.config), args.profile)
    if args.n_realizations is not None:
        config = replace(config, n_realizations=args.n_realizations)
    if args.n_steps is not None:
        config = replace(config, n_steps=args.n_steps)
    lang = resolve_lang(config.profile, args.lang)
    workers = default_workers() if args.workers is None else max(1, int(args.workers))
    pipeline = InformationAuditPipeline(
        config,
        cache_dir=args.cache_dir,
        lang=lang,
        n_workers=workers,
    )
    if args.plot_only:
        logger.info("Plot-only information audit output=%s lang=%s", args.output_dir, lang)
        pipeline.plot_only(args.output_dir, lang=lang)
        return
    logger.info(
        "Starting information audit profile=%s R=%d N=%d cache=%s workers=%d lang=%s",
        config.profile,
        config.n_realizations,
        config.n_steps,
        args.cache_dir,
        workers,
        lang,
    )
    result = pipeline.run(args.output_dir)
    logger.info("\n%s", summary_text(result))


if __name__ == "__main__":
    main()
