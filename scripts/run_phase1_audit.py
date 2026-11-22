#!/usr/bin/env python3
"""CLI: Phase-1 discriminability audit of descriptor matrix D."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from tsg.audit.config import AuditConfig
from tsg.audit.pipeline import AuditPipeline
from tsg.utils.logging import setup_logging

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "phase1_audit.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether descriptor matrix D separates the 6 synthetic regimes."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "reports" / "phase1_audit",
    )
    parser.add_argument(
        "--n-realizations",
        type=int,
        default=None,
        help="Override R (default from YAML, typically 30).",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Override series length N (default from YAML).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke mode: R=3, N=400, without RN (fast sanity check).",
    )
    parser.add_argument(
        "--skip-rn",
        action="store_true",
        help="Drop RN from methods (keeps kNN for proximity; much faster).",
    )
    parser.add_argument(
        "--regimes",
        type=str,
        default=None,
        help=(
            "Comma-separated regimes to FORCE recompute. "
            "The report still uses all regimes from the YAML (others loaded from cache)."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Ensemble cache directory (default: <output-dir>/ensemble_cache).",
    )
    parser.add_argument(
        "--only-forced",
        action="store_true",
        help="Restrict the report to --regimes only (legacy partial audit).",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    config = AuditConfig.from_yaml(args.config)

    overrides: dict = {}
    if args.n_realizations is not None:
        overrides["n_realizations"] = args.n_realizations
    if args.n_steps is not None:
        overrides["n_steps"] = args.n_steps
    if args.quick:
        overrides.update(
            {
                "n_realizations": 3,
                "n_steps": 400,
                "methods": (
                    "NVG",
                    "HVG",
                    "kNN",
                    "STN",
                    "OPN",
                    "Correlation",
                    "Segment",
                ),
                "families": {
                    "visibility": ("NVG", "HVG"),
                    "transition": ("STN", "OPN"),
                    "proximity": ("kNN",),
                },
            }
        )
    if overrides:
        from dataclasses import replace

        config = replace(config, **overrides)

    if args.skip_rn:
        from dataclasses import replace

        methods = tuple(m for m in config.methods if m != "RN")
        families = {
            name: tuple(m for m in meths if m != "RN")
            for name, meths in config.families.items()
        }
        config = replace(config, methods=methods, families=families)

    force_regimes: tuple[str, ...] | None = None
    if args.regimes:
        from dataclasses import replace

        wanted = tuple(r.strip() for r in args.regimes.split(",") if r.strip())
        unknown = [r for r in wanted if r not in config.regimes]
        if unknown:
            raise SystemExit(f"Unknown regimes: {unknown}. Valid: {config.regimes}")
        force_regimes = wanted
        if args.only_forced:
            crit = tuple(
                pair
                for pair in config.critical_pairs
                if pair[0] in wanted and pair[1] in wanted
            )
            config = replace(config, regimes=wanted, critical_pairs=crit)

    logger.info(
        "Starting audit R=%d N=%d regimes=%s methods=%s brownian_mode=%s force=%s",
        config.n_realizations,
        config.n_steps,
        config.regimes,
        config.methods,
        config.brownian_mode,
        force_regimes,
    )
    result = AuditPipeline(config).run(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        force_regimes=force_regimes,
    )
    logger.info(
        "Verdict=%s macro_f1_rf=%s",
        result.verdict.verdict.value,
        None
        if result.classification.full_rf() is None
        else f"{result.classification.full_rf().macro_f1_mean:.4f}",
    )


if __name__ == "__main__":
    main()
