#!/usr/bin/env python3
"""CLI: Bloque-4 invariance audit under signal transforms."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from tsg.audit.artifacts import default_workers, resolve_lang
from tsg.audit.invariance import InvarianceAuditConfig, InvarianceAuditPipeline
from tsg.utils.logging import setup_logging

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "phase2_invariance.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit TS2G family invariances under affine / monotone / outlier / "
            "quantize / downsample / SNR transforms (descriptor stability + OOD F1)."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--profile",
        choices=("dev", "paper"),
        default="dev",
        help="Benchmark profile. dev: R=10, N=1000, 6 regimes. paper: R=30, N=1000, 12 regimes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "reports" / "phase2_invariance",
    )
    parser.add_argument("--n-realizations", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke mode: R=3, N=400, drop RN (fast sanity check).",
    )
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
        help="Regenerate figures from saved artifacts without recomputing graphs.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    config = InvarianceAuditConfig.from_yaml(args.config)
    config = config.with_profile(args.profile)

    audit = config.audit
    if args.quick:
        methods = tuple(m for m in audit.methods if m != "RN")
        families = {
            name: tuple(m for m in meths if m != "RN")
            for name, meths in audit.families.items()
        }
        audit = replace(
            audit,
            profile="quick",
            n_realizations=3,
            n_steps=400,
            methods=methods,
            families=families,
        )
    if args.n_realizations is not None:
        audit = replace(audit, n_realizations=args.n_realizations)
    if args.n_steps is not None:
        audit = replace(audit, n_steps=args.n_steps)
    config = replace(config, audit=audit)

    lang = resolve_lang(config.audit.profile, args.lang)
    workers = default_workers() if args.workers is None else max(1, int(args.workers))
    pipeline = InvarianceAuditPipeline(config, lang=lang, n_workers=workers)

    if args.plot_only:
        logger.info("Plot-only invariance audit output=%s lang=%s", args.output_dir, lang)
        pipeline.plot_only(args.output_dir, lang=lang)
        return

    logger.info(
        "Starting invariance audit profile=%s R=%d N=%d regimes=%s transforms=%s "
        "workers=%d lang=%s",
        config.audit.profile,
        config.audit.n_realizations,
        config.audit.n_steps,
        config.audit.regimes,
        config.transforms,
        workers,
        lang,
    )
    pipeline.run(output_dir=args.output_dir)
    logger.info("Done. Artifacts in %s", args.output_dir)


if __name__ == "__main__":
    main()
