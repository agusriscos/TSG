#!/usr/bin/env python3
"""CLI: Bloque-2 surrogate audit (Shuffled / FT / IAAFT) + dHVG."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from tsg.audit.artifacts import default_workers, resolve_lang
from tsg.audit.surrogates import SurrogateAuditConfig, SurrogateAuditPipeline
from tsg.utils.logging import setup_logging

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "phase2_surrogates.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit temporal sensitivity (Δ_T) and genuine nonlinearity (Δ_NL) "
            "via shuffled / FT / IAAFT surrogates, including directed dHVG."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--profile",
        choices=("dev", "paper"),
        default="dev",
        help="Benchmark profile. dev: R=10, N=1000. paper: R=30, N=1000.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "reports" / "phase2_surrogates",
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
    config = SurrogateAuditConfig.from_yaml(args.config)
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
    # Explicit size flags always win over --quick defaults.
    if args.n_realizations is not None:
        audit = replace(audit, n_realizations=args.n_realizations)
    if args.n_steps is not None:
        audit = replace(audit, n_steps=args.n_steps)
    config = replace(config, audit=audit)

    lang = resolve_lang(config.audit.profile, args.lang)
    workers = default_workers() if args.workers is None else max(1, int(args.workers))
    pipeline = SurrogateAuditPipeline(config, lang=lang, n_workers=workers)

    if args.plot_only:
        logger.info("Plot-only surrogate audit output=%s lang=%s", args.output_dir, lang)
        pipeline.plot_only(args.output_dir, lang=lang)
        return

    logger.info(
        "Starting surrogate audit profile=%s R=%d N=%d regimes=%s methods=%s "
        "workers=%d lang=%s",
        config.audit.profile,
        config.audit.n_realizations,
        config.audit.n_steps,
        config.audit.regimes,
        config.audit.methods,
        workers,
        lang,
    )
    result = pipeline.run(output_dir=args.output_dir)

    multi = result.nonlinearity_multiclass.get("original_shuffled_ft_iaaft", {})
    logger.info(
        "Done. full RF Macro-F1 (surrogate-type multiclass)=%.3f → %s",
        multi.get("full", {}).get("random_forest", {}).get("macro_f1_mean", float("nan")),
        args.output_dir,
    )


if __name__ == "__main__":
    main()
