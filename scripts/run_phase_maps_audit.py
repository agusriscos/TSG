#!/usr/bin/env python3
"""CLI: Bloque-5 N × SNR phase maps + topological vs spectral failure modes."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from tsg.audit.artifacts import default_workers, resolve_lang
from tsg.audit.phase_maps import PhaseMapsAuditConfig, PhaseMapsAuditPipeline
from tsg.utils.logging import setup_logging

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "phase2_phase_maps.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep Macro-F1 over the (N, SNR_dB) plane for TS2G families and "
            "contrast topological_only vs spectral_only failure modes."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--profile",
        choices=("dev", "paper"),
        default="dev",
        help=(
            "Benchmark profile. "
            "dev: N∈{200,500,1000}, SNR∈{∞,20,10,5,0}, R=8, 6 regimes. "
            "paper: N∈{100,250,500,1000}, SNR∈{∞,20,10,5,0,-5}, R=20, 8 regimes."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "reports" / "phase2_phase_maps",
    )
    parser.add_argument("--n-realizations", type=int, default=None)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke mode: R=3, N∈{200,500}, SNR∈{∞,10,0}, drop RN.",
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
        help="Regenerate figures from saved metrics JSON without recomputing graphs.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    config = PhaseMapsAuditConfig.from_yaml(args.config)
    config = config.with_profile(args.profile)

    audit = config.audit
    n_grid = config.n_grid
    snr_grid = config.snr_db_grid
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
            methods=methods,
            families=families,
        )
        n_grid = (200, 500)
        snr_grid = (None, 10.0, 0.0)
    if args.n_realizations is not None:
        audit = replace(audit, n_realizations=args.n_realizations)
    config = replace(config, audit=audit, n_grid=n_grid, snr_db_grid=snr_grid)

    lang = resolve_lang(config.audit.profile, args.lang)
    workers = default_workers() if args.workers is None else max(1, int(args.workers))
    pipeline = PhaseMapsAuditPipeline(config, lang=lang, n_workers=workers)

    if args.plot_only:
        logger.info("Plot-only phase-maps audit output=%s lang=%s", args.output_dir, lang)
        pipeline.plot_only(args.output_dir, lang=lang)
        return

    logger.info(
        "Starting phase-maps audit profile=%s R=%d N_grid=%s SNR=%s regimes=%s "
        "workers=%d lang=%s",
        config.audit.profile,
        config.audit.n_realizations,
        config.n_grid,
        config.snr_db_grid,
        config.audit.regimes,
        workers,
        lang,
    )
    pipeline.run(output_dir=args.output_dir)
    logger.info("Done. Artifacts in %s", args.output_dir)


if __name__ == "__main__":
    main()
