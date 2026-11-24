#!/usr/bin/env python3
"""CLI: Phase-1 discriminability audit of descriptor matrix D."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from tsg.audit.artifacts import default_workers, resolve_lang
from tsg.audit.config import AuditConfig, apply_profile
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
        "--profile",
        choices=("dev", "paper"),
        default=None,
        help=(
            "Benchmark profile. dev: R=10, N=1000. paper: R=30, N=1500 (YAML may override). "
            "A YAML profiles block overrides these defaults for the same name."
        ),
    )
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
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel graph/descriptor workers (default: max(1, cpu_count-1)).",
    )
    parser.add_argument(
        "--lang",
        choices=("en", "es"),
        default=None,
        help="Figure language (default: en when --profile paper, else en).",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate figures from saved NPZ/JSON/pickle without recomputing graphs.",
    )
    parser.add_argument(
        "--difficulty-sweep",
        action="store_true",
        help=(
            "B3 non-saturated classification protocol from the YAML difficulty: block. "
            "Sweeps N × SNR, scores headline binary critical pairs, and, when "
            "data/oof_predictions.npz or data/features_clean.npz already exist, "
            "writes clean-panel proper scores. Does not rebuild the clean N=1500 "
            "ensemble or its cache. Copies b3_*.csv and b3_hard_08/09 into "
            "outputs/paper_bundle."
        ),
    )
    parser.add_argument(
        "--difficulty-force",
        action="store_true",
        help="Ignore difficulty feature/metric checkpoints and recompute those cells.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    config = AuditConfig.from_yaml(args.config)
    if args.profile is not None:
        config = apply_profile(config, args.profile)

    overrides: dict = {}
    if args.quick:
        overrides.update(
            {
                "profile": "quick",
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
    # Explicit size flags always win over --quick defaults.
    if args.n_realizations is not None:
        overrides["n_realizations"] = args.n_realizations
    if args.n_steps is not None:
        overrides["n_steps"] = args.n_steps
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

    lang = resolve_lang(config.profile, args.lang)
    workers = default_workers() if args.workers is None else max(1, int(args.workers))
    pipeline = AuditPipeline(config, lang=lang, n_workers=workers)

    if args.plot_only:
        logger.info("Plot-only mode output=%s lang=%s", args.output_dir, lang)
        pipeline.plot_only(args.output_dir, lang=lang)
        return

    if args.difficulty_sweep:
        from tsg.audit.hard_frontiers_difficulty import run_difficulty_protocol

        logger.info(
            "B3 difficulty sweep output=%s lang=%s workers=%d force=%s",
            args.output_dir,
            lang,
            workers,
            args.difficulty_force,
        )
        result = run_difficulty_protocol(
            config,
            config_path=args.config,
            output_dir=args.output_dir,
            n_workers=workers,
            lang=lang,
            force=bool(args.difficulty_force),
        )
        logger.info(
            "Difficulty headline N=%d SNR=%s full_rf_macro_f1=%.4f fallback=%s (%s)",
            result.operating_point.n_steps,
            result.operating_point.snr_db,
            result.operating_point.macro_f1_mean,
            result.fallback_applied,
            result.fallback_reason,
        )
        return

    logger.info(
        "Starting audit profile=%s R=%d N=%d regimes=%s methods=%s "
        "brownian_mode=%s force=%s workers=%d lang=%s",
        config.profile,
        config.n_realizations,
        config.n_steps,
        config.regimes,
        config.methods,
        config.brownian_mode,
        force_regimes,
        workers,
        lang,
    )
    result = pipeline.run(
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
