#!/usr/bin/env python3
"""CLI: RQ1 advanced PID (BROJA) + Ω bootstrap + SII + optional stress.

PID/Ω reuse ``features_clean.npz``. SII reuses ``information_metrics.json``.
Stress regenerates raw series before TS2G (SNR / short-N).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from tsg.audit.o_information_bootstrap import OmegaBootstrapConfig
from tsg.audit.pid_advanced import AdvancedPIDConfig
from tsg.audit.rq1_pipeline import RQ1InformationPipeline, RQ1PipelineConfig
from tsg.audit.stress_eval import StressConfig
from tsg.utils.logging import setup_logging

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = (
    ROOT / "outputs" / "reports" / "phase2_information" / "data" / "features_clean.npz"
)
DEFAULT_METRICS = ROOT / "outputs" / "reports" / "phase2_information" / "information_metrics.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "reports" / "rq1_information_advanced"
DEFAULT_AUDIT_YAML = ROOT / "configs" / "phase2_hard_frontiers.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "RQ1: BROJA PID by dynamic group, Ω bootstrap, SII from subset_scores, "
            "optional raw-series stress (Macro-F1 + log-loss)."
        )
    )
    parser.add_argument("--features-npz", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--mmi-metrics-json",
        type=Path,
        default=DEFAULT_METRICS,
        help="information_metrics.json (MMI pid_pairs + subset_scores).",
    )
    parser.add_argument(
        "--subset-scores-json",
        type=Path,
        default=None,
        help="Override JSON for SII (defaults to --mmi-metrics-json).",
    )
    parser.add_argument(
        "--estimator",
        choices=("broja", "ccs", "mmi_discrete"),
        default="broja",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=3,
        help="Quantile bins per family PCA score (default 3; hard cap 4).",
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=1000,
        help="Column-wise permutations for p_perm (H0: Ω=0).",
    )
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lang", choices=("en", "es"), default="en")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--skip-sii",
        action="store_true",
        help="Skip SII stage (on by default).",
    )
    parser.add_argument(
        "--run-stress",
        action="store_true",
        help="Regenerate raw series with SNR / short-N before TS2G.",
    )
    parser.add_argument(
        "--stress-realizations",
        type=int,
        default=None,
        help="Override R for stress (default: audit profile / quick→2).",
    )
    parser.add_argument(
        "--audit-config",
        type=Path,
        default=DEFAULT_AUDIT_YAML,
        help="Audit YAML used by the stress generator.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke: B=50, 3 bins, stress R=2 + RF trees=40.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    if not args.features_npz.exists():
        raise SystemExit(
            f"Features NPZ not found: {args.features_npz}. "
            "Run scripts/run_information_audit.py first."
        )

    metrics_json = args.subset_scores_json or args.mmi_metrics_json
    n_bootstrap = 50 if args.quick else int(args.n_bootstrap)
    n_permutations = 50 if args.quick else int(args.n_permutations)
    n_bins = min(3, int(args.n_bins)) if args.quick else int(args.n_bins)
    stress_r = (
        2
        if args.quick and args.stress_realizations is None
        else args.stress_realizations
    )
    # Quick stress: 4 representative regimes × R=2 keeps wall-time manageable
    # while still exercising raw-series SNR / short-N → TS2G → metrics.
    quick_regimes = (
        "white_noise",
        "lorenz",
        "logistic",
        "regime_switching",
    )
    stress = StressConfig(
        seed=int(args.seed) or 42,
        n_realizations=stress_r,
        rf_estimators=40 if args.quick else None,
        snr_db_levels=(10.0,) if args.quick else (10.0, 5.0),
        short_n_steps=(200,),
        regimes=quick_regimes if args.quick else None,
        model_names=("random_forest", "logistic"),
    )

    config = RQ1PipelineConfig(
        features_npz=args.features_npz,
        output_dir=args.output_dir,
        pid=AdvancedPIDConfig(
            estimator=args.estimator,
            n_bins=n_bins,
            max_bins=4,
            seed=int(args.seed),
        ),
        omega=OmegaBootstrapConfig(
            n_bootstrap=n_bootstrap,
            n_permutations=n_permutations,
            ci_level=float(args.ci_level),
            seed=int(args.seed),
        ),
        lang=args.lang,
        figure_dpi=int(args.dpi),
        compute_sii=not bool(args.skip_sii),
        run_stress=bool(args.run_stress),
        subset_scores_json=metrics_json,
        mmi_metrics_json=args.mmi_metrics_json,
        stress=stress,
        audit_config_yaml=args.audit_config,
    )
    logger.info(
        "RQ1 features=%s estimator=%s bins=%d B=%d n_perm=%d sii=%s stress=%s out=%s",
        config.features_npz,
        config.pid.estimator,
        config.pid.n_bins,
        config.omega.n_bootstrap,
        config.omega.n_permutations,
        config.compute_sii,
        config.run_stress,
        config.output_dir,
    )
    result = RQ1InformationPipeline(config).run()
    global_omega = next(r for r in result.omega_results if r.scope == "global")
    logger.info(
        "Done. PID=%d one_vs_rest=%d | Ω_global=%+.4f CI[%.4f, %.4f] "
        "p_boot=%.4g p_perm=%.4g p_holm=%.4g | coverage=%s",
        len(result.pid_atoms),
        len(result.one_vs_rest),
        global_omega.omega_bits,
        global_omega.ci_low,
        global_omega.ci_high,
        global_omega.p_value,
        global_omega.p_perm,
        global_omega.p_holm,
        result.group_coverage.get("counts_by_group"),
    )


if __name__ == "__main__":
    main()
