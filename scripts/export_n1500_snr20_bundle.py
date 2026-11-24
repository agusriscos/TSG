#!/usr/bin/env python3
"""Export graphs/tables/resources for the N=1500, SNR=20 dB operating point.

Writes a self-contained tree under ``outputs/n1500_snr20_bundle/`` that is
separate from the curated ``outputs/paper_bundle/`` and the mixed report dirs.

Blocks:
  B3 hard frontiers — rebuilt from the difficulty-cell ensemble cache
  B1 information    — same preloaded records (SNR=20 panel)
  B5 phase maps     — single-cell grid N={1500} × SNR={20}
  B2 / B4           — STATUS note (pipelines lack observational SNR=20 hooks)

Usage
-----
  .venv/bin/python scripts/export_n1500_snr20_bundle.py
  .venv/bin/python scripts/export_n1500_snr20_bundle.py --skip-phase-maps
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from tsg.audit.artifacts import atomic_write_json, atomic_write_text, default_workers
from tsg.audit.config import AuditConfig, apply_profile
from tsg.audit.ensemble import load_generator_suite_config, versioned_cache_dir
from tsg.audit.hard_frontiers_difficulty import (
    GridPoint,
    _generate_cell_records,
    _load_checkpoint_frame,
    evaluate_views,
    pairwise_f1_differences,
    plot_binary_critical_pairs,
    plot_difficulty_curve,
    write_headline_ablation_artifacts,
)
from tsg.audit.health import diagnose_and_clean
from tsg.audit.information import InformationAuditPipeline
from tsg.audit.phase_maps import PhaseMapsAuditConfig, PhaseMapsAuditPipeline
from tsg.audit.pipeline import AuditPipeline
from tsg.descriptors.config import DescriptorConfig
from tsg.utils.logging import setup_logging

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs" / "n1500_snr20_bundle"
HF_REPORT = ROOT / "outputs" / "reports" / "phase2_hard_frontiers"
HF_CONFIG = ROOT / "configs" / "phase2_hard_frontiers.yaml"
PM_CONFIG = ROOT / "configs" / "phase2_phase_maps.yaml"

N_STEPS = 1500
SNR_DB = 20.0
DIFFICULTY_R = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package N=1500 / SNR=20 dB artifacts into a separate output tree."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--source-hf",
        type=Path,
        default=HF_REPORT,
        help="Existing phase2_hard_frontiers report (difficulty cache + curve).",
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--skip-phase-maps",
        action="store_true",
        help="Do not recompute the B5 single cell (still writes a STATUS note).",
    )
    parser.add_argument(
        "--skip-information",
        action="store_true",
        help="Skip B1 information audit on the SNR=20 records.",
    )
    parser.add_argument(
        "--skip-hard-frontiers",
        action="store_true",
        help="Skip rebuilding the B3 audit panel (still copies difficulty slices).",
    )
    return parser.parse_args()


def _operating_point_from_curve(curve: pd.DataFrame) -> GridPoint:
    cell = curve[
        (curve["n_steps"] == N_STEPS)
        & (curve["snr_db"].astype(float) == float(SNR_DB))
        & (curve["view"] == "full")
        & (curve["model"] == "random_forest")
    ]
    if cell.empty:
        raise RuntimeError(
            f"difficulty_curve.csv has no full/RF row for N={N_STEPS} SNR={SNR_DB}"
        )
    row = cell.iloc[0]
    return GridPoint(
        n_steps=N_STEPS,
        snr_db=float(SNR_DB),
        macro_f1_mean=float(row["macro_f1_mean"]),
        macro_f1_std=float(row["macro_f1_std"]),
        ci_low=float(row["macro_f1_ci_low"]),
        ci_high=float(row["macro_f1_ci_high"]),
    )


def _write_status(path: Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, f"# {title}\n\n{body.rstrip()}\n")


def _flat_publish(reports: Path, bundle: Path) -> None:
    """Mirror run_paper_suite packaging: PNG figures + CSV tables with block prefixes."""
    fig_dst = bundle / "figures"
    tab_dst = bundle / "tables"
    fig_dst.mkdir(parents=True, exist_ok=True)
    tab_dst.mkdir(parents=True, exist_ok=True)
    mapping = {
        "phase2_hard_frontiers": "b3_hard",
        "phase2_information": "b1",
        "phase2_surrogates": "b2_surr",
        "phase2_invariance": "b4_inv",
        "phase2_phase_maps": "b5_phase",
    }
    for dirname, prefix in mapping.items():
        src = reports / dirname
        if not src.exists():
            continue
        fig_dirs = [src / "figures", src]
        seen: set[str] = set()
        for fig_dir in fig_dirs:
            if not fig_dir.exists():
                continue
            for fig in sorted(fig_dir.glob("*.png")):
                if fig.name in seen:
                    continue
                seen.add(fig.name)
                shutil.copy2(fig, fig_dst / f"{prefix}_{fig.name}")
        tables = src / "data" / "tables"
        if tables.exists():
            for csv in sorted(tables.glob("*.csv")):
                shutil.copy2(csv, tab_dst / f"{prefix}_{csv.name}")


def export_difficulty_slice(source_hf: Path, dest_hf: Path, operating: GridPoint) -> None:
    """Copy/filter difficulty-grid resources for the requested cell into dest_hf."""
    tables_src = source_hf / "data" / "tables"
    tables_dst = dest_hf / "data" / "tables"
    figures_dst = dest_hf / "figures"
    diff_src = source_hf / "data" / "difficulty"
    diff_dst = dest_hf / "data" / "difficulty"
    tables_dst.mkdir(parents=True, exist_ok=True)
    figures_dst.mkdir(parents=True, exist_ok=True)
    diff_dst.mkdir(parents=True, exist_ok=True)

    curve = pd.read_csv(tables_src / "difficulty_curve.csv")
    cell = curve[
        (curve["n_steps"] == N_STEPS) & (curve["snr_db"].astype(float) == float(SNR_DB))
    ].copy()
    cell.to_csv(tables_dst / "difficulty_curve_n1500_snr20.csv", index=False)
    # Keep the full curve too so fig 08 can mark the star at this OP.
    shutil.copy2(tables_src / "difficulty_curve.csv", tables_dst / "difficulty_curve.csv")

    for name in (
        "ablation.csv",
        "critical_pairs_binary_hard.csv",
        "headline_operating_point.csv",
        "proper_scores_n1500_clean.csv",
        "ablation_n1500_clean.csv",
    ):
        src = tables_src / name
        if src.exists():
            shutil.copy2(src, tables_dst / name)

    for stem in (
        f"features_n{N_STEPS}_snr{int(SNR_DB)}",
        f"metrics_n{N_STEPS}_snr{int(SNR_DB)}",
    ):
        for path in diff_src.glob(f"{stem}.*"):
            shutil.copy2(path, diff_dst / path.name)

    # Rebuild difficulty figures with the star on N=1500 / SNR=20.
    plot_difficulty_curve(
        curve,
        operating,
        fallback_reason="export_fixed_operating_point",
        output_dir=figures_dst,
        dpi=300,
    )
    binary_path = tables_dst / "critical_pairs_binary_hard.csv"
    if binary_path.exists():
        binary = pd.read_csv(binary_path)
        plot_binary_critical_pairs(
            binary,
            operating,
            output_dir=figures_dst,
            dpi=300,
        )

    # Refresh ablation figure from the SNR=20 checkpoint when present.
    features_path = diff_dst / f"features_n{N_STEPS}_snr{int(SNR_DB)}.npz"
    if features_path.exists():
        audit = apply_profile(AuditConfig.from_yaml(HF_CONFIG), "paper")
        cell_audit = replace(audit, n_steps=N_STEPS, n_realizations=DIFFICULTY_R)
        x, y = _load_checkpoint_frame(features_path)
        health = diagnose_and_clean(x, cell_audit)
        views = evaluate_views(health.x_clean, y, cell_audit, rf_n_jobs=1)
        write_headline_ablation_artifacts(
            x=health.x_clean,
            y=y,
            views=views,
            audit=cell_audit,
            operating=operating,
            tables_dir=tables_dst,
            figures_dir=figures_dst,
            bundle_dir=None,
        )
        differences = pairwise_f1_differences(
            views,
            n_steps=N_STEPS,
            snr_db=SNR_DB,
            n_bootstrap=1000,
            bootstrap_seed=42,
        )
        pd.DataFrame(differences).to_csv(
            tables_dst / "view_f1_differences_n1500_snr20.csv", index=False
        )


def export_hard_frontiers(
    source_hf: Path,
    dest_hf: Path,
    *,
    workers: int,
) -> list[dict[str, Any]]:
    """Rebuild the B3 audit panel from the difficulty-cell cache records."""
    audit = apply_profile(AuditConfig.from_yaml(HF_CONFIG), "paper")
    cell_audit = replace(audit, n_steps=N_STEPS, n_realizations=DIFFICULTY_R)
    generator_config = load_generator_suite_config()
    descriptor_config = DescriptorConfig()
    cache_root = source_hf / "ensemble_cache_difficulty"
    records = _generate_cell_records(
        cell_audit,
        n_steps=N_STEPS,
        snr_db=SNR_DB,
        n_realizations=DIFFICULTY_R,
        cache_root=cache_root,
        generator_config=generator_config,
        descriptor_config=descriptor_config,
        force=False,
    )
    # Persist a local pointer to the source cache fingerprint used.
    cache = versioned_cache_dir(
        cache_root, cell_audit, generator_config, snr_db=SNR_DB
    )
    dest_hf.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        dest_hf / "data" / "source_cache.json",
        {
            "n_steps": N_STEPS,
            "snr_db": SNR_DB,
            "n_realizations": DIFFICULTY_R,
            "source_cache": str(cache),
            "n_records": len(records),
        },
    )
    pipeline = AuditPipeline(cell_audit, lang="en", n_workers=workers)
    pipeline.run(records=records, output_dir=dest_hf, cache_dir=dest_hf / "ensemble_cache")
    return records


def export_information(
    dest_info: Path,
    records: list[dict[str, Any]],
    *,
    workers: int,
) -> None:
    audit = apply_profile(AuditConfig.from_yaml(HF_CONFIG), "paper")
    cell_audit = replace(audit, n_steps=N_STEPS, n_realizations=DIFFICULTY_R)
    pipeline = InformationAuditPipeline(
        cell_audit,
        cache_dir=dest_info / "ensemble_cache_unused",
        lang="en",
        n_workers=workers,
    )
    pipeline.run(dest_info, records=records)


def export_phase_maps(dest_pm: Path, *, workers: int) -> None:
    config = PhaseMapsAuditConfig.from_yaml(PM_CONFIG).with_profile("paper")
    audit = replace(config.audit, n_realizations=DIFFICULTY_R, n_steps=N_STEPS)
    config = replace(
        config,
        audit=audit,
        n_grid=(N_STEPS,),
        snr_db_grid=(SNR_DB,),
    )
    pipeline = PhaseMapsAuditPipeline(config, lang="en", n_workers=workers)
    pipeline.run(output_dir=dest_pm)


def main() -> None:
    setup_logging()
    args = parse_args()
    workers = default_workers() if args.workers is None else max(1, int(args.workers))
    out = Path(args.output_root)
    source_hf = Path(args.source_hf)
    reports = out / "reports"
    bundle = out / "paper_bundle"
    meta = out / "meta"

    if out.exists():
        logger.info("Refreshing existing tree at %s", out)
    out.mkdir(parents=True, exist_ok=True)
    meta.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    curve_path = source_hf / "data" / "tables" / "difficulty_curve.csv"
    if not curve_path.exists():
        raise FileNotFoundError(f"Missing difficulty curve at {curve_path}")
    curve = pd.read_csv(curve_path)
    operating = _operating_point_from_curve(curve)

    atomic_write_json(
        meta / "operating_point.json",
        {
            "n_steps": N_STEPS,
            "snr_db": SNR_DB,
            "n_realizations_difficulty": DIFFICULTY_R,
            "macro_f1_full_rf_mean": operating.macro_f1_mean,
            "macro_f1_full_rf_std": operating.macro_f1_std,
            "macro_f1_ci_low": operating.ci_low,
            "macro_f1_ci_high": operating.ci_high,
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source_hard_frontiers": str(source_hf),
        },
    )

    dest_hf = reports / "phase2_hard_frontiers"
    dest_info = reports / "phase2_information"
    dest_pm = reports / "phase2_phase_maps"
    dest_surr = reports / "phase2_surrogates"
    dest_inv = reports / "phase2_invariance"

    logger.info("Exporting difficulty slice N=%d SNR=%s → %s", N_STEPS, SNR_DB, dest_hf)
    export_difficulty_slice(source_hf, dest_hf, operating)

    records: list[dict[str, Any]] | None = None
    if not args.skip_hard_frontiers:
        logger.info("Rebuilding B3 hard-frontiers panel from difficulty cache")
        records = export_hard_frontiers(source_hf, dest_hf, workers=workers)
        # Difficulty figures/tables may have been overwritten by AuditPipeline
        # figure numbering; re-apply the difficulty slice on top.
        export_difficulty_slice(source_hf, dest_hf, operating)
    else:
        logger.info("Skipped B3 panel rebuild (--skip-hard-frontiers)")

    if not args.skip_information:
        if records is None:
            audit = apply_profile(AuditConfig.from_yaml(HF_CONFIG), "paper")
            cell_audit = replace(audit, n_steps=N_STEPS, n_realizations=DIFFICULTY_R)
            records = _generate_cell_records(
                cell_audit,
                n_steps=N_STEPS,
                snr_db=SNR_DB,
                n_realizations=DIFFICULTY_R,
                cache_root=source_hf / "ensemble_cache_difficulty",
                generator_config=load_generator_suite_config(),
                descriptor_config=DescriptorConfig(),
                force=False,
            )
        logger.info("Running B1 information audit on SNR=20 records")
        export_information(dest_info, records, workers=workers)
    else:
        _write_status(
            dest_info / "STATUS.md",
            "B1 information — skipped",
            "Re-run without `--skip-information` to compute PID / O-info / Shapley "
            "on the N=1500 SNR=20 dB difficulty-cell records.",
        )

    if not args.skip_phase_maps:
        logger.info("Running B5 phase-maps single cell N=%d SNR=%s", N_STEPS, SNR_DB)
        export_phase_maps(dest_pm, workers=workers)
    else:
        _write_status(
            dest_pm / "STATUS.md",
            "B5 phase maps — skipped",
            "Re-run without `--skip-phase-maps` to compute the single-cell "
            f"grid N={{{N_STEPS}}} × SNR={{{int(SNR_DB)}}} into this folder.",
        )

    _write_status(
        dest_surr / "STATUS.md",
        "B2 surrogates — not exported at SNR=20",
        (
            "The surrogate pipeline builds clean originals then Shuffled/FT/IAAFT. "
            "It has no observational SNR=20 injection hook yet, so this operating "
            "point is not packaged here. Existing paper run is N=1000 clean under "
            "`outputs/reports/phase2_surrogates/`.\n\n"
            "To add it later: build originals with "
            "`generate_one_matrix(..., values_transform=_SnrTransform(20, seed))` "
            f"at N={N_STEPS}, then pass `records=` into `SurrogateAuditPipeline.run`."
        ),
    )
    _write_status(
        dest_inv / "STATUS.md",
        "B4 invariance — not exported at SNR=20",
        (
            "Invariance audits a transform suite (including `snr_10`) on clean "
            "series at the configured N. There is no global observational SNR=20 "
            "panel matching B3/B5. Existing paper run is N=1000 under "
            "`outputs/reports/phase2_invariance/`.\n\n"
            "To add it later: generate the original/transform jobs on series "
            f"pre-noised at SNR={int(SNR_DB)} dB with N={N_STEPS}."
        ),
    )

    _flat_publish(reports, bundle)
    readme = f"""# N=1500 / SNR=20 dB bundle

Separate output tree for the joint operating point **N={N_STEPS}**, **SNR={int(SNR_DB)} dB**.

Generated (UTC): `{datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}`

## Layout

- `meta/operating_point.json` — fixed OP metadata from the B3 difficulty curve
- `reports/phase2_hard_frontiers/` — B3 panel + difficulty slice for this cell
- `reports/phase2_information/` — B1 on the same SNR=20 records
- `reports/phase2_phase_maps/` — B5 single-cell grid (or STATUS if skipped)
- `reports/phase2_surrogates/` — STATUS (no SNR=20 hook yet)
- `reports/phase2_invariance/` — STATUS (no SNR=20 hook yet)
- `paper_bundle/` — flat PNG + CSV copies with `b*_` prefixes

## Notes

- Difficulty-cell ensembles use R={DIFFICULTY_R} (not the clean-panel R=30).
- This folder is independent of `outputs/paper_bundle/` (headline there is N=500 / SNR=20).
"""
    atomic_write_text(out / "README.md", readme)
    logger.info("Done. Bundle root: %s", out)


if __name__ == "__main__":
    main()
