"""End-to-end Phase-1 discriminability audit pipeline."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from tsg.audit.classification import ClassificationResult, evaluate_classification
from tsg.audit.config import AuditConfig
from tsg.audit.ensemble import build_ensemble_records
from tsg.audit.features import build_feature_table
from tsg.audit.geometry import GeometryResult, analyze_geometry
from tsg.audit.health import HealthReport, diagnose_and_clean
from tsg.audit.plotting import generate_all_figures
from tsg.audit.verdict import VerdictResult, decide_verdict
from tsg.descriptors.config import DescriptorConfig

logger = logging.getLogger(__name__)


@dataclass
class AuditResult:
    config: AuditConfig
    health: HealthReport
    geometry: GeometryResult
    classification: ClassificationResult
    verdict: VerdictResult
    records: list[dict[str, Any]] = field(default_factory=list)
    x: pd.DataFrame | None = None
    y: pd.Series | None = None
    figure_paths: list[Path] = field(default_factory=list)

    def metrics_dict(self) -> dict[str, Any]:
        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": {
                "n_steps": self.config.n_steps,
                "n_realizations": self.config.n_realizations,
                "seed": self.config.seed,
                "brownian_mode": self.config.brownian_mode,
                "n_folds": self.config.n_folds,
                "regimes": list(self.config.regimes),
                "methods": list(self.config.methods),
                "critical_pairs": [list(p) for p in self.config.critical_pairs],
            },
            "health": self.health.to_dict(),
            "geometry": self.geometry.to_dict(),
            "classification": self.classification.to_dict(),
            "verdict": self.verdict.to_dict(),
            "figures": [str(p) for p in self.figure_paths],
        }


class AuditPipeline:
    """Orchestrate ensemble → features → geometry → classification → verdict."""

    def __init__(
        self,
        config: AuditConfig | None = None,
        descriptor_config: DescriptorConfig | None = None,
    ) -> None:
        self.config = config or AuditConfig()
        self.descriptor_config = descriptor_config or DescriptorConfig()

    def run(
        self,
        *,
        records: list[dict[str, Any]] | None = None,
        output_dir: Path | str | None = None,
        cache_dir: Path | str | None = None,
        force_regimes: tuple[str, ...] | None = None,
    ) -> AuditResult:
        cfg = self.config
        out = Path(output_dir) if output_dir is not None else None
        cache = Path(cache_dir) if cache_dir is not None else (
            out / "ensemble_cache" if out is not None else None
        )
        if records is None:
            logger.info(
                "Building ensemble: regimes=%d R=%d N=%d cache=%s force=%s",
                len(cfg.regimes),
                cfg.n_realizations,
                cfg.n_steps,
                cache,
                force_regimes,
            )
            records = build_ensemble_records(
                cfg,
                descriptor_config=self.descriptor_config,
                cache_dir=cache,
                force_regimes=force_regimes,
            )

        x, y = build_feature_table(records, cfg)
        health = diagnose_and_clean(x, cfg)
        geometry = analyze_geometry(health.x_clean, y, cfg)
        classification = evaluate_classification(health.x_clean, y, cfg)
        verdict = decide_verdict(classification, cfg)

        result = AuditResult(
            config=cfg,
            health=health,
            geometry=geometry,
            classification=classification,
            verdict=verdict,
            records=records,
            x=health.x_clean,
            y=y,
        )

        if out is not None:
            self.persist(result, out)
        return result

    def persist(self, result: AuditResult, output_dir: Path) -> Path:
        """Write metrics JSON, figures, markdown and HTML reports."""
        from tsg.reporting.audit_report import write_audit_reports

        output_dir.mkdir(parents=True, exist_ok=True)
        fig_dir = output_dir / "figures"
        result.figure_paths = generate_all_figures(
            records=result.records,
            geometry=result.geometry,
            classification=result.classification,
            config=result.config,
            output_dir=fig_dir,
        )
        metrics_path = output_dir / "metrics_summary.json"
        payload = result.metrics_dict()
        payload["config"]["regimes"] = list(result.config.regimes)
        metrics_path.write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
        write_audit_reports(result, output_dir)
        logger.info("Audit artifacts written to %s", output_dir)
        return output_dir
