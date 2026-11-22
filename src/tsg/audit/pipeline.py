"""End-to-end Phase-1 discriminability audit pipeline."""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from tsg.audit.artifacts import (
    atomic_write_text,
    meta_from_records,
    save_features_npz,
    save_oof_predictions,
    save_table_csv,
    write_config_snapshot,
    write_metrics_alias,
)

# Backward-compatible alias for reporting modules.
_atomic_write_text = atomic_write_text
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
    x_raw: pd.DataFrame | None = None
    figure_paths: list[Path] = field(default_factory=list)
    lang: str = "en"
    generated_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def metrics_dict(self) -> dict[str, Any]:
        health = self.health.to_dict()
        n_raw = int(health["n_features_raw"])
        n_clean = int(health["n_features_clean"])
        return {
            "generated_at_utc": self.generated_at_utc,
            "n_features_raw": n_raw,
            "n_features_clean": n_clean,
            "lang": self.lang,
            "config": {
                "n_steps": self.config.n_steps,
                "n_realizations": self.config.n_realizations,
                "seed": self.config.seed,
                "brownian_mode": self.config.brownian_mode,
                "profile": self.config.profile,
                "brownian_mode_note": (
                    "STN/OPN operate on the integrated Brownian path (levels) and, "
                    "consistently, on integrated fBm paths; "
                    "visibility/proximity/similarity always see the raw series."
                    if self.config.brownian_mode == "levels"
                    else f"brownian_mode={self.config.brownian_mode}"
                ),
                "n_folds": self.config.n_folds,
                "regimes": list(self.config.regimes),
                "methods": list(self.config.methods),
                "families": {k: list(v) for k, v in self.config.families.items()},
                "critical_pairs": [list(p) for p in self.config.critical_pairs],
            },
            "health": health,
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
        *,
        lang: str = "en",
        n_workers: int = 1,
    ) -> None:
        self.config = config or AuditConfig()
        self.descriptor_config = descriptor_config or DescriptorConfig()
        self.lang = lang
        self.n_workers = max(1, int(n_workers))

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
                "Building ensemble: regimes=%d R=%d N=%d cache=%s force=%s workers=%d",
                len(cfg.regimes),
                cfg.n_realizations,
                cfg.n_steps,
                cache,
                force_regimes,
                self.n_workers,
            )
            records = build_ensemble_records(
                cfg,
                descriptor_config=self.descriptor_config,
                cache_dir=cache,
                force_regimes=force_regimes,
                n_workers=self.n_workers,
            )

        x_raw, y = build_feature_table(records, cfg)
        health = diagnose_and_clean(x_raw, cfg)
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
            x_raw=x_raw,
            y=y,
            lang=self.lang,
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
        )

        if out is not None:
            self.persist(result, out)
        return result

    def persist(self, result: AuditResult, output_dir: Path) -> Path:
        """Write metrics, NPZ/CSV data, figures, markdown and HTML."""
        from tsg.reporting.audit_report import write_audit_reports

        output_dir.mkdir(parents=True, exist_ok=True)
        data_dir = output_dir / "data"
        tables_dir = data_dir / "tables"
        data_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)

        self._persist_data(result, data_dir, tables_dir)

        fig_dir = output_dir / "figures"
        result.figure_paths = generate_all_figures(
            records=result.records,
            geometry=result.geometry,
            classification=result.classification,
            config=result.config,
            output_dir=fig_dir,
            lang=result.lang,
        )
        payload = result.metrics_dict()
        metrics_path = output_dir / "metrics_summary.json"
        atomic_write_text(
            metrics_path,
            json.dumps(payload, indent=2, default=str),
        )
        write_metrics_alias(output_dir, metrics_path)
        write_config_snapshot(
            output_dir,
            block="phase2_hard_frontiers",
            profile=result.config.profile,
            seed=result.config.seed,
            config=payload["config"],
            extra={"lang": result.lang, "n_workers": self.n_workers},
        )
        # Keep a pickle of records for plot-only figure regeneration.
        records_path = data_dir / "records.pkl"
        with records_path.open("wb") as fh:
            pickle.dump(result.records, fh, protocol=pickle.HIGHEST_PROTOCOL)

        write_audit_reports(result, output_dir, metrics=payload)
        logger.info("Audit artifacts written to %s", output_dir)
        return output_dir

    def plot_only(self, output_dir: Path | str, *, lang: str | None = None) -> AuditResult:
        """Regenerate figures from persisted NPZ/JSON/pickle without recomputing graphs."""
        from dataclasses import replace

        out = Path(output_dir)
        data_dir = out / "data"
        metrics_path = out / "metrics_summary.json"
        if not metrics_path.exists():
            metrics_path = out / "metrics.json"
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        records_path = data_dir / "records.pkl"
        if not records_path.exists():
            raise FileNotFoundError(
                f"Missing {records_path}; cannot regenerate figures that need records"
            )
        with records_path.open("rb") as fh:
            records = pickle.load(fh)

        # Prefer methods/regimes from the saved run so CLI YAML overrides
        # (e.g. --skip-rn) do not break plot-only regeneration.
        saved = payload.get("config", {})
        cfg = self.config
        overrides: dict[str, Any] = {}
        if saved.get("methods"):
            overrides["methods"] = tuple(saved["methods"])
        if saved.get("regimes"):
            overrides["regimes"] = tuple(saved["regimes"])
        if saved.get("families"):
            overrides["families"] = {k: tuple(v) for k, v in saved["families"].items()}
        if saved.get("critical_pairs"):
            overrides["critical_pairs"] = tuple(
                tuple(pair) for pair in saved["critical_pairs"]
            )
        if overrides:
            cfg = replace(cfg, **overrides)
        x_raw, y = build_feature_table(records, cfg)
        health = diagnose_and_clean(x_raw, cfg)
        geometry = analyze_geometry(health.x_clean, y, cfg)
        classification = evaluate_classification(health.x_clean, y, cfg)
        verdict = decide_verdict(classification, cfg)
        use_lang = lang or payload.get("lang") or self.lang
        result = AuditResult(
            config=cfg,
            health=health,
            geometry=geometry,
            classification=classification,
            verdict=verdict,
            records=records,
            x=health.x_clean,
            x_raw=x_raw,
            y=y,
            lang=use_lang,
            generated_at_utc=payload.get(
                "generated_at_utc", datetime.now(timezone.utc).isoformat()
            ),
        )
        fig_dir = out / "figures"
        result.figure_paths = generate_all_figures(
            records=result.records,
            geometry=result.geometry,
            classification=result.classification,
            config=result.config,
            output_dir=fig_dir,
            lang=use_lang,
        )
        payload["figures"] = [str(p) for p in result.figure_paths]
        payload["lang"] = use_lang
        atomic_write_text(
            out / "metrics_summary.json",
            json.dumps(payload, indent=2, default=str),
        )
        write_metrics_alias(out, out / "metrics_summary.json")
        logger.info("Plot-only figures regenerated in %s (lang=%s)", out, use_lang)
        return result

    def _persist_data(
        self,
        result: AuditResult,
        data_dir: Path,
        tables_dir: Path,
    ) -> None:
        meta = meta_from_records(result.records)
        y = result.y if result.y is not None else pd.Series(dtype=object)
        x_raw = result.x_raw
        x_clean = result.health.x_clean
        if x_raw is None:
            x_raw = x_clean
        save_features_npz(
            data_dir / "features_raw.npz",
            x=x_raw,
            y=y,
            regime_names=list(result.config.regimes),
            realization_ids=meta["realization_ids"],
            transform_or_kind=meta["transform_or_kind"],
            n_steps=meta["n_steps"],
            snr_db=meta["snr_db"],
        )
        save_features_npz(
            data_dir / "features_clean.npz",
            x=x_clean,
            y=y,
            regime_names=list(result.config.regimes),
            realization_ids=meta["realization_ids"],
            transform_or_kind=meta["transform_or_kind"],
            n_steps=meta["n_steps"],
            snr_db=meta["snr_db"],
        )
        save_oof_predictions(
            data_dir / "oof_predictions.npz",
            result.classification.oof_bundles(),
        )

        abl_rows = []
        for view in result.classification.views:
            abl_rows.append(
                {
                    "view": view.view,
                    "model": view.model,
                    "macro_f1_mean": view.macro_f1_mean,
                    "macro_f1_std": view.macro_f1_std,
                    "accuracy_mean": view.accuracy_mean,
                    "accuracy_std": view.accuracy_std,
                }
            )
        save_table_csv(tables_dir / "ablation.csv", pd.DataFrame(abl_rows))
        if result.classification.rf_gini_importance is not None:
            save_table_csv(
                tables_dir / "rf_gini_importance.csv",
                result.classification.rf_gini_importance,
            )
        if result.classification.logistic_l2_importance is not None:
            save_table_csv(
                tables_dir / "logistic_l2_importance.csv",
                result.classification.logistic_l2_importance,
            )
        crit = result.classification.critical_pair_confusion_rate
        if crit:
            save_table_csv(
                tables_dir / "critical_pairs.csv",
                pd.DataFrame(
                    [{"pair": k, "confusion_rate": v} for k, v in crit.items()]
                ),
            )
