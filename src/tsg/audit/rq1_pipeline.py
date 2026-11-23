"""RQ1 advanced information pipeline (PID-BROJA, Ω bootstrap, SII, stress).

Loads a persisted ``features_clean.npz`` for PID/Ω (no graph recompute).
SII reuses ``subset_scores`` from ``information_metrics.json``. Stress
regenerates raw series → optional SNR/truncate → TS2G → Macro-F1 + log-loss.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from tsg.audit.artifacts import (
    atomic_write_text,
    load_features_npz,
    save_table_csv,
    write_config_snapshot,
)
from tsg.audit.information import DYNAMIC_GROUPS, FAMILY_ORDER, family_principal_scores
from tsg.audit.o_information_bootstrap import (
    OmegaBootstrapConfig,
    OmegaBootstrapResult,
    bootstrap_report_from_features,
    results_to_frame as omega_to_frame,
)
from tsg.audit.pid_advanced import (
    AdvancedPIDAtom,
    AdvancedPIDConfig,
    OneVsRestPIDAtom,
    atoms_to_frame,
    one_vs_rest_pid,
    one_vs_rest_to_frame,
    pairwise_pid_advanced,
)
from tsg.audit.pid_contrast import write_mmi_broja_contrast
from tsg.audit.rq1_plots import write_rq1_figures
from tsg.audit.shapley_interaction import (
    SIIReport,
    sii_report_from_metrics_json,
    sii_to_frames,
)
from tsg.audit.stress_eval import StressConfig, StressSuiteResult, run_stress_suite

logger = logging.getLogger(__name__)


@dataclass
class RQ1PipelineConfig:
    """CLI-facing knobs for the advanced RQ1 suite."""

    features_npz: Path
    output_dir: Path
    pid: AdvancedPIDConfig = field(default_factory=AdvancedPIDConfig)
    omega: OmegaBootstrapConfig = field(default_factory=OmegaBootstrapConfig)
    lang: str = "en"
    figure_dpi: int = 300
    compute_sii: bool = True
    run_stress: bool = False
    subset_scores_json: Path | None = None
    mmi_metrics_json: Path | None = None
    stress: StressConfig = field(default_factory=StressConfig)
    audit_config_yaml: Path | None = None


@dataclass
class RQ1Result:
    """Artifacts produced by :class:`RQ1InformationPipeline`."""

    pid_atoms: list[AdvancedPIDAtom]
    omega_results: list[OmegaBootstrapResult]
    n_samples: int
    n_features: int
    features_path: str
    generated_at_utc: str
    figure_paths: list[Path] = field(default_factory=list)
    notes: dict[str, str] = field(default_factory=dict)
    sii_report: SIIReport | None = None
    stress_result: StressSuiteResult | None = None
    group_coverage: dict[str, Any] = field(default_factory=dict)
    mmi_broja_contrast: list[dict[str, Any]] = field(default_factory=list)
    one_vs_rest: list[OneVsRestPIDAtom] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "generated_at_utc": self.generated_at_utc,
            "features_path": self.features_path,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "group_coverage": self.group_coverage,
            "pid_pairs": [atom.as_dict() for atom in self.pid_atoms],
            "pid_one_vs_rest": [atom.as_dict() for atom in self.one_vs_rest],
            "o_information_bootstrap": [row.as_dict() for row in self.omega_results],
            "mmi_broja_contrast": self.mmi_broja_contrast,
            "figures": [str(p) for p in self.figure_paths],
            "notes": self.notes,
        }
        if self.sii_report is not None:
            payload["sii"] = self.sii_report.as_dict()
        if self.stress_result is not None:
            payload["stress"] = self.stress_result.as_dict()
        return payload


RQ1_NOTES: dict[str, str] = {
    "pid": (
        "Pairwise PID uses dit's I_broja (default) on quantile-binned leading "
        "PCA scores (default ≤3 bins; hard cap 4; groups with n<120 capped at 3). "
        "Scopes: global + stochastic + chaotic_flows + discrete_maps + "
        "structured_other (quasiperiodic, regime_switching). The four "
        "macro-groups partition the full panel."
    ),
    "o_information": (
        "Gaussian O-information of the four family PCA scores, with B≥1000 "
        "bootstrap resamples of panel rows (PCA fixed once). Percentile CI, "
        "bootstrap p-value, column-wise permutation p_perm for H0: Ω=0, and "
        "Holm–Bonferroni p_holm across all scopes. Macro-groups partition the "
        "full panel (structured_other = quasiperiodic + regime_switching)."
    ),
    "sii": (
        "Pairwise Shapley Interaction Index from the 16 Macro-F1 coalitions in "
        "information_metrics.json, including the majority baseline "
        "v(∅)=2/(K(K+1))=1/78 for K=12 balanced regimes "
        "(Macro-F1 of a constant/majority predictor; not 1/K). "
        "φ uncertainty = bootstrap over v(S)~N(μ_CV, σ_CV). "
        "Efficiency: Σφ + v(∅) = v(N)."
    ),
    "stress": (
        "Ceiling-breaking suite: SNR noise / short-N applied to raw series "
        "before TS2G extraction. Reports Macro-F1 and log-loss (primary)."
    ),
    "mmi_contrast": (
        "pid_mmi_vs_broja.csv/md contrasts predictive-MMI atoms (Bloque 1) "
        "with BROJA atoms on the same panel. Absolute bit scales differ by "
        "estimator; the unique-atom pattern is the scientifically relevant contrast."
    ),
    "pid_one_vs_rest": (
        "Four bivariate PIDs X_i | X_{-i}: each family PC vs the leading PC of "
        "the other three. Both discrete MMI and BROJA on the same quantile bins."
    ),
}


def features_frame_from_npz(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load ``features_clean.npz`` into ``(X, y)`` without touching the cache."""
    payload = load_features_npz(path)
    frame = pd.DataFrame(payload["X"], columns=list(payload["feature_names"]))
    labels = pd.Series(payload["y"], name="regime")
    if len(frame) != len(labels):
        raise ValueError(
            f"Feature/label length mismatch: X={len(frame)} y={len(labels)}"
        )
    return frame, labels


def verify_dynamic_group_coverage(
    labels: pd.Series,
    *,
    groups: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    """Assert macro-groups are disjoint and cover every sample label."""
    group_map = groups or DYNAMIC_GROUPS
    counts = {name: int(labels.isin(members).sum()) for name, members in group_map.items()}
    covered = {r for members in group_map.values() for r in members}
    observed = set(labels.astype(str))
    missing = sorted(observed - covered)
    extra = sorted(covered - observed)
    overlap_pairs: list[str] = []
    names = list(group_map)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            inter = set(group_map[left]) & set(group_map[right])
            if inter:
                overlap_pairs.append(f"{left}∩{right}={sorted(inter)}")
    total = int(sum(counts.values()))
    ok = not missing and not overlap_pairs and total == int(len(labels))
    report = {
        "ok": ok,
        "n_samples": int(len(labels)),
        "n_covered": total,
        "counts_by_group": counts,
        "missing_regimes": missing,
        "unused_group_regimes": extra,
        "overlaps": overlap_pairs,
    }
    if ok:
        logger.info(
            "Dynamic-group coverage OK: n=%d counts=%s",
            report["n_samples"],
            counts,
        )
    else:
        logger.error("Dynamic-group coverage FAILED: %s", report)
    return report


class RQ1InformationPipeline:
    """Orchestrate advanced PID, Ω bootstrap, SII and optional stress tests."""

    def __init__(self, config: RQ1PipelineConfig) -> None:
        self.config = config

    def run(self) -> RQ1Result:
        cfg = self.config
        features, labels = features_frame_from_npz(cfg.features_npz)
        coverage = verify_dynamic_group_coverage(labels)
        if not coverage["ok"]:
            raise RuntimeError(
                "DYNAMIC_GROUPS do not partition the feature panel: "
                f"{coverage}"
            )
        logger.info(
            "Loaded features %s shape=%s estimator=%s bins≤%d B=%d",
            cfg.features_npz,
            features.shape,
            cfg.pid.estimator,
            cfg.pid.max_bins,
            cfg.omega.n_bootstrap,
        )

        scores = family_principal_scores(features)
        pid_atoms = pairwise_pid_advanced(
            features,
            labels,
            config=cfg.pid,
            scores=scores,
        )
        # Sanity: every non-global scope used ≤ max adaptive bins.
        for atom in pid_atoms:
            if atom.scope == "group" and atom.n_samples < 120 and atom.n_bins > 3:
                raise RuntimeError(
                    f"Non-conservative bins={atom.n_bins} for {atom.group} "
                    f"(n={atom.n_samples})"
                )

        one_vs_rest = one_vs_rest_pid(
            features,
            labels,
            config=cfg.pid,
            scores=scores,
            estimators=("broja", "mmi_discrete"),
        )

        omega_results, _ = bootstrap_report_from_features(
            features,
            labels,
            config=cfg.omega,
            scores=scores,
        )

        sii_report: SIIReport | None = None
        if cfg.compute_sii:
            metrics_path = cfg.subset_scores_json or cfg.mmi_metrics_json
            if metrics_path is None or not Path(metrics_path).exists():
                raise FileNotFoundError(
                    "SII requires --subset-scores-json / --mmi-metrics-json "
                    "pointing to information_metrics.json"
                )
            sii_report = sii_report_from_metrics_json(
                metrics_path,
                n_bootstrap=int(cfg.omega.n_bootstrap),
                seed=int(cfg.pid.seed),
                ci_level=float(cfg.omega.ci_level),
            )

        stress_result: StressSuiteResult | None = None
        if cfg.run_stress:
            stress_result = self._run_stress(cfg)

        result = RQ1Result(
            pid_atoms=pid_atoms,
            omega_results=omega_results,
            n_samples=int(len(labels)),
            n_features=int(features.shape[1]),
            features_path=str(cfg.features_npz),
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
            notes=dict(RQ1_NOTES),
            sii_report=sii_report,
            stress_result=stress_result,
            group_coverage=coverage,
            one_vs_rest=one_vs_rest,
        )
        self.persist(result)
        return result

    def _run_stress(self, cfg: RQ1PipelineConfig) -> StressSuiteResult:
        from tsg.audit.config import AuditConfig, apply_profile

        yaml_path = cfg.audit_config_yaml
        if yaml_path is None:
            yaml_path = (
                Path(__file__).resolve().parents[3]
                / "configs"
                / "phase2_hard_frontiers.yaml"
            )
        audit = apply_profile(AuditConfig.from_yaml(yaml_path), "paper")
        # Stress uses its own realization count / seed; keep paper N_steps for generation.
        logger.info(
            "Starting stress suite on raw series (N_gen=%d, stress_cfg=%s)",
            audit.n_steps,
            asdict(cfg.stress),
        )
        return run_stress_suite(audit, stress_config=cfg.stress)

    def persist(self, result: RQ1Result) -> Path:
        """Write CSV tables, JSON metrics and RQ1 figures."""
        cfg = self.config
        out = Path(cfg.output_dir)
        data_dir = out / "data"
        tables_dir = data_dir / "tables"
        fig_dir = out / "figures"
        for path in (out, data_dir, tables_dir, fig_dir):
            path.mkdir(parents=True, exist_ok=True)

        save_table_csv(tables_dir / "pid_broja.csv", atoms_to_frame(result.pid_atoms))
        save_table_csv(
            tables_dir / "pid_one_vs_rest.csv",
            one_vs_rest_to_frame(result.one_vs_rest),
        )
        save_table_csv(
            tables_dir / "o_information_bootstrap.csv",
            omega_to_frame(result.omega_results),
        )
        save_table_csv(
            tables_dir / "dynamic_group_coverage.csv",
            pd.DataFrame(
                [
                    {"group": g, "n_samples": n}
                    for g, n in result.group_coverage.get("counts_by_group", {}).items()
                ]
            ),
        )

        mmi_path = cfg.mmi_metrics_json or cfg.subset_scores_json
        if mmi_path is not None and Path(mmi_path).exists():
            contrast, _, _ = write_mmi_broja_contrast(
                mmi_metrics_json=mmi_path,
                broja_atoms=result.pid_atoms,
                tables_dir=tables_dir,
            )
            result.mmi_broja_contrast = contrast.to_dict(orient="records")

        if result.sii_report is not None:
            phi_df, sii_df, base_df = sii_to_frames(result.sii_report)
            save_table_csv(tables_dir / "shapley_phi.csv", phi_df)
            save_table_csv(tables_dir / "shapley_sii.csv", sii_df)
            save_table_csv(tables_dir / "shapley_v_empty.csv", base_df)

        if result.stress_result is not None:
            save_table_csv(tables_dir / "stress_scores.csv", result.stress_result.to_frame())

        result.figure_paths = write_rq1_figures(
            pid_atoms=result.pid_atoms,
            omega_results=result.omega_results,
            output_dir=fig_dir,
            dpi=cfg.figure_dpi,
            lang=cfg.lang,
            sii_report=result.sii_report,
            one_vs_rest=result.one_vs_rest,
        )

        metrics_path = out / "rq1_metrics.json"
        atomic_write_text(
            metrics_path,
            json.dumps(result.to_dict(), indent=2, default=str),
        )
        write_config_snapshot(
            out,
            block="rq1_information",
            profile="reuse_features",
            seed=cfg.pid.seed,
            config={
                "features_npz": str(cfg.features_npz),
                "pid": asdict(cfg.pid),
                "omega": asdict(cfg.omega),
                "lang": cfg.lang,
                "families": list(FAMILY_ORDER),
                "dynamic_groups": {k: list(v) for k, v in DYNAMIC_GROUPS.items()},
                "compute_sii": cfg.compute_sii,
                "run_stress": cfg.run_stress,
                "stress": asdict(cfg.stress),
            },
        )
        self._write_html_summary(result, out / "rq1_report.html")
        logger.info("RQ1 artifacts written to %s", out)
        return out

    def _write_html_summary(self, result: RQ1Result, path: Path) -> None:
        """Self-contained HTML summary (project reporting rule)."""
        pid_html = atoms_to_frame(result.pid_atoms).to_html(
            index=False, float_format=lambda x: f"{x:.4f}"
        )
        one_vs_html = ""
        if result.one_vs_rest:
            one_vs_html = (
                "<h2>PID one-vs-rest (family | rest)</h2>"
                + one_vs_rest_to_frame(result.one_vs_rest).to_html(
                    index=False, float_format=lambda x: f"{x:.4f}"
                )
            )
        omega_html = omega_to_frame(result.omega_results).to_html(
            index=False, float_format=lambda x: f"{x:.4f}"
        )
        contrast_html = ""
        if result.mmi_broja_contrast:
            contrast_html = (
                "<h2>MMI vs BROJA</h2>"
                + pd.DataFrame(result.mmi_broja_contrast).to_html(
                    index=False, float_format=lambda x: f"{x:.4f}"
                )
            )
        sii_html = ""
        if result.sii_report is not None:
            phi_df, sii_df, base_df = sii_to_frames(result.sii_report)
            sii_html = (
                "<h2>Shapley Interaction Index</h2>"
                + base_df.to_html(index=False, float_format=lambda x: f"{x:.4f}")
                + phi_df.to_html(index=False, float_format=lambda x: f"{x:.4f}")
                + sii_df.to_html(index=False, float_format=lambda x: f"{x:.4f}")
            )
        stress_html = ""
        if result.stress_result is not None:
            stress_html = (
                "<h2>Stress test (Macro-F1 + log-loss)</h2>"
                + result.stress_result.to_frame().to_html(
                    index=False, float_format=lambda x: f"{x:.4f}"
                )
            )
        fig_tags = []
        for fig in result.figure_paths:
            if fig.suffix.lower() == ".png":
                fig_tags.append(
                    f'<h3>{fig.stem}</h3><img src="figures/{fig.name}" '
                    f'style="max-width:100%;height:auto;" alt="{fig.stem}"/>'
                )
        coverage = result.group_coverage.get("counts_by_group", {})
        html = f"""<!DOCTYPE html>
<html lang="{self.config.lang}">
<head>
<meta charset="utf-8"/>
<title>RQ1 advanced information</title>
<style>
body {{ font-family: Georgia, serif; margin: 2rem; color: #222; }}
table {{ border-collapse: collapse; font-size: 0.85rem; margin-bottom: 1.5rem; }}
th, td {{ border: 1px solid #ddd; padding: 0.35rem 0.5rem; }}
th {{ background: #f5f5f5; }}
code {{ background: #f0f0f0; padding: 0.1rem 0.3rem; }}
</style>
</head>
<body>
<h1>RQ1 — advanced PID (BROJA), Ω bootstrap, SII &amp; stress</h1>
<p>Generated (UTC): {result.generated_at_utc}<br/>
Features: <code>{result.features_path}</code><br/>
n={result.n_samples}, features={result.n_features}<br/>
Group coverage: {coverage}</p>
<h2>Notes</h2>
<ul>
{"".join(f"<li><b>{k}</b>: {v}</li>" for k, v in result.notes.items())}
</ul>
<h2>PID atoms (BROJA)</h2>
{pid_html}
{one_vs_html}
{contrast_html}
<h2>O-information bootstrap</h2>
{omega_html}
{sii_html}
{stress_html}
<h2>Figures</h2>
{"".join(fig_tags)}
</body>
</html>
"""
        atomic_write_text(path, html)
