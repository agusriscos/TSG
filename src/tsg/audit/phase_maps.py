"""Bloque 5 — N × SNR phase maps and topological vs spectral failure modes."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import clone

from tsg.audit.artifacts import (
    atomic_write_text,
    save_features_npz,
    save_figure_trio,
    save_oof_predictions,
    save_table_csv,
    write_config_snapshot,
    write_metrics_alias,
)
from tsg.audit.classification import (
    ViewMetrics,
    _evaluate_view,
    _make_models,
    evaluate_classification,
)
from tsg.audit.config import AuditConfig, apply_profile
from tsg.audit.ensemble import (
    _prepare_series_for_method,
    load_generator_suite_config,
    realization_seed,
)
from tsg.audit.features import (
    all_feature_names,
    descriptor_kind_mask,
    family_feature_mask,
    matrix_to_feature_row,
)
from tsg.audit.health import diagnose_and_clean
from tsg.audit.i18n import t as _t
from tsg.descriptors.base import DESCRIPTOR_IDS
from tsg.descriptors.config import DescriptorConfig
from tsg.descriptors.pipeline import DescriptorPipeline, results_to_matrix
from tsg.generators.config import GeneratorSuiteConfig
from tsg.generators.perturbations import add_white_noise_snr
from tsg.generators.registry import REGIME_LABELS, build_generator
from tsg.graphs.config import TS2GSuiteConfig
from tsg.graphs.registry import build_transformers

logger = logging.getLogger(__name__)

FAMILY_VIEWS: tuple[str, ...] = (
    "visibility",
    "transition",
    "proximity",
    "similarity",
    "full",
)

KIND_VIEWS: tuple[str, ...] = ("topological_only", "spectral_only")

DEFAULT_N_GRID_DEV: tuple[int, ...] = (200, 500, 1000)
DEFAULT_SNR_DEV: tuple[float | None, ...] = (None, 20.0, 10.0, 5.0, 0.0)
DEFAULT_N_GRID_PAPER: tuple[int, ...] = (100, 250, 500, 1000, 2500)
DEFAULT_SNR_PAPER: tuple[float | None, ...] = (
    None,
    30.0,
    20.0,
    10.0,
    5.0,
    0.0,
    -5.0,
)

PAPER_REGIMES: tuple[str, ...] = (
    "white_noise",
    "fbm_antipersistent",
    "brownian",
    "fbm_persistent",
    "arma_correlated",
    "quasiperiodic",
    "lorenz",
    "rossler",
    "henon",
    "logistic_weak",
    "logistic",
    "regime_switching",
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PHASE_MAPS_CONFIG = _REPO_ROOT / "configs" / "phase2_phase_maps.yaml"

# Sentinel used in JSON / axis labels for clean (infinite SNR) cells.
SNR_CLEAN_LABEL = "inf"


def snr_label(snr_db: float | None) -> str:
    """Human-readable SNR tick (``inf`` for clean)."""
    if snr_db is None or not np.isfinite(snr_db):
        return SNR_CLEAN_LABEL
    if float(snr_db).is_integer():
        return str(int(snr_db))
    return f"{float(snr_db):g}"


def _parse_snr_list(raw: Sequence[Any]) -> tuple[float | None, ...]:
    out: list[float | None] = []
    for item in raw:
        if item is None or (isinstance(item, str) and item.lower() in {"inf", "null", "none", "∞"}):
            out.append(None)
        else:
            out.append(float(item))
    return tuple(out)


@dataclass(frozen=True)
class PhaseMapsAuditConfig:
    """Configuration for the Bloque-5 N × SNR phase-map audit."""

    audit: AuditConfig = field(default_factory=AuditConfig)
    n_grid: tuple[int, ...] = DEFAULT_N_GRID_DEV
    snr_db_grid: tuple[float | None, ...] = DEFAULT_SNR_DEV
    n_grid_dev: tuple[int, ...] = DEFAULT_N_GRID_DEV
    snr_db_dev: tuple[float | None, ...] = DEFAULT_SNR_DEV
    n_grid_paper: tuple[int, ...] = DEFAULT_N_GRID_PAPER
    snr_db_paper: tuple[float | None, ...] = DEFAULT_SNR_PAPER
    paper_regimes: tuple[str, ...] = PAPER_REGIMES

    @classmethod
    def from_yaml(cls, path: Path | str) -> PhaseMapsAuditConfig:
        import yaml

        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Expected mapping in {path}, got {type(raw)!r}")
        audit = AuditConfig.from_mapping(raw.get("audit", raw))
        block = raw.get("phase_maps", {})
        n_dev = tuple(int(v) for v in block.get("n_grid_dev", DEFAULT_N_GRID_DEV))
        snr_dev = _parse_snr_list(block.get("snr_db_dev", DEFAULT_SNR_DEV))
        n_paper = tuple(int(v) for v in block.get("n_grid_paper", DEFAULT_N_GRID_PAPER))
        snr_paper = _parse_snr_list(block.get("snr_db_paper", DEFAULT_SNR_PAPER))
        paper_regimes = tuple(block.get("paper_regimes", PAPER_REGIMES))
        return cls(
            audit=audit,
            n_grid=n_dev,
            snr_db_grid=snr_dev,
            n_grid_dev=n_dev,
            snr_db_dev=snr_dev,
            n_grid_paper=n_paper,
            snr_db_paper=snr_paper,
            paper_regimes=paper_regimes,
        )

    def with_profile(self, profile: str) -> PhaseMapsAuditConfig:
        audit = apply_profile(self.audit, profile)
        if profile == "paper":
            audit = replace(audit, regimes=self.paper_regimes)
            return replace(
                self,
                audit=audit,
                n_grid=self.n_grid_paper,
                snr_db_grid=self.snr_db_paper,
            )
        return replace(
            self,
            audit=audit,
            n_grid=self.n_grid_dev,
            snr_db_grid=self.snr_db_dev,
        )


@dataclass
class PhaseMapsAuditResult:
    """Grid metrics and figure paths for the phase-map audit."""

    config: PhaseMapsAuditConfig
    cells: dict[str, Any]
    collapse_thresholds: dict[str, Any]
    figure_paths: list[Path] = field(default_factory=list)
    # Reference cell (max N, clean SNR) used for features/OOF persistence.
    records: list[dict[str, Any]] = field(default_factory=list)
    generated_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def metrics_dict(self) -> dict[str, Any]:
        audit = self.config.audit
        return {
            "generated_at_utc": self.generated_at_utc,
            "config": {
                "n_steps_nominal": audit.n_steps,
                "n_realizations": audit.n_realizations,
                "seed": audit.seed,
                "profile": audit.profile,
                "regimes": list(audit.regimes),
                "methods": list(audit.methods),
                "families": {k: list(v) for k, v in audit.families.items()},
                "n_grid": list(self.config.n_grid),
                "snr_db_grid": [snr_label(s) for s in self.config.snr_db_grid],
            },
            "cells": self.cells,
            "collapse_thresholds": self.collapse_thresholds,
            "figures": [str(p) for p in self.figure_paths],
        }


def _noise_seed(base_seed: int, n_steps: int, snr_db: float | None, realization: int) -> int:
    snr_tag = 0 if snr_db is None else int(round(1000 * float(snr_db) + 5000))
    return int(base_seed + 17_000 * n_steps + 31 * snr_tag + realization)


def _extract_matrix_for_series(
    values: NDArray[np.floating],
    *,
    dataset_id: str,
    audit_config: AuditConfig,
    transformers: dict[str, Any],
    pipeline: DescriptorPipeline,
) -> pd.DataFrame:
    desc_results = []
    for method in audit_config.methods:
        tr = transformers[method]
        series_m = _prepare_series_for_method(values, dataset_id, method, audit_config)
        graph_result = tr.transform(
            series_m,
            dataset_id,
            regime=REGIME_LABELS[dataset_id],
        )
        desc_results.append(
            pipeline.compute(
                graph_result.graph,
                dataset_id=dataset_id,
                method=method,
                regime=REGIME_LABELS[dataset_id],
                family=graph_result.family,
            )
        )
    return results_to_matrix(desc_results, method_order=audit_config.methods)


def _build_cell_records(
    config: PhaseMapsAuditConfig,
    *,
    n_steps: int,
    snr_db: float | None,
    generator_config: GeneratorSuiteConfig,
    transformers: dict[str, Any],
    pipeline: DescriptorPipeline,
) -> list[dict[str, Any]]:
    audit = config.audit
    gen_cfg = replace(
        generator_config,
        global_params=replace(
            generator_config.global_params,
            n_steps=n_steps,
            seed=audit.seed,
            n_samples=1,
        ),
    )
    records: list[dict[str, Any]] = []
    for dataset_id in audit.regimes:
        generator = build_generator(dataset_id, gen_cfg)
        for r in range(audit.n_realizations):
            seed = realization_seed(audit.seed, dataset_id, r)
            series = generator.generate(n_steps, seed=seed)
            values = np.asarray(series.values, dtype=float).ravel()
            if snr_db is not None:
                rng = np.random.default_rng(_noise_seed(seed, n_steps, snr_db, r))
                values = add_white_noise_snr(values, snr_db, rng)
            matrix = _extract_matrix_for_series(
                values,
                dataset_id=dataset_id,
                audit_config=audit,
                transformers=transformers,
                pipeline=pipeline,
            )
            records.append(
                {
                    "dataset_id": dataset_id,
                    "regime": REGIME_LABELS[dataset_id],
                    "realization": r,
                    "seed": seed,
                    "n_steps": n_steps,
                    "snr_db": snr_db,
                    "matrix": matrix,
                }
            )
    return records


def _records_to_xy(
    records: Sequence[dict[str, Any]],
    methods: Sequence[str],
) -> tuple[pd.DataFrame, pd.Series]:
    rows: list[dict[str, float]] = []
    labels: list[str] = []
    for rec in records:
        rows.append(matrix_to_feature_row(rec["matrix"], methods=methods))
        labels.append(str(rec["dataset_id"]))
    x = pd.DataFrame(rows, columns=all_feature_names(methods, DESCRIPTOR_IDS))
    y = pd.Series(labels, name="regime")
    return x, y


def _view_specs(columns: Sequence[str]) -> list[tuple[str, np.ndarray]]:
    specs: list[tuple[str, np.ndarray]] = [
        ("visibility", family_feature_mask(columns, "visibility")),
        ("transition", family_feature_mask(columns, "transition")),
        ("proximity", family_feature_mask(columns, "proximity")),
        ("similarity", family_feature_mask(columns, "similarity")),
        ("full", np.ones(len(columns), dtype=bool)),
        ("topological_only", descriptor_kind_mask(columns, "topological")),
        ("spectral_only", descriptor_kind_mask(columns, "spectral")),
    ]
    return [(name, mask) for name, mask in specs if mask.any()]


def _nan_failure_rate(x: pd.DataFrame) -> dict[str, float]:
    """Fraction of NaN / Inf entries and spectral-column NaN rate."""
    arr = x.to_numpy(dtype=float)
    total = arr.size
    bad = int(np.logical_or(~np.isfinite(arr), np.isnan(arr)).sum()) if total else 0
    spectral_mask = descriptor_kind_mask(list(x.columns), "spectral")
    if spectral_mask.any():
        s_arr = x.loc[:, spectral_mask].to_numpy(dtype=float)
        s_total = s_arr.size
        s_bad = int((~np.isfinite(s_arr)).sum()) if s_total else 0
        spectral_nan_rate = float(s_bad / s_total) if s_total else float("nan")
    else:
        spectral_nan_rate = float("nan")
    # Proxy for disconnection: fraction of rows where lambda2_g is 0 or NaN.
    disc_cols = [c for c in x.columns if c.endswith("__lambda2_g")]
    if disc_cols:
        disc = x[disc_cols].to_numpy(dtype=float)
        disc_rate = float(np.mean(np.any((~np.isfinite(disc)) | (disc <= 1e-12), axis=1)))
    else:
        disc_rate = float("nan")
    return {
        "nan_inf_rate": float(bad / total) if total else float("nan"),
        "spectral_nan_rate": spectral_nan_rate,
        "disconnection_proxy_rate": disc_rate,
    }


def _evaluate_cell(
    records: Sequence[dict[str, Any]],
    audit: AuditConfig,
) -> dict[str, Any]:
    x, y = _records_to_xy(records, audit.methods)
    failure = _nan_failure_rate(x)
    health = diagnose_and_clean(x, audit)
    x_clean = health.x_clean
    models = _make_models(audit.seed)
    views: dict[str, Any] = {}
    for view_name, mask in _view_specs(list(x_clean.columns)):
        x_view = x_clean.loc[:, mask]
        if x_view.shape[1] == 0:
            continue
        views[view_name] = {}
        for model_name, model in models.items():
            metrics: ViewMetrics = _evaluate_view(
                x_view,
                y,
                view=view_name,
                model_name=model_name,
                model=clone(model),
                config=audit,
            )
            views[view_name][model_name] = {
                "macro_f1_mean": metrics.macro_f1_mean,
                "macro_f1_std": metrics.macro_f1_std,
                "accuracy_mean": metrics.accuracy_mean,
            }
    return {"views": views, "failure_modes": failure}


def _phase_map_cell_job(
    n_steps: int,
    snr_db: float | None,
    *,
    config: PhaseMapsAuditConfig,
    generator_config: GeneratorSuiteConfig,
    descriptor_config: DescriptorConfig,
    keep_records: bool = False,
) -> tuple[str, str, dict[str, Any], list[dict[str, Any]] | None]:
    """Top-level picklable task for one (N, SNR) grid cell.

    Returns
    -------
    tuple
        ``(n_key, snr_key, cell_metrics, records_or_None)``.
    """
    audit = config.audit
    ts2g_cfg = TS2GSuiteConfig(methods=audit.methods)
    transformers = {
        t.method: t
        for t in build_transformers(ts2g_cfg)
        if t.method in audit.methods
    }
    pipeline = DescriptorPipeline(config=descriptor_config)

    records = _build_cell_records(
        config,
        n_steps=int(n_steps),
        snr_db=snr_db,
        generator_config=generator_config,
        transformers=transformers,
        pipeline=pipeline,
    )
    cell_audit = replace(audit, n_steps=int(n_steps))
    n_key = str(int(n_steps))
    s_key = snr_label(snr_db)
    metrics = _evaluate_cell(records, cell_audit)
    return n_key, s_key, metrics, (records if keep_records else None)


def compute_phase_map_grid(
    config: PhaseMapsAuditConfig,
    *,
    generator_config: GeneratorSuiteConfig | None = None,
    descriptor_config: DescriptorConfig | None = None,
    n_workers: int = 1,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Sweep the (N, SNR) grid and return nested cell metrics + reference records.

    The reference records come from the largest ``N`` and clean SNR (``None``)
    cell, so features / OOF can be persisted without keeping the whole grid.
    """
    audit = config.audit
    gen_cfg = generator_config or load_generator_suite_config()
    desc_cfg = descriptor_config or DescriptorConfig()
    ref_n = int(max(config.n_grid))

    grid_jobs: list[tuple[int, float | None]] = [
        (int(n_steps), snr_db)
        for n_steps in config.n_grid
        for snr_db in config.snr_db_grid
    ]
    total = len(grid_jobs)
    n_workers = max(1, int(n_workers))

    if n_workers > 1:
        from joblib import Parallel, delayed

        logger.info("Computing %d phase-map cells with %d workers", total, n_workers)
        raw_results: list[
            tuple[str, str, dict[str, Any], list[dict[str, Any]] | None]
        ] = Parallel(n_jobs=n_workers)(
            delayed(_phase_map_cell_job)(
                n_steps,
                snr_db,
                config=config,
                generator_config=gen_cfg,
                descriptor_config=desc_cfg,
                keep_records=(int(n_steps) == ref_n and snr_db is None),
            )
            for n_steps, snr_db in grid_jobs
        )
    else:
        raw_results = []
        for idx, (n_steps, snr_db) in enumerate(grid_jobs, 1):
            s_key = snr_label(snr_db)
            logger.info(
                "Phase map cell %d/%d N=%d SNR=%s R=%d regimes=%d",
                idx, total, n_steps, s_key,
                audit.n_realizations, len(audit.regimes),
            )
            raw_results.append(
                _phase_map_cell_job(
                    n_steps,
                    snr_db,
                    config=config,
                    generator_config=gen_cfg,
                    descriptor_config=desc_cfg,
                    keep_records=(int(n_steps) == ref_n and snr_db is None),
                )
            )

    cells: dict[str, Any] = {}
    reference_records: list[dict[str, Any]] = []
    for n_key, s_key, cell_data, maybe_records in raw_results:
        cells.setdefault(n_key, {})[s_key] = cell_data
        if maybe_records is not None:
            reference_records = maybe_records
    return cells, reference_records


def collapse_snr_thresholds(
    cells: dict[str, Any],
    *,
    n_grid: Sequence[int],
    snr_grid: Sequence[float | None],
    f1_threshold: float = 0.80,
    model: str = "random_forest",
) -> dict[str, Any]:
    """Approximate SNR where Macro-F1 first falls below ``f1_threshold``.

    For each family / kind view and each N, scan SNR from clean → noisy and
    report the first SNR at which RF Macro-F1 < threshold (or ``None`` if it
    never collapses).
    """
    views = list(FAMILY_VIEWS) + list(KIND_VIEWS)
    # Sort SNR from high (clean) to low.
    ordered = sorted(
        snr_grid,
        key=lambda s: float("inf") if s is None else float(s),
        reverse=True,
    )
    out: dict[str, Any] = {"f1_threshold": f1_threshold, "model": model, "per_view": {}}
    for view in views:
        per_n: dict[str, Any] = {}
        for n in n_grid:
            n_key = str(int(n))
            collapse_at: str | None = None
            f1_curve: dict[str, float] = {}
            for snr in ordered:
                s_key = snr_label(snr)
                f1 = (
                    cells.get(n_key, {})
                    .get(s_key, {})
                    .get("views", {})
                    .get(view, {})
                    .get(model, {})
                    .get("macro_f1_mean", float("nan"))
                )
                f1_curve[s_key] = float(f1) if np.isfinite(f1) else float("nan")
                if collapse_at is None and np.isfinite(f1) and f1 < f1_threshold:
                    collapse_at = s_key
            per_n[n_key] = {"collapse_snr_db": collapse_at, "f1_curve": f1_curve}
        out["per_view"][view] = per_n
    return out


def _plot_family_phase_maps(
    result: PhaseMapsAuditResult,
    output_dir: Path,
    lang: str = "en",
) -> list[Path]:
    """Heatmap panels for each TS2G family view.

    Parameters
    ----------
    result : PhaseMapsAuditResult
        Computed phase-map audit result.
    output_dir : Path
        Directory for figure files.
    lang : str
        Language code for i18n titles.

    Returns
    -------
    list[Path]
        Paths written (pdf + svg + png).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n_labels = [str(n) for n in result.config.n_grid]
    snr_labels = [snr_label(s) for s in result.config.snr_db_grid]
    views = list(FAMILY_VIEWS)

    n_views = len(views)
    fig = plt.figure(figsize=(3.15 * n_views + 1.6, 4.4))
    width_ratios = [1.0] * n_views + [0.12, 0.05]
    gs = fig.add_gridspec(
        1,
        n_views + 2,
        width_ratios=width_ratios,
        wspace=0.35,
        left=0.05,
        right=0.98,
        top=0.86,
        bottom=0.20,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_views)]
    for i, ax in enumerate(axes[1:], start=1):
        ax.sharey(axes[0])
    for ax, view in zip(axes, views):
        mat = np.full((len(n_labels), len(snr_labels)), np.nan)
        for i, n_key in enumerate(n_labels):
            for j, s_key in enumerate(snr_labels):
                mat[i, j] = (
                    result.cells.get(n_key, {})
                    .get(s_key, {})
                    .get("views", {})
                    .get(view, {})
                    .get("random_forest", {})
                    .get("macro_f1_mean", float("nan"))
                )
        im = ax.imshow(mat, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_title(view)
        ax.set_xticks(range(len(snr_labels)))
        ax.set_xticklabels(snr_labels, rotation=40, ha="right")
        ax.set_yticks(range(len(n_labels)))
        ax.set_yticklabels(n_labels)
        ax.set_xlabel("SNR [dB]")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="w")
    axes[0].set_ylabel("N")
    cax = fig.add_subplot(gs[0, -1])
    fig.colorbar(im, cax=cax, label="Macro-F1 RF")
    fig.suptitle(_t("phase_maps_families_title", lang), fontsize=11)

    return save_figure_trio(fig, output_dir / "01_phase_maps_families")


def _plot_topo_vs_spectral(
    result: PhaseMapsAuditResult,
    output_dir: Path,
    lang: str = "en",
) -> list[Path]:
    """Line plots comparing topological and spectral failure modes vs SNR.

    Parameters
    ----------
    result : PhaseMapsAuditResult
        Computed phase-map audit result.
    output_dir : Path
        Directory for figure files.
    lang : str
        Language code for i18n titles.

    Returns
    -------
    list[Path]
        Paths written (pdf + svg + png).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n_ref = max(result.config.n_grid)
    n_key = str(int(n_ref))
    snr_labels_list = [snr_label(s) for s in result.config.snr_db_grid]

    f1_topo: list[float] = []
    f1_spec: list[float] = []
    nan_rates: list[float] = []
    disc_rates: list[float] = []
    for s_key in snr_labels_list:
        cell = result.cells.get(n_key, {}).get(s_key, {})
        f1_topo.append(
            cell.get("views", {})
            .get("topological_only", {})
            .get("random_forest", {})
            .get("macro_f1_mean", float("nan"))
        )
        f1_spec.append(
            cell.get("views", {})
            .get("spectral_only", {})
            .get("random_forest", {})
            .get("macro_f1_mean", float("nan"))
        )
        fail = cell.get("failure_modes", {})
        nan_rates.append(fail.get("spectral_nan_rate", float("nan")))
        disc_rates.append(fail.get("disconnection_proxy_rate", float("nan")))

    x = np.arange(len(snr_labels_list))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(x, f1_topo, "o-", label="topological_only", color="#0f5c4c")
    axes[0].plot(x, f1_spec, "s--", label="spectral_only", color="#8b2e1a")
    axes[0].axhline(0.80, color="#666", lw=0.8, ls=":", label="F1=0.80")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(snr_labels_list)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_xlabel("SNR [dB]")
    axes[0].set_ylabel("Macro-F1 RF")
    axes[0].set_title(f"Topological vs spectral (N={n_ref})")
    axes[0].legend(fontsize=8)

    axes[1].plot(x, nan_rates, "o-", label="spectral NaN rate", color="#3d7ea6")
    axes[1].plot(x, disc_rates, "s--", label="disconnection proxy", color="#9a6b00")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(snr_labels_list)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_xlabel("SNR [dB]")
    axes[1].set_ylabel("Rate")
    axes[1].set_title(f"Failure modes (N={n_ref})")
    axes[1].legend(fontsize=8)

    fig.suptitle(_t("phase_topo_vs_spec_title", lang), fontsize=11)
    fig.tight_layout()

    return save_figure_trio(fig, output_dir / "02_topological_vs_spectral_snr")


def _format_markdown(result: PhaseMapsAuditResult) -> str:
    lines: list[str] = [
        "# Mapas de fase N × SNR (Bloque 5)",
        "",
        f"- Generado: `{result.generated_at_utc}`",
        f"- Profile: `{result.config.audit.profile}`",
        f"- R={result.config.audit.n_realizations}",
        f"- Regímenes: `{', '.join(result.config.audit.regimes)}`",
        f"- N grid: `{', '.join(str(n) for n in result.config.n_grid)}`",
        f"- SNR grid [dB]: `{', '.join(snr_label(s) for s in result.config.snr_db_grid)}`",
        "",
        "## Umbral aproximado de colapso (RF Macro-F1 < 0.80)",
        "",
        "| Vista | " + " | ".join(f"N={n}" for n in result.config.n_grid) + " |",
        "|---|" + "|".join(["---:" for _ in result.config.n_grid]) + "|",
    ]
    for view in list(FAMILY_VIEWS) + list(KIND_VIEWS):
        cells = []
        for n in result.config.n_grid:
            thr = (
                result.collapse_thresholds.get("per_view", {})
                .get(view, {})
                .get(str(int(n)), {})
                .get("collapse_snr_db")
            )
            cells.append("—" if thr is None else str(thr))
        lines.append(f"| {view} | " + " | ".join(cells) + " |")

    n_ref = max(result.config.n_grid)
    n_key = str(int(n_ref))
    lines.extend(
        [
            "",
            f"## Macro-F1 RF en N={n_ref} (familias)",
            "",
            "| SNR | " + " | ".join(FAMILY_VIEWS) + " |",
            "|---|" + "|".join(["---:" for _ in FAMILY_VIEWS]) + "|",
        ]
    )
    for snr in result.config.snr_db_grid:
        s_key = snr_label(snr)
        cells = []
        for view in FAMILY_VIEWS:
            f1 = (
                result.cells.get(n_key, {})
                .get(s_key, {})
                .get("views", {})
                .get(view, {})
                .get("random_forest", {})
                .get("macro_f1_mean", float("nan"))
            )
            cells.append(f"{f1:.3f}")
        lines.append(f"| {s_key} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            f"## Topológico vs espectral (N={n_ref})",
            "",
            "| SNR | topo F1 | spectral F1 | spectral NaN | disconnect |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for snr in result.config.snr_db_grid:
        s_key = snr_label(snr)
        cell = result.cells.get(n_key, {}).get(s_key, {})
        topo = (
            cell.get("views", {})
            .get("topological_only", {})
            .get("random_forest", {})
            .get("macro_f1_mean", float("nan"))
        )
        spec = (
            cell.get("views", {})
            .get("spectral_only", {})
            .get("random_forest", {})
            .get("macro_f1_mean", float("nan"))
        )
        fail = cell.get("failure_modes", {})
        lines.append(
            f"| {s_key} | {topo:.3f} | {spec:.3f} | "
            f"{fail.get('spectral_nan_rate', float('nan')):.3f} | "
            f"{fail.get('disconnection_proxy_rate', float('nan')):.3f} |"
        )

    lines.extend(["", "## Figuras", ""])
    for p in result.figure_paths:
        lines.append(f"- `{p.name}`")
    lines.append("")
    return "\n".join(lines)


def _markdown_to_html(result: PhaseMapsAuditResult) -> str:
    import base64

    def _img(path: Path) -> str:
        if not path.exists() or path.suffix.lower() != ".png":
            return ""
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'<img class="fig" alt="{path.name}" src="data:image/png;base64,{b64}"/>'

    md = _format_markdown(result)
    imgs = "".join(_img(p) for p in result.figure_paths if p.suffix == ".png")
    return f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"/>
<title>Mapas de fase N × SNR</title>
<style>
body{{font-family:IBM Plex Sans,Segoe UI,sans-serif;max-width:1100px;margin:1.5rem auto;padding:0 1rem;background:#f7f5f1;color:#1c1a16}}
pre{{white-space:pre-wrap;background:#fff;border:1px solid #ddd6c8;border-radius:10px;padding:1rem}}
img.fig{{width:100%;margin:1rem 0;border:1px solid #ddd6c8;border-radius:8px}}
</style></head><body>
<h1>Mapas de fase N × SNR (Bloque 5)</h1>
{imgs}
<pre>{md}</pre>
</body></html>
"""


class PhaseMapsAuditPipeline:
    """Orchestrate the N × SNR grid sweep and persist phase-map artifacts."""

    def __init__(
        self,
        config: PhaseMapsAuditConfig | None = None,
        descriptor_config: DescriptorConfig | None = None,
        *,
        lang: str = "en",
        n_workers: int = 1,
    ) -> None:
        self.config = config or PhaseMapsAuditConfig()
        self.descriptor_config = descriptor_config or DescriptorConfig()
        self.lang = lang
        self.n_workers = max(1, int(n_workers))

    def run(
        self,
        *,
        cells: dict[str, Any] | None = None,
        output_dir: Path | str | None = None,
    ) -> PhaseMapsAuditResult:
        cfg = self.config
        logger.info(
            "Phase maps profile=%s R=%d N_grid=%s SNR_grid=%s regimes=%s workers=%d",
            cfg.audit.profile,
            cfg.audit.n_realizations,
            cfg.n_grid,
            [snr_label(s) for s in cfg.snr_db_grid],
            cfg.audit.regimes,
            self.n_workers,
        )
        reference_records: list[dict[str, Any]] = []
        if cells is None:
            cells, reference_records = compute_phase_map_grid(
                cfg,
                descriptor_config=self.descriptor_config,
                n_workers=self.n_workers,
            )
        thresholds = collapse_snr_thresholds(
            cells,
            n_grid=cfg.n_grid,
            snr_grid=cfg.snr_db_grid,
        )
        result = PhaseMapsAuditResult(
            config=cfg,
            cells=cells,
            collapse_thresholds=thresholds,
            records=reference_records,
        )
        if output_dir is not None:
            self.persist(result, Path(output_dir))
        return result

    def persist(self, result: PhaseMapsAuditResult, output_dir: Path) -> Path:
        """Write metrics, CSV tables, figures, markdown and HTML."""
        output_dir.mkdir(parents=True, exist_ok=True)
        data_dir = output_dir / "data"
        tables_dir = data_dir / "tables"
        data_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)

        self._persist_data(result, data_dir, tables_dir)

        fig_dir = output_dir / "figures"
        paths = _plot_family_phase_maps(result, fig_dir, lang=self.lang)
        paths.extend(_plot_topo_vs_spectral(result, fig_dir, lang=self.lang))
        result.figure_paths = paths

        metrics_path = output_dir / "phase_maps_metrics.json"
        atomic_write_text(
            metrics_path,
            json.dumps(result.metrics_dict(), indent=2, default=str),
        )
        write_metrics_alias(output_dir, metrics_path)
        write_config_snapshot(
            output_dir,
            block="phase2_phase_maps",
            profile=result.config.audit.profile,
            seed=result.config.audit.seed,
            config=result.metrics_dict().get("config", {}),
            extra={"lang": self.lang, "n_workers": self.n_workers},
        )

        atomic_write_text(output_dir / "phase_maps_report.md", _format_markdown(result))
        html = _markdown_to_html(result)
        atomic_write_text(output_dir / "latest.html", html)
        atomic_write_text(output_dir / "phase_maps_report.html", html)
        logger.info("Phase-map audit artifacts written to %s", output_dir)
        return output_dir

    def plot_only(
        self,
        output_dir: Path | str,
        *,
        lang: str | None = None,
    ) -> PhaseMapsAuditResult:
        """Regenerate figures from persisted metrics JSON without recomputing graphs.

        Parameters
        ----------
        output_dir : Path | str
            Directory previously written by :meth:`persist`.
        lang : str | None
            Override language; falls back to stored value or ``self.lang``.

        Returns
        -------
        PhaseMapsAuditResult
            Reconstructed result with updated ``figure_paths``.
        """
        out = Path(output_dir)
        metrics_path = out / "phase_maps_metrics.json"
        if not metrics_path.exists():
            metrics_path = out / "metrics.json"
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))

        cells = payload.get("cells", {})
        thresholds = collapse_snr_thresholds(
            cells,
            n_grid=self.config.n_grid,
            snr_grid=self.config.snr_db_grid,
        )
        use_lang = lang or payload.get("lang") or self.lang
        result = PhaseMapsAuditResult(
            config=self.config,
            cells=cells,
            collapse_thresholds=thresholds,
            generated_at_utc=payload.get(
                "generated_at_utc", datetime.now(timezone.utc).isoformat()
            ),
        )
        fig_dir = out / "figures"
        paths = _plot_family_phase_maps(result, fig_dir, lang=use_lang)
        paths.extend(_plot_topo_vs_spectral(result, fig_dir, lang=use_lang))
        result.figure_paths = paths
        payload["figures"] = [str(p) for p in result.figure_paths]
        payload["lang"] = use_lang
        atomic_write_text(
            out / "phase_maps_metrics.json",
            json.dumps(payload, indent=2, default=str),
        )
        write_metrics_alias(out, out / "phase_maps_metrics.json")
        logger.info("Plot-only figures regenerated in %s (lang=%s)", out, use_lang)
        return result

    def _persist_data(
        self,
        result: PhaseMapsAuditResult,
        data_dir: Path,
        tables_dir: Path,
    ) -> None:
        """Write features/OOF for the reference cell plus CSV tables."""
        audit = result.config.audit
        if result.records:
            rows = [
                matrix_to_feature_row(rec["matrix"], methods=audit.methods)
                for rec in result.records
            ]
            y = pd.Series(
                [str(rec["dataset_id"]) for rec in result.records],
                name="regime",
            )
            realization_ids = [int(rec["realization"]) for rec in result.records]
            n_steps = [int(rec.get("n_steps", -1)) for rec in result.records]
            snr_db = [
                np.inf if rec.get("snr_db") is None else float(rec["snr_db"])
                for rec in result.records
            ]
            x_raw = pd.DataFrame(
                rows, columns=all_feature_names(audit.methods, DESCRIPTOR_IDS)
            )
            save_features_npz(
                data_dir / "features_raw.npz",
                x=x_raw,
                y=y,
                regime_names=list(audit.regimes),
                realization_ids=realization_ids,
                n_steps=n_steps,
                snr_db=snr_db,
            )
            health = diagnose_and_clean(x_raw, audit)
            save_features_npz(
                data_dir / "features_clean.npz",
                x=health.x_clean,
                y=y,
                regime_names=list(audit.regimes),
                realization_ids=realization_ids,
                n_steps=n_steps,
                snr_db=snr_db,
            )
            if health.x_clean.shape[1] > 0 and int(y.nunique()) >= 2:
                clf = evaluate_classification(health.x_clean, y, audit)
                save_oof_predictions(
                    data_dir / "oof_predictions.npz",
                    clf.oof_bundles(),
                )

        f1_rows: list[dict[str, Any]] = []
        nan_rows: list[dict[str, Any]] = []
        for n_steps in result.config.n_grid:
            n_key = str(int(n_steps))
            for snr_db in result.config.snr_db_grid:
                s_key = snr_label(snr_db)
                cell = result.cells.get(n_key, {}).get(s_key, {})
                fail = cell.get("failure_modes", {})
                nan_rows.append({
                    "N": int(n_steps),
                    "SNR": s_key,
                    "nan_inf_rate": fail.get("nan_inf_rate", float("nan")),
                    "spectral_nan_rate": fail.get("spectral_nan_rate", float("nan")),
                    "disconnection_proxy_rate": fail.get(
                        "disconnection_proxy_rate", float("nan")
                    ),
                })
                for view in list(FAMILY_VIEWS) + list(KIND_VIEWS):
                    for model_name in ("random_forest", "logistic"):
                        m = (
                            cell.get("views", {})
                            .get(view, {})
                            .get(model_name, {})
                        )
                        f1_rows.append({
                            "N": int(n_steps),
                            "SNR": s_key,
                            "view": view,
                            "model": model_name,
                            "macro_f1_mean": m.get("macro_f1_mean", float("nan")),
                            "macro_f1_std": m.get("macro_f1_std", float("nan")),
                            "accuracy_mean": m.get("accuracy_mean", float("nan")),
                        })
        save_table_csv(tables_dir / "phase_map_f1.csv", pd.DataFrame(f1_rows))
        save_table_csv(tables_dir / "nan_rates.csv", pd.DataFrame(nan_rows))
        logger.info("Persisted phase-map tables to %s", tables_dir)
