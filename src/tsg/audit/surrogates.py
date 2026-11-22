"""Surrogate audit: temporal sensitivity and genuine nonlinearity (Bloque 2)."""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

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
    family_feature_mask,
    matrix_to_feature_row,
)
from tsg.audit.health import diagnose_and_clean
from tsg.audit.i18n import t as _t
from tsg.descriptors.base import DESCRIPTOR_IDS
from tsg.descriptors.config import DescriptorConfig
from tsg.descriptors.pipeline import DescriptorPipeline, results_to_matrix
from tsg.generators.config import GeneratorSuiteConfig
from tsg.generators.registry import REGIME_LABELS, build_generator
from tsg.generators.surrogates import (
    IAAFT_CONVERGENCE_CRITERION,
    IAAFT_DEFAULT_MAX_ITER,
    IAAFT_DEFAULT_MIN_ITER,
    IAAFT_DEFAULT_TOL,
    SurrogateKind,
    make_surrogate,
)
from tsg.graphs.config import TS2GSuiteConfig
from tsg.graphs.registry import build_transformers

logger = logging.getLogger(__name__)

SURROGATE_KINDS: tuple[SurrogateKind, ...] = ("original", "shuffled", "ft", "iaaft")
FAMILY_VIEWS: tuple[str, ...] = (
    "visibility",
    "transition",
    "proximity",
    "similarity",
    "full",
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SURROGATE_CONFIG = _REPO_ROOT / "configs" / "phase2_surrogates.yaml"


@dataclass(frozen=True)
class SurrogateAuditConfig:
    """Configuration for the Bloque-2 surrogate audit."""

    audit: AuditConfig = field(default_factory=AuditConfig)
    kinds: tuple[SurrogateKind, ...] = SURROGATE_KINDS
    iaaft_max_iter: int = IAAFT_DEFAULT_MAX_ITER
    iaaft_min_iter: int = IAAFT_DEFAULT_MIN_ITER
    iaaft_tol: float = IAAFT_DEFAULT_TOL

    @classmethod
    def from_yaml(cls, path: Path | str) -> SurrogateAuditConfig:
        import yaml

        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Expected mapping in {path}, got {type(raw)!r}")
        audit = AuditConfig.from_mapping(raw.get("audit", raw))
        surr = raw.get("surrogates", {})
        kinds = tuple(surr.get("kinds", SURROGATE_KINDS))
        return cls(
            audit=audit,
            kinds=kinds,
            iaaft_max_iter=int(surr.get("iaaft_max_iter", IAAFT_DEFAULT_MAX_ITER)),
            iaaft_min_iter=int(surr.get("iaaft_min_iter", IAAFT_DEFAULT_MIN_ITER)),
            iaaft_tol=float(surr.get("iaaft_tol", IAAFT_DEFAULT_TOL)),
        )

    def with_profile(self, profile: str) -> SurrogateAuditConfig:
        return replace(self, audit=apply_profile(self.audit, profile))


@dataclass
class SurrogateAuditResult:
    """Aggregated metrics and artifacts from a surrogate audit run."""

    config: SurrogateAuditConfig
    records: list[dict[str, Any]]
    temporal_sensitivity: dict[str, Any]
    nonlinearity_binary: dict[str, Any]
    nonlinearity_multiclass: dict[str, Any]
    regime_multiclass_by_surrogate: dict[str, Any]
    figure_paths: list[Path] = field(default_factory=list)
    generated_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def metrics_dict(self) -> dict[str, Any]:
        audit = self.config.audit
        return {
            "generated_at_utc": self.generated_at_utc,
            "config": {
                "n_steps": audit.n_steps,
                "n_realizations": audit.n_realizations,
                "seed": audit.seed,
                "profile": audit.profile,
                "regimes": list(audit.regimes),
                "methods": list(audit.methods),
                "families": {k: list(v) for k, v in audit.families.items()},
                "surrogate_kinds": list(self.config.kinds),
                "iaaft_max_iter": self.config.iaaft_max_iter,
                "iaaft_min_iter": self.config.iaaft_min_iter,
                "iaaft_tol": self.config.iaaft_tol,
                "iaaft_convergence_criterion": IAAFT_CONVERGENCE_CRITERION,
            },
            "temporal_sensitivity_delta_T": self.temporal_sensitivity,
            "nonlinearity_binary_original_vs_iaaft": self.nonlinearity_binary,
            "nonlinearity_multiclass_surrogate_type": self.nonlinearity_multiclass,
            "regime_multiclass_by_surrogate": self.regime_multiclass_by_surrogate,
            "figures": [str(p) for p in self.figure_paths],
        }


def _surrogate_seed(base_seed: int, kind: str, realization: int) -> int:
    """Independent but deterministic RNG stream per surrogate kind."""
    offset = {"original": 0, "shuffled": 17, "ft": 29, "iaaft": 41}.get(kind, 53)
    return int(base_seed + 10_000 * offset + realization)


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


def _surrogate_job(
    dataset_id: str,
    r: int,
    kind: str,
    *,
    config: SurrogateAuditConfig,
    generator_config: GeneratorSuiteConfig,
    descriptor_config: DescriptorConfig,
) -> dict[str, Any]:
    """Top-level picklable task for one (regime, realization, kind) triple.

    Determinism is guaranteed because the RNG seed is derived from the
    realization seed, independent of execution order.

    Parameters
    ----------
    dataset_id : str
        Regime identifier.
    r : int
        Realization index.
    kind : str
        Surrogate kind (``original``, ``shuffled``, ``ft``, ``iaaft``).
    config : SurrogateAuditConfig
        Full surrogate audit configuration.
    generator_config : GeneratorSuiteConfig
        Generator suite configuration.
    descriptor_config : DescriptorConfig
        Descriptor extraction configuration.

    Returns
    -------
    dict[str, Any]
        Record with keys ``dataset_id``, ``regime``, ``realization``, ``seed``,
        ``surrogate_kind``, ``matrix``, ``series_length``.
    """
    audit = config.audit
    gen_cfg = replace(
        generator_config,
        global_params=replace(
            generator_config.global_params,
            n_steps=audit.n_steps,
            seed=audit.seed,
            n_samples=1,
        ),
    )
    ts2g_cfg = TS2GSuiteConfig(methods=audit.methods)
    transformers = {
        t.method: t
        for t in build_transformers(ts2g_cfg)
        if t.method in audit.methods
    }
    pipeline = DescriptorPipeline(config=descriptor_config)

    generator = build_generator(dataset_id, gen_cfg)
    seed = realization_seed(audit.seed, dataset_id, r)
    series = generator.generate(audit.n_steps, seed=seed)
    values = np.asarray(series.values, dtype=float).ravel()

    rng = np.random.default_rng(_surrogate_seed(seed, kind, r))
    surr = make_surrogate(
        values,
        kind,
        rng,
        max_iter=config.iaaft_max_iter,
        min_iter=config.iaaft_min_iter,
        tol=config.iaaft_tol,
    )
    matrix = _extract_matrix_for_series(
        surr,
        dataset_id=dataset_id,
        audit_config=audit,
        transformers=transformers,
        pipeline=pipeline,
    )
    return {
        "dataset_id": dataset_id,
        "regime": REGIME_LABELS[dataset_id],
        "realization": r,
        "seed": seed,
        "surrogate_kind": kind,
        "matrix": matrix,
        "series_length": int(values.size),
    }


def build_surrogate_records(
    config: SurrogateAuditConfig,
    *,
    generator_config: GeneratorSuiteConfig | None = None,
    descriptor_config: DescriptorConfig | None = None,
    n_workers: int = 1,
) -> list[dict[str, Any]]:
    """Generate original + surrogate descriptor matrices for the panel.

    Parameters
    ----------
    config : SurrogateAuditConfig
        Full surrogate audit configuration.
    generator_config : GeneratorSuiteConfig | None
        Generator configuration; loaded from defaults when ``None``.
    descriptor_config : DescriptorConfig | None
        Descriptor extraction configuration.
    n_workers : int
        Number of parallel workers.  ``1`` disables parallelism.
    """
    audit = config.audit
    gen_cfg = generator_config or load_generator_suite_config()
    desc_cfg = descriptor_config or DescriptorConfig()

    jobs: list[tuple[str, int, str]] = [
        (dataset_id, r, kind)
        for dataset_id in audit.regimes
        for r in range(audit.n_realizations)
        for kind in config.kinds
    ]
    total = len(jobs)
    n_workers = max(1, int(n_workers))

    if n_workers > 1:
        from joblib import Parallel, delayed

        logger.info("Building %d surrogate records with %d workers", total, n_workers)
        records: list[dict[str, Any]] = Parallel(n_jobs=n_workers)(
            delayed(_surrogate_job)(
                did, r, kind,
                config=config,
                generator_config=gen_cfg,
                descriptor_config=desc_cfg,
            )
            for did, r, kind in jobs
        )
    else:
        records = []
        for idx, (did, r, kind) in enumerate(jobs, 1):
            logger.info("Surrogate %d/%d regime=%s r=%d kind=%s", idx, total, did, r, kind)
            records.append(
                _surrogate_job(
                    did, r, kind,
                    config=config,
                    generator_config=gen_cfg,
                    descriptor_config=desc_cfg,
                )
            )
    return records


def _records_to_xy(
    records: Sequence[dict[str, Any]],
    methods: Sequence[str],
    *,
    label_key: str,
) -> tuple[pd.DataFrame, pd.Series]:
    rows: list[dict[str, float]] = []
    labels: list[str] = []
    for rec in records:
        rows.append(matrix_to_feature_row(rec["matrix"], methods=methods))
        labels.append(str(rec[label_key]))
    x = pd.DataFrame(rows, columns=all_feature_names(methods, DESCRIPTOR_IDS))
    y = pd.Series(labels, name=label_key)
    return x, y


def _view_specs(columns: Sequence[str]) -> list[tuple[str, np.ndarray]]:
    specs: list[tuple[str, np.ndarray]] = [
        ("visibility", family_feature_mask(columns, "visibility")),
        ("transition", family_feature_mask(columns, "transition")),
        ("proximity", family_feature_mask(columns, "proximity")),
        ("similarity", family_feature_mask(columns, "similarity")),
        ("full", np.ones(len(columns), dtype=bool)),
    ]
    return [(name, mask) for name, mask in specs if mask.any()]


def _evaluate_family_views(
    x: pd.DataFrame,
    y: pd.Series,
    audit: AuditConfig,
    *,
    labels: Sequence[str],
) -> dict[str, dict[str, dict[str, float]]]:
    """Macro-F1 (RF + LR) for each family view."""
    cfg = replace(audit, regimes=tuple(labels))
    models = _make_models(cfg.seed)
    out: dict[str, dict[str, dict[str, float]]] = {}
    for view_name, mask in _view_specs(list(x.columns)):
        x_view = x.loc[:, mask]
        out[view_name] = {}
        for model_name, model in models.items():
            metrics: ViewMetrics = _evaluate_view(
                x_view,
                y,
                view=view_name,
                model_name=model_name,
                model=clone(model),
                config=cfg,
            )
            out[view_name][model_name] = {
                "macro_f1_mean": metrics.macro_f1_mean,
                "macro_f1_std": metrics.macro_f1_std,
                "accuracy_mean": metrics.accuracy_mean,
            }
    return out


def _binary_original_vs_iaaft(
    records: Sequence[dict[str, Any]],
    audit: AuditConfig,
) -> dict[str, Any]:
    """Per-regime binary detection of original vs IAAFT (Macro-F1 + AUROC)."""
    models = _make_models(audit.seed)
    by_regime: dict[str, Any] = {}
    for regime in audit.regimes:
        subset = [
            r
            for r in records
            if r["dataset_id"] == regime and r["surrogate_kind"] in {"original", "iaaft"}
        ]
        if len(subset) < 4:
            continue
        x, y = _records_to_xy(subset, audit.methods, label_key="surrogate_kind")
        # Map to {0,1} with original=1 for AUROC orientation.
        y_bin = (y == "original").astype(int)
        health = diagnose_and_clean(x, audit)
        x_clean = health.x_clean
        regime_views: dict[str, Any] = {}
        for view_name, mask in _view_specs(list(x_clean.columns)):
            x_view = x_clean.loc[:, mask]
            view_metrics: dict[str, Any] = {}
            for model_name, model in models.items():
                f1s: list[float] = []
                aucs: list[float] = []
                n_splits = min(audit.n_folds, int(y_bin.value_counts().min()))
                if n_splits < 2:
                    continue
                skf = StratifiedKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=audit.seed,
                )
                xa = x_view.to_numpy(dtype=float)
                ya = y_bin.to_numpy()
                for train_idx, test_idx in skf.split(xa, ya):
                    clf: Pipeline = clone(model)
                    clf.fit(xa[train_idx], ya[train_idx])
                    pred = clf.predict(xa[test_idx])
                    f1s.append(
                        float(
                            f1_score(
                                ya[test_idx],
                                pred,
                                average="macro",
                                zero_division=0,
                            )
                        )
                    )
                    if hasattr(clf, "predict_proba"):
                        proba = clf.predict_proba(xa[test_idx])[:, 1]
                    else:
                        proba = clf.decision_function(xa[test_idx])
                    # AUROC needs both classes in the fold.
                    if len(np.unique(ya[test_idx])) < 2:
                        continue
                    aucs.append(float(roc_auc_score(ya[test_idx], proba)))
                view_metrics[model_name] = {
                    "macro_f1_mean": float(np.mean(f1s)) if f1s else float("nan"),
                    "macro_f1_std": float(np.std(f1s)) if f1s else float("nan"),
                    "auroc_mean": float(np.mean(aucs)) if aucs else float("nan"),
                    "auroc_std": float(np.std(aucs)) if aucs else float("nan"),
                }
            regime_views[view_name] = view_metrics
        by_regime[regime] = regime_views
    return by_regime


def compute_surrogate_metrics(
    records: Sequence[dict[str, Any]],
    config: SurrogateAuditConfig,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Compute Δ_T, binary Δ_NL and multiclass surrogate-type metrics."""
    audit = config.audit

    # --- Regime multiclass per surrogate kind (basis for Δ_T) ---
    regime_by_surr: dict[str, Any] = {}
    for kind in config.kinds:
        subset = [r for r in records if r["surrogate_kind"] == kind]
        x, y = _records_to_xy(subset, audit.methods, label_key="dataset_id")
        health = diagnose_and_clean(x, audit)
        regime_by_surr[kind] = _evaluate_family_views(
            health.x_clean,
            y,
            audit,
            labels=audit.regimes,
        )

    temporal: dict[str, Any] = {"per_family": {}, "definition": "F1(original) - F1(shuffled)"}
    for family in FAMILY_VIEWS:
        temporal["per_family"][family] = {}
        for model in ("random_forest", "logistic"):
            f1_orig = (
                regime_by_surr.get("original", {})
                .get(family, {})
                .get(model, {})
                .get("macro_f1_mean", float("nan"))
            )
            f1_shuf = (
                regime_by_surr.get("shuffled", {})
                .get(family, {})
                .get(model, {})
                .get("macro_f1_mean", float("nan"))
            )
            temporal["per_family"][family][model] = {
                "macro_f1_original": f1_orig,
                "macro_f1_shuffled": f1_shuf,
                "delta_T": float(f1_orig - f1_shuf)
                if np.isfinite(f1_orig) and np.isfinite(f1_shuf)
                else float("nan"),
            }

    # --- Multiclass: classify surrogate kind (original/shuffled/ft/iaaft) ---
    x_all, y_kind = _records_to_xy(records, audit.methods, label_key="surrogate_kind")
    health_all = diagnose_and_clean(x_all, audit)
    multiclass = _evaluate_family_views(
        health_all.x_clean,
        y_kind,
        audit,
        labels=list(config.kinds),
    )

    # Also original vs ft vs iaaft only (Δ_NL-b without shuffled).
    subset_nl = [r for r in records if r["surrogate_kind"] in {"original", "ft", "iaaft"}]
    x_nl, y_nl = _records_to_xy(subset_nl, audit.methods, label_key="surrogate_kind")
    health_nl = diagnose_and_clean(x_nl, audit)
    multiclass_nl = _evaluate_family_views(
        health_nl.x_clean,
        y_nl,
        audit,
        labels=("original", "ft", "iaaft"),
    )

    nonlinearity_multiclass = {
        "original_shuffled_ft_iaaft": multiclass,
        "original_ft_iaaft": multiclass_nl,
    }

    binary = _binary_original_vs_iaaft(records, audit)
    return temporal, binary, nonlinearity_multiclass, regime_by_surr


def _plot_summary_figure(
    result: SurrogateAuditResult,
    output_dir: Path,
    lang: str = "en",
) -> list[Path]:
    """Bar charts for Δ_T and multiclass Macro-F1 by family.

    Parameters
    ----------
    result : SurrogateAuditResult
        Computed surrogate audit result.
    output_dir : Path
        Directory for figure files.
    lang : str
        Language code for i18n titles (``'en'`` or ``'es'``).

    Returns
    -------
    list[Path]
        Paths written (pdf + svg + png per figure).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    families = list(FAMILY_VIEWS)
    delta_t = [
        result.temporal_sensitivity["per_family"]
        .get(f, {})
        .get("random_forest", {})
        .get("delta_T", float("nan"))
        for f in families
    ]
    multi = result.nonlinearity_multiclass.get("original_shuffled_ft_iaaft", {})
    multi_f1 = [
        multi.get(f, {}).get("random_forest", {}).get("macro_f1_mean", float("nan"))
        for f in families
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = ["#0f5c4c", "#1b7f4a", "#3d7ea6", "#9a6b00", "#8b2e1a"]
    axes[0].bar(families, delta_t, color=colors[: len(families)])
    axes[0].axhline(0.0, color="#333", lw=0.8)
    axes[0].set_title(
        r"$\Delta_T = \mathrm{F1}(\mathrm{original}) - \mathrm{F1}(\mathrm{shuffled})$ [RF]"
    )
    axes[0].set_ylabel(_t("surrogate_delta_ylabel", lang))
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(families, multi_f1, color=colors[: len(families)])
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title(_t("surrogate_multi_title", lang))
    axes[1].set_ylabel("Macro-F1")
    axes[1].tick_params(axis="x", rotation=20)

    fig.suptitle(_t("surrogate_summary_suptitle", lang), fontsize=11)
    fig.tight_layout()
    paths.extend(save_figure_trio(fig, output_dir / "surrogate_summary"))

    # Heatmap: binary AUROC (RF) regime × family
    regimes = list(result.config.audit.regimes)
    fam_core = ["visibility", "transition", "proximity", "similarity", "full"]
    mat = np.full((len(regimes), len(fam_core)), np.nan)
    for i, regime in enumerate(regimes):
        for j, fam in enumerate(fam_core):
            mat[i, j] = (
                result.nonlinearity_binary.get(regime, {})
                .get(fam, {})
                .get("random_forest", {})
                .get("auroc_mean", float("nan"))
            )

    fig2, ax = plt.subplots(figsize=(8.5, 4.5))
    im = ax.imshow(mat, aspect="auto", vmin=0.4, vmax=1.0, cmap="YlGn")
    ax.set_xticks(range(len(fam_core)))
    ax.set_xticklabels(fam_core, rotation=20)
    ax.set_yticks(range(len(regimes)))
    ax.set_yticklabels(regimes)
    ax.set_title(_t("surrogate_auroc_title", lang))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig2.tight_layout()
    paths.extend(save_figure_trio(fig2, output_dir / "surrogate_binary_auroc"))
    return paths


def _format_markdown(result: SurrogateAuditResult) -> str:
    lines: list[str] = [
        "# Auditoría de subrogados (Bloque 2)",
        "",
        f"- Generado: `{result.generated_at_utc}`",
        f"- Profile: `{result.config.audit.profile}`",
        f"- R={result.config.audit.n_realizations}, N={result.config.audit.n_steps}",
        f"- Métodos: `{', '.join(result.config.audit.methods)}`",
        f"- Regímenes: `{', '.join(result.config.audit.regimes)}`",
        (
            f"- IAAFT: `max_iter={result.config.iaaft_max_iter}`, "
            f"`min_iter={result.config.iaaft_min_iter}`, "
            f"`tol={result.config.iaaft_tol:g}`, "
            f"criterio de convergencia: `{IAAFT_CONVERGENCE_CRITERION}` "
            "(error espectral relativo "
            r"`|| |rFFT(y)| - |rFFT(x)| ||_2 / || |rFFT(x)| ||_2`)."
        ),
        "",
        "## Sensibilidad temporal (Δ_T)",
        "",
        "| Familia | RF Δ_T | LR Δ_T | RF F1 orig | RF F1 shuf |",
        "|---|---:|---:|---:|---:|",
    ]
    for fam in FAMILY_VIEWS:
        block = result.temporal_sensitivity["per_family"].get(fam, {})
        rf = block.get("random_forest", {})
        lr = block.get("logistic", {})
        lines.append(
            f"| {fam} | {rf.get('delta_T', float('nan')):.3f} | "
            f"{lr.get('delta_T', float('nan')):.3f} | "
            f"{rf.get('macro_f1_original', float('nan')):.3f} | "
            f"{rf.get('macro_f1_shuffled', float('nan')):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Macro-F1 multiclase (original / shuffled / ft / iaaft)",
            "",
            "| Familia | RF Macro-F1 | LR Macro-F1 |",
            "|---|---:|---:|",
        ]
    )
    multi = result.nonlinearity_multiclass.get("original_shuffled_ft_iaaft", {})
    for fam in FAMILY_VIEWS:
        rf = multi.get(fam, {}).get("random_forest", {})
        lr = multi.get(fam, {}).get("logistic", {})
        lines.append(
            f"| {fam} | {rf.get('macro_f1_mean', float('nan')):.3f} | "
            f"{lr.get('macro_f1_mean', float('nan')):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Detección binaria original vs IAAFT (RF AUROC)",
            "",
            "| Régimen | visibility | transition | proximity | similarity | full |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for regime in result.config.audit.regimes:
        cells = []
        for fam in ("visibility", "transition", "proximity", "similarity", "full"):
            auroc = (
                result.nonlinearity_binary.get(regime, {})
                .get(fam, {})
                .get("random_forest", {})
                .get("auroc_mean", float("nan"))
            )
            cells.append(f"{auroc:.3f}")
        lines.append(f"| {regime} | " + " | ".join(cells) + " |")

    lines.extend(["", "## Figuras", ""])
    for p in result.figure_paths:
        lines.append(f"- `{p.name}`")
    lines.append("")
    return "\n".join(lines)


class SurrogateAuditPipeline:
    """Orchestrate surrogate generation → D matrices → Δ_T / Δ_NL metrics."""

    def __init__(
        self,
        config: SurrogateAuditConfig | None = None,
        descriptor_config: DescriptorConfig | None = None,
        *,
        lang: str = "en",
        n_workers: int = 1,
    ) -> None:
        self.config = config or SurrogateAuditConfig()
        self.descriptor_config = descriptor_config or DescriptorConfig()
        self.lang = lang
        self.n_workers = max(1, int(n_workers))

    def run(
        self,
        *,
        records: list[dict[str, Any]] | None = None,
        output_dir: Path | str | None = None,
    ) -> SurrogateAuditResult:
        cfg = self.config
        logger.info(
            "Surrogate audit profile=%s R=%d N=%d regimes=%s methods=%s kinds=%s "
            "iaaft_max_iter=%d iaaft_min_iter=%d iaaft_tol=%g criterion=%s workers=%d",
            cfg.audit.profile,
            cfg.audit.n_realizations,
            cfg.audit.n_steps,
            cfg.audit.regimes,
            cfg.audit.methods,
            cfg.kinds,
            cfg.iaaft_max_iter,
            cfg.iaaft_min_iter,
            cfg.iaaft_tol,
            IAAFT_CONVERGENCE_CRITERION,
            self.n_workers,
        )
        if records is None:
            records = build_surrogate_records(
                cfg,
                descriptor_config=self.descriptor_config,
                n_workers=self.n_workers,
            )
        temporal, binary, multiclass, regime_by_surr = compute_surrogate_metrics(
            records, cfg
        )
        result = SurrogateAuditResult(
            config=cfg,
            records=records,
            temporal_sensitivity=temporal,
            nonlinearity_binary=binary,
            nonlinearity_multiclass=multiclass,
            regime_multiclass_by_surrogate=regime_by_surr,
        )
        if output_dir is not None:
            self.persist(result, Path(output_dir))
        return result

    def persist(self, result: SurrogateAuditResult, output_dir: Path) -> Path:
        """Write metrics, NPZ/CSV data, figures, markdown and HTML."""
        output_dir.mkdir(parents=True, exist_ok=True)
        data_dir = output_dir / "data"
        tables_dir = data_dir / "tables"
        data_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)

        # --- data persistence ---
        self._persist_data(result, data_dir, tables_dir)

        # --- figures ---
        fig_dir = output_dir / "figures"
        result.figure_paths = _plot_summary_figure(result, fig_dir, lang=self.lang)

        # --- JSON / reports ---
        metrics_path = output_dir / "surrogate_metrics.json"
        atomic_write_text(
            metrics_path,
            json.dumps(result.metrics_dict(), indent=2, default=str),
        )
        write_metrics_alias(output_dir, metrics_path)
        write_config_snapshot(
            output_dir,
            block="phase2_surrogates",
            profile=result.config.audit.profile,
            seed=result.config.audit.seed,
            config=result.metrics_dict().get("config", {}),
            extra={"lang": self.lang, "n_workers": self.n_workers},
        )
        md_path = output_dir / "surrogate_report.md"
        atomic_write_text(md_path, _format_markdown(result))

        html = _markdown_tables_to_html(result)
        atomic_write_text(output_dir / "latest.html", html)
        atomic_write_text(output_dir / "surrogate_report.html", html)
        logger.info("Surrogate audit artifacts written to %s", output_dir)
        return output_dir

    def plot_only(
        self,
        output_dir: Path | str,
        *,
        lang: str | None = None,
    ) -> SurrogateAuditResult:
        """Regenerate figures from persisted records without recomputing graphs.

        Parameters
        ----------
        output_dir : Path | str
            Directory that was previously written by :meth:`persist`.
        lang : str | None
            Override language; falls back to the stored value or ``self.lang``.

        Returns
        -------
        SurrogateAuditResult
            Reconstructed result with updated ``figure_paths``.
        """
        out = Path(output_dir)
        data_dir = out / "data"
        metrics_path = out / "surrogate_metrics.json"
        if not metrics_path.exists():
            metrics_path = out / "metrics.json"
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))

        records_path = data_dir / "records.pkl"
        if not records_path.exists():
            raise FileNotFoundError(
                f"Missing {records_path}; cannot regenerate figures"
            )
        with records_path.open("rb") as fh:
            records = pickle.load(fh)

        temporal, binary, multiclass, regime_by_surr = compute_surrogate_metrics(
            records, self.config
        )
        use_lang = lang or payload.get("lang") or self.lang
        result = SurrogateAuditResult(
            config=self.config,
            records=records,
            temporal_sensitivity=temporal,
            nonlinearity_binary=binary,
            nonlinearity_multiclass=multiclass,
            regime_multiclass_by_surrogate=regime_by_surr,
            generated_at_utc=payload.get(
                "generated_at_utc", datetime.now(timezone.utc).isoformat()
            ),
        )
        fig_dir = out / "figures"
        result.figure_paths = _plot_summary_figure(result, fig_dir, lang=use_lang)
        payload["figures"] = [str(p) for p in result.figure_paths]
        payload["lang"] = use_lang
        atomic_write_text(
            out / "surrogate_metrics.json",
            json.dumps(payload, indent=2, default=str),
        )
        write_metrics_alias(out, out / "surrogate_metrics.json")
        logger.info("Plot-only figures regenerated in %s (lang=%s)", out, use_lang)
        return result

    def _persist_data(
        self,
        result: SurrogateAuditResult,
        data_dir: Path,
        tables_dir: Path,
    ) -> None:
        """Write features NPZ, CSV tables and records pickle."""
        records = result.records
        audit = result.config.audit

        # Features with y=dataset_id and transform_or_kind=surrogate_kind
        rows: list[dict[str, float]] = []
        y_labels: list[str] = []
        kinds: list[str] = []
        realization_ids: list[int] = []
        for rec in records:
            rows.append(matrix_to_feature_row(rec["matrix"], methods=audit.methods))
            y_labels.append(str(rec["dataset_id"]))
            kinds.append(str(rec["surrogate_kind"]))
            realization_ids.append(int(rec["realization"]))

        x_raw = pd.DataFrame(
            rows, columns=all_feature_names(audit.methods, DESCRIPTOR_IDS)
        )
        y = pd.Series(y_labels, name="regime")
        save_features_npz(
            data_dir / "features_raw.npz",
            x=x_raw,
            y=y,
            regime_names=list(audit.regimes),
            realization_ids=realization_ids,
            transform_or_kind=kinds,
        )
        health = diagnose_and_clean(x_raw, audit)
        save_features_npz(
            data_dir / "features_clean.npz",
            x=health.x_clean,
            y=y,
            regime_names=list(audit.regimes),
            realization_ids=realization_ids,
            transform_or_kind=kinds,
        )
        # OOF on original-series rows only (regime discrimination baseline).
        original_mask = np.asarray([k == "original" for k in kinds], dtype=bool)
        if original_mask.any() and health.x_clean.shape[1] > 0:
            x_orig = health.x_clean.iloc[original_mask]
            y_orig = y.iloc[original_mask]
            keep = x_orig.nunique(dropna=False) > 1
            x_orig = x_orig.loc[:, keep]
            if x_orig.shape[1] > 0 and int(y_orig.nunique()) >= 2:
                clf = evaluate_classification(x_orig, y_orig, audit)
                save_oof_predictions(
                    data_dir / "oof_predictions.npz",
                    clf.oof_bundles(),
                )

        # CSV tables
        delta_rows = []
        for fam in FAMILY_VIEWS:
            block = result.temporal_sensitivity["per_family"].get(fam, {})
            for model_name in ("random_forest", "logistic"):
                m = block.get(model_name, {})
                delta_rows.append({
                    "family": fam,
                    "model": model_name,
                    "macro_f1_original": m.get("macro_f1_original", float("nan")),
                    "macro_f1_shuffled": m.get("macro_f1_shuffled", float("nan")),
                    "delta_T": m.get("delta_T", float("nan")),
                })
        save_table_csv(tables_dir / "delta_T.csv", pd.DataFrame(delta_rows))

        auroc_rows = []
        for regime in audit.regimes:
            for fam in FAMILY_VIEWS:
                for model_name in ("random_forest", "logistic"):
                    m = (
                        result.nonlinearity_binary.get(regime, {})
                        .get(fam, {})
                        .get(model_name, {})
                    )
                    auroc_rows.append({
                        "regime": regime,
                        "family": fam,
                        "model": model_name,
                        "auroc_mean": m.get("auroc_mean", float("nan")),
                        "auroc_std": m.get("auroc_std", float("nan")),
                        "macro_f1_mean": m.get("macro_f1_mean", float("nan")),
                    })
        save_table_csv(tables_dir / "auroc_binary.csv", pd.DataFrame(auroc_rows))

        multi = result.nonlinearity_multiclass.get("original_shuffled_ft_iaaft", {})
        multi_rows = []
        for fam in FAMILY_VIEWS:
            for model_name in ("random_forest", "logistic"):
                m = multi.get(fam, {}).get(model_name, {})
                multi_rows.append({
                    "family": fam,
                    "model": model_name,
                    "macro_f1_mean": m.get("macro_f1_mean", float("nan")),
                    "macro_f1_std": m.get("macro_f1_std", float("nan")),
                })
        save_table_csv(tables_dir / "multiclass_f1.csv", pd.DataFrame(multi_rows))

        # Records pickle for plot_only
        records_path = data_dir / "records.pkl"
        with records_path.open("wb") as fh:
            pickle.dump(records, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Persisted surrogate data to %s", data_dir)


def _markdown_tables_to_html(result: SurrogateAuditResult) -> str:
    import base64

    def _img(path: Path) -> str:
        if not path.exists() or path.suffix.lower() != ".png":
            return ""
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'<img class="fig" alt="{path.name}" src="data:image/png;base64,{b64}"/>'

    md = _format_markdown(result)
    # Very small converter: wrap preformatted markdown in <pre> plus figures.
    imgs = "".join(_img(p) for p in result.figure_paths if p.suffix == ".png")
    return f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"/>
<title>Auditoría de subrogados</title>
<style>
body{{font-family:IBM Plex Sans,Segoe UI,sans-serif;max-width:1000px;margin:1.5rem auto;padding:0 1rem;background:#f7f5f1;color:#1c1a16}}
pre{{white-space:pre-wrap;background:#fff;border:1px solid #ddd6c8;border-radius:10px;padding:1rem}}
img.fig{{width:100%;margin:1rem 0;border:1px solid #ddd6c8;border-radius:8px}}
</style></head><body>
<h1>Auditoría de subrogados (Bloque 2)</h1>
{imgs}
<pre>{md}</pre>
</body></html>
"""
