"""Surrogate audit: temporal sensitivity and genuine nonlinearity (Bloque 2)."""

from __future__ import annotations

import json
import logging
import pickle
import time
import warnings
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import rankdata, wilcoxon
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
    feature_name,
    matrix_to_feature_row,
)
from tsg.audit.health import diagnose_and_clean
from tsg.audit.i18n import t as _t
from tsg.descriptors.base import DESCRIPTOR_IDS
from tsg.descriptors.config import DescriptorConfig
from tsg.descriptors.directional import temporal_irreversibility_kl
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
from tsg.graphs.geometry import DHVGTransformer
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
    dhvg_irreversibility: pd.DataFrame | None = None
    dhvg_vs_hvg_auroc: pd.DataFrame | None = None

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
            "dhvg_irreversibility": _frame_records(self.dhvg_irreversibility),
            "dhvg_vs_hvg_auroc": _frame_records(self.dhvg_vs_hvg_auroc),
            "figures": [str(p) for p in self.figure_paths],
        }


def _frame_records(frame: pd.DataFrame | None) -> list[dict[str, Any]] | None:
    """JSON-ready records for a table, or ``None`` when the table was not built."""
    if frame is None:
        return None
    return frame.to_dict(orient="records")


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


def _finite_diff(a: float, b: float) -> float:
    """Return ``a - b`` when both are finite, else NaN."""
    if np.isfinite(a) and np.isfinite(b):
        return float(a - b)
    return float("nan")


def _temporal_row(
    *,
    f1_orig: float,
    f1_shuf: float,
    f1_ft: float,
    f1_iaaft: float,
) -> dict[str, float]:
    """Additive Δ_T decomposition for one (family, model) cell."""
    return {
        "macro_f1_original": float(f1_orig),
        "macro_f1_shuffled": float(f1_shuf),
        "macro_f1_ft": float(f1_ft),
        "macro_f1_iaaft": float(f1_iaaft),
        "delta_T": _finite_diff(f1_orig, f1_shuf),
        "delta_marginal": float(f1_shuf),
        "delta_linear": _finite_diff(f1_ft, f1_shuf),
        "delta_iaaft_minus_ft": _finite_diff(f1_iaaft, f1_ft),
        "delta_nonlinear": _finite_diff(f1_orig, f1_iaaft),
    }


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

    temporal: dict[str, Any] = {
        "per_family": {},
        "definition": "F1(original) - F1(shuffled)",
        "decomposition": (
            "delta_marginal = F1(shuffled); "
            "delta_linear = F1(ft) - F1(shuffled); "
            "delta_iaaft_minus_ft = F1(iaaft) - F1(ft); "
            "delta_nonlinear = F1(original) - F1(iaaft); "
            "F1(original) = sum of the four terms"
        ),
    }
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
            f1_ft = (
                regime_by_surr.get("ft", {})
                .get(family, {})
                .get(model, {})
                .get("macro_f1_mean", float("nan"))
            )
            f1_iaaft = (
                regime_by_surr.get("iaaft", {})
                .get(family, {})
                .get(model, {})
                .get("macro_f1_mean", float("nan"))
            )
            temporal["per_family"][family][model] = _temporal_row(
                f1_orig=f1_orig,
                f1_shuf=f1_shuf,
                f1_ft=f1_ft,
                f1_iaaft=f1_iaaft,
            )

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
    paths.extend(_plot_delta_t_decomposition(result, output_dir, lang=lang))
    return paths


def _plot_delta_t_decomposition(
    result: SurrogateAuditResult,
    output_dir: Path,
    *,
    lang: str = "en",
) -> list[Path]:
    """Stacked bars of the additive Macro-F1 decomposition by family."""
    del lang  # titles are equation-based; kept for API symmetry with other plots
    families = list(FAMILY_VIEWS)
    components = (
        ("delta_marginal", "#9e9e9e", r"$F1_{\mathrm{shuf}}$ (marginal)"),
        ("delta_linear", "#3d7ea6", r"$\Delta_{\mathrm{lin}}=F1_{\mathrm{ft}}-F1_{\mathrm{shuf}}$"),
        (
            "delta_iaaft_minus_ft",
            "#9a6b00",
            r"$\Delta_{\mathrm{IAAFT-FT}}=F1_{\mathrm{iaaft}}-F1_{\mathrm{ft}}$",
        ),
        (
            "delta_nonlinear",
            "#8b2e1a",
            r"$\Delta_{\mathrm{NL}}=F1_{\mathrm{orig}}-F1_{\mathrm{iaaft}}$",
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), sharey=True)
    x = np.arange(len(families))
    width = 0.62
    for ax, model, title in zip(
        axes,
        ("random_forest", "logistic"),
        ("Random Forest", "Logistic L2"),
        strict=True,
    ):
        bottoms = np.zeros(len(families), dtype=float)
        for key, color, label in components:
            heights = np.array(
                [
                    float(
                        result.temporal_sensitivity["per_family"]
                        .get(fam, {})
                        .get(model, {})
                        .get(key, float("nan"))
                    )
                    for fam in families
                ],
                dtype=float,
            )
            heights = np.nan_to_num(heights, nan=0.0)
            ax.bar(
                x,
                heights,
                width=width,
                bottom=bottoms,
                color=color,
                edgecolor="#333333",
                linewidth=0.35,
                label=label,
            )
            bottoms = bottoms + heights
        f1_orig = np.array(
            [
                float(
                    result.temporal_sensitivity["per_family"]
                    .get(fam, {})
                    .get(model, {})
                    .get("macro_f1_original", float("nan"))
                )
                for fam in families
            ],
            dtype=float,
        )
        ax.scatter(
            x,
            f1_orig,
            marker="D",
            s=28,
            color="#111111",
            zorder=5,
            label=r"$F1_{\mathrm{orig}}$",
        )
        ax.set_xticks(x, families, rotation=20, ha="right")
        ax.set_title(title)
        ax.set_ylim(0.0, 1.08)
        ax.yaxis.grid(True, linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Macro-F1 (additive stack)")
    axes[1].legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.suptitle(
        r"Additive $\Delta_T$ decomposition "
        r"($F1_{\mathrm{orig}}=F1_{\mathrm{shuf}}+\Delta_{\mathrm{lin}}"
        r"+\Delta_{\mathrm{IAAFT-FT}}+\Delta_{\mathrm{NL}}$)",
        fontsize=11,
    )
    fig.tight_layout()
    return list(save_figure_trio(fig, output_dir / "delta_T_decomposition"))


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

    _append_dhvg_markdown(lines, result)
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
        result.figure_paths.extend(
            write_dhvg_irreversibility_artifacts(
                result,
                result.records,
                output_dir,
                lang=self.lang,
                n_workers=self.n_workers,
            )
        )

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
        """Regenerate figures and the dHVG irreversibility table.

        Saved Δ_T / AUROC blocks are reused when present. ``kl_inout`` is read
        from the matrices when available; otherwise only dHVG is rebuilt.

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

        use_lang = lang or payload.get("lang") or self.lang
        temporal = payload.get("temporal_sensitivity_delta_T")
        binary = payload.get("nonlinearity_binary_original_vs_iaaft")
        multiclass = payload.get("nonlinearity_multiclass_surrogate_type")
        regime_by_surr = payload.get("regime_multiclass_by_surrogate")
        records: list[dict[str, Any]] = []
        persisted = (temporal, binary, multiclass, regime_by_surr)
        if all(isinstance(block, dict) for block in persisted):
            logger.info(
                "Plot-only surrogate figures from persisted metrics (no recompute)"
            )
        else:
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
        records_path = data_dir / "records.pkl"
        irr_records = records
        if records_path.exists():
            with records_path.open("rb") as fh:
                irr_records = pickle.load(fh)
        result.figure_paths.extend(
            write_dhvg_irreversibility_artifacts(
                result,
                irr_records,
                out,
                lang=use_lang,
                n_workers=self.n_workers,
            )
        )
        payload["figures"] = [str(p) for p in result.figure_paths]
        payload["lang"] = use_lang
        payload["dhvg_irreversibility_table"] = str(
            data_dir / "tables" / "dhvg_irreversibility.csv"
        )
        payload["dhvg_vs_hvg_auroc_table"] = str(
            data_dir / "tables" / "dhvg_vs_hvg_auroc.csv"
        )
        payload["dhvg_irreversibility"] = _frame_records(result.dhvg_irreversibility)
        payload["dhvg_vs_hvg_auroc"] = _frame_records(result.dhvg_vs_hvg_auroc)
        atomic_write_text(
            out / "surrogate_metrics.json",
            json.dumps(payload, indent=2, default=str),
        )
        write_metrics_alias(out, out / "surrogate_metrics.json")
        md_path = out / "surrogate_report.md"
        atomic_write_text(md_path, _format_markdown(result))
        html = _markdown_tables_to_html(result)
        atomic_write_text(out / "latest.html", html)
        atomic_write_text(out / "surrogate_report.html", html)
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
        decomposed_rows = []
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
                decomposed_rows.append({
                    "family": fam,
                    "model": model_name,
                    "macro_f1_original": m.get("macro_f1_original", float("nan")),
                    "macro_f1_shuffled": m.get("macro_f1_shuffled", float("nan")),
                    "macro_f1_ft": m.get("macro_f1_ft", float("nan")),
                    "macro_f1_iaaft": m.get("macro_f1_iaaft", float("nan")),
                    "delta_T": m.get("delta_T", float("nan")),
                    "delta_marginal": m.get("delta_marginal", float("nan")),
                    "delta_linear": m.get("delta_linear", float("nan")),
                    "delta_iaaft_minus_ft": m.get(
                        "delta_iaaft_minus_ft", float("nan")
                    ),
                    "delta_nonlinear": m.get("delta_nonlinear", float("nan")),
                })
        save_table_csv(tables_dir / "delta_T.csv", pd.DataFrame(delta_rows))
        save_table_csv(
            tables_dir / "delta_T_decomposed.csv",
            pd.DataFrame(decomposed_rows),
        )

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


_DHVG_METRIC_COLUMNS: tuple[tuple[str, str], ...] = (
    ("dHVG__kl_inout", "kl_inout"),
    ("dHVG__i_dir", "i_dir"),
)
_DHVG_CONTRASTS: tuple[str, ...] = ("shuffled", "ft", "iaaft")
_DHVG_KIND_STYLE: tuple[tuple[str, str, str], ...] = (
    ("original", "#1b3a4b", "original"),
    ("shuffled", "#8d8d8d", "shuffled"),
    ("ft", "#3d7ea6", "FT"),
    ("iaaft", "#c47b2b", "IAAFT"),
)
_DHVG_LONG_COLUMNS: tuple[str, ...] = (
    "dataset_id",
    "regime",
    "realization",
    "surrogate_kind",
    "i_dir",
    "kl_inout",
    "kl_source",
)
_ISOLATED_VISIBILITY_FAMILIES: tuple[str, ...] = (
    "HVG",
    "dHVG",
    "dHVG_plus_kl",
    "kl_inout_only",
)


def _sample_std(values: NDArray[np.floating]) -> float:
    """Sample standard deviation (``ddof=1``); NaN when ``n < 2``."""
    if values.size < 2:
        return float("nan")
    return float(np.std(values, ddof=1))


def _fmt_num(value: object, spec: str = ".3f") -> str:
    """Format a finite float, otherwise ``nan``."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(number):
        return "nan"
    return format(number, spec)


def _records_store_kl(records: Sequence[dict[str, Any]]) -> bool:
    """True when every saved matrix already carries finite ``dHVG`` ``kl_inout``."""
    if not records:
        return False
    for rec in records:
        matrix = rec.get("matrix")
        if matrix is None or "kl_inout" not in matrix.index or "dHVG" not in matrix.columns:
            return False
        value = float(matrix.loc["kl_inout", "dHVG"])
        if not np.isfinite(value):
            return False
    return True


def _paired_wilcoxon(
    original: NDArray[np.floating],
    surrogate: NDArray[np.floating],
) -> dict[str, float | str | int]:
    """Paired Wilcoxon signed-rank test of ``original`` against ``surrogate``.

    Effect size is the matched-pairs rank-biserial correlation
    ``(W+ − W−) / (W+ + W−)``. Positive values mean the original scores higher.
    """
    diff = np.asarray(original, dtype=float) - np.asarray(surrogate, dtype=float)
    finite = np.isfinite(diff)
    diff = diff[finite]
    n_pairs = int(diff.size)
    nonzero = diff[diff != 0.0]
    n_eff = int(nonzero.size)
    empty = {
        "test": "wilcoxon_signed_rank",
        "statistic": 0.0,
        "p_value": 1.0,
        "effect_size": 0.0,
        "effect_size_name": "rank_biserial",
        "n_pairs": n_pairs,
    }
    if n_eff == 0:
        return empty
    ranks = rankdata(np.abs(nonzero))
    w_plus = float(ranks[nonzero > 0.0].sum())
    w_minus = float(ranks[nonzero < 0.0].sum())
    effect = (w_plus - w_minus) / (w_plus + w_minus)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            result = wilcoxon(
                diff,
                zero_method="wilcox",
                alternative="two-sided",
                method="auto",
            )
        p_value = float(result.pvalue)
    except ValueError:
        p_value = float("nan")
    return {
        "test": "wilcoxon_signed_rank",
        "statistic": w_plus,
        "p_value": p_value,
        "effect_size": float(effect),
        "effect_size_name": "rank_biserial",
        "n_pairs": n_pairs,
    }


def _dhvg_kl_realization(
    dataset_id: str,
    realization: int,
    seed: int,
    kinds: tuple[str, ...],
    series_length: int,
    generator_config: GeneratorSuiteConfig,
    n_steps: int,
    iaaft_max_iter: int,
    iaaft_min_iter: int,
    iaaft_tol: float,
) -> dict[str, Any]:
    """Rebuild one realization and return ``kl_inout`` for each surrogate kind."""
    generator = build_generator(dataset_id, generator_config)
    series = generator.generate(n_steps, seed=seed)
    values = np.asarray(series.values, dtype=float).ravel()
    if int(values.size) != int(series_length):
        raise RuntimeError(
            f"Series length mismatch for {dataset_id} r={realization}: "
            f"generated {values.size}, record says {series_length}"
        )
    transformer = DHVGTransformer()
    kl_by_kind: dict[str, float] = {}
    for kind in kinds:
        rng = np.random.default_rng(_surrogate_seed(seed, kind, realization))
        surrogate = make_surrogate(
            values,
            kind,
            rng,
            max_iter=iaaft_max_iter,
            min_iter=iaaft_min_iter,
            tol=iaaft_tol,
        )
        graph = transformer.transform(surrogate, dataset_id).graph
        kl_by_kind[kind] = float(temporal_irreversibility_kl(graph))
    return {
        "dataset_id": dataset_id,
        "realization": int(realization),
        "kl": kl_by_kind,
    }


def _recompute_kl_map(
    records: Sequence[dict[str, Any]],
    config: SurrogateAuditConfig,
    generator_config: GeneratorSuiteConfig | None,
    n_workers: int,
) -> dict[tuple[str, int, str], float]:
    """Map ``(dataset_id, realization, kind)`` → ``kl_inout`` via dHVG only."""
    audit = config.audit
    base_cfg = generator_config or load_generator_suite_config()
    gen_cfg = replace(
        base_cfg,
        global_params=replace(
            base_cfg.global_params,
            n_steps=audit.n_steps,
            seed=audit.seed,
            n_samples=1,
        ),
    )
    grouped: dict[tuple[str, int], dict[str, Any]] = {}
    for rec in records:
        key = (str(rec["dataset_id"]), int(rec["realization"]))
        slot = grouped.setdefault(
            key,
            {
                "seed": int(rec.get("seed", realization_seed(audit.seed, key[0], key[1]))),
                "series_length": int(rec["series_length"]),
                "kinds": [],
            },
        )
        slot["kinds"].append(str(rec["surrogate_kind"]))

    tasks = [
        (
            dataset_id,
            realization,
            int(payload["seed"]),
            tuple(payload["kinds"]),
            int(payload["series_length"]),
        )
        for (dataset_id, realization), payload in grouped.items()
    ]
    n_workers = max(1, int(n_workers))
    logger.info(
        "dHVG kl_inout is not in records.pkl; recomputing dHVG only "
        "for %d realizations with %d workers",
        len(tasks),
        n_workers,
    )
    t0 = time.perf_counter()
    if n_workers > 1:
        from joblib import Parallel, delayed

        computed = Parallel(n_jobs=n_workers)(
            delayed(_dhvg_kl_realization)(
                dataset_id,
                realization,
                seed,
                kinds,
                series_length,
                gen_cfg,
                audit.n_steps,
                config.iaaft_max_iter,
                config.iaaft_min_iter,
                config.iaaft_tol,
            )
            for dataset_id, realization, seed, kinds, series_length in tasks
        )
    else:
        computed = [
            _dhvg_kl_realization(
                dataset_id,
                realization,
                seed,
                kinds,
                series_length,
                gen_cfg,
                audit.n_steps,
                config.iaaft_max_iter,
                config.iaaft_min_iter,
                config.iaaft_tol,
            )
            for dataset_id, realization, seed, kinds, series_length in tasks
        ]
    kl_map: dict[tuple[str, int, str], float] = {}
    for item in computed:
        for kind, value in item["kl"].items():
            kl_map[(item["dataset_id"], int(item["realization"]), kind)] = float(value)
    logger.info(
        "Recomputed dHVG kl_inout for %d series in %.2fs",
        len(kl_map),
        time.perf_counter() - t0,
    )
    return kl_map


def collect_dhvg_irreversibility(
    records: Sequence[dict[str, Any]],
    config: SurrogateAuditConfig,
    *,
    generator_config: GeneratorSuiteConfig | None = None,
    n_workers: int = 1,
) -> pd.DataFrame:
    """Long table of ``dHVG`` ``i_dir`` and ``kl_inout`` per surrogate record."""
    if not records:
        return pd.DataFrame(columns=list(_DHVG_LONG_COLUMNS))

    keys = [
        (str(rec["dataset_id"]), int(rec["realization"]), str(rec["surrogate_kind"]))
        for rec in records
    ]
    if len(keys) != len(set(keys)):
        raise ValueError(
            "Duplicate (dataset_id, realization, surrogate_kind) in surrogate records"
        )

    if _records_store_kl(records):
        logger.info("dHVG kl_inout already stored in records; skipping graph rebuild")
        kl_map = {
            (
                str(rec["dataset_id"]),
                int(rec["realization"]),
                str(rec["surrogate_kind"]),
            ): float(rec["matrix"].loc["kl_inout", "dHVG"])
            for rec in records
        }
        source = "records"
    else:
        kl_map = _recompute_kl_map(records, config, generator_config, n_workers)
        source = "recomputed_dhvg"

    rows: list[dict[str, Any]] = []
    for rec in records:
        key = (str(rec["dataset_id"]), int(rec["realization"]), str(rec["surrogate_kind"]))
        matrix = rec["matrix"]
        if "dHVG" not in matrix.columns or "i_dir" not in matrix.index:
            raise KeyError(f"Record {key} has no dHVG i_dir column")
        if key not in kl_map:
            raise KeyError(f"Missing kl_inout for {key}")
        rows.append(
            {
                "dataset_id": key[0],
                "regime": rec.get("regime", ""),
                "realization": key[1],
                "surrogate_kind": key[2],
                "i_dir": float(matrix.loc["i_dir", "dHVG"]),
                "kl_inout": float(kl_map[key]),
                "kl_source": source,
            }
        )
    return pd.DataFrame(rows, columns=list(_DHVG_LONG_COLUMNS))


def summarize_dhvg_irreversibility(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean, std, median and paired Wilcoxon tests of original vs surrogates."""
    columns = [
        "regime",
        "metric",
        "contrast",
        "n_pairs",
        "mean_original",
        "std_original",
        "median_original",
        "mean_surrogate",
        "std_surrogate",
        "median_surrogate",
        "test",
        "statistic",
        "p_value",
        "effect_size",
        "effect_size_name",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    regimes = list(dict.fromkeys(frame["dataset_id"].astype(str).tolist()))
    rows: list[dict[str, Any]] = []
    for regime in regimes:
        subset = frame[frame["dataset_id"].astype(str) == regime]
        original = subset[subset["surrogate_kind"] == "original"]
        if original.empty:
            continue
        for metric_name, column in _DHVG_METRIC_COLUMNS:
            original_by_r = original.set_index("realization")[column]
            for contrast in _DHVG_CONTRASTS:
                other = subset[subset["surrogate_kind"] == contrast]
                if other.empty:
                    continue
                paired = pd.concat(
                    [
                        original_by_r.rename("original"),
                        other.set_index("realization")[column].rename("surrogate"),
                    ],
                    axis=1,
                    join="inner",
                ).dropna()
                if paired.empty:
                    continue
                original_values = paired["original"].to_numpy(dtype=float)
                surrogate_values = paired["surrogate"].to_numpy(dtype=float)
                test = _paired_wilcoxon(original_values, surrogate_values)
                rows.append(
                    {
                        "regime": regime,
                        "metric": metric_name,
                        "contrast": contrast,
                        "n_pairs": int(test["n_pairs"]),
                        "mean_original": float(np.mean(original_values)),
                        "std_original": _sample_std(original_values),
                        "median_original": float(np.median(original_values)),
                        "mean_surrogate": float(np.mean(surrogate_values)),
                        "std_surrogate": _sample_std(surrogate_values),
                        "median_surrogate": float(np.median(surrogate_values)),
                        "test": test["test"],
                        "statistic": test["statistic"],
                        "p_value": test["p_value"],
                        "effect_size": test["effect_size"],
                        "effect_size_name": test["effect_size_name"],
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def _indexed_method_frame(
    records: Sequence[dict[str, Any]],
    method: str,
) -> pd.DataFrame:
    """Descriptor columns of one TS2G method, indexed by record identity."""
    rows: list[dict[str, float]] = []
    index: list[tuple[str, int, str]] = []
    for rec in records:
        matrix = rec["matrix"]
        if method not in matrix.columns:
            raise KeyError(f"Method {method!r} missing from a surrogate matrix")
        rows.append(
            {
                feature_name(method, desc): float(matrix.loc[desc, method])
                for desc in DESCRIPTOR_IDS
            }
        )
        index.append(
            (str(rec["dataset_id"]), int(rec["realization"]), str(rec["surrogate_kind"]))
        )
    frame = pd.DataFrame(rows)
    frame.index = pd.MultiIndex.from_tuples(
        index,
        names=("dataset_id", "realization", "surrogate_kind"),
    )
    return frame


def _cv_binary_view(
    x: pd.DataFrame,
    y_bin: pd.Series,
    audit: AuditConfig,
) -> dict[str, dict[str, float]]:
    """Stratified CV AUROC and macro-F1 for logistic L2 and random forest."""
    models = _make_models(audit.seed)
    nan_block = {
        name: {
            "auroc_mean": float("nan"),
            "auroc_std": float("nan"),
            "macro_f1_mean": float("nan"),
            "n_features": 0,
        }
        for name in models
    }
    if x.shape[1] == 0 or int(y_bin.nunique()) < 2:
        return nan_block
    health = diagnose_and_clean(x, audit)
    x_clean = health.x_clean
    n_features = int(x_clean.shape[1])
    if n_features == 0:
        return nan_block
    n_splits = min(audit.n_folds, int(y_bin.value_counts().min()))
    if n_splits < 2:
        return nan_block
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=audit.seed)
    xa = x_clean.to_numpy(dtype=float)
    ya = y_bin.to_numpy(dtype=int)
    out: dict[str, dict[str, float]] = {}
    for model_name, model in models.items():
        f1s: list[float] = []
        aucs: list[float] = []
        for train_idx, test_idx in skf.split(xa, ya):
            clf: Pipeline = clone(model)
            clf.fit(xa[train_idx], ya[train_idx])
            pred = clf.predict(xa[test_idx])
            f1s.append(
                float(f1_score(ya[test_idx], pred, average="macro", zero_division=0))
            )
            if len(np.unique(ya[test_idx])) < 2:
                continue
            if hasattr(clf, "predict_proba"):
                score = clf.predict_proba(xa[test_idx])[:, 1]
            else:
                score = clf.decision_function(xa[test_idx])
            aucs.append(float(roc_auc_score(ya[test_idx], score)))
        out[model_name] = {
            "auroc_mean": float(np.mean(aucs)) if aucs else float("nan"),
            "auroc_std": float(np.std(aucs)) if aucs else float("nan"),
            "macro_f1_mean": float(np.mean(f1s)) if f1s else float("nan"),
            "n_features": n_features,
        }
    return out


def dhvg_vs_hvg_auroc(
    records: Sequence[dict[str, Any]],
    kl_frame: pd.DataFrame,
    config: SurrogateAuditConfig,
) -> pd.DataFrame:
    """Binary AUROC of original vs each surrogate using only HVG or only dHVG."""
    columns = [
        "regime",
        "contrast",
        "feature_set",
        "model",
        "n_features",
        "auroc_mean",
        "auroc_std",
        "macro_f1_mean",
    ]
    if not records or kl_frame.empty:
        return pd.DataFrame(columns=columns)

    hvg = _indexed_method_frame(records, "HVG")
    dhvg = _indexed_method_frame(records, "dHVG")
    kl_index = kl_frame.set_index(["dataset_id", "realization", "surrogate_kind"])
    kl_column = feature_name("dHVG", "kl_inout")
    dhvg_plus = dhvg.join(kl_index["kl_inout"].rename(kl_column), how="left")
    feature_sets = {
        "HVG": hvg,
        "dHVG": dhvg,
        "dHVG_plus_kl": dhvg_plus,
        "kl_inout_only": dhvg_plus[[kl_column]],
    }
    audit = config.audit
    regimes = [r for r in audit.regimes if r in set(hvg.index.get_level_values("dataset_id"))]
    rows: list[dict[str, Any]] = []
    scored: dict[tuple[str, str, str, str], dict[str, float]] = {}
    for regime in regimes:
        for contrast in _DHVG_CONTRASTS:
            for set_name, frame in feature_sets.items():
                kind_level = frame.index.get_level_values("surrogate_kind")
                regime_level = frame.index.get_level_values("dataset_id")
                mask = (regime_level == regime) & kind_level.isin(("original", contrast))
                subset = frame.loc[mask]
                y_bin = pd.Series(
                    (subset.index.get_level_values("surrogate_kind") == "original").astype(int),
                    name="original",
                )
                metrics = _cv_binary_view(
                    subset.reset_index(drop=True),
                    y_bin.reset_index(drop=True),
                    audit,
                )
                for model_name, block in metrics.items():
                    scored[(regime, contrast, set_name, model_name)] = block
                    rows.append(
                        {
                            "regime": regime,
                            "contrast": contrast,
                            "feature_set": set_name,
                            "model": model_name,
                            "n_features": block["n_features"],
                            "auroc_mean": block["auroc_mean"],
                            "auroc_std": block["auroc_std"],
                            "macro_f1_mean": block["macro_f1_mean"],
                        }
                    )
            for model_name in ("logistic", "random_forest"):
                for gain_name, left_name in (
                    ("gain_dHVG_minus_HVG", "dHVG"),
                    ("gain_dHVG_plus_kl_minus_HVG", "dHVG_plus_kl"),
                    ("gain_kl_only_minus_HVG", "kl_inout_only"),
                ):
                    left = scored[(regime, contrast, left_name, model_name)]["auroc_mean"]
                    right = scored[(regime, contrast, "HVG", model_name)]["auroc_mean"]
                    gain = (
                        float(left - right)
                        if np.isfinite(left) and np.isfinite(right)
                        else float("nan")
                    )
                    rows.append(
                        {
                            "regime": regime,
                            "contrast": contrast,
                            "feature_set": gain_name,
                            "model": model_name,
                            "n_features": float("nan"),
                            "auroc_mean": gain,
                            "auroc_std": float("nan"),
                            "macro_f1_mean": float("nan"),
                        }
                    )
    return pd.DataFrame(rows, columns=columns)


def plot_dhvg_irreversibility(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    lang: str = "en",
    regimes: Sequence[str] | None = None,
) -> list[Path]:
    """Boxplots of ``kl_inout`` by regime and surrogate kind.

    ``i_dir`` is omitted: on every dHVG it saturates at 1 (all edges follow
    the time arrow), so a second panel carries no information.
    """
    if frame.empty:
        return []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if regimes is None:
        regime_order = list(dict.fromkeys(frame["dataset_id"].astype(str).tolist()))
    else:
        present = set(frame["dataset_id"].astype(str))
        regime_order = [str(r) for r in regimes if str(r) in present]
        if not regime_order:
            regime_order = list(dict.fromkeys(frame["dataset_id"].astype(str).tolist()))

    fig, ax = plt.subplots(1, 1, figsize=(8.6, 4.8))
    _grouped_kind_boxes(ax, frame, "kl_inout", regime_order)
    kl_values = frame["kl_inout"].to_numpy(dtype=float)
    finite_kl = kl_values[np.isfinite(kl_values)]
    if (
        finite_kl.size
        and float(finite_kl.min()) > 0.0
        and float(finite_kl.max()) / float(finite_kl.min()) > 25.0
    ):
        ax.set_yscale("log")
    ax.set_title(_t("surrogate_dhvg_suptitle", lang), fontsize=12)
    ax.set_ylabel(_t("surrogate_dhvg_ylabel_kl", lang))
    ax.set_xlabel(_t("surrogate_dhvg_xlabel", lang))

    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=color, edgecolor="#333333", label=label)
        for _kind, color, label in _DHVG_KIND_STYLE
    ]
    ax.legend(handles=handles, frameon=False, ncol=4, loc="upper right", fontsize=8)
    fig.tight_layout()
    return save_figure_trio(fig, output_dir / "surrogate_dhvg_irreversibility")


def _grouped_kind_boxes(
    ax: plt.Axes,
    frame: pd.DataFrame,
    column: str,
    regimes: Sequence[str],
) -> None:
    """Draw one box per surrogate kind inside each regime group."""
    width = 0.18
    for j, (kind, color, _label) in enumerate(_DHVG_KIND_STYLE):
        positions: list[float] = []
        data: list[NDArray[np.floating]] = []
        for i, regime in enumerate(regimes):
            vals = frame.loc[
                (frame["dataset_id"].astype(str) == regime) & (frame["surrogate_kind"] == kind),
                column,
            ].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            positions.append(i + (j - 1.5) * width)
            data.append(vals)
        if not data:
            continue
        drawn = ax.boxplot(
            data,
            positions=positions,
            widths=width * 0.92,
            patch_artist=True,
            manage_ticks=False,
            medianprops={"color": "white", "linewidth": 1.2},
            whiskerprops={"color": "#333333"},
            capprops={"color": "#333333"},
            boxprops={"edgecolor": "#333333", "linewidth": 0.6},
            flierprops={
                "marker": ".",
                "markersize": 3,
                "markerfacecolor": color,
                "markeredgecolor": color,
                "alpha": 0.45,
            },
        )
        for patch in drawn["boxes"]:
            patch.set_facecolor(color)
    ax.set_xticks(range(len(regimes)))
    ax.set_xticklabels(list(regimes), rotation=25, ha="right")
    ax.grid(axis="y", color="#e4e0d8", lw=0.6)


def _merge_isolated_visibility_auroc(tables_dir: Path, auroc: pd.DataFrame) -> None:
    """Append original-vs-IAAFT HVG/dHVG rows to ``auroc_binary.csv``."""
    if auroc.empty:
        return
    subset = auroc[
        (auroc["contrast"] == "iaaft")
        & (auroc["feature_set"].isin(_ISOLATED_VISIBILITY_FAMILIES))
    ]
    if subset.empty:
        return
    extra = pd.DataFrame(
        {
            "regime": subset["regime"].to_numpy(),
            "family": subset["feature_set"].to_numpy(),
            "model": subset["model"].to_numpy(),
            "auroc_mean": subset["auroc_mean"].to_numpy(dtype=float),
            "auroc_std": subset["auroc_std"].to_numpy(dtype=float),
            "macro_f1_mean": subset["macro_f1_mean"].to_numpy(dtype=float),
        }
    )
    path = tables_dir / "auroc_binary.csv"
    if path.exists():
        base = pd.read_csv(path)
        base = base.loc[~base["family"].isin(_ISOLATED_VISIBILITY_FAMILIES)]
        merged = pd.concat([base, extra], ignore_index=True)
    else:
        merged = extra
    save_table_csv(path, merged)


def write_dhvg_irreversibility_artifacts(
    result: SurrogateAuditResult,
    records: Sequence[dict[str, Any]],
    output_dir: Path,
    *,
    lang: str = "en",
    n_workers: int = 1,
) -> list[Path]:
    """Write the dHVG irreversibility table, the HVG-vs-dHVG AUROC and the figure."""
    if not records:
        logger.warning("No surrogate records available; skipping dHVG irreversibility")
        return []
    t0 = time.perf_counter()
    long_frame = collect_dhvg_irreversibility(
        records,
        result.config,
        n_workers=n_workers,
    )
    summary = summarize_dhvg_irreversibility(long_frame)
    tables_dir = Path(output_dir) / "data" / "tables"
    save_table_csv(tables_dir / "dhvg_irreversibility.csv", summary)
    save_table_csv(tables_dir / "dhvg_irreversibility_long.csv", long_frame)
    result.dhvg_irreversibility = summary
    auroc = dhvg_vs_hvg_auroc(records, long_frame, result.config)
    save_table_csv(tables_dir / "dhvg_vs_hvg_auroc.csv", auroc)
    _merge_isolated_visibility_auroc(tables_dir, auroc)
    figures = plot_dhvg_irreversibility(
        long_frame,
        Path(output_dir) / "figures",
        lang=lang,
        regimes=result.config.audit.regimes,
    )
    result.dhvg_vs_hvg_auroc = auroc
    logger.info(
        "dHVG irreversibility table + figure written in %.2fs (%d rows)",
        time.perf_counter() - t0,
        len(summary),
    )
    return figures


def _append_dhvg_markdown(lines: list[str], result: SurrogateAuditResult) -> None:
    """Append the dHVG irreversibility and HVG-vs-dHVG sections when present."""
    summary = result.dhvg_irreversibility
    if summary is None or summary.empty:
        return
    lines.extend(
        [
            "",
            "## Irreversibilidad temporal dHVG",
            "",
            "Wilcoxon de rangos con signo, pareado por realización "
            "(original frente a cada subrogado). "
            "Tamaño de efecto: correlación rank-biserial "
            "(positivo ⇒ el original es mayor que el subrogado). "
            "`dHVG__i_dir` vale 1 en todo dHVG con aristas solo hacia el futuro, "
            "así que no discrimina la flecha del tiempo; la métrica informativa es "
            "`dHVG__kl_inout`.",
            "",
            "| Régimen | métrica | contraste | n | media orig | media surr | p | r_rb |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary.itertuples(index=False):
        lines.append(
            "| {regime} | {metric} | {contrast} | {n_pairs} | {mean_o} | {mean_s} | {p} | {effect} |".format(
                regime=row.regime,
                metric=row.metric,
                contrast=row.contrast,
                n_pairs=int(row.n_pairs),
                mean_o=_fmt_num(row.mean_original, ".4f"),
                mean_s=_fmt_num(row.mean_surrogate, ".4f"),
                p=_fmt_num(row.p_value, ".2e"),
                effect=_fmt_num(row.effect_size, ".3f"),
            )
        )

    auroc = result.dhvg_vs_hvg_auroc
    if auroc is None or auroc.empty:
        return
    shown = auroc[
        (auroc["model"] == "random_forest")
        & (auroc["contrast"].isin(["shuffled", "iaaft"]))
        & (
            auroc["feature_set"].isin(
                [
                    "HVG",
                    "dHVG",
                    "dHVG_plus_kl",
                    "kl_inout_only",
                    "gain_dHVG_minus_HVG",
                    "gain_kl_only_minus_HVG",
                ]
            )
        )
    ]
    lines.extend(
        [
            "",
            "## Flecha del tiempo: dHVG solo frente a HVG solo (RF AUROC)",
            "",
            "Clase positiva = serie original. "
            "`gain_dHVG_minus_HVG` = AUROC(dHVG) − AUROC(HVG). "
            "El contraste `shuffled` destruye el orden temporal; "
            "`iaaft` conserva espectro y amplitudes. "
            "La tabla completa está en `data/tables/dhvg_vs_hvg_auroc.csv`.",
            "",
            "| Régimen | contraste | features | AUROC |",
            "|---|---|---|---:|",
        ]
    )
    for row in shown.itertuples(index=False):
        lines.append(
            f"| {row.regime} | {row.contrast} | {row.feature_set} | "
            f"{_fmt_num(row.auroc_mean, '.3f')} |"
        )
