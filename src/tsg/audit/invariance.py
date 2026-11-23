"""Bloque 4 — empirical invariance audit under signal transforms."""

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
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.metrics import f1_score
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
from tsg.audit.classification import _make_models, evaluate_classification
from tsg.audit.config import AuditConfig, apply_profile
from tsg.audit.ensemble import (
    _prepare_series_for_method,
    load_generator_suite_config,
    realization_seed,
)
from tsg.audit.features import (
    METHOD_TO_FAMILY,
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
from tsg.generators.perturbations import apply_transform
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

# Heatmap row order: methods grouped by TS2G family.
METHOD_DISPLAY_ORDER: tuple[str, ...] = (
    "HVG",
    "NVG",
    "OPN",
    "STN",
    "RN",
    "kNN",
    "Correlation",
    "Segment",
)

DEFAULT_TRANSFORMS: tuple[str, ...] = (
    "affine",
    "monotone",
    "outliers",
    "quantize",
    "downsample",
    "snr_10",
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
DEFAULT_INVARIANCE_CONFIG = _REPO_ROOT / "configs" / "phase2_invariance.yaml"


@dataclass(frozen=True)
class InvarianceAuditConfig:
    """Configuration for the Bloque-4 invariance audit."""

    audit: AuditConfig = field(default_factory=AuditConfig)
    transforms: tuple[str, ...] = DEFAULT_TRANSFORMS
    paper_regimes: tuple[str, ...] = PAPER_REGIMES

    @classmethod
    def from_yaml(cls, path: Path | str) -> InvarianceAuditConfig:
        import yaml

        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Expected mapping in {path}, got {type(raw)!r}")
        audit = AuditConfig.from_mapping(raw.get("audit", raw))
        block = raw.get("invariance", {})
        transforms = tuple(block.get("transforms", DEFAULT_TRANSFORMS))
        paper_regimes = tuple(block.get("paper_regimes", PAPER_REGIMES))
        return cls(audit=audit, transforms=transforms, paper_regimes=paper_regimes)

    def with_profile(self, profile: str) -> InvarianceAuditConfig:
        audit = apply_profile(self.audit, profile)
        if profile == "paper":
            audit = replace(audit, regimes=self.paper_regimes)
        return replace(self, audit=audit)


@dataclass
class InvarianceAuditResult:
    """Aggregated invariance metrics and figure paths."""

    config: InvarianceAuditConfig
    descriptor_stability: dict[str, Any]
    classification_ood: dict[str, Any]
    records: list[dict[str, Any]] = field(default_factory=list)
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
                "transforms": list(self.config.transforms),
            },
            "descriptor_stability": self.descriptor_stability,
            "classification_ood": self.classification_ood,
            "figures": [str(p) for p in self.figure_paths],
        }


def _transform_seed(base_seed: int, transform: str, realization: int) -> int:
    offset = {
        "affine": 0,
        "monotone": 11,
        "outliers": 23,
        "quantize": 37,
        "downsample": 53,
        "snr_10": 71,
        "snr_5": 83,
        "snr_0": 97,
    }.get(transform, 113 + abs(hash(transform)) % 997)
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


def _invariance_job(
    dataset_id: str,
    r: int,
    kind: str,
    *,
    config: InvarianceAuditConfig,
    generator_config: GeneratorSuiteConfig,
    descriptor_config: DescriptorConfig,
) -> dict[str, Any]:
    """Top-level picklable task for one (regime, realization, transform) triple.

    Parameters
    ----------
    dataset_id : str
        Regime identifier.
    r : int
        Realization index.
    kind : str
        Transform name or ``'original'``.
    config : InvarianceAuditConfig
        Full invariance audit configuration.
    generator_config : GeneratorSuiteConfig
        Generator suite configuration.
    descriptor_config : DescriptorConfig
        Descriptor extraction configuration.

    Returns
    -------
    dict[str, Any]
        Record with dataset_id, regime, realization, seed, transform,
        matrix, series_length.
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

    if kind == "original":
        transformed = values
    else:
        rng = np.random.default_rng(_transform_seed(seed, kind, r))
        transformed = apply_transform(values, kind, rng)

    matrix = _extract_matrix_for_series(
        transformed,
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
        "transform": kind,
        "matrix": matrix,
        "series_length": int(values.size),
    }


def build_invariance_records(
    config: InvarianceAuditConfig,
    *,
    generator_config: GeneratorSuiteConfig | None = None,
    descriptor_config: DescriptorConfig | None = None,
    n_workers: int = 1,
) -> list[dict[str, Any]]:
    """Generate clean + transformed descriptor matrices for the panel.

    Parameters
    ----------
    config : InvarianceAuditConfig
        Full invariance audit configuration.
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

    kinds = ("original", *config.transforms)
    jobs: list[tuple[str, int, str]] = [
        (dataset_id, r, kind)
        for dataset_id in audit.regimes
        for r in range(audit.n_realizations)
        for kind in kinds
    ]
    total = len(jobs)
    n_workers = max(1, int(n_workers))

    if n_workers > 1:
        from joblib import Parallel, delayed

        logger.info("Building %d invariance records with %d workers", total, n_workers)
        records: list[dict[str, Any]] = Parallel(n_jobs=n_workers)(
            delayed(_invariance_job)(
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
            logger.info("Invariance %d/%d regime=%s r=%d kind=%s", idx, total, did, r, kind)
            records.append(
                _invariance_job(
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
    ]
    return [(name, mask) for name, mask in specs if mask.any()]


def _method_view_specs(
    columns: Sequence[str],
    methods: Sequence[str],
) -> list[tuple[str, np.ndarray]]:
    """Boolean column masks for each individual TS2G method that is present.

    Canonical feature names are ``{family}__{method}__{descriptor}``. A column
    belongs to ``method`` when it starts with the prefix ``{method}__`` or
    contains that prefix after a ``__`` separator.

    Parameters
    ----------
    columns : Sequence[str]
        Feature names, typically after health filtering.
    methods : Sequence[str]
        Method ids to look up, in the order they should be emitted.

    Returns
    -------
    list[tuple[str, np.ndarray]]
        ``(method, mask)`` pairs whose mask selects at least one column.
    """
    specs: list[tuple[str, np.ndarray]] = []
    for method in methods:
        prefix = f"{method}__"
        token = f"__{prefix}"
        mask = np.asarray(
            [str(col).startswith(prefix) or token in str(col) for col in columns],
            dtype=bool,
        )
        if mask.any():
            specs.append((method, mask))
    return specs


def _methods_in_display_order(methods: Sequence[str]) -> list[str]:
    """Return ``methods`` grouped by family (HVG, NVG | OPN, STN | ...)."""
    chosen = set(methods)
    ordered = [name for name in METHOD_DISPLAY_ORDER if name in chosen]
    ordered.extend(name for name in methods if name not in set(ordered))
    return ordered


def _aligned_transform_tables(
    records: Sequence[dict[str, Any]],
    methods: Sequence[str],
    transform: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Pair original / transformed feature rows by (dataset_id, realization)."""
    orig = {
        (r["dataset_id"], r["realization"]): r
        for r in records
        if r["transform"] == "original"
    }
    pert = {
        (r["dataset_id"], r["realization"]): r
        for r in records
        if r["transform"] == transform
    }
    keys = sorted(set(orig) & set(pert))
    if not keys:
        raise ValueError(f"No aligned pairs for transform={transform!r}")
    rows_o: list[dict[str, float]] = []
    rows_p: list[dict[str, float]] = []
    labels: list[str] = []
    for key in keys:
        rows_o.append(matrix_to_feature_row(orig[key]["matrix"], methods=methods))
        rows_p.append(matrix_to_feature_row(pert[key]["matrix"], methods=methods))
        labels.append(str(key[0]))
    cols = all_feature_names(methods, DESCRIPTOR_IDS)
    return (
        pd.DataFrame(rows_o, columns=cols),
        pd.DataFrame(rows_p, columns=cols),
        pd.Series(labels, name="regime"),
    )


def _normalized_descriptor_error(
    x_clean: pd.DataFrame,
    x_pert: pd.DataFrame,
    mask: np.ndarray,
) -> dict[str, float]:
    """Median |Δ| / σ_intra-panel and mean Spearman correlation over features."""
    cols = x_clean.columns[mask]
    if len(cols) == 0:
        return {
            "normalized_error_median": float("nan"),
            "spearman_mean": float("nan"),
            "n_features": 0,
        }
    xc = x_clean.loc[:, cols].to_numpy(dtype=float)
    xp = x_pert.loc[:, cols].to_numpy(dtype=float)
    sigma = np.nanstd(xc, axis=0, ddof=0)
    sigma = np.where(sigma < 1e-12, np.nan, sigma)
    abs_err = np.abs(xp - xc)
    normed = abs_err / sigma[None, :]
    per_sample = np.nanmedian(normed, axis=1)
    spearman_vals: list[float] = []
    for j in range(xc.shape[1]):
        a = xc[:, j]
        b = xp[:, j]
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            continue
        if np.std(a) < 1e-12 and np.std(b) < 1e-12:
            spearman_vals.append(1.0)
            continue
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            continue
        rho, _ = spearmanr(a, b)
        if np.isfinite(rho):
            spearman_vals.append(float(rho))
    return {
        "normalized_error_median": float(np.nanmedian(per_sample)),
        "spearman_mean": float(np.mean(spearman_vals)) if spearman_vals else float("nan"),
        "n_features": int(len(cols)),
    }


def _ood_macro_f1(
    x_clean: pd.DataFrame,
    x_pert: pd.DataFrame,
    y: pd.Series,
    *,
    audit: AuditConfig,
    model: Pipeline,
) -> dict[str, float]:
    """Train on clean folds; evaluate test indices on clean and perturbed X."""
    labels = list(audit.regimes)
    n_splits = min(audit.n_folds, int(y.value_counts().min()))
    if n_splits < 2:
        return {
            "f1_clean": float("nan"),
            "f1_perturbed": float("nan"),
            "delta_f1": float("nan"),
        }
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=audit.seed,
    )
    xa = x_clean.to_numpy(dtype=float)
    xp = x_pert.to_numpy(dtype=float)
    ya = y.to_numpy()
    f1_clean: list[float] = []
    f1_pert: list[float] = []
    for train_idx, test_idx in skf.split(xa, ya):
        clf = clone(model)
        clf.fit(xa[train_idx], ya[train_idx])
        pred_c = clf.predict(xa[test_idx])
        pred_p = clf.predict(xp[test_idx])
        f1_clean.append(
            float(
                f1_score(
                    ya[test_idx],
                    pred_c,
                    average="macro",
                    labels=labels,
                    zero_division=0,
                )
            )
        )
        f1_pert.append(
            float(
                f1_score(
                    ya[test_idx],
                    pred_p,
                    average="macro",
                    labels=labels,
                    zero_division=0,
                )
            )
        )
    fc = float(np.mean(f1_clean))
    fp = float(np.mean(f1_pert))
    return {
        "f1_clean": fc,
        "f1_perturbed": fp,
        "delta_f1": float(fp - fc),
        "f1_clean_std": float(np.std(f1_clean)),
        "f1_perturbed_std": float(np.std(f1_pert)),
    }


def compute_invariance_metrics(
    records: Sequence[dict[str, Any]],
    config: InvarianceAuditConfig,
    *,
    include_families: bool = True,
    include_methods: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Descriptor stability and OOD F1 drop by family, method and transform.

    ``per_family`` keeps the historical family-level audit. ``per_method``
    repeats the same estimators on each TS2G method so that monotone
    invariance of ``HVG`` is not averaged together with ``NVG``.

    Parameters
    ----------
    records : Sequence[dict[str, Any]]
        Original and transformed descriptor records.
    config : InvarianceAuditConfig
        Invariance audit configuration (methods, transforms, seed, folds).
    include_families : bool, optional
        Fill ``per_family``. Default is True.
    include_methods : bool, optional
        Fill ``per_method``. Default is True.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        ``(descriptor_stability, classification_ood)``. Each mapping contains
        ``per_family`` and ``per_method``. Method blocks are nested as
        ``[method][transform]`` and, for classification, ``[model_name]``.
    """
    audit = config.audit
    descriptor_stability: dict[str, Any] = {"per_family": {}, "per_method": {}}
    classification_ood: dict[str, Any] = {"per_family": {}, "per_method": {}}

    for transform in config.transforms:
        x_o, x_p, y = _aligned_transform_tables(records, audit.methods, transform)
        # Drop columns that are unhealthy on either table; keep shared survivors.
        health_o = diagnose_and_clean(x_o, audit)
        health_p = diagnose_and_clean(x_p, audit)
        shared = [c for c in health_o.x_clean.columns if c in health_p.x_clean.columns]
        if not shared:
            logger.warning("No shared clean features for transform=%s", transform)
            continue
        xo = health_o.x_clean.loc[:, shared]
        xp = health_p.x_clean.loc[:, shared]
        columns = list(xo.columns)
        view_groups: list[tuple[str, list[tuple[str, np.ndarray]]]] = []
        if include_families:
            view_groups.append(("per_family", _view_specs(columns)))
        if include_methods:
            view_groups.append(
                ("per_method", _method_view_specs(columns, audit.methods))
            )
        if not view_groups:
            continue
        logger.info(
            "Invariance metrics transform=%s samples=%d features=%d views=%s",
            transform,
            len(y),
            len(columns),
            ",".join(name for name, _ in view_groups),
        )
        models = _make_models(audit.seed)
        for bucket, views in view_groups:
            if bucket == "per_method":
                present = {name for name, _ in views}
                missing = [method for method in audit.methods if method not in present]
                if missing:
                    logger.warning(
                        "No clean features for methods %s under transform=%s",
                        missing,
                        transform,
                    )
            for view_name, mask in views:
                descriptor_stability[bucket].setdefault(view_name, {})
                classification_ood[bucket].setdefault(view_name, {})
                descriptor_stability[bucket][view_name][transform] = (
                    _normalized_descriptor_error(xo, xp, mask)
                )
                x_view_o = xo.loc[:, mask]
                x_view_p = xp.loc[:, mask]
                class_block: dict[str, Any] = {}
                for model_name, model in models.items():
                    class_block[model_name] = _ood_macro_f1(
                        x_view_o,
                        x_view_p,
                        y,
                        audit=audit,
                        model=model,
                    )
                classification_ood[bucket][view_name][transform] = class_block

    return descriptor_stability, classification_ood


def _contrasting_text_color(rgba: Sequence[float]) -> str:
    """Return white or black text for readable contrast over ``rgba``."""
    r, g, b = float(rgba[0]), float(rgba[1]), float(rgba[2])
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 0.55 else "black"


def _plot_invariance_matrix(
    result: InvarianceAuditResult,
    output_dir: Path,
    lang: str = "en",
) -> list[Path]:
    """Double heatmap: normalised descriptor error | Δ Macro-F1 RF OOD.

    Parameters
    ----------
    result : InvarianceAuditResult
        Computed invariance audit result.
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
    families = list(FAMILY_VIEWS)
    transforms = list(result.config.transforms)
    err = np.full((len(families), len(transforms)), np.nan)
    d_f1 = np.full((len(families), len(transforms)), np.nan)
    for i, fam in enumerate(families):
        for j, tr in enumerate(transforms):
            err[i, j] = (
                result.descriptor_stability["per_family"]
                .get(fam, {})
                .get(tr, {})
                .get("normalized_error_median", float("nan"))
            )
            d_f1[i, j] = (
                result.classification_ood["per_family"]
                .get(fam, {})
                .get(tr, {})
                .get("random_forest", {})
                .get("delta_f1", float("nan"))
            )

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    im0 = axes[0].imshow(err, aspect="auto", cmap="YlOrRd", vmin=0.0)
    axes[0].set_title(_t("invariance_error_title", lang))
    im1 = axes[1].imshow(d_f1, aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.1)
    axes[1].set_title(_t("invariance_delta_title", lang))

    for ax, mat, im in ((axes[0], err, im0), (axes[1], d_f1, im1)):
        ax.set_xticks(range(len(transforms)))
        ax.set_xticklabels(transforms, rotation=25, ha="right")
        ax.set_yticks(range(len(families)))
        ax.set_yticklabels(families)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if np.isfinite(val):
                    rgba = im.cmap(im.norm(val))
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=_contrasting_text_color(rgba),
                    )

    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle(_t("invariance_matrix_title", lang), fontsize=11)
    fig.tight_layout()

    return save_figure_trio(fig, output_dir / "01_invariance_matrix")


def _plot_invariance_matrix_by_method(
    result: InvarianceAuditResult,
    output_dir: Path,
    lang: str = "en",
) -> list[Path]:
    """Double heatmap of descriptor error and Δ Macro-F1, one row per method.

    Rows follow :data:`METHOD_DISPLAY_ORDER` so each TS2G family stays in a
    contiguous block. Subtle horizontal lines mark the family boundaries.

    Parameters
    ----------
    result : InvarianceAuditResult
        Computed invariance audit result. Reads ``per_method`` blocks.
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
    methods = _methods_in_display_order(list(result.config.audit.methods))
    transforms = list(result.config.transforms)
    err = np.full((len(methods), len(transforms)), np.nan)
    d_f1 = np.full((len(methods), len(transforms)), np.nan)
    stability = result.descriptor_stability.get("per_method", {})
    ood = result.classification_ood.get("per_method", {})
    for i, method in enumerate(methods):
        for j, transform in enumerate(transforms):
            err[i, j] = (
                stability.get(method, {})
                .get(transform, {})
                .get("normalized_error_median", float("nan"))
            )
            d_f1[i, j] = (
                ood.get(method, {})
                .get(transform, {})
                .get("random_forest", {})
                .get("delta_f1", float("nan"))
            )

    fig_h = max(4.8, 0.62 * len(methods) + 1.6)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, fig_h))
    im0 = axes[0].imshow(err, aspect="auto", cmap="YlOrRd", vmin=0.0)
    axes[0].set_title(_t("invariance_method_error_title", lang))
    im1 = axes[1].imshow(d_f1, aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.1)
    axes[1].set_title(_t("invariance_method_delta_title", lang))

    families = [METHOD_TO_FAMILY.get(method, "other") for method in methods]
    boundaries = [
        i + 0.5
        for i in range(len(methods) - 1)
        if families[i] != families[i + 1]
    ]
    for ax, mat, im in ((axes[0], err, im0), (axes[1], d_f1, im1)):
        ax.set_xticks(range(len(transforms)))
        ax.set_xticklabels(transforms, rotation=25, ha="right")
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods)
        for boundary in boundaries:
            ax.axhline(boundary, color="#1c1a16", linewidth=0.8, alpha=0.35, zorder=2)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if np.isfinite(val):
                    rgba = im.cmap(im.norm(val))
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=_contrasting_text_color(rgba),
                        zorder=3,
                    )

    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle(_t("invariance_matrix_by_method_title", lang), fontsize=11)
    fig.tight_layout()

    return save_figure_trio(fig, output_dir / "02_invariance_matrix_by_method")


def _format_markdown(result: InvarianceAuditResult) -> str:
    lines: list[str] = [
        "# Auditoría de invarianzas (Bloque 4)",
        "",
        f"- Generado: `{result.generated_at_utc}`",
        f"- Profile: `{result.config.audit.profile}`",
        f"- R={result.config.audit.n_realizations}, N={result.config.audit.n_steps}",
        f"- Métodos: `{', '.join(result.config.audit.methods)}`",
        f"- Regímenes: `{', '.join(result.config.audit.regimes)}`",
        f"- Transformaciones: `{', '.join(result.config.transforms)}`",
        "",
        "## Estabilidad de descriptores (error normalizado mediano)",
        "",
        "| Familia | " + " | ".join(result.config.transforms) + " |",
        "|---|" + "|".join(["---:" for _ in result.config.transforms]) + "|",
    ]
    for fam in FAMILY_VIEWS:
        cells = []
        for tr in result.config.transforms:
            val = (
                result.descriptor_stability["per_family"]
                .get(fam, {})
                .get(tr, {})
                .get("normalized_error_median", float("nan"))
            )
            cells.append(f"{val:.3f}")
        lines.append(f"| {fam} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Δ Macro-F1 RF OOD (perturbado − clean)",
            "",
            "| Familia | " + " | ".join(result.config.transforms) + " |",
            "|---|" + "|".join(["---:" for _ in result.config.transforms]) + "|",
        ]
    )
    for fam in FAMILY_VIEWS:
        cells = []
        for tr in result.config.transforms:
            val = (
                result.classification_ood["per_family"]
                .get(fam, {})
                .get(tr, {})
                .get("random_forest", {})
                .get("delta_f1", float("nan"))
            )
            cells.append(f"{val:.3f}")
        lines.append(f"| {fam} | " + " | ".join(cells) + " |")

    per_method_stab = result.descriptor_stability.get("per_method") or {}
    per_method_ood = result.classification_ood.get("per_method") or {}
    if per_method_stab or per_method_ood:
        method_names = _methods_in_display_order(
            list(per_method_stab or per_method_ood)
        )
        lines.extend(
            [
                "",
                "## Estabilidad de descriptores por método (error normalizado mediano)",
                "",
                "| Método | Familia | " + " | ".join(result.config.transforms) + " |",
                "|---|---|" + "|".join(["---:" for _ in result.config.transforms]) + "|",
            ]
        )
        for method in method_names:
            cells = []
            for tr in result.config.transforms:
                val = (
                    per_method_stab.get(method, {})
                    .get(tr, {})
                    .get("normalized_error_median", float("nan"))
                )
                cells.append(f"{val:.3f}")
            family = METHOD_TO_FAMILY.get(method, "other")
            lines.append(f"| {method} | {family} | " + " | ".join(cells) + " |")
        lines.extend(
            [
                "",
                "## Δ Macro-F1 RF OOD por método (perturbado − clean)",
                "",
                "| Método | Familia | " + " | ".join(result.config.transforms) + " |",
                "|---|---|" + "|".join(["---:" for _ in result.config.transforms]) + "|",
            ]
        )
        for method in method_names:
            cells = []
            for tr in result.config.transforms:
                val = (
                    per_method_ood.get(method, {})
                    .get(tr, {})
                    .get("random_forest", {})
                    .get("delta_f1", float("nan"))
                )
                cells.append(f"{val:.3f}")
            family = METHOD_TO_FAMILY.get(method, "other")
            lines.append(f"| {method} | {family} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## F1 clean / F1 perturbado (RF, vista full)",
            "",
            "| Transform | F1 clean | F1 pert | ΔF1 |",
            "|---|---:|---:|---:|",
        ]
    )
    for tr in result.config.transforms:
        block = (
            result.classification_ood["per_family"]
            .get("full", {})
            .get(tr, {})
            .get("random_forest", {})
        )
        lines.append(
            f"| {tr} | {block.get('f1_clean', float('nan')):.3f} | "
            f"{block.get('f1_perturbed', float('nan')):.3f} | "
            f"{block.get('delta_f1', float('nan')):.3f} |"
        )

    lines.extend(["", "## Figuras", ""])
    for p in result.figure_paths:
        lines.append(f"- `{p.name}`")
    lines.append("")
    return "\n".join(lines)


def _markdown_to_html(result: InvarianceAuditResult) -> str:
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
<title>Auditoría de invarianzas</title>
<style>
body{{font-family:IBM Plex Sans,Segoe UI,sans-serif;max-width:1000px;margin:1.5rem auto;padding:0 1rem;background:#f7f5f1;color:#1c1a16}}
pre{{white-space:pre-wrap;background:#fff;border:1px solid #ddd6c8;border-radius:10px;padding:1rem}}
img.fig{{width:100%;margin:1rem 0;border:1px solid #ddd6c8;border-radius:8px}}
</style></head><body>
<h1>Auditoría de invarianzas (Bloque 4)</h1>
{imgs}
<pre>{md}</pre>
</body></html>
"""


_STABILITY_METHOD_COLUMNS: tuple[str, ...] = (
    "method",
    "family",
    "transform",
    "normalized_error_median",
    "spearman_mean",
    "n_features",
)
_DELTA_METHOD_COLUMNS: tuple[str, ...] = (
    "method",
    "family",
    "transform",
    "model",
    "f1_clean",
    "f1_perturbed",
    "delta_f1",
)


def _has_named_block(block: Any, key: str) -> bool:
    """True when ``block[key]`` is a non-empty mapping."""
    if not isinstance(block, dict):
        return False
    inner = block.get(key)
    return isinstance(inner, dict) and bool(inner)


def _load_invariance_records(data_dir: Path) -> list[dict[str, Any]]:
    """Load descriptor records saved by :meth:`InvarianceAuditPipeline.persist`."""
    records_path = data_dir / "records.pkl"
    if not records_path.exists():
        raise FileNotFoundError(
            f"Missing {records_path}; cannot regenerate per-method metrics"
        )
    with records_path.open("rb") as fh:
        loaded = pickle.load(fh)
    if not isinstance(loaded, list):
        raise TypeError(f"Expected a list of records in {records_path}")
    return loaded


def write_invariance_tables(
    result: InvarianceAuditResult,
    tables_dir: Path,
    *,
    include_families: bool = True,
    include_methods: bool = True,
) -> None:
    """Write family and per-method invariance tables.

    Family files (``stability_matrix.csv``, ``delta_f1.csv``) stay in their
    historical schema. Method files add a ``method`` column and the TS2G
    family of that method.

    Parameters
    ----------
    result : InvarianceAuditResult
        Metrics to export. ``per_method`` may be empty.
    tables_dir : Path
        Destination directory (created when missing).
    include_families : bool, optional
        Rewrite the family-level CSVs.
    include_methods : bool, optional
        Rewrite ``stability_matrix_by_method.csv`` and ``delta_f1_by_method.csv``.
    """
    tables_dir.mkdir(parents=True, exist_ok=True)
    transforms = list(result.config.transforms)

    if include_families:
        stab_rows: list[dict[str, Any]] = []
        for fam in FAMILY_VIEWS:
            for transform in transforms:
                stats = (
                    result.descriptor_stability.get("per_family", {})
                    .get(fam, {})
                    .get(transform, {})
                )
                stab_rows.append(
                    {
                        "family": fam,
                        "transform": transform,
                        "normalized_error_median": stats.get(
                            "normalized_error_median", float("nan")
                        ),
                        "spearman_mean": stats.get("spearman_mean", float("nan")),
                        "n_features": stats.get("n_features", 0),
                    }
                )
        save_table_csv(tables_dir / "stability_matrix.csv", pd.DataFrame(stab_rows))

        df1_rows: list[dict[str, Any]] = []
        for fam in FAMILY_VIEWS:
            for transform in transforms:
                for model_name in ("random_forest", "logistic"):
                    metrics = (
                        result.classification_ood.get("per_family", {})
                        .get(fam, {})
                        .get(transform, {})
                        .get(model_name, {})
                    )
                    df1_rows.append(
                        {
                            "family": fam,
                            "transform": transform,
                            "model": model_name,
                            "f1_clean": metrics.get("f1_clean", float("nan")),
                            "f1_perturbed": metrics.get("f1_perturbed", float("nan")),
                            "delta_f1": metrics.get("delta_f1", float("nan")),
                        }
                    )
        save_table_csv(tables_dir / "delta_f1.csv", pd.DataFrame(df1_rows))

    if include_methods:
        per_stab = result.descriptor_stability.get("per_method") or {}
        per_ood = result.classification_ood.get("per_method") or {}
        methods = _methods_in_display_order(list(per_stab or per_ood))
        method_stab_rows: list[dict[str, Any]] = []
        method_df1_rows: list[dict[str, Any]] = []
        for method in methods:
            family = METHOD_TO_FAMILY.get(method, "other")
            for transform in transforms:
                stats = per_stab.get(method, {}).get(transform, {})
                method_stab_rows.append(
                    {
                        "method": method,
                        "family": family,
                        "transform": transform,
                        "normalized_error_median": stats.get(
                            "normalized_error_median", float("nan")
                        ),
                        "spearman_mean": stats.get("spearman_mean", float("nan")),
                        "n_features": stats.get("n_features", 0),
                    }
                )
                for model_name in ("random_forest", "logistic"):
                    metrics = (
                        per_ood.get(method, {})
                        .get(transform, {})
                        .get(model_name, {})
                    )
                    method_df1_rows.append(
                        {
                            "method": method,
                            "family": family,
                            "transform": transform,
                            "model": model_name,
                            "f1_clean": metrics.get("f1_clean", float("nan")),
                            "f1_perturbed": metrics.get("f1_perturbed", float("nan")),
                            "delta_f1": metrics.get("delta_f1", float("nan")),
                        }
                    )
        save_table_csv(
            tables_dir / "stability_matrix_by_method.csv",
            pd.DataFrame(method_stab_rows, columns=list(_STABILITY_METHOD_COLUMNS)),
        )
        save_table_csv(
            tables_dir / "delta_f1_by_method.csv",
            pd.DataFrame(method_df1_rows, columns=list(_DELTA_METHOD_COLUMNS)),
        )


def _plot_invariance_figures(
    result: InvarianceAuditResult,
    fig_dir: Path,
    lang: str,
) -> list[Path]:
    """Write the family heatmap and the per-method heatmap."""
    paths = _plot_invariance_matrix(result, fig_dir, lang=lang)
    paths.extend(_plot_invariance_matrix_by_method(result, fig_dir, lang=lang))
    return paths


class InvarianceAuditPipeline:
    """Orchestrate transforms → descriptor matrices → invariance metrics."""

    def __init__(
        self,
        config: InvarianceAuditConfig | None = None,
        descriptor_config: DescriptorConfig | None = None,
        *,
        lang: str = "en",
        n_workers: int = 1,
    ) -> None:
        self.config = config or InvarianceAuditConfig()
        self.descriptor_config = descriptor_config or DescriptorConfig()
        self.lang = lang
        self.n_workers = max(1, int(n_workers))

    def run(
        self,
        *,
        records: list[dict[str, Any]] | None = None,
        output_dir: Path | str | None = None,
    ) -> InvarianceAuditResult:
        cfg = self.config
        logger.info(
            "Invariance audit profile=%s R=%d N=%d regimes=%s transforms=%s workers=%d",
            cfg.audit.profile,
            cfg.audit.n_realizations,
            cfg.audit.n_steps,
            cfg.audit.regimes,
            cfg.transforms,
            self.n_workers,
        )
        if records is None:
            records = build_invariance_records(
                cfg,
                descriptor_config=self.descriptor_config,
                n_workers=self.n_workers,
            )
        stability, ood = compute_invariance_metrics(records, cfg)
        result = InvarianceAuditResult(
            config=cfg,
            records=records,
            descriptor_stability=stability,
            classification_ood=ood,
        )
        if output_dir is not None:
            self.persist(result, Path(output_dir))
        return result

    def persist(self, result: InvarianceAuditResult, output_dir: Path) -> Path:
        """Write metrics, NPZ/CSV data, figures, markdown and HTML."""
        output_dir.mkdir(parents=True, exist_ok=True)
        data_dir = output_dir / "data"
        tables_dir = data_dir / "tables"
        data_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)

        self._persist_data(result, data_dir, tables_dir)

        fig_dir = output_dir / "figures"
        result.figure_paths = _plot_invariance_figures(result, fig_dir, self.lang)

        metrics_path = output_dir / "invariance_metrics.json"
        atomic_write_text(
            metrics_path,
            json.dumps(result.metrics_dict(), indent=2, default=str),
        )
        write_metrics_alias(output_dir, metrics_path)
        write_config_snapshot(
            output_dir,
            block="phase2_invariance",
            profile=result.config.audit.profile,
            seed=result.config.audit.seed,
            config=result.metrics_dict().get("config", {}),
            extra={"lang": self.lang, "n_workers": self.n_workers},
        )

        atomic_write_text(output_dir / "invariance_report.md", _format_markdown(result))
        html = _markdown_to_html(result)
        atomic_write_text(output_dir / "latest.html", html)
        atomic_write_text(output_dir / "invariance_report.html", html)
        logger.info("Invariance audit artifacts written to %s", output_dir)
        return output_dir

    def plot_only(
        self,
        output_dir: Path | str,
        *,
        lang: str | None = None,
    ) -> InvarianceAuditResult:
        """Regenerate figures from persisted metrics without recomputing graphs.

        Family blocks already stored in ``invariance_metrics.json`` are reused.
        When ``per_method`` is missing, descriptor matrices are reloaded from
        ``data/records.pkl`` and only the method-level metrics are scored.
        Both family and method metrics are recomputed from that pickle when
        the family snapshot itself is absent.

        Parameters
        ----------
        output_dir : Path | str
            Directory previously written by :meth:`persist`.
        lang : str | None
            Override language; falls back to stored value or ``self.lang``.

        Returns
        -------
        InvarianceAuditResult
            Reconstructed result with updated ``figure_paths``.
        """
        out = Path(output_dir)
        data_dir = out / "data"
        metrics_path = out / "invariance_metrics.json"
        if not metrics_path.exists():
            metrics_path = out / "metrics.json"
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))

        use_lang = lang or payload.get("lang") or self.lang
        stability = payload.get("descriptor_stability")
        ood = payload.get("classification_ood")
        records: list[dict[str, Any]] = []
        family_ready = _has_named_block(stability, "per_family") and _has_named_block(
            ood, "per_family"
        )
        method_ready = _has_named_block(stability, "per_method") and _has_named_block(
            ood, "per_method"
        )
        recomputed_families = False
        if family_ready and method_ready:
            logger.info(
                "Plot-only invariance figures from persisted metrics (no recompute)"
            )
        else:
            records = _load_invariance_records(data_dir)
            if family_ready:
                logger.info(
                    "Plot-only: scoring per_method from records.pkl "
                    "(reusing persisted per_family)"
                )
                stability = dict(stability)
                ood = dict(ood)
                method_stability, method_ood = compute_invariance_metrics(
                    records,
                    self.config,
                    include_families=False,
                    include_methods=True,
                )
                stability["per_method"] = method_stability["per_method"]
                ood["per_method"] = method_ood["per_method"]
            else:
                logger.info(
                    "Plot-only: scoring per_family and per_method from records.pkl"
                )
                stability, ood = compute_invariance_metrics(records, self.config)
                recomputed_families = True
        result = InvarianceAuditResult(
            config=self.config,
            records=records,
            descriptor_stability=stability,
            classification_ood=ood,
            generated_at_utc=payload.get(
                "generated_at_utc", datetime.now(timezone.utc).isoformat()
            ),
        )
        fig_dir = out / "figures"
        result.figure_paths = _plot_invariance_figures(result, fig_dir, use_lang)
        payload["descriptor_stability"] = result.descriptor_stability
        payload["classification_ood"] = result.classification_ood
        payload["figures"] = [str(p) for p in result.figure_paths]
        payload["lang"] = use_lang
        atomic_write_text(
            out / "invariance_metrics.json",
            json.dumps(payload, indent=2, default=str),
        )
        write_metrics_alias(out, out / "invariance_metrics.json")
        write_invariance_tables(
            result,
            data_dir / "tables",
            include_families=recomputed_families,
            include_methods=True,
        )
        atomic_write_text(out / "invariance_report.md", _format_markdown(result))
        html = _markdown_to_html(result)
        atomic_write_text(out / "latest.html", html)
        atomic_write_text(out / "invariance_report.html", html)
        logger.info("Plot-only figures regenerated in %s (lang=%s)", out, use_lang)
        return result

    def _persist_data(
        self,
        result: InvarianceAuditResult,
        data_dir: Path,
        tables_dir: Path,
    ) -> None:
        """Write features NPZ, CSV tables and records pickle."""
        records = result.records
        audit = result.config.audit

        rows: list[dict[str, float]] = []
        y_labels: list[str] = []
        transforms: list[str] = []
        realization_ids: list[int] = []
        for rec in records:
            rows.append(matrix_to_feature_row(rec["matrix"], methods=audit.methods))
            y_labels.append(str(rec["dataset_id"]))
            transforms.append(str(rec["transform"]))
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
            transform_or_kind=transforms,
        )
        health = diagnose_and_clean(x_raw, audit)
        save_features_npz(
            data_dir / "features_clean.npz",
            x=health.x_clean,
            y=y,
            regime_names=list(audit.regimes),
            realization_ids=realization_ids,
            transform_or_kind=transforms,
        )
        original_mask = np.asarray([t == "original" for t in transforms], dtype=bool)
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

        write_invariance_tables(result, tables_dir)

        # Records pickle
        records_path = data_dir / "records.pkl"
        with records_path.open("wb") as fh:
            pickle.dump(records, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Persisted invariance data to %s", data_dir)
