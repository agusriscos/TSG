"""Mandatory Phase-1 audit figures (matplotlib, 300 dpi)."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from sklearn.covariance import EmpiricalCovariance

from tsg.audit.classification import ClassificationResult
from tsg.audit.config import AuditConfig
from tsg.audit.geometry import GeometryResult

logger = logging.getLogger(__name__)

REGIME_COLORS = {
    "white_noise": "#4c78a8",
    "brownian": "#9ecae9",
    "sine": "#59a14f",
    "quasiperiodic": "#8cd17d",
    "lorenz": "#e15759",
    "logistic": "#ff9d9a",
}


def _save(fig: plt.Figure, path: Path, dpi: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    svg = path.with_suffix(".svg")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote figure %s", path)
    return path


def _confidence_ellipse(ax, x, y, color, n_std: float = 2.0) -> None:
    if len(x) < 3:
        return
    cov = EmpiricalCovariance().fit(np.column_stack([x, y])).covariance_
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(np.maximum(vals, 1e-12))
    ellipse = Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=width,
        height=height,
        angle=theta,
        edgecolor=color,
        facecolor=color,
        alpha=0.15,
        lw=1.2,
    )
    ax.add_patch(ellipse)


def _short_feature_label(name: str) -> str:
    """Compress ``family__method__descriptor`` for axis ticks."""
    parts = str(name).split("__")
    if len(parts) >= 3:
        return f"{parts[-1]} · {parts[1]}"
    return str(name)


def _family_of(name: str) -> str:
    return str(name).split("__", 1)[0]


FAMILY_COLORS = {
    "visibility": "#4c78a8",
    "transition": "#f28e2b",
    "proximity": "#59a14f",
    "similarity": "#b07aa1",
}


def _plot_loading_bars(ax: plt.Axes, loadings: pd.DataFrame, pc: str, title: str) -> None:
    """Signed horizontal bars of the top-|loading| features on one PC."""
    ranked = loadings.reindex(loadings[pc].abs().sort_values(ascending=False).index).head(8)
    ranked = ranked.iloc[::-1]  # largest |loading| at top after barh
    colors = [FAMILY_COLORS.get(_family_of(i), "#777") for i in ranked.index]
    labels = [_short_feature_label(i) for i in ranked.index]
    vals = ranked[pc].to_numpy(dtype=float)
    ax.barh(labels, vals, color=colors, edgecolor="white", linewidth=0.4)
    ax.axvline(0.0, color="#444", lw=0.9)
    ax.set_xlabel(f"loading {pc}")
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
    xmax = float(np.nanmax(np.abs(vals))) if vals.size else 0.1
    ax.set_xlim(-1.15 * xmax, 1.15 * xmax)


def plot_pca_projection(
    geometry: GeometryResult,
    config: AuditConfig,
    output_dir: Path,
) -> Path:
    """PCA scatter + ranked PC loadings (best panel for interpretation)."""
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch

    emb = geometry.embedding
    var = geometry.explained_variance_ratio
    pc1_var = 100.0 * var[0] if len(var) > 0 else 0.0
    pc2_var = 100.0 * var[1] if len(var) > 1 else 0.0

    fig = plt.figure(figsize=(14, 6.2))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.15, 1.0], wspace=0.35, hspace=0.35)

    ax_a = fig.add_subplot(gs[:, 0])
    for regime in config.regimes:
        mask = emb["regime"] == regime
        ax_a.scatter(
            emb.loc[mask, "PC1"],
            emb.loc[mask, "PC2"],
            s=28,
            alpha=0.85,
            label=regime,
            color=REGIME_COLORS.get(regime, "#333"),
        )
        _confidence_ellipse(
            ax_a,
            emb.loc[mask, "PC1"].to_numpy(),
            emb.loc[mask, "PC2"].to_numpy(),
            REGIME_COLORS.get(regime, "#333"),
        )
    ax_a.set_xlabel(f"PC1 ({pc1_var:.1f}% var.)")
    ax_a.set_ylabel(f"PC2 ({pc2_var:.1f}% var.)")
    ax_a.set_title("A) Proyección PCA (elipses ~95%)")
    ax_a.legend(fontsize=8, frameon=False, loc="best")

    loadings = geometry.loadings.copy()
    ax_b1 = fig.add_subplot(gs[0, 1])
    ax_b2 = fig.add_subplot(gs[1, 1])
    _plot_loading_bars(
        ax_b1,
        loadings,
        "PC1",
        f"B1) Top-8 loadings en PC1 ({pc1_var:.1f}% var.)",
    )
    _plot_loading_bars(
        ax_b2,
        loadings,
        "PC2",
        f"B2) Top-8 loadings en PC2 ({pc2_var:.1f}% var.)",
    )
    legend_handles = [
        Patch(facecolor=c, edgecolor="none", label=fam)
        for fam, c in FAMILY_COLORS.items()
    ]
    ax_b1.legend(
        handles=legend_handles,
        title="familia",
        fontsize=7,
        title_fontsize=8,
        loc="lower right",
        frameon=False,
    )
    fig.suptitle(
        "PCA: geometría de clases (A) y descriptores que empujan cada eje (B)",
        fontsize=11,
        y=1.01,
    )
    return _save(fig, output_dir / "01_pca_projection.png", config.figure_dpi)


def plot_confusion_grid(
    classification: ClassificationResult,
    config: AuditConfig,
    output_dir: Path,
) -> Path:
    wanted = [
        ("visibility", "random_forest"),
        ("transition", "random_forest"),
        ("proximity", "random_forest"),
        ("full", "random_forest"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.ravel()
    for ax, (view, model) in zip(axes, wanted):
        metrics = next(
            (v for v in classification.views if v.view == view and v.model == model),
            None,
        )
        ax.set_title(f"{view} · {model}")
        if metrics is None:
            ax.text(0.5, 0.5, "N/A", ha="center")
            continue
        cm = metrics.confusion_normalized
        im = ax.imshow(cm.to_numpy(), cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(cm.columns)))
        ax.set_yticks(range(len(cm.index)))
        ax.set_xticklabels(cm.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(cm.index, fontsize=8)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{100*cm.iloc[i, j]:.0f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Matrices de confusión normalizadas (%) — ablación por familia")
    fig.tight_layout()
    return _save(fig, output_dir / "02_confusion_matrices_grid.png", config.figure_dpi)


def plot_regime_fingerprints(
    records: list[dict],
    config: AuditConfig,
    output_dir: Path,
) -> Path:
    # Mean standardized fingerprint: regimes × features (method__descriptor)
    from tsg.audit.features import matrix_to_feature_row
    from tsg.descriptors.base import DESCRIPTOR_IDS

    rows = []
    for rec in records:
        flat = matrix_to_feature_row(rec["matrix"], methods=config.methods)
        flat["dataset_id"] = rec["dataset_id"]
        rows.append(flat)
    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns if c != "dataset_id"]
    means = df.groupby("dataset_id")[feature_cols].mean().reindex(config.regimes)
    # z-score across regimes for visual contrast
    z = (means - means.mean(axis=0)) / means.std(axis=0).replace(0, np.nan)
    z = z.fillna(0.0)

    # Order columns by family/method
    ordered = []
    for method in config.methods:
        for desc in DESCRIPTOR_IDS:
            col = next((c for c in feature_cols if c.endswith(f"__{method}__{desc}")), None)
            if col:
                ordered.append(col)
    z = z[ordered]

    fig, ax = plt.subplots(figsize=(max(10, 0.12 * len(ordered)), 4.8))
    im = ax.imshow(z.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_yticks(range(len(z.index)))
    ax.set_yticklabels(z.index)
    ax.set_xticks(range(len(z.columns)))
    ax.set_xticklabels(
        [c.split("__", 1)[1] for c in z.columns],
        rotation=90,
        fontsize=5,
    )
    ax.set_title("Huella media estandarizada por régimen (z-score)")
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    return _save(fig, output_dir / "03_regime_fingerprints_heatmap.png", config.figure_dpi)


def plot_family_ablation(
    classification: ClassificationResult,
    geometry: GeometryResult,
    config: AuditConfig,
    output_dir: Path,
) -> Path:
    views = ["visibility", "transition", "proximity", "full"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    ax = axes[0]
    x = np.arange(len(views))
    width = 0.35
    f1_means, f1_stds = [], []
    for view in views:
        m = next(
            (v for v in classification.views if v.view == view and v.model == "random_forest"),
            None,
        )
        f1_means.append(0.0 if m is None else m.macro_f1_mean)
        f1_stds.append(0.0 if m is None else m.macro_f1_std)
    ax.bar(x - width / 2, f1_means, width, yerr=f1_stds, label="Macro-F1 (RF)", color="#4c78a8")
    # Silhouette only global for full; show same reference line for context.
    ax.axhline(geometry.silhouette_global, color="#e15759", ls="--", label="Silhouette global")
    ax.set_xticks(x)
    ax.set_xticklabels(views)
    ax.set_ylim(0, 1.05)
    ax.set_title("Ablación por familia")
    ax.legend(fontsize=8)

    ax = axes[1]
    full = classification.full_rf()
    if full is not None:
        regimes = list(config.regimes)
        vals = [full.f1_by_class.get(r, 0.0) for r in regimes]
        colors = [REGIME_COLORS.get(r, "#333") for r in regimes]
        ax.bar(regimes, vals, color=colors)
        ax.axhline(0.90, color="#333", ls="--", lw=1, label="umbral 0.90")
        ax.set_ylim(0, 1.05)
        ax.set_title("F1 por régimen (matriz completa · RF)")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, output_dir / "04_family_ablation_scores.png", config.figure_dpi)


def plot_critical_pairs_boxplots(
    records: list[dict],
    config: AuditConfig,
    output_dir: Path,
) -> Path:
    """Compare degree-like descriptors for white_noise / logistic / lorenz."""
    focus = ["white_noise", "logistic", "lorenz"]
    descriptors = ["cv_k", "h_pk", "skew_k", "c_avg"]
    # Aggregate across visibility+proximity methods for degree stats.
    methods = ["NVG", "HVG", "RN", "kNN"]
    rows = []
    for rec in records:
        if rec["dataset_id"] not in focus:
            continue
        mat: pd.DataFrame = rec["matrix"]
        for desc in descriptors:
            vals = [float(mat.loc[desc, m]) for m in methods if m in mat.columns]
            rows.append(
                {
                    "dataset_id": rec["dataset_id"],
                    "descriptor": desc,
                    "value": float(np.nanmean(vals)),
                }
            )
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    for ax, desc in zip(axes, descriptors):
        data = [df.loc[(df.descriptor == desc) & (df.dataset_id == r), "value"].to_numpy() for r in focus]
        bp = ax.boxplot(data, patch_artist=True)
        ax.set_xticks(range(1, len(focus) + 1))
        ax.set_xticklabels(focus)
        for patch, regime in zip(bp["boxes"], focus):
            patch.set_facecolor(REGIME_COLORS.get(regime, "#ccc"))
            patch.set_alpha(0.7)
        ax.set_title(desc)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Pares críticos en descriptores de grado/cohesión (media NVG/HVG/RN/kNN)")
    fig.tight_layout()
    return _save(fig, output_dir / "05_critical_pairs_boxplots.png", config.figure_dpi)


def _plot_importance_panel(
    ax: plt.Axes,
    frame: pd.DataFrame | None,
    *,
    xlabel: str,
    title: str,
) -> None:
    """Horizontal top-15 importance bars colored by TS2G family."""
    from matplotlib.patches import Patch

    if frame is None or frame.empty:
        ax.text(0.5, 0.5, "Sin importancias", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    top = frame.head(15).iloc[::-1]
    colors = [FAMILY_COLORS.get(str(f), "#777") for f in top["family"]]
    labels = [_short_feature_label(f) for f in top["feature"]]
    ax.barh(
        labels,
        top["importance_mean"].to_numpy(dtype=float),
        xerr=top["importance_std"].to_numpy(dtype=float),
        color=colors,
        ecolor="#555",
        capsize=2,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
    legend_handles = [
        Patch(facecolor=c, edgecolor="none", label=fam) for fam, c in FAMILY_COLORS.items()
    ]
    ax.legend(
        handles=legend_handles,
        title="familia",
        fontsize=7,
        title_fontsize=8,
        loc="lower right",
        frameon=False,
    )


def plot_feature_importance(
    classification: ClassificationResult,
    config: AuditConfig,
    output_dir: Path,
) -> Path:
    """Top-15 RF Gini and logistic L2 importances side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    _plot_importance_panel(
        axes[0],
        classification.rf_gini_importance,
        xlabel="Gini impurity importance",
        title="A) Random Forest (Gini)",
    )
    _plot_importance_panel(
        axes[1],
        classification.logistic_l2_importance,
        xlabel=r"$\|\beta\|_2$ across classes",
        title="B) Logistic Regression L2 (coef. norm)",
    )
    fig.suptitle(
        "Top-15 descriptores (vista full) — Gini + norma L2\n"
        "(Permutation Importance omitida: Macro-F1=1 + colinealidad → Δscore≈0)",
        fontsize=11,
    )
    fig.tight_layout()
    return _save(fig, output_dir / "06_feature_importance_top15.png", config.figure_dpi)


def generate_all_figures(
    *,
    records: list[dict],
    geometry: GeometryResult,
    classification: ClassificationResult,
    config: AuditConfig,
    output_dir: Path,
) -> list[Path]:
    paths = [
        plot_pca_projection(geometry, config, output_dir),
        plot_confusion_grid(classification, config, output_dir),
        plot_regime_fingerprints(records, config, output_dir),
        plot_family_ablation(classification, geometry, config, output_dir),
        plot_critical_pairs_boxplots(records, config, output_dir),
        plot_feature_importance(classification, config, output_dir),
    ]
    return paths
