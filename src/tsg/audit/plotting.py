"""Mandatory Phase-1 audit figures (matplotlib, 300 dpi)."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from sklearn.covariance import EmpiricalCovariance

from tsg.audit.artifacts import save_figure_trio
from tsg.audit.classification import ClassificationResult
from tsg.audit.config import AuditConfig
from tsg.audit.geometry import GeometryResult
from tsg.audit.i18n import t

logger = logging.getLogger(__name__)

REGIME_COLORS = {
    "white_noise": "#4c78a8",
    "brownian": "#9ecae9",
    "sine": "#59a14f",
    "quasiperiodic": "#8cd17d",
    "lorenz": "#e15759",
    "logistic": "#ff9d9a",
    "fbm_antipersistent": "#1f4e79",
    "fbm_persistent": "#6baed6",
    "arma_correlated": "#f2cf5b",
    "rossler": "#b279a2",
    "henon": "#d62728",
    "logistic_weak": "#d4a373",
    "regime_switching": "#9d755d",
}


def _save(fig: plt.Figure, path: Path, dpi: int) -> Path:
    """Save PDF + SVG + PNG; return the PNG path for HTML embedding."""
    stem = path.with_suffix("")
    written = save_figure_trio(fig, stem, dpi=dpi)
    png = next((p for p in written if p.suffix == ".png"), written[0])
    return png


def _text_color_from_rgba(rgba: tuple[float, ...] | list[float]) -> str:
    """Pick black/white cell text from perceived luminance of an RGBA tuple."""
    r, g, b = float(rgba[0]), float(rgba[1]), float(rgba[2])
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 0.50 else "black"


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
    *,
    lang: str = "en",
) -> Path:
    """PCA scatter + ranked PC loadings (best panel for interpretation)."""
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch

    emb = geometry.embedding
    var = geometry.explained_variance_ratio
    pc1_var = 100.0 * var[0] if len(var) > 0 else 0.0
    pc2_var = 100.0 * var[1] if len(var) > 1 else 0.0

    fig = plt.figure(figsize=(14.8, 7.0))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[1.2, 1.0],
        wspace=0.38,
        hspace=0.45,
        top=0.78,
        bottom=0.10,
        left=0.07,
        right=0.98,
    )

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
    ax_a.set_title(t("pca_panel_a", lang))
    ax_a.legend(fontsize=8, frameon=False, loc="best")

    loadings = geometry.loadings.copy()
    ax_b1 = fig.add_subplot(gs[0, 1])
    ax_b2 = fig.add_subplot(gs[1, 1])
    _plot_loading_bars(
        ax_b1,
        loadings,
        "PC1",
        t("pca_panel_b1", lang, pc1_var=pc1_var),
    )
    _plot_loading_bars(
        ax_b2,
        loadings,
        "PC2",
        t("pca_panel_b2", lang, pc2_var=pc2_var),
    )
    legend_handles = [
        Patch(facecolor=c, edgecolor="none", label=fam)
        for fam, c in FAMILY_COLORS.items()
    ]
    # Horizontal family legend above the B panels — leave air vs B1 title.
    fig.legend(
        handles=legend_handles,
        title=t("pca_family_legend", lang),
        fontsize=8,
        title_fontsize=9,
        loc="lower center",
        bbox_to_anchor=(0.74, 0.88),
        ncol=4,
        frameon=False,
        borderaxespad=0.4,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    fig.suptitle(t("pca_suptitle", lang), fontsize=11, y=0.98)
    return _save(fig, output_dir / "01_pca_projection.png", config.figure_dpi)


def plot_confusion_grid(
    classification: ClassificationResult,
    config: AuditConfig,
    output_dir: Path,
    *,
    lang: str = "en",
) -> Path:
    """Normalized confusion matrices for the five ablation views (RF)."""
    wanted = [
        ("visibility", "random_forest"),
        ("transition", "random_forest"),
        ("proximity", "random_forest"),
        ("similarity", "random_forest"),
        ("full", "random_forest"),
    ]
    n_labels = max(len(config.regimes), 1)
    # Scale cell annotations and tick labels with the number of regimes so a
    # 12-class panel stays readable without overlapping names.
    tick_fs = 7 if n_labels <= 8 else 5
    cell_fs = 7 if n_labels <= 8 else 5
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.2))
    axes_flat = axes.ravel()
    for ax, (view, model) in zip(axes_flat, wanted):
        metrics = next(
            (v for v in classification.views if v.view == view and v.model == model),
            None,
        )
        ax.set_title(f"{view} · {model}", fontsize=10)
        if metrics is None:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            continue
        cm = metrics.confusion_normalized
        im = ax.imshow(cm.to_numpy(), cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(cm.columns)))
        ax.set_yticks(range(len(cm.index)))
        ax.set_xticklabels(cm.columns, rotation=45, ha="right", fontsize=tick_fs)
        ax.set_yticklabels(cm.index, fontsize=tick_fs)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{100 * cm.iloc[i, j]:.0f}",
                    ha="center",
                    va="center",
                    fontsize=cell_fs,
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    # Hide the unused 6th cell of the 2×3 grid.
    axes_flat[-1].set_visible(False)
    fig.suptitle(t("confusion_suptitle", lang), fontsize=12)
    fig.tight_layout()
    return _save(fig, output_dir / "02_confusion_matrices_grid.png", config.figure_dpi)


def plot_regime_fingerprints(
    records: list[dict],
    config: AuditConfig,
    output_dir: Path,
    *,
    lang: str = "en",
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
    ax.set_title(t("fingerprint_title", lang))
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    return _save(fig, output_dir / "03_regime_fingerprints_heatmap.png", config.figure_dpi)


def plot_family_ablation(
    classification: ClassificationResult,
    geometry: GeometryResult,
    config: AuditConfig,
    output_dir: Path,
    *,
    lang: str = "en",
) -> Path:
    """Macro-F1 by ablation view (incl. similarity) and per-regime F1 bars."""
    views = ["visibility", "transition", "proximity", "similarity", "full"]
    n_regimes = len(config.regimes)
    # Wide canvas so 12 rotated regime names never collide.
    fig_w = max(14.0, 1.35 * n_regimes + 8.0)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(fig_w, 6.2),
        gridspec_kw={"width_ratios": [1.0, 1.55]},
    )

    ax = axes[0]
    x = np.arange(len(views))
    width = 0.55
    f1_means, f1_stds = [], []
    for view in views:
        m = next(
            (v for v in classification.views if v.view == view and v.model == "random_forest"),
            None,
        )
        f1_means.append(0.0 if m is None else m.macro_f1_mean)
        f1_stds.append(0.0 if m is None else m.macro_f1_std)
    ax.bar(
        x,
        f1_means,
        width,
        yerr=f1_stds,
        label=t("ablation_f1_label", lang),
        color="#4c78a8",
        capsize=3,
    )
    # Silhouette only global for full; show same reference line for context.
    ax.axhline(
        geometry.silhouette_global,
        color="#e15759",
        ls="--",
        label=t("ablation_silhouette", lang),
    )
    ax.set_xticks(x)
    ax.set_xticklabels(views, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title(t("ablation_title", lang))
    ax.legend(fontsize=8, frameon=False, loc="lower left")

    ax = axes[1]
    full = classification.full_rf()
    if full is not None:
        regimes = list(config.regimes)
        vals = [full.f1_by_class.get(r, 0.0) for r in regimes]
        colors = [REGIME_COLORS.get(r, "#333") for r in regimes]
        xpos = np.arange(len(regimes), dtype=float)
        ax.bar(xpos, vals, color=colors, width=0.65)
        ax.axhline(
            0.90,
            color="#333",
            ls="--",
            lw=1,
            label=t("ablation_threshold", lang),
        )
        ax.set_ylim(0, 1.05)
        ax.set_xlim(-0.7, len(regimes) - 0.3)
        ax.set_title(t("ablation_per_regime", lang))
        ax.set_xticks(xpos)
        tick_fs = 9 if n_regimes <= 8 else 7.5
        ax.set_xticklabels(regimes, rotation=45, ha="right", rotation_mode="anchor", fontsize=tick_fs)
        ax.tick_params(axis="x", pad=1)
        ax.legend(
            fontsize=8,
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
        )
    fig.tight_layout()
    return _save(fig, output_dir / "04_family_ablation_scores.png", config.figure_dpi)


def plot_critical_pairs_boxplots(
    records: list[dict],
    config: AuditConfig,
    output_dir: Path,
    *,
    lang: str = "en",
) -> Path:
    """Compare degree-like descriptors for regimes appearing in ``critical_pairs``."""
    # Preserve first-seen order from the configured critical pairs.
    focus = list(
        dict.fromkeys(regime for pair in config.critical_pairs for regime in pair)
    )
    if not focus:
        focus = list(config.regimes[:3])
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
        ax.set_xticklabels(focus, fontsize=8)
        for patch, regime in zip(bp["boxes"], focus):
            patch.set_facecolor(REGIME_COLORS.get(regime, "#ccc"))
            patch.set_alpha(0.7)
        ax.set_title(desc)
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle(t("critical_pairs_suptitle", lang))
    fig.tight_layout()
    return _save(fig, output_dir / "05_critical_pairs_boxplots.png", config.figure_dpi)


def _plot_importance_panel(
    ax: plt.Axes,
    frame: pd.DataFrame | None,
    *,
    xlabel: str,
    title: str,
    lang: str = "en",
) -> None:
    """Horizontal top-15 importance bars colored by TS2G family."""
    from matplotlib.patches import Patch

    if frame is None or frame.empty:
        ax.text(
            0.5,
            0.5,
            t("importance_no_data", lang),
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
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
        title=t("importance_family", lang),
        fontsize=7,
        title_fontsize=8,
        loc="lower right",
        frameon=False,
    )


def plot_feature_importance(
    classification: ClassificationResult,
    config: AuditConfig,
    output_dir: Path,
    *,
    lang: str = "en",
) -> Path:
    """Top-15 RF Gini and logistic L2 importances side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    _plot_importance_panel(
        axes[0],
        classification.rf_gini_importance,
        xlabel="Gini impurity importance",
        title="A) Random Forest (Gini)",
        lang=lang,
    )
    _plot_importance_panel(
        axes[1],
        classification.logistic_l2_importance,
        xlabel=r"$\|\beta\|_2$ across classes",
        title="B) Logistic Regression L2 (coef. norm)",
        lang=lang,
    )
    fig.suptitle(t("importance_suptitle", lang), fontsize=11)
    fig.tight_layout()
    return _save(fig, output_dir / "06_feature_importance_top15.png", config.figure_dpi)


_FAMILY_VIEWS = ("visibility", "transition", "proximity", "similarity", "full")


def _short_pair_label(pair: str) -> str:
    """Compact critical-pair label for heatmap axes."""
    return pair.replace("__", " ↔ ")


def plot_critical_pairs_family_analysis(
    classification: ClassificationResult,
    geometry: GeometryResult,
    config: AuditConfig,
    output_dir: Path,
    *,
    lang: str = "en",
) -> Path:
    """Two-panel heatmaps: logistic confusion vs pair silhouette by family view."""
    pairs = [f"{a}__{b}" for a, b in config.critical_pairs]
    details = classification.critical_pair_details
    geo = geometry.critical_pair_geometry_by_view

    conf_logistic = np.full((len(pairs), len(_FAMILY_VIEWS)), np.nan, dtype=float)
    conf_rf = np.full((len(pairs), len(_FAMILY_VIEWS)), np.nan, dtype=float)
    sil = np.full((len(pairs), len(_FAMILY_VIEWS)), np.nan, dtype=float)
    fisher = np.full((len(pairs), len(_FAMILY_VIEWS)), np.nan, dtype=float)

    for i, pair in enumerate(pairs):
        for j, view in enumerate(_FAMILY_VIEWS):
            by_model = details.get(pair, {}).get(view, {})
            if "logistic" in by_model:
                conf_logistic[i, j] = float(by_model["logistic"].get("confusion_rate", np.nan))
            if "random_forest" in by_model:
                conf_rf[i, j] = float(
                    by_model["random_forest"].get("confusion_rate", np.nan)
                )
            g = geo.get(view, {}).get(pair, {})
            sil[i, j] = float(g.get("pair_silhouette", np.nan))
            fisher[i, j] = float(g.get("fisher_ratio", np.nan))

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4.5, 0.55 * len(pairs) + 2.5)))
    pair_labels = [_short_pair_label(p) for p in pairs]

    # Panel A — confusion rates (%); annotate RF when > 0.
    ax0 = axes[0]
    conf_pct = conf_logistic * 100.0
    im0 = ax0.imshow(conf_pct, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=50.0)
    ax0.set_xticks(range(len(_FAMILY_VIEWS)))
    ax0.set_xticklabels(list(_FAMILY_VIEWS), rotation=30, ha="right")
    ax0.set_yticks(range(len(pairs)))
    ax0.set_yticklabels(pair_labels, fontsize=8)
    ax0.set_xlabel(t("critical_pairs_family_xlabel", lang))
    ax0.set_ylabel(t("critical_pairs_family_ylabel", lang))
    ax0.set_title(t("critical_pairs_family_panel_a", lang), fontsize=10)
    for i in range(len(pairs)):
        for j in range(len(_FAMILY_VIEWS)):
            val = conf_pct[i, j]
            if np.isnan(val):
                text = "—"
            else:
                rf_val = conf_rf[i, j]
                if not np.isnan(rf_val) and rf_val > 0:
                    text = f"{val:.1f}\n({rf_val * 100.0:.1f})"
                else:
                    text = f"{val:.1f}"
            ax0.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=7,
                color="black" if (np.isnan(val) or val < 25) else "white",
            )
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04, label="%")

    # Panel B — pair silhouette with Fisher ratio annotation.
    ax1 = axes[1]
    im1 = ax1.imshow(sil, aspect="auto", cmap="RdYlGn", vmin=-0.2, vmax=1.0)
    ax1.set_xticks(range(len(_FAMILY_VIEWS)))
    ax1.set_xticklabels(list(_FAMILY_VIEWS), rotation=30, ha="right")
    ax1.set_yticks(range(len(pairs)))
    ax1.set_yticklabels(pair_labels, fontsize=8)
    ax1.set_xlabel(t("critical_pairs_family_xlabel", lang))
    ax1.set_title(t("critical_pairs_family_panel_b", lang), fontsize=10)
    for i in range(len(pairs)):
        for j in range(len(_FAMILY_VIEWS)):
            s_val = sil[i, j]
            f_val = fisher[i, j]
            if np.isnan(s_val):
                text = "—"
                text_color = "black"
            else:
                if np.isnan(f_val):
                    text = f"{s_val:.2f}"
                elif np.isinf(f_val):
                    text = f"{s_val:.2f}\n(∞)"
                else:
                    text = f"{s_val:.2f}\n({f_val:.1f})"
                rgba = im1.cmap(im1.norm(float(s_val)))
                text_color = _text_color_from_rgba(rgba)
            ax1.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
            )
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="silhouette")

    fig.suptitle(t("critical_pairs_family_suptitle", lang), fontsize=11)
    fig.tight_layout()
    return _save(
        fig,
        output_dir / "07_critical_pairs_family_analysis.png",
        config.figure_dpi,
    )


def generate_all_figures(
    *,
    records: list[dict],
    geometry: GeometryResult,
    classification: ClassificationResult,
    config: AuditConfig,
    output_dir: Path,
    lang: str = "en",
) -> list[Path]:
    paths = [
        plot_pca_projection(geometry, config, output_dir, lang=lang),
        plot_confusion_grid(classification, config, output_dir, lang=lang),
        plot_regime_fingerprints(records, config, output_dir, lang=lang),
        plot_family_ablation(classification, geometry, config, output_dir, lang=lang),
        plot_critical_pairs_boxplots(records, config, output_dir, lang=lang),
        plot_feature_importance(classification, config, output_dir, lang=lang),
        plot_critical_pairs_family_analysis(
            classification, geometry, config, output_dir, lang=lang
        ),
    ]
    return paths
