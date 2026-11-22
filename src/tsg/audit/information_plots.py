"""Figures for the information audit: PID, Shapley and the Pareto front."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from tsg.audit.artifacts import save_figure_trio
from tsg.audit.i18n import t as _t
from tsg.audit.information import (
    FAMILY_ORDER,
    FAMILY_SHORT,
    MODEL_LABELS,
    InformationAuditResult,
)

logger = logging.getLogger(__name__)

_PID_COLORS = {
    "redundancy": "#4c78a8",
    "unique_i": "#f28e2b",
    "unique_j": "#59a14f",
    "synergy": "#e15759",
    "synergy_negative": "#b07aa1",
}
_SIZE_COLORS = {1: "#4c78a8", 2: "#59a14f", 3: "#f28e2b", 4: "#e15759"}


def _save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    """Write PDF + SVG + PNG via :func:`save_figure_trio`."""
    stem = path.with_suffix("")
    save_figure_trio(fig, stem, dpi=dpi)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.unicode_minus": True,
        }
    )


def plot_pid(result: InformationAuditResult, path: Path, dpi: int, lang: str = "en") -> None:
    """Stacked MMI atoms for the six family pairs."""
    _style()
    atoms = result.pid
    labels = [atom.pair_label for atom in atoms]
    redundancy = np.array([atom.redundancy for atom in atoms], dtype=float)
    unique_i = np.array([atom.unique_i for atom in atoms], dtype=float)
    unique_j = np.array([atom.unique_j for atom in atoms], dtype=float)
    synergy = np.array([atom.synergy for atom in atoms], dtype=float)
    x = np.arange(len(atoms))
    fig, ax = plt.subplots(figsize=(10.6, 5.6))
    ax.bar(x, redundancy, color=_PID_COLORS["redundancy"], label="Redundancia", width=0.72)
    ax.bar(
        x,
        unique_i,
        bottom=redundancy,
        color=_PID_COLORS["unique_i"],
        label="Única (1ª familia)",
        width=0.72,
    )
    ax.bar(
        x,
        unique_j,
        bottom=redundancy + unique_i,
        color=_PID_COLORS["unique_j"],
        label="Única (2ª familia)",
        width=0.72,
    )
    positive = np.clip(synergy, 0.0, None)
    negative = np.clip(synergy, None, 0.0)
    ax.bar(
        x,
        positive,
        bottom=redundancy + unique_i + unique_j,
        color=_PID_COLORS["synergy"],
        label="Sinergia",
        width=0.72,
    )
    if np.any(negative < -1e-4):
        ax.bar(
            x,
            negative,
            color=_PID_COLORS["synergy_negative"],
            label="Sinergia negativa",
            width=0.72,
        )
    ax.axhline(0.0, color="#444444", lw=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Información (bits)")
    ax.set_xlabel("Par de familias (1ª | 2ª)")
    ax.set_title(_t("pid_title", lang))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    _save_figure(fig, path, dpi)


def plot_shapley(result: InformationAuditResult, path: Path, dpi: int, lang: str = "en") -> None:
    """Grouped bars of exact Shapley values for random forest and logistic."""
    _style()
    labels = [FAMILY_SHORT[family] for family in FAMILY_ORDER]
    x = np.arange(len(FAMILY_ORDER))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    series = [
        ("random_forest", -width / 2, "#4c78a8"),
        ("logistic", width / 2, "#f28e2b"),
    ]
    collected: list[np.ndarray] = []
    for model_name, offset, color in series:
        phi = result.shapley["by_model"][model_name]["phi"]
        values = np.array([phi[family] for family in FAMILY_ORDER], dtype=float)
        collected.append(values)
        ax.bar(x + offset, values, width=width, color=color, label=MODEL_LABELS[model_name])
    ax.axhline(0.0, color="#444444", lw=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Valor de Shapley (Macro-F1)")
    ax.set_xlabel("Familia TS2G")
    ax.set_title(_t("shapley_title", lang))
    stacked = np.concatenate(collected)
    pad = 0.08 * (float(np.max(np.abs(stacked))) + 1e-3)
    ax.set_ylim(min(0.0, float(stacked.min())) - pad, max(0.0, float(stacked.max())) + pad)
    ax.legend(frameon=False)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    _save_figure(fig, path, dpi)


def _front_caption(ax: plt.Axes, rows: list[dict], front: list[dict]) -> None:
    """List the front, cheapest first, in the empty part of the axes.

    Point-wise labels collide once several combinations sit near Macro-F1 = 1.
    A single caption keeps those names readable and off the markers.
    """
    ordered = sorted(front, key=lambda row: row["time_ms_mean"])
    text = "Frontera\n" + "\n".join(row["label"] for row in ordered)
    figure = ax.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    axes_box = ax.get_window_extent(renderer)
    points = [
        ax.transData.transform((row["time_ms_mean"], row["macro_f1_mean"]))
        for row in rows
    ]
    boxstyle = dict(boxstyle="round,pad=0.35", fc="white", ec="#ddd6c8", lw=0.6)
    for x_pos in (0.40, 0.34, 0.48, 0.28, 0.55):
        for y_pos in (0.16, 0.08, 0.28, 0.38):
            caption = ax.text(
                x_pos,
                y_pos,
                text,
                transform=ax.transAxes,
                fontsize=8,
                va="bottom",
                ha="left",
                color="#222222",
                zorder=5,
                bbox=boxstyle,
            )
            figure.canvas.draw()
            box = caption.get_window_extent(renderer).expanded(1.04, 1.08)
            inside = (
                box.x0 >= axes_box.x0
                and box.x1 <= axes_box.x1
                and box.y0 >= axes_box.y0
                and box.y1 <= axes_box.y1
            )
            covered = any(box.contains(px, py) for px, py in points)
            if inside and not covered:
                return
            caption.remove()
    ax.text(
        0.38,
        0.12,
        text,
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        color="#222222",
        zorder=5,
        bbox=boxstyle,
    )


def plot_pareto(result: InformationAuditResult, path: Path, dpi: int, lang: str = "en") -> None:
    """Macro-F1 against per-series cost, with the non-dominated front highlighted."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.8), sharey=True)
    n_steps = result.reference_n_steps
    all_scores = [row["macro_f1_mean"] for row in result.pareto]
    ymin = max(0.0, min(all_scores) - 0.07)
    ymax = min(1.08, max(all_scores) + 0.07)
    for ax, model_name in zip(axes, ("random_forest", "logistic")):
        rows = [row for row in result.pareto if row["model"] == model_name]
        costs = np.array([row["time_ms_mean"] for row in rows], dtype=float)
        scores = np.array([row["macro_f1_mean"] for row in rows], dtype=float)
        colors = [_SIZE_COLORS[int(row["n_families"])] for row in rows]
        ax.scatter(costs, scores, c=colors, s=48, zorder=3, linewidths=0)
        front = sorted(
            (row for row in rows if row["on_pareto_front"]),
            key=lambda row: row["time_ms_mean"],
        )
        if front:
            ax.plot(
                [row["time_ms_mean"] for row in front],
                [row["macro_f1_mean"] for row in front],
                color="#222222",
                lw=1.0,
                zorder=2,
            )
            ax.scatter(
                [row["time_ms_mean"] for row in front],
                [row["macro_f1_mean"] for row in front],
                c=[_SIZE_COLORS[int(row["n_families"])] for row in front],
                s=160,
                marker="*",
                zorder=4,
                edgecolors="#222222",
                linewidths=0.5,
            )
        ax.set_xscale("log")
        ax.set_xlim(float(costs.min()) / 1.7, float(costs.max()) * 1.45)
        ax.set_title(MODEL_LABELS[model_name])
        ax.set_xlabel(f"Tiempo por serie (ms, N={n_steps})")
        ax.set_ylim(ymin, ymax)
        ax.grid(True, which="both", linestyle=":", alpha=0.45)
        ax.set_axisbelow(True)
        if front:
            _front_caption(ax, rows, front)
    axes[0].set_ylabel("Macro-F1")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=_SIZE_COLORS[size],
            markeredgecolor="none",
            markersize=8,
            label=f"{size} familia" if size == 1 else f"{size} familias",
        )
        for size in (1, 2, 3, 4)
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor="#333333",
            markeredgecolor="#333333",
            markersize=12,
            label="frontera",
        )
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(_t("pareto_title", lang), y=1.02)
    _save_figure(fig, path, dpi)


def write_information_figures(
    result: InformationAuditResult,
    output_dir: Path,
    *,
    dpi: int,
    lang: str = "en",
) -> list[Path]:
    """Write the three audit figures (PDF + SVG + PNG) and return the PNG paths.

    Parameters
    ----------
    result : InformationAuditResult
        Computed information audit result.
    output_dir : Path
        Directory for figure files.
    dpi : int
        Resolution for PNG export.
    lang : str
        Language code for i18n titles (``'en'`` or ``'es'``).
    """
    specs = (
        ("01_pid_decomposition_pairs.png", plot_pid),
        ("02_shapley_family_importance.png", plot_shapley),
        ("03_pareto_frontier_cost_f1.png", plot_pareto),
    )
    paths: list[Path] = []
    for filename, drawer in specs:
        path = output_dir / filename
        drawer(result, path, dpi, lang=lang)
        paths.append(path)
    return paths
