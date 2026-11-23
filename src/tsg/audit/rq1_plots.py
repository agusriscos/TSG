"""Publication figures for the RQ1 advanced information suite."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tsg.audit.artifacts import save_figure_trio
from tsg.audit.i18n import t as _t
from tsg.audit.information import DYNAMIC_GROUPS, FAMILY_ORDER, FAMILY_SHORT, GROUP_LABELS
from tsg.audit.o_information_bootstrap import OmegaBootstrapResult
from tsg.audit.pid_advanced import AdvancedPIDAtom, OneVsRestPIDAtom
from tsg.audit.shapley_interaction import SIIReport

logger = logging.getLogger(__name__)

_PID_COLORS = {
    "redundancy": "#4c78a8",
    "unique_i": "#f28e2b",
    "unique_j": "#59a14f",
    "synergy": "#e15759",
}
_GROUP_ORDER = (
    "stochastic",
    "chaotic_flows",
    "discrete_maps",
    "structured_other",
)
_GROUP_COLORS = {
    "stochastic": "#4c78a8",
    "chaotic_flows": "#f28e2b",
    "discrete_maps": "#59a14f",
    "structured_other": "#e15759",
    "global": "#b07aa1",
}


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


def _save(fig: plt.Figure, path: Path, dpi: int) -> None:
    save_figure_trio(fig, path.with_suffix(""), dpi=dpi)


def _pair_label(family_i: str, family_j: str) -> str:
    return f"{FAMILY_SHORT[family_i]}|{FAMILY_SHORT[family_j]}"


def plot_pid_by_scope(
    atoms: Sequence[AdvancedPIDAtom],
    path: Path,
    *,
    dpi: int = 300,
    lang: str = "en",
) -> None:
    """Faceted stacked bars: one panel per scope (global + 3 macro-groups)."""
    _style()
    frame = pd.DataFrame([a.as_dict() for a in atoms])
    if frame.empty:
        logger.warning("No PID atoms to plot")
        return

    def scope_key(row: pd.Series) -> str:
        return "global" if row["scope"] == "global" else str(row["group"])

    frame = frame.copy()
    frame["scope_key"] = frame.apply(scope_key, axis=1)
    order = ["global", *[g for g in _GROUP_ORDER if g in set(frame["scope_key"])]]
    order = [k for k in order if k in set(frame["scope_key"])]
    n_panels = len(order)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(3.6 * n_panels, 4.8),
        sharey=True,
        squeeze=False,
    )
    axes_flat = list(axes.ravel())

    for ax, key in zip(axes_flat, order):
        sub = frame[frame["scope_key"] == key].copy()
        sub["pair"] = [
            _pair_label(i, j) for i, j in zip(sub["family_i"], sub["family_j"])
        ]
        sub = sub.sort_values("pair")
        x = np.arange(len(sub))
        red = sub["redundancy"].to_numpy(dtype=float)
        u_i = sub["unique_i"].to_numpy(dtype=float)
        u_j = sub["unique_j"].to_numpy(dtype=float)
        syn = np.clip(sub["synergy"].to_numpy(dtype=float), 0.0, None)
        ax.bar(x, red, color=_PID_COLORS["redundancy"], width=0.72, label=_t("pid_redundancy", lang))
        ax.bar(x, u_i, bottom=red, color=_PID_COLORS["unique_i"], width=0.72, label=_t("pid_unique_i", lang))
        ax.bar(
            x,
            u_j,
            bottom=red + u_i,
            color=_PID_COLORS["unique_j"],
            width=0.72,
            label=_t("pid_unique_j", lang),
        )
        ax.bar(
            x,
            syn,
            bottom=red + u_i + u_j,
            color=_PID_COLORS["synergy"],
            width=0.72,
            label=_t("pid_synergy", lang),
        )
        ax.set_xticks(x, sub["pair"], rotation=45, ha="right")
        title = _t("rq1_scope_global", lang) if key == "global" else (
            GROUP_LABELS.get(key, key) if lang == "es" else key.replace("_", " ")
        )
        ax.set_title(title)
        ax.yaxis.grid(True, linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)
        ax.axhline(0.0, color="#444444", lw=0.7)

    axes_flat[0].set_ylabel(_t("pid_ylabel", lang))
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(_t("rq1_pid_title", lang), y=1.08)
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_omega_bootstrap(
    results: Sequence[OmegaBootstrapResult],
    path: Path,
    *,
    dpi: int = 300,
    lang: str = "en",
) -> None:
    """Bar plot of Ω by regime / macro-group with bootstrap error bars."""
    _style()
    rows = [r for r in results if r.scope in {"group", "regime"}]
    if not rows:
        logger.warning("No Ω bootstrap rows to plot")
        return

    regime_to_group: dict[str, str] = {}
    for group, members in DYNAMIC_GROUPS.items():
        for member in members:
            regime_to_group[member] = group

    group_rows = [r for r in rows if r.scope == "group"]
    regime_rows = [r for r in rows if r.scope == "regime"]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, 5.2),
        gridspec_kw={"width_ratios": [1.0, 2.4]},
    )

    # --- macro-groups ---
    ax = axes[0]
    ordered_groups = [g for g in _GROUP_ORDER if any(r.group == g for r in group_rows)]
    g_map = {r.group: r for r in group_rows}
    y_pos = np.arange(len(ordered_groups))
    omega = np.array([g_map[g].omega_bits for g in ordered_groups], dtype=float)
    yerr = np.vstack(
        [
            omega - np.array([g_map[g].ci_low for g in ordered_groups]),
            np.array([g_map[g].ci_high for g in ordered_groups]) - omega,
        ]
    )
    colors = [_GROUP_COLORS.get(g, "#888888") for g in ordered_groups]
    ax.barh(
        y_pos,
        omega,
        xerr=yerr,
        color=colors,
        edgecolor="#333333",
        linewidth=0.4,
        capsize=3,
        height=0.55,
        error_kw={"ecolor": "#333333", "lw": 1.0},
    )
    ax.axvline(0.0, color="#444444", lw=0.9)
    ax.set_yticks(
        y_pos,
        [
            GROUP_LABELS.get(g, g) if lang == "es" else g.replace("_", " ")
            for g in ordered_groups
        ],
    )
    ax.set_xlabel(_t("rq1_omega_xlabel", lang))
    ax.set_title(_t("rq1_omega_groups_title", lang))
    ax.xaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    # --- regimes, coloured by macro-group ---
    ax = axes[1]
    # Keep DYNAMIC_GROUPS order, then leftover regimes alphabetically.
    ordered_regimes: list[OmegaBootstrapResult] = []
    seen: set[str] = set()
    for group in _GROUP_ORDER:
        for row in regime_rows:
            if regime_to_group.get(row.group) == group and row.group not in seen:
                ordered_regimes.append(row)
                seen.add(row.group)
    for row in sorted(regime_rows, key=lambda r: r.group):
        if row.group not in seen:
            ordered_regimes.append(row)
            seen.add(row.group)

    y_pos = np.arange(len(ordered_regimes))
    omega = np.array([r.omega_bits for r in ordered_regimes], dtype=float)
    yerr = np.vstack(
        [
            omega - np.array([r.ci_low for r in ordered_regimes]),
            np.array([r.ci_high for r in ordered_regimes]) - omega,
        ]
    )
    colors = [
        _GROUP_COLORS.get(regime_to_group.get(r.group, ""), "#9e9e9e")
        for r in ordered_regimes
    ]
    ax.barh(
        y_pos,
        omega,
        xerr=yerr,
        color=colors,
        edgecolor="#333333",
        linewidth=0.35,
        capsize=2.5,
        height=0.7,
        error_kw={"ecolor": "#333333", "lw": 0.9},
    )
    ax.axvline(0.0, color="#444444", lw=0.9)
    ax.set_yticks(y_pos, [r.group for r in ordered_regimes], fontsize=9)
    ax.set_xlabel(_t("rq1_omega_xlabel", lang))
    ax.set_title(_t("rq1_omega_regimes_title", lang))
    ax.xaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    # Mark non-significant near-zero regimes with an open circle.
    for y, row in enumerate(ordered_regimes):
        if not row.significant_vs_zero:
            ax.plot(
                row.omega_bits,
                y,
                marker="o",
                markersize=5,
                markerfacecolor="white",
                markeredgecolor="#222222",
                linestyle="None",
                zorder=5,
            )

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=_GROUP_COLORS[g], label=(
            GROUP_LABELS.get(g, g) if lang == "es" else g.replace("_", " ")
        ))
        for g in _GROUP_ORDER
    ]
    legend_handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            markerfacecolor="white",
            markeredgecolor="#222222",
            linestyle="None",
            markersize=6,
            label=_t("rq1_omega_nonsig", lang),
        )
    )
    axes[1].legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        fontsize=8,
        borderaxespad=0.0,
    )
    fig.suptitle(_t("rq1_omega_suptitle", lang), y=1.02)
    fig.tight_layout(rect=(0.0, 0.0, 0.88, 1.0))
    _save(fig, path, dpi)


def plot_sii(
    report: SIIReport,
    path: Path,
    *,
    dpi: int = 300,
    lang: str = "en",
) -> None:
    """Grouped bars: φ with CI error bars, pairwise SII, and v(∅) baselines."""
    _style()
    models = list(report.by_model)
    if not models:
        logger.warning("No SII models to plot")
        return
    from itertools import combinations

    pair_keys = [
        f"{FAMILY_SHORT[a]}+{FAMILY_SHORT[b]}"
        for a, b in combinations(FAMILY_ORDER, 2)
    ]
    fam_labels = [FAMILY_SHORT[f] for f in FAMILY_ORDER]
    x_fam = np.arange(len(FAMILY_ORDER))
    x_pair = np.arange(len(pair_keys))
    width = 0.36 if len(models) == 2 else 0.8 / max(len(models), 1)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14.5, 4.8),
        gridspec_kw={"width_ratios": [1.4, 2.0, 1.0]},
    )
    colors = {"random_forest": "#4c78a8", "logistic": "#f28e2b"}

    ax = axes[0]
    for idx, model in enumerate(models):
        block = report.by_model[model]
        values = np.array([block.phi[f] for f in FAMILY_ORDER], dtype=float)
        lo = np.array(
            [block.phi_ci_low.get(f, values[i]) for i, f in enumerate(FAMILY_ORDER)],
            dtype=float,
        )
        hi = np.array(
            [block.phi_ci_high.get(f, values[i]) for i, f in enumerate(FAMILY_ORDER)],
            dtype=float,
        )
        yerr = np.vstack([np.maximum(0.0, values - lo), np.maximum(0.0, hi - values)])
        offset = (idx - (len(models) - 1) / 2.0) * width
        ax.bar(
            x_fam + offset,
            values,
            width=width * 0.92,
            color=colors.get(model, "#888888"),
            yerr=yerr,
            capsize=3,
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
            label=_t(f"model_{model}", lang)
            if _t(f"model_{model}", lang) != f"model_{model}"
            else model,
        )
    ax.axhline(0.0, color="#444444", lw=0.8)
    ax.set_xticks(x_fam, fam_labels)
    ax.set_ylabel(_t("rq1_shapley_phi_ylabel", lang))
    ax.set_title(_t("rq1_shapley_phi_title", lang))
    ax.legend(frameon=False, fontsize=8)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    ax = axes[1]
    for idx, model in enumerate(models):
        block = report.by_model[model]
        values = np.array([block.sii.get(k, 0.0) for k in pair_keys], dtype=float)
        offset = (idx - (len(models) - 1) / 2.0) * width
        ax.bar(
            x_pair + offset,
            values,
            width=width * 0.92,
            color=colors.get(model, "#888888"),
            label=_t(f"model_{model}", lang)
            if _t(f"model_{model}", lang) != f"model_{model}"
            else model,
        )
    ax.axhline(0.0, color="#444444", lw=0.8)
    ax.set_xticks(x_pair, pair_keys, rotation=30, ha="right")
    ax.set_ylabel(_t("rq1_sii_ylabel", lang))
    ax.set_title(_t("rq1_sii_title", lang))
    ax.legend(frameon=False, fontsize=8)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    ax = axes[2]
    v_empty = [report.by_model[m].v_empty for m in models]
    v_full = [report.by_model[m].v_full for m in models]
    xpos = np.arange(len(models))
    ax.bar(xpos - 0.18, v_empty, width=0.35, color="#9e9e9e", label=r"$v(\emptyset)=1/78$")
    ax.bar(xpos + 0.18, v_full, width=0.35, color="#59a14f", label=r"$v(N)$")
    ax.set_xticks(
        xpos,
        [
            _t(f"model_{m}", lang) if _t(f"model_{m}", lang) != f"model_{m}" else m
            for m in models
        ],
        rotation=15,
        ha="right",
    )
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Macro-F1")
    ax.set_title(_t("rq1_sii_baseline_title", lang))
    ax.legend(frameon=False, fontsize=8)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    _save(fig, path, dpi)


def plot_one_vs_rest(
    atoms: Sequence[OneVsRestPIDAtom],
    path: Path,
    *,
    dpi: int = 300,
    lang: str = "en",
) -> None:
    """Stacked PID bars for each family vs rest, faceted by estimator."""
    _style()
    if not atoms:
        logger.warning("No one-vs-rest PID atoms to plot")
        return
    frame = pd.DataFrame([a.as_dict() for a in atoms])
    estimators = list(dict.fromkeys(frame["estimator"].tolist()))
    fig, axes = plt.subplots(
        1,
        len(estimators),
        figsize=(5.2 * len(estimators), 4.6),
        sharey=True,
        squeeze=False,
    )
    fam_order = list(FAMILY_ORDER)
    for ax, estimator in zip(axes.ravel(), estimators):
        sub = frame[frame["estimator"] == estimator].set_index("family")
        labels = [FAMILY_SHORT[f] for f in fam_order if f in sub.index]
        x = np.arange(len(labels))
        red = np.array([sub.loc[f, "redundancy"] for f in fam_order if f in sub.index])
        u_f = np.array([sub.loc[f, "unique_family"] for f in fam_order if f in sub.index])
        u_r = np.array([sub.loc[f, "unique_rest"] for f in fam_order if f in sub.index])
        syn = np.array([sub.loc[f, "synergy"] for f in fam_order if f in sub.index])
        ax.bar(x, red, color=_PID_COLORS["redundancy"], width=0.72, label=_t("pid_redundancy", lang))
        ax.bar(x, u_f, bottom=red, color=_PID_COLORS["unique_i"], width=0.72, label=_t("pid_unique_i", lang))
        ax.bar(
            x,
            u_r,
            bottom=red + u_f,
            color=_PID_COLORS["unique_j"],
            width=0.72,
            label=_t("pid_unique_j", lang),
        )
        ax.bar(
            x,
            syn,
            bottom=red + u_f + u_r,
            color=_PID_COLORS["synergy"],
            width=0.72,
            label=_t("pid_synergy", lang),
        )
        ax.set_xticks(x, [f"{lab}|rest" for lab in labels], rotation=20, ha="right")
        ax.set_title(f"{_t('rq1_one_vs_rest_title', lang)} ({estimator})")
        ax.yaxis.grid(True, linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)
        if ax is axes.ravel()[0]:
            ax.set_ylabel("bits")
            ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, path, dpi)


def write_rq1_figures(
    *,
    pid_atoms: Sequence[AdvancedPIDAtom],
    omega_results: Sequence[OmegaBootstrapResult],
    output_dir: Path,
    dpi: int = 300,
    lang: str = "en",
    sii_report: SIIReport | None = None,
    one_vs_rest: Sequence[OneVsRestPIDAtom] | None = None,
) -> list[Path]:
    """Write RQ1 figure stems under ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stems = [
        output_dir / "01_pid_broja_by_scope.png",
        output_dir / "02_omega_bootstrap_bars.png",
    ]
    plot_pid_by_scope(pid_atoms, stems[0], dpi=dpi, lang=lang)
    plot_omega_bootstrap(omega_results, stems[1], dpi=dpi, lang=lang)
    if one_vs_rest:
        ovr_stem = output_dir / "04_pid_one_vs_rest.png"
        plot_one_vs_rest(one_vs_rest, ovr_stem, dpi=dpi, lang=lang)
        stems.append(ovr_stem)
    if sii_report is not None:
        sii_stem = output_dir / "03_shapley_interaction.png"
        plot_sii(sii_report, sii_stem, dpi=dpi, lang=lang)
        stems.append(sii_stem)
    written: list[Path] = []
    for stem in stems:
        for ext in (".pdf", ".svg", ".png"):
            candidate = stem.with_suffix(ext)
            if candidate.exists():
                written.append(candidate)
    logger.info("Wrote %d RQ1 figure files under %s", len(written), output_dir)
    return written
