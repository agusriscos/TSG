"""Markdown + HTML reports for Phase-1 discriminability audit."""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

from jinja2 import Template

from tsg.audit.pipeline import AuditResult, _atomic_write_text

logger = logging.getLogger(__name__)

_HTML_TEMPLATE = Template(
    """<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{{ title }}</title>
  <style>
    :root {
      --bg:#f7f5f1; --ink:#1c1a16; --muted:#5c574e; --card:#fff; --line:#ddd6c8;
      --green:#1b7f4a; --yellow:#9a6b00; --red:#8b2e1a; --accent:#0f5c4c;
    }
    body {
      margin:0; font-family:"IBM Plex Sans","Segoe UI",sans-serif;
      background:linear-gradient(180deg,#efeae2 0%,var(--bg) 40%);
      color:var(--ink); line-height:1.5;
    }
    header, main { max-width:1100px; margin:0 auto; padding:1.5rem; }
    h1 { margin:0 0 .4rem; font-size:1.8rem; }
    h2 { color:var(--accent); margin-top:1.5rem; }
    .meta { color:var(--muted); }
    .verdict {
      display:inline-block; padding:.4rem .8rem; border-radius:8px;
      color:#fff; font-weight:700; letter-spacing:.02em;
    }
    .GREEN { background:var(--green); }
    .YELLOW { background:var(--yellow); }
    .RED { background:var(--red); }
    section {
      background:var(--card); border:1px solid var(--line);
      border-radius:12px; padding:1.1rem 1.3rem; margin:1rem 0;
    }
    table { width:100%; border-collapse:collapse; font-size:.85rem; }
    th, td { padding:.4rem .35rem; border-bottom:1px solid var(--line); text-align:left; }
    img.fig { width:100%; max-width:1000px; border:1px solid var(--line); border-radius:8px; margin:.6rem 0; }
    ul { margin:.4rem 0 .8rem 1.1rem; }
  </style>
</head>
<body>
  <header>
    <h1>{{ title }}</h1>
    <p class="meta">{{ subtitle }}</p>
    <p>Veredicto: <span class="verdict {{ verdict }}">{{ verdict }}</span></p>
  </header>
  <main>
    <section>
      <h2>Resumen</h2>
      <ul>
        {% for r in reasons %}<li>{{ r }}</li>{% endfor %}
      </ul>
      <h3>Recomendaciones</h3>
      <ul>
        {% for r in recommendations %}<li>{{ r }}</li>{% endfor %}
      </ul>
    </section>
    <section>
      <h2>KPIs</h2>
      {{ kpi_table | safe }}
    </section>
    <section>
      <h2>Salud de descriptores</h2>
      {{ health_table | safe }}
    </section>
    <section>
      <h2>Clasificación (vista completa)</h2>
      {{ class_table | safe }}
    </section>
    <section>
      <h2>Importancia de variables</h2>
      <p class="meta">{{ importance_note }}</p>
      <h3>Top-10 Random Forest (Gini)</h3>
      {{ rf_importance_table | safe }}
      <h3>Top-10 Regresión Logística L2</h3>
      {{ lr_importance_table | safe }}
    </section>
    <section>
      <h2>Figuras</h2>
      {% for fig in figures %}
        <h3>{{ fig.name }}</h3>
        <img class="fig" src="data:image/png;base64,{{ fig.b64 }}" alt="{{ fig.name }}"/>
      {% endfor %}
    </section>
  </main>
</body>
</html>
"""
)


def _df_html(df) -> str:
    return df.to_html(border=0, float_format=lambda x: f"{x:.4f}")


def _health_markdown_lines(result: AuditResult) -> list[str]:
    """Drop taxonomy whose counts add up to ``dropped_features``.

    ``low_variance_features`` here lists only non-structural columns (for
    example ``transition__OPN__d_lcc``). Structural constants are counted
    separately, so

    ``numerical_nan + numerical_inf + structural_constant
    + low_variance_non_structural == dropped``.
    """
    health = result.health
    n_nan = len(health.numerical_nan_features)
    n_inf = len(health.numerical_inf_features)
    n_struct = len(health.structural_constant_features)
    non_struct = list(health.low_variance_non_structural_features)
    n_low = len(non_struct)
    n_drop = len(health.dropped_features)
    lines = [
        f"- numerical_nan: {health.numerical_nan_features or '(ninguno)'}",
        f"- numerical_inf: {health.numerical_inf_features or '(ninguno)'}",
        f"- structural_constant: {n_struct} features",
    ]
    for name in health.structural_constant_features[:12]:
        lines.append(f"  - `{name}`: {health.drop_reasons.get(name, '')}")
    if n_struct > 12:
        lines.append(f"  - … +{n_struct - 12} más")
    lines.append(f"- low_variance_features (no estructurales): {n_low} features")
    for name in non_struct[:16]:
        lines.append(f"  - `{name}`: {health.drop_reasons.get(name, '')}")
    if n_low > 16:
        lines.append(f"  - … +{n_low - 16} más")
    lines.append(
        f"- dropped: {n_drop} = numerical_nan ({n_nan}) "
        f"+ structural_constant ({n_struct}) "
        f"+ low_variance_non_structural ({n_low}) "
        f"+ numerical_inf ({n_inf})"
    )
    return lines


def write_audit_reports(
    result: AuditResult,
    output_dir: Path,
    *,
    metrics: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write ``report_phase1.md`` and autocontained HTML from one results payload."""
    output_dir = Path(output_dir)
    md_path = output_dir / "report_phase1.md"
    html_path = output_dir / "report_phase1.html"
    latest_html = output_dir / "latest.html"

    payload = metrics if metrics is not None else result.metrics_dict()
    generated_at = str(payload.get("generated_at_utc", result.generated_at_utc))
    n_raw = int(payload.get("n_features_raw", result.health.n_features))
    n_clean = int(
        payload.get("n_features_clean", result.health.x_clean.shape[1])
    )

    v = result.verdict
    full_rf = result.classification.full_rf()
    full_lin = result.classification.full_linear()
    brownian_note = (
        "STN/OPN operan sobre la **trayectoria browniana integrada** (`levels`) "
        "y, de forma consistente, sobre las trayectorias fBm integradas; "
        "visibility/proximity/similarity siempre ven la serie en bruto."
        if result.config.brownian_mode == "levels"
        else f"Modo activo: `{result.config.brownian_mode}`."
    )

    # Markdown
    lines = [
        "# Fase 1 — Auditoría de discriminabilidad de la matriz D",
        "",
        f"- Generado (UTC): `{generated_at}`",
        f"- Perfil: `{result.config.profile}`",
        f"- Realizaciones por régimen: **{result.config.n_realizations}**",
        f"- N: **{result.config.n_steps}**",
        f"- brownian_mode: `{result.config.brownian_mode}`",
        f"- Nota brownian: {brownian_note}",
        f"- Semilla: `{result.config.seed}`",
        f"- Features brutas / limpias: **{n_raw}** / **{n_clean}**",
        "",
        f"## Veredicto: {v.verdict.value}",
        "",
        "### Razones",
    ]
    lines.extend([f"- {r}" for r in v.reasons] or ["- (sin razones)"])
    lines += ["", "### Recomendaciones"]
    lines.extend([f"- {r}" for r in v.recommendations] or ["- (ninguna)"])

    lines += ["", "## Métricas clave", ""]
    if full_rf is not None:
        lines.append(
            f"- Macro-F1 RF (full): **{full_rf.macro_f1_mean:.4f} ± {full_rf.macro_f1_std:.4f}**"
        )
    if full_lin is not None:
        lines.append(
            f"- Macro-F1 lineal (full): **{full_lin.macro_f1_mean:.4f} ± {full_lin.macro_f1_std:.4f}**"
        )
    lines.append(f"- Silhouette global: **{result.geometry.silhouette_global:.4f}**")
    lines.append(
        f"- Features usadas: **{n_clean}** "
        f"(raw {n_raw}, dropped {len(result.health.dropped_features)})"
    )

    lines += ["", "### Silhouette por régimen", ""]
    for k, val in result.geometry.silhouette_by_regime.items():
        lines.append(f"- `{k}`: {val:.4f}")

    lines += ["", "### Macro-F1 por vista (RF / LR)", ""]
    view_names = sorted({m.view for m in result.classification.views})
    for view in view_names:
        rf = next(
            (
                m
                for m in result.classification.views
                if m.view == view and m.model == "random_forest"
            ),
            None,
        )
        lr = next(
            (
                m
                for m in result.classification.views
                if m.view == view and m.model == "logistic"
            ),
            None,
        )
        rf_s = f"{rf.macro_f1_mean:.4f}" if rf is not None else "N/A"
        lr_s = f"{lr.macro_f1_mean:.4f}" if lr is not None else "N/A"
        lines.append(f"- `{view}`: RF={rf_s}, LR={lr_s}")

    lines += ["", "### F1 por régimen (RF full)", ""]
    if full_rf is not None:
        for k, val in full_rf.f1_by_class.items():
            lines.append(f"- `{k}`: {val:.4f}")

    lines += ["", "### Confusión en pares críticos", ""]
    for k, val in result.classification.critical_pair_confusion_rate.items():
        lines.append(f"- `{k}`: {100*val:.2f}%")

    lines += ["", "### Salud de descriptores (clasificación de drops)", ""]
    lines.extend(_health_markdown_lines(result))

    lines += [
        "",
        "### Importancia de variables (vista full)",
        "",
        "> **Nota:** *Permutation Importance* univariante se sustituyó por **Gini (RF)** "
        "y **norma L2 de coeficientes (LR)** porque, con Macro-F1 = 1.00 y descriptores "
        "colineales, permutar una sola columna no degrada el score (importancias ≈ 0).",
        "",
    ]
    rf_imp = result.classification.rf_gini_importance
    lr_imp = result.classification.logistic_l2_importance
    lines.append("#### Top-10 Random Forest (Gini)")
    lines.append("")
    if rf_imp is not None and not rf_imp.empty:
        for _, row in rf_imp.head(10).iterrows():
            lines.append(
                f"- `{row['feature']}` [{row['family']}]: "
                f"{row['importance_mean']:.4f} ± {row['importance_std']:.4f}"
            )
    else:
        lines.append("- (no disponible)")
    lines += ["", "#### Top-10 Regresión Logística L2 (norma de coeficientes)", ""]
    if lr_imp is not None and not lr_imp.empty:
        for _, row in lr_imp.head(10).iterrows():
            lines.append(
                f"- `{row['feature']}` [{row['family']}]: "
                f"{row['importance_mean']:.4f} ± {row['importance_std']:.4f}"
            )
    else:
        lines.append("- (no disponible)")

    lines += ["", "## Figuras", ""]
    for p in result.figure_paths:
        rel = p.relative_to(output_dir) if p.is_relative_to(output_dir) else p
        lines.append(f"### {p.stem}")
        lines.append(f"![{p.stem}]({rel.as_posix()})")
        lines.append("")

    _atomic_write_text(md_path, "\n".join(lines))

    # HTML with embedded figures
    import pandas as pd

    kpi_rows = []
    if full_rf is not None:
        kpi_rows.append(
            {
                "metric": "macro_f1_rf_full",
                "value": full_rf.macro_f1_mean,
                "std": full_rf.macro_f1_std,
            }
        )
    if full_lin is not None:
        kpi_rows.append(
            {
                "metric": "macro_f1_linear_full",
                "value": full_lin.macro_f1_mean,
                "std": full_lin.macro_f1_std,
            }
        )
    kpi_rows.append(
        {
            "metric": "silhouette_global",
            "value": result.geometry.silhouette_global,
            "std": 0.0,
        }
    )
    kpi_rows.append({"metric": "n_features_raw", "value": float(n_raw), "std": 0.0})
    kpi_rows.append(
        {"metric": "n_features_clean", "value": float(n_clean), "std": 0.0}
    )
    kpi_table = _df_html(pd.DataFrame(kpi_rows))
    health_table = _df_html(
        pd.DataFrame(
            [
                {
                    "n_features_raw": n_raw,
                    "n_features_clean": n_clean,
                    "n_nan_features": len(result.health.nan_features),
                    "n_numerical_nan": len(result.health.numerical_nan_features),
                    "n_structural_constant": len(
                        result.health.structural_constant_features
                    ),
                    "n_inf_features": len(result.health.inf_features),
                    "n_low_variance": len(result.health.low_variance_features),
                    "n_low_variance_non_structural": len(
                        result.health.low_variance_non_structural_features
                    ),
                    "n_numerical_inf": len(result.health.numerical_inf_features),
                    "n_dropped": len(result.health.dropped_features),
                    "n_samples": result.health.n_samples,
                }
            ]
        )
    )
    if full_rf is not None:
        class_table = _df_html(
            pd.DataFrame(
                [{"regime": k, "f1": v} for k, v in full_rf.f1_by_class.items()]
            )
        )
    else:
        class_table = "<p>N/A</p>"

    def _importance_html(frame) -> str:
        if frame is None or getattr(frame, "empty", True):
            return "<p>N/A</p>"
        cols = ["feature", "family", "importance_mean", "importance_std"]
        return _df_html(frame.loc[:, cols].head(10).reset_index(drop=True))

    importance_note = (
        "Permutation Importance univariante se sustituyó por Gini (RF) y norma L2 "
        "de coeficientes (LR): con Macro-F1 = 1.00 y descriptores colineales, "
        "permutar una sola columna no degrada el score (importancias ≈ 0). "
        f"{brownian_note}"
    )
    rf_importance_table = _importance_html(result.classification.rf_gini_importance)
    lr_importance_table = _importance_html(result.classification.logistic_l2_importance)

    figures = []
    for p in result.figure_paths:
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        figures.append({"name": p.stem, "b64": b64})

    html = _HTML_TEMPLATE.render(
        title="Fase 1 — Auditoría de discriminabilidad",
        subtitle=(
            f"R={result.config.n_realizations}, N={result.config.n_steps}, "
            f"brownian_mode={result.config.brownian_mode}, "
            f"features={n_raw}/{n_clean}, generated={generated_at}"
        ),
        verdict=v.verdict.value,
        reasons=v.reasons,
        recommendations=v.recommendations,
        kpi_table=kpi_table,
        health_table=health_table,
        class_table=class_table,
        importance_note=importance_note,
        rf_importance_table=rf_importance_table,
        lr_importance_table=lr_importance_table,
        figures=figures,
    )
    _atomic_write_text(html_path, html)
    _atomic_write_text(latest_html, html)
    logger.info("Wrote audit reports %s and %s", md_path, html_path)
    return md_path, html_path
