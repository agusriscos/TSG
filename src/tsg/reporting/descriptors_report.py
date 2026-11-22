"""HTML report for Table 3.3 descriptor matrices."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Template

from tsg.descriptors.base import DESCRIPTOR_IDS
from tsg.descriptors.pipeline import DescriptorResult, results_to_long_table

logger = logging.getLogger(__name__)

_REPORT_TEMPLATE = Template(
    """<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{{ title }}</title>
  <style>
    :root {
      --bg: #f7f5f1; --ink: #1c1a16; --muted: #5c574e;
      --accent: #0f5c4c; --card: #ffffff; --line: #ddd6c8;
      --warn: #8b2e1a; --ok: #0f5c4c;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #efeae2 0%, var(--bg) 40%);
      color: var(--ink); line-height: 1.5;
    }
    header { padding: 2.5rem 1.5rem 1.5rem; max-width: 1200px; margin: 0 auto; }
    h1 { margin: 0 0 0.4rem; font-size: 1.8rem; letter-spacing: -0.02em; }
    .meta { color: var(--muted); font-size: 0.95rem; }
    main { max-width: 1200px; margin: 0 auto; padding: 0 1.5rem 3rem; }
    section {
      background: var(--card); border: 1px solid var(--line);
      border-radius: 12px; padding: 1.25rem 1.4rem; margin-bottom: 1.25rem;
    }
    h2 { margin: 0 0 0.85rem; font-size: 1.15rem; color: var(--accent); }
    h3 { margin: 1rem 0 0.5rem; font-size: 1rem; }
    .kpi-grid {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 0.75rem;
    }
    .kpi { background: #f3efe7; border-radius: 10px; padding: 0.85rem 1rem; }
    .kpi .label { font-size: 0.75rem; text-transform: uppercase; color: var(--muted); }
    .kpi .value { font-size: 1.25rem; font-weight: 650; }
    table { width: 100%; border-collapse: collapse; font-size: 0.78rem; }
    th, td { text-align: left; padding: 0.4rem 0.3rem; border-bottom: 1px solid var(--line); }
    th { color: var(--muted); font-weight: 600; }
    .alert { color: var(--warn); font-weight: 650; }
    .ok { color: var(--ok); }
    .plot { margin-top: 0.5rem; }
  </style>
</head>
<body>
  <header>
    <h1>{{ title }}</h1>
    <p class="meta">{{ subtitle }}</p>
  </header>
  <main>
    <section>
      <h2>Resumen ejecutivo</h2>
      <div class="kpi-grid">
        {% for kpi in kpis %}
        <div class="kpi">
          <div class="label">{{ kpi.label }}</div>
          <div class="value">{{ kpi.value }}</div>
        </div>
        {% endfor %}
      </div>
    </section>
    {% for block in series_blocks %}
    <section>
      <h2>{{ block.title }}</h2>
      <p class="meta">{{ block.alerts | safe }}</p>
      {{ block.table_html | safe }}
      <div class="plot">{{ block.heatmap_html | safe }}</div>
    </section>
    {% endfor %}
  </main>
</body>
</html>
"""
)


def _fmt_num(x: float) -> str:
    if not np.isfinite(x):
        return "NaN"
    ax = abs(x)
    if ax != 0.0 and ax < 1e-3:
        return f"{x:.2e}"
    return f"{x:.4f}"


class DescriptorsReportBuilder:
    """Build a portable HTML summary of the 6 descriptor matrices."""

    def __init__(self, title: str = "Descriptores topológicos y espectrales — Tabla 3.3") -> None:
        self.title = title

    def build(
        self,
        matrices: dict[str, pd.DataFrame],
        results: Sequence[DescriptorResult],
        *,
        regimes: dict[str, str] | None = None,
    ) -> str:
        regimes = regimes or {}
        long_df = results_to_long_table(results)
        total_time = float(sum(r.build_time_sec for r in results))
        nan_count = int(np.isnan(long_df[list(DESCRIPTOR_IDS)].to_numpy(dtype=float)).sum())
        kpis = [
            {"label": "Grafos", "value": f"{len(results)}/48"},
            {"label": "Matrices", "value": f"{len(matrices)}/6"},
            {"label": "Tiempo", "value": f"{total_time:.1f}s"},
            {"label": "NaN", "value": str(nan_count)},
            {
                "label": "Fecha UTC",
                "value": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            },
        ]

        series_blocks = []
        for i, (dataset_id, matrix) in enumerate(matrices.items()):
            alerts = self._alerts(matrix)
            table_html = matrix.copy().map(_fmt_num).to_html(border=0)
            series_blocks.append(
                {
                    "title": f"{dataset_id} ({regimes.get(dataset_id, '')})".strip(),
                    "alerts": alerts,
                    "table_html": table_html,
                    "heatmap_html": self._row_zscore_heatmap(
                        matrix,
                        dataset_id,
                        include_plotlyjs=(i == 0),
                    ),
                }
            )

        return _REPORT_TEMPLATE.render(
            title=self.title,
            subtitle="Fase 1 — 6 matrices 16×8 sobre la suite sintética",
            kpis=kpis,
            series_blocks=series_blocks,
        )

    def write(
        self,
        matrices: dict[str, pd.DataFrame],
        results: Sequence[DescriptorResult],
        output_path: Path | str,
        *,
        regimes: dict[str, str] | None = None,
        also_latest: bool = True,
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        html = self.build(matrices, results, regimes=regimes)
        path.write_text(html, encoding="utf-8")
        logger.info("Wrote descriptors report to %s", path)
        if also_latest:
            (path.parent / "latest.html").write_text(html, encoding="utf-8")
        return path

    @staticmethod
    def _alerts(matrix: pd.DataFrame) -> str:
        bits: list[str] = []
        if "lcc" in matrix.index:
            frag = [c for c in matrix.columns if float(matrix.loc["lcc", c]) < 0.8]
            if frag:
                bits.append(
                    f'<span class="alert">Fragmentación LCC&lt;0.8:</span> {", ".join(frag)}'
                )
        if "assortativity_r" in matrix.index:
            over_s = [
                c
                for c in matrix.columns
                if np.isfinite(matrix.loc["assortativity_r", c])
                and float(matrix.loc["assortativity_r", c]) <= -0.1
            ]
            if over_s:
                bits.append(
                    f'<span class="alert">Over-smoothing r≤-0.1:</span> {", ".join(over_s)}'
                )
        if "lambda2_lcc" in matrix.index:
            over_q = [
                c
                for c in matrix.columns
                if float(matrix.loc["lambda2_lcc", c]) < 1e-2
            ]
            if over_q:
                bits.append(
                    f'<span class="alert">Over-squashing λ₂(LCC)&lt;1e-2:</span> {", ".join(over_q)}'
                )
        if not bits:
            return '<span class="ok">Sin alertas GNN destacadas.</span>'
        return " · ".join(bits)

    @staticmethod
    def _row_zscore_heatmap(
        matrix: pd.DataFrame,
        dataset_id: str,
        *,
        include_plotlyjs: bool = False,
    ) -> str:
        """Row-wise z-score heatmap across TS2G methods for one series."""
        arr = matrix.to_numpy(dtype=float)
        mu = np.nanmean(arr, axis=1, keepdims=True)
        sd = np.nanstd(arr, axis=1, keepdims=True)
        sd = np.where(sd < 1e-12, 1.0, sd)
        z = (arr - mu) / sd
        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=list(matrix.columns),
                y=list(matrix.index),
                colorscale="RdBu",
                zmid=0.0,
                colorbar={"title": "z"},
            )
        )
        fig.update_layout(
            title=f"Perfil z-score por descriptor — {dataset_id}",
            height=520,
            margin={"l": 100, "r": 20, "t": 40, "b": 40},
            template="plotly_white",
        )
        return fig.to_html(
            full_html=False,
            include_plotlyjs="cdn" if include_plotlyjs else False,
        )
