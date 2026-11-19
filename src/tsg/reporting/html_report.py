"""Self-contained HTML reports for synthetic suite validation."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Template
from plotly.subplots import make_subplots

from tsg.generators.base import GeneratedSeries

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
      --bg: #f7f5f1;
      --ink: #1c1a16;
      --muted: #5c574e;
      --accent: #0f5c4c;
      --card: #ffffff;
      --line: #ddd6c8;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #efeae2 0%, var(--bg) 40%);
      color: var(--ink);
      line-height: 1.5;
    }
    header {
      padding: 2.5rem 1.5rem 1.5rem;
      max-width: 1100px;
      margin: 0 auto;
    }
    h1 { margin: 0 0 0.4rem; font-size: 1.8rem; letter-spacing: -0.02em; }
    .meta { color: var(--muted); font-size: 0.95rem; }
    main { max-width: 1100px; margin: 0 auto; padding: 0 1.5rem 3rem; }
    section {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 1.25rem 1.4rem;
      margin-bottom: 1.25rem;
    }
    h2 { margin: 0 0 0.85rem; font-size: 1.15rem; color: var(--accent); }
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 0.75rem;
    }
    .kpi {
      background: #f3efe7;
      border-radius: 10px;
      padding: 0.85rem 1rem;
    }
    .kpi .label { font-size: 0.75rem; text-transform: uppercase; color: var(--muted); }
    .kpi .value { font-size: 1.25rem; font-weight: 650; }
    table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
    th, td { text-align: left; padding: 0.55rem 0.45rem; border-bottom: 1px solid var(--line); }
    th { color: var(--muted); font-weight: 600; }
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
    <section>
      <h2>Tabla comparativa</h2>
      {{ table_html | safe }}
    </section>
    <section>
      <h2>Series temporales</h2>
      <div class="plot">{{ plot_html | safe }}</div>
    </section>
  </main>
</body>
</html>
"""
)


class HTMLReportBuilder:
    """Build a portable HTML summary for a generated synthetic suite."""

    def __init__(self, title: str = "Suite sintética multirrégimen") -> None:
        self.title = title

    def build(
        self,
        series_list: Sequence[GeneratedSeries],
        *,
        seed: int,
        n_steps: int,
        config_summary: dict[str, Any] | None = None,
    ) -> str:
        """Return a self-contained HTML document as a string."""
        stats = self._stats_frame(series_list)
        kpis = [
            {"label": "Series", "value": str(len(series_list))},
            {"label": "Longitud", "value": str(n_steps)},
            {"label": "Semilla", "value": str(seed)},
            {
                "label": "NaN/Inf",
                "value": "OK" if bool(stats["finite"].all()) else "FAIL",
            },
            {
                "label": "Fecha UTC",
                "value": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            },
        ]
        if config_summary:
            kpis.append(
                {
                    "label": "Datasets",
                    "value": str(config_summary.get("n_datasets", len(series_list))),
                }
            )

        table_html = stats.drop(columns=["finite"], errors="ignore").to_html(
            index=False,
            float_format=lambda x: f"{x:.4g}",
            border=0,
            classes="stats",
        )
        plot_html = self._plot_suite(series_list)
        return _REPORT_TEMPLATE.render(
            title=self.title,
            subtitle="Validación Fase 1 — 1 serie por Dataset ID (Tabla 3.1)",
            kpis=kpis,
            table_html=table_html,
            plot_html=plot_html,
        )

    def write(
        self,
        series_list: Sequence[GeneratedSeries],
        output_path: Path | str,
        *,
        seed: int,
        n_steps: int,
        config_summary: dict[str, Any] | None = None,
        also_latest: bool = True,
    ) -> Path:
        """Write the HTML report and optionally refresh ``latest.html``."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        html = self.build(
            series_list,
            seed=seed,
            n_steps=n_steps,
            config_summary=config_summary,
        )
        path.write_text(html, encoding="utf-8")
        logger.info("Wrote report to %s", path)
        if also_latest:
            latest = path.parent / "latest.html"
            latest.write_text(html, encoding="utf-8")
        return path

    @staticmethod
    def _stats_frame(series_list: Sequence[GeneratedSeries]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for s in series_list:
            v = np.asarray(s.values, dtype=float)
            rows.append(
                {
                    "dataset_id": s.dataset_id,
                    "regime": s.regime,
                    "variable": s.variable,
                    "length": s.length,
                    "mean": float(np.mean(v)),
                    "std": float(np.std(v)),
                    "min": float(np.min(v)),
                    "max": float(np.max(v)),
                    "finite": bool(np.isfinite(v).all()),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _plot_suite(series_list: Sequence[GeneratedSeries]) -> str:
        n = len(series_list)
        fig = make_subplots(
            rows=n,
            cols=1,
            shared_xaxes=False,
            subplot_titles=[f"{s.dataset_id} ({s.regime})" for s in series_list],
            vertical_spacing=0.04,
        )
        for i, s in enumerate(series_list, start=1):
            fig.add_trace(
                go.Scatter(
                    y=s.values,
                    mode="lines",
                    name=s.dataset_id,
                    line={"width": 1.1},
                    showlegend=False,
                ),
                row=i,
                col=1,
            )
        fig.update_layout(
            height=max(240 * n, 480),
            margin={"l": 40, "r": 20, "t": 40, "b": 30},
            template="plotly_white",
        )
        return fig.to_html(full_html=False, include_plotlyjs="cdn")
