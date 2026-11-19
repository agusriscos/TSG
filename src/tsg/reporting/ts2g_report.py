"""HTML report for the 6×8 TS2G mapping suite."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Template
from plotly.subplots import make_subplots

from tsg.graphs.base import BaseGraphTransformer, GraphResult
from tsg.graphs.registry import build_transformers, expected_nodes_ok

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
    .kpi-grid {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 0.75rem;
    }
    .kpi { background: #f3efe7; border-radius: 10px; padding: 0.85rem 1rem; }
    .kpi .label { font-size: 0.75rem; text-transform: uppercase; color: var(--muted); }
    .kpi .value { font-size: 1.25rem; font-weight: 650; }
    table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
    th, td { text-align: left; padding: 0.45rem 0.35rem; border-bottom: 1px solid var(--line); }
    th { color: var(--muted); font-weight: 600; }
    .ok { color: #0f5c4c; font-weight: 650; }
    .fail { color: #8b2e1a; font-weight: 650; }
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
      <h2>Matriz resumen (6 × 8)</h2>
      {{ table_html | safe }}
    </section>
    <section>
      <h2>Verificación de nodos (Tabla 3.2)</h2>
      {{ verify_html | safe }}
    </section>
    <section>
      <h2>Heatmap de densidad</h2>
      <div class="plot">{{ density_html | safe }}</div>
    </section>
    <section>
      <h2>Grafos representativos (STN / OPN / Correlation / Segment)</h2>
      <div class="plot">{{ graphs_html | safe }}</div>
    </section>
  </main>
</body>
</html>
"""
)


class TS2GReportBuilder:
    """Build a portable HTML summary for the TS2G mapping matrix."""

    def __init__(self, title: str = "Mapeo TS2G — Tabla 3.2") -> None:
        self.title = title

    def build(
        self,
        results: Sequence[GraphResult],
        *,
        transformers: Sequence[BaseGraphTransformer] | None = None,
    ) -> str:
        transformers = list(transformers) if transformers is not None else build_transformers()
        by_method = {t.method: t for t in transformers}
        frame = self._summary_frame(results)
        verify = self._verify_frame(results, by_method)

        total_time = float(sum(r.build_time_sec for r in results))
        n_datasets = frame["dataset_id"].nunique() if not frame.empty else 0
        n_methods = frame["method"].nunique() if not frame.empty else 0
        expected_graphs = int(n_datasets * n_methods)
        integrity_ok = bool(
            len(results) == expected_graphs
            and expected_graphs > 0
            and verify["nodes_ok"].all()
            and frame["n_nodes"].ge(1).all()
        )
        kpis = [
            {"label": "Grafos", "value": f"{len(results)}/{expected_graphs}"},
            {"label": "Tiempo total", "value": f"{total_time:.2f}s"},
            {"label": "Integridad", "value": "OK" if integrity_ok else "FAIL"},
            {
                "label": "Fecha UTC",
                "value": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            },
        ]

        table_html = frame.to_html(index=False, float_format=lambda x: f"{x:.4g}", border=0)
        verify_view = verify.copy()
        verify_view["nodes_ok"] = verify_view["nodes_ok"].map(
            lambda ok: '<span class="ok">OK</span>' if ok else '<span class="fail">FAIL</span>'
        )
        verify_html = verify_view.to_html(index=False, escape=False, border=0)

        return _REPORT_TEMPLATE.render(
            title=self.title,
            subtitle="Fase 1 — 6 series × 8 métodos = 48 grafos",
            kpis=kpis,
            table_html=table_html,
            verify_html=verify_html,
            density_html=self._density_heatmap(frame),
            graphs_html=self._sample_graph_plots(results),
        )

    def write(
        self,
        results: Sequence[GraphResult],
        output_path: Path | str,
        *,
        transformers: Sequence[BaseGraphTransformer] | None = None,
        also_latest: bool = True,
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        html = self.build(results, transformers=transformers)
        path.write_text(html, encoding="utf-8")
        logger.info("Wrote TS2G report to %s", path)
        if also_latest:
            (path.parent / "latest.html").write_text(html, encoding="utf-8")
        return path

    @staticmethod
    def _summary_frame(results: Sequence[GraphResult]) -> pd.DataFrame:
        rows = [
            {
                "dataset_id": r.dataset_id,
                "method": r.method,
                "family": r.family,
                "n_nodes": r.n_nodes,
                "n_edges": r.n_edges,
                "density": r.density,
                "mean_degree": r.mean_degree,
                "build_time_sec": r.build_time_sec,
                "directed": r.directed,
                "weighted": r.weighted,
            }
            for r in results
        ]
        return pd.DataFrame(rows)

    @staticmethod
    def _verify_frame(
        results: Sequence[GraphResult],
        by_method: dict[str, BaseGraphTransformer],
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for r in results:
            tr = by_method[r.method]
            expected = tr.expected_n_nodes(r.series_length)
            if expected is None:
                expected_label = f"≤ {r.params.get('max_patterns', math.factorial(4))}"
            else:
                expected_label = str(expected)
            rows.append(
                {
                    "dataset_id": r.dataset_id,
                    "method": r.method,
                    "n_nodes": r.n_nodes,
                    "expected_n": expected_label,
                    "nodes_ok": expected_nodes_ok(r, tr),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _density_heatmap(frame: pd.DataFrame) -> str:
        pivot = frame.pivot(index="dataset_id", columns="method", values="density")
        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=list(pivot.columns),
                y=list(pivot.index),
                colorscale="Viridis",
                colorbar={"title": "density"},
            )
        )
        fig.update_layout(
            height=420,
            margin={"l": 80, "r": 20, "t": 30, "b": 60},
            template="plotly_white",
        )
        return fig.to_html(full_html=False, include_plotlyjs="cdn")

    @staticmethod
    def _sample_graph_plots(results: Sequence[GraphResult]) -> str:
        wanted = {"STN", "OPN", "Correlation", "Segment"}
        # Prefer sine as a visually clean prototype when available.
        preferred = [r for r in results if r.dataset_id == "sine" and r.method in wanted]
        if len(preferred) < 4:
            preferred = []
            seen: set[str] = set()
            for r in results:
                if r.method in wanted and r.method not in seen:
                    preferred.append(r)
                    seen.add(r.method)
                if len(preferred) == 4:
                    break

        cols = len(preferred) or 1
        fig = make_subplots(
            rows=1,
            cols=cols,
            subplot_titles=[f"{r.dataset_id} · {r.method}" for r in preferred],
        )
        for i, r in enumerate(preferred, start=1):
            g = r.graph
            pos = nx.spring_layout(g, seed=0)
            edge_x: list[float | None] = []
            edge_y: list[float | None] = []
            for u, v in g.edges():
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]
            fig.add_trace(
                go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    mode="lines",
                    line={"width": 0.8, "color": "#8a8174"},
                    hoverinfo="none",
                    showlegend=False,
                ),
                row=1,
                col=i,
            )
            nodes = list(g.nodes())
            fig.add_trace(
                go.Scatter(
                    x=[pos[n][0] for n in nodes],
                    y=[pos[n][1] for n in nodes],
                    mode="markers",
                    marker={"size": 7, "color": "#0f5c4c"},
                    text=[str(n) for n in nodes],
                    hoverinfo="text",
                    showlegend=False,
                ),
                row=1,
                col=i,
            )
            fig.update_xaxes(visible=False, row=1, col=i)
            fig.update_yaxes(visible=False, row=1, col=i)

        fig.update_layout(
            height=360,
            margin={"l": 10, "r": 10, "t": 40, "b": 10},
            template="plotly_white",
        )
        return fig.to_html(full_html=False, include_plotlyjs=False)
