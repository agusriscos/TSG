"""Markdown and self-contained HTML report for the information audit."""

from __future__ import annotations

import base64
import logging
from html import escape
from pathlib import Path

import pandas as pd
from jinja2 import Template

from tsg.audit.information import (
    FAMILY_ORDER,
    FAMILY_SHORT,
    GROUP_LABELS,
    MODEL_LABELS,
    NOTES,
    InformationAuditResult,
)
from tsg.audit.pipeline import _atomic_write_text

logger = logging.getLogger(__name__)

_HTML = Template(
    """<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{{ title }}</title>
  <style>
    :root {
      --bg:#f7f5f1; --ink:#1c1a16; --muted:#5c574e; --card:#fff; --line:#ddd6c8;
      --accent:#0f5c4c;
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
    section {
      background:var(--card); border:1px solid var(--line);
      border-radius:12px; padding:1.1rem 1.3rem; margin:1rem 0;
    }
    table { width:100%; border-collapse:collapse; font-size:.85rem; }
    th, td { padding:.4rem .35rem; border-bottom:1px solid var(--line); text-align:left; }
    img.fig { width:100%; max-width:1000px; border:1px solid var(--line); border-radius:8px; margin:.6rem 0; }
    ul { margin:.4rem 0 .8rem 1.1rem; }
    code { font-size:.9em; }
  </style>
</head>
<body>
  <header>
    <h1>{{ title }}</h1>
    <p class="meta">{{ subtitle }}</p>
  </header>
  <main>
    <section>
      <h2>KPIs</h2>
      {{ kpi_table | safe }}
    </section>
    <section>
      <h2>Notas de estimador</h2>
      <ul>
        {% for note in notes %}<li>{{ note }}</li>{% endfor %}
      </ul>
    </section>
    <section>
      <h2>PID por pares (MMI)</h2>
      {{ pid_table | safe }}
    </section>
    <section>
      <h2>O-información</h2>
      {{ omega_table | safe }}
    </section>
    <section>
      <h2>Shapley exacto</h2>
      {{ shapley_table | safe }}
    </section>
    <section>
      <h2>Coste por familia</h2>
      {{ timing_table | safe }}
    </section>
    <section>
      <h2>Frontera de Pareto</h2>
      <p>{{ pareto_lead }}</p>
      {{ pareto_table | safe }}
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


def _fmt(value: float, digits: int) -> str:
    if value is None or value != value:
        return "NA"
    return f"{value:.{digits}f}"


def _frame_html(frame: pd.DataFrame) -> str:
    return frame.to_html(index=False, border=0, escape=True)


def _md_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(str(row[column]) for column in frame.columns) + " |"
        for _, row in frame.iterrows()
    ]
    return "\n".join([header, separator, *body])


def _pid_frame(result: InformationAuditResult) -> pd.DataFrame:
    rows = []
    for atom in result.pid:
        rows.append(
            {
                "par": f"{atom.family_i} · {atom.family_j}",
                "I(Fi;Y)": _fmt(atom.mi_i, 3),
                "I(Fj;Y)": _fmt(atom.mi_j, 3),
                "I(joint)": _fmt(atom.mi_joint, 3),
                "Red": _fmt(atom.redundancy, 3),
                "Un(1ª)": _fmt(atom.unique_i, 3),
                "Un(2ª)": _fmt(atom.unique_j, 3),
                "Syn": _fmt(atom.synergy, 3),
            }
        )
    return pd.DataFrame(rows)


def _omega_frame(result: InformationAuditResult) -> pd.DataFrame:
    rows = []
    global_row = result.o_information["global"]
    rows.append(
        {
            "ámbito": "global",
            "n": global_row["n_samples"],
            "Ω (bits)": _fmt(global_row["omega_bits"], 3),
            "lectura": global_row["reading"],
        }
    )
    for name, payload in result.o_information["by_group"].items():
        rows.append(
            {
                "ámbito": GROUP_LABELS.get(name, name),
                "n": payload["n_samples"],
                "Ω (bits)": _fmt(payload["omega_bits"], 3),
                "lectura": payload["reading"],
            }
        )
    for regime, payload in result.o_information["by_regime"].items():
        rows.append(
            {
                "ámbito": regime,
                "n": payload["n_samples"],
                "Ω (bits)": _fmt(payload["omega_bits"], 3),
                "lectura": payload["reading"],
            }
        )
    return pd.DataFrame(rows)


def _shapley_frame(result: InformationAuditResult) -> pd.DataFrame:
    rows = []
    empty = result.shapley["empty_macro_f1"]
    for model_name, payload in result.shapley["by_model"].items():
        row = {
            "modelo": MODEL_LABELS.get(model_name, model_name),
            "v(∅)": _fmt(empty, 4),
            "v(full)": _fmt(payload["full_macro_f1"], 4),
            "Σ φ": _fmt(payload["sum_phi"], 4),
            "full−∅": _fmt(payload["target_full_minus_empty"], 4),
            "gap": f"{payload['efficiency_gap']:.2e}",
        }
        for family in FAMILY_ORDER:
            row[FAMILY_SHORT[family]] = _fmt(payload["phi"][family], 4)
        rows.append(row)
    return pd.DataFrame(rows)


def _timing_frame(result: InformationAuditResult) -> pd.DataFrame:
    rows = []
    for record in result.timing:
        rows.append(
            {
                "N": record["n_steps"],
                "familia": record["family"],
                "series": record["n_series"],
                "media (ms)": _fmt(record["time_ms_mean"], 1),
                "desv. (ms)": _fmt(record["time_ms_std"], 1),
            }
        )
    return pd.DataFrame(rows)


def _pareto_frame(result: InformationAuditResult) -> pd.DataFrame:
    rows = []
    ordered = sorted(
        result.pareto,
        key=lambda row: (row["model"] != "random_forest", row["time_ms_mean"], row["label"]),
    )
    for row in ordered:
        rows.append(
            {
                "modelo": MODEL_LABELS.get(row["model"], row["model"]),
                "combinación": row["label"],
                "familias": row["n_families"],
                "Macro-F1": _fmt(row["macro_f1_mean"], 4),
                "± pliegues": _fmt(row["macro_f1_std"], 4),
                "ms/serie": _fmt(row["time_ms_mean"], 1),
                "± series": _fmt(row["time_ms_std"], 1),
                "frontera": "sí" if row["on_pareto_front"] else "",
            }
        )
    return pd.DataFrame(rows)


def _front_sentence(result: InformationAuditResult, model: str) -> str:
    front = result.pareto_front(model)
    if not front:
        return f"{MODEL_LABELS.get(model, model)}: sin puntos."
    parts = [
        f"{row['label']} (F1={row['macro_f1_mean']:.4f}, {row['time_ms_mean']:.1f} ms)"
        for row in front
    ]
    return f"{MODEL_LABELS.get(model, model)}: " + "; ".join(parts)


def summary_text(result: InformationAuditResult) -> str:
    """Plain-text tables used in the log and as the body of the Markdown report."""
    pid = _pid_frame(result)
    shapley = _shapley_frame(result)
    omega = result.o_information["global"]
    lines = [
        "PID (MMI), bits. Un(1ª)/Un(2ª) siguen el orden del par.",
        pid.to_string(index=False),
        "",
        (
            f"O-información global: Ω={omega['omega_bits']:+.4f} bits "
            f"({omega['reading']}, n={omega['n_samples']}). "
            f"H(Y)={result.h_y_bits:.3f} bits "
            f"(plugin {result.h_y_plugin_bits:.3f}, Miller–Madow "
            f"+{result.h_y_bits - result.h_y_plugin_bits:.3f})."
        ),
        "",
        "Shapley exacto (Macro-F1)",
        shapley.to_string(index=False),
        "",
        f"Frontera de Pareto (N={result.reference_n_steps}):",
        _front_sentence(result, "random_forest"),
        _front_sentence(result, "logistic"),
    ]
    return "\n".join(lines)


def _markdown(result: InformationAuditResult) -> str:
    config = result.config
    excluded = ", ".join(result.o_information["excluded_from_dynamic_groups"]) or "(ninguno)"
    widths = ", ".join(f"{FAMILY_SHORT[name]}={result.family_widths[name]}" for name in FAMILY_ORDER)
    sections = [
        "# Bloque 1 — Auditoría teórico-informacional y frontera de Pareto",
        "",
        f"- Generado (UTC): `{result.generated_at_utc}`",
        f"- Perfil: `{config.profile}`",
        f"- Semilla: `{config.seed}`",
        f"- Realizaciones por régimen: **{config.n_realizations}**",
        f"- N de la matriz D (caché): **{config.n_steps}**",
        f"- N de referencia del coste: **{result.reference_n_steps}**",
        f"- Muestras / features brutas / limpias: **{result.n_samples}** / "
        f"**{result.n_features_raw}** / **{result.n_features_clean}**",
        f"- Ancho por familia (tras limpieza): {widths}",
        f"- Caché: `{result.cache_dir}` (hits={result.cache_hits}, misses={result.cache_misses})",
        f"- Regímenes fuera de los tres grupos dinámicos: {excluded}",
        "",
        "## Estimadores",
        "",
        *[f"- {note}" for note in NOTES.values()],
        "",
        "## PID por pares",
        "",
        "La primera familia del par es la referencia de Un(1ª).",
        "",
        _md_table(_pid_frame(result)),
        "",
        "## O-información",
        "",
        _md_table(_omega_frame(result)),
        "",
        "## Shapley exacto",
        "",
        _md_table(_shapley_frame(result)),
        "",
        "## Coste por familia",
        "",
        "Media y desviación estándar muestral entre regímenes (una serie por régimen, "
        "realización 0). El cronómetro cubre grafo + descriptores.",
        "",
        _md_table(_timing_frame(result)),
        "",
        "## Frontera de Pareto",
        "",
        f"- {_front_sentence(result, 'random_forest')}",
        f"- {_front_sentence(result, 'logistic')}",
        "",
        _md_table(_pareto_frame(result)),
        "",
        "## Figuras",
        "",
        *[f"- `{path.name}`" for path in result.figure_paths],
        "",
    ]
    return "\n".join(sections)


def write_information_reports(result: InformationAuditResult, output_dir: Path) -> None:
    """Write ``information_report.md``, ``information_report.html`` and ``latest.html``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown = _markdown(result)
    _atomic_write_text(output_dir / "information_report.md", markdown)

    figures = []
    for path in result.figure_paths:
        if path.suffix.lower() != ".png" or not path.exists():
            continue
        figures.append(
            {
                "name": path.name,
                "b64": base64.b64encode(path.read_bytes()).decode("ascii"),
            }
        )
    config = result.config
    html = _HTML.render(
        title="Bloque 1 — Información parcial, O-información y frontera de Pareto",
        subtitle=(
            f"UTC {result.generated_at_utc} · perfil {config.profile} · "
            f"semilla {config.seed} · R={config.n_realizations} · N={config.n_steps}"
        ),
        kpi_table=_frame_html(
            pd.DataFrame(
                [
                    {
                        "muestras": result.n_samples,
                        "features limpias": result.n_features_clean,
                        "H(Y) bits": _fmt(result.h_y_bits, 3),
                        "Ω global": _fmt(result.o_information["global"]["omega_bits"], 3),
                        "lectura Ω": result.o_information["global"]["reading"],
                        "F1 RF full": _fmt(
                            result.shapley["by_model"]["random_forest"]["full_macro_f1"], 4
                        ),
                        "F1 LR full": _fmt(
                            result.shapley["by_model"]["logistic"]["full_macro_f1"], 4
                        ),
                    }
                ]
            )
        ),
        notes=[escape(note) for note in NOTES.values()],
        pid_table=_frame_html(_pid_frame(result)),
        omega_table=_frame_html(_omega_frame(result)),
        shapley_table=_frame_html(_shapley_frame(result)),
        timing_table=_frame_html(_timing_frame(result)),
        pareto_lead=escape(
            _front_sentence(result, "random_forest")
            + " "
            + _front_sentence(result, "logistic")
        ),
        pareto_table=_frame_html(_pareto_frame(result)),
        figures=figures,
    )
    _atomic_write_text(output_dir / "information_report.html", html)
    _atomic_write_text(output_dir / "latest.html", html)
    logger.info("Wrote information reports in %s", output_dir)
