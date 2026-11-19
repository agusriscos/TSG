# TSG — Generador de series sintéticas multirrégimen

Implementación Fase 1 de la Tabla 3.1: **1 serie controlada** por cada `Dataset ID`
(`white_noise`, `brownian`, `sine`, `quasiperiodic`, `lorenz`, `logistic`).

## Instalación

```bash
uv venv .venv
uv pip install -e ".[dev]"
source .venv/bin/activate
```

## Generar la suite (Fase 1)

```bash
python scripts/generate_synthetic_suite.py
```

Salidas:

- `outputs/data/synthetic_phase1/*.csv` + `metadata.json`
- `outputs/reports/synthetic_suite/<timestamp>_summary.html` y `latest.html`

## Tests

```bash
pytest
```

## Estructura

```text
src/tsg/generators/   # ABC, configs, estocástico / periódico / caótico, registry
src/tsg/reporting/    # HTMLReportBuilder (Plotly + Jinja2)
src/tsg/utils/        # logging e I/O
configs/              # YAML Fase 1
scripts/              # CLI de generación
tests/
```
