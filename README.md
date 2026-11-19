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

## Mapeo TS2G (Tabla 3.2)

```bash
python scripts/build_ts2g_graphs.py
```

Salidas:

- `outputs/graphs/synthetic_phase1/<dataset>/<method>.graphml` + `metadata.json`
- `outputs/reports/ts2g_mapping/latest.html`

## Tests

```bash
pytest
```

## Estructura

```text
src/tsg/generators/   # ABC, configs, estocástico / periódico / caótico, registry
src/tsg/graphs/       # 8 transformadores TS2G (Tabla 3.2)
src/tsg/reporting/    # HTMLReportBuilder / TS2GReportBuilder
src/tsg/utils/        # logging e I/O
configs/              # YAML Fase 1
scripts/              # CLI de generación y mapeo
tests/
```
