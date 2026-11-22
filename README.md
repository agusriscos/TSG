# TSG — Series sintéticas, TS2G y auditoría de la matriz D

Alcance acotado a la **Fase 1**: generar series controladas → mapear a grafos (TS2G) →
extraer la matriz de descriptores \(D\) → auditar su discriminabilidad entre los 6 regímenes
(`white_noise`, `brownian`, `sine`, `quasiperiodic`, `lorenz`, `logistic`).

## Instalación

```bash
uv venv .venv
uv pip install -e ".[dev]"
source .venv/bin/activate
```

## Generar la suite (Tabla 3.1)

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

## Descriptores — matriz D (Tabla 3.3)

```bash
python scripts/extract_descriptors.py
```

Salidas:

- `outputs/descriptors/synthetic_phase1/*_matrix.csv` (6 matrices 16×8)
- `outputs/descriptors/synthetic_phase1/all_descriptors_long.csv`
- `outputs/reports/descriptors/latest.html`

## Auditoría de discriminabilidad

```bash
python scripts/run_phase1_audit.py          # R=30, N=5000 (costoso)
python scripts/run_phase1_audit.py --quick  # smoke test rápido
```

Salidas en `outputs/reports/phase1_audit/`: veredicto Go/No-Go, figuras, `metrics_summary.json`, HTML/MD.

## Tests

```bash
pytest
```

## Estructura

```text
src/tsg/generators/    # Series sintéticas multirrégimen
src/tsg/graphs/        # Transformadores TS2G (Tabla 3.2)
src/tsg/descriptors/   # Descriptores topológicos/espectrales → matriz D
src/tsg/audit/         # Auditoría de discriminabilidad de D
src/tsg/reporting/     # Reportes HTML
src/tsg/utils/         # Logging e I/O
configs/               # YAML de la Fase 1
scripts/               # CLI de generación, TS2G, descriptores y auditoría
plan/                  # Especificaciones de alcance
tests/
```
