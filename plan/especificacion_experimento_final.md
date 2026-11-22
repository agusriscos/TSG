# Objetivo: Infraestructura de producción `--profile paper` (paralelización, persistencia total de datos, `--plot-only`, inglés y script nocturno)

Los Bloques 0–5 están implementados y validados en modo `dev`. Antes de lanzar la corrida nocturna definitiva (`--profile paper`) para el manuscrito en inglés (entrega martes 12:00), prepara la infraestructura de ejecución y persistencia siguiendo `plan/OLD/gu_a_y_reglas_de_desarrollo_para_cursor.md`:

---

## Paso 1: Persistencia total de datos brutos y derivados (cero recomputación futura)
En **todos** los bloques (`phase2_hard_frontiers`, `phase2_information`, `phase2_surrogates`, `phase2_invariance`, `phase2_phase_maps`), guarda atómicamente en disco **todos los datos necesarios para regenerar cualquier figura, tabla LaTeX o análisis nuevo sin volver a calcular grafos**:
1. **Matrices y metadatos (`data/` dentro de cada carpeta de reporte):**
   - `features_raw.npz` y `features_clean.npz` (con `X`, `feature_names`, `y`, `regime_names`, `realization_ids`, `transform_or_kind`, `n_steps`, `snr_db` según aplique).
   - `oof_predictions.npz`: predicciones *out-of-fold* (`y_true`, `y_pred`, `y_proba`, `fold_id`) por vista (`visibility`, `transition`, `proximity`, `similarity`, `full`) y modelo (`random_forest`, `logistic`).
   - `tables/*.csv`: todas las tablas numéricas clave (ablación, PID, O-información, Shapley, Pareto + tiempos por serie, subrogados $\Delta_T$ y AUROC, matriz de invarianzas, mapas de fase `N × SNR` y tasas de NaN).
2. **Trazabilidad exacta (`config_snapshot.json`):**
   - Parámetros completos, semillas, `profile`, `git_commit`, timestamp UTC y versiones de `numpy`/`scipy`/`scikit-learn`/`networkx`.

---

## Paso 2: Modo `--plot-only` y figuras de publicación en inglés (`--lang en`)
1. Añade el flag **`--plot-only`** a los 5 scripts CLI (`run_phase1_audit.py`, `run_information_audit.py`, `run_surrogate_audit.py`, `run_invariance_audit.py`, `run_phase_maps_audit.py`) para que lean directamente los `.json`/`.npz` guardados y regeneren las figuras en segundos sin recomputar nada.
2. Añade **`--lang {en,es}`** (por defecto `en` cuando `--profile paper`) para que **todos los títulos, ejes, leyendas y anotaciones** salgan en inglés limpio de publicación (usando mathtext para símbolos como `$\Delta_T$`, `$\Omega$`, `$\lambda_2$`).
3. Guarda todas las figuras en **3 formatos**: `.pdf` (vectorial para LaTeX), `.svg` (vectorial editable) y `.png` (300 DPI).

---

## Paso 3: Paralelización determinista (`--workers`) y calibración de `--profile paper`
1. Añade `--workers` (por defecto `max(1, os.cpu_count() - 1)`) usando `ProcessPoolExecutor` o `joblib.Parallel` en los bucles de extracción de grafos/descriptores de los bloques 3, 2, 4 y 5.
   - **Determinismo estricto:** cada tarea `(regime, r, variant)` debe derivar su propia semilla determinista (`SeedSequence`) para que el resultado en paralelo sea idéntico al secuencial.
2. Calibra `--profile paper` para mantener rigor Q1 completando la corrida en una sola noche:
   - **Bloque 3 + Bloque 1:** 12 regímenes, `R=30`, `N=1500` (Bloque 1 reutiliza el `ensemble_cache` del Bloque 3 con 0 misses + cronómetro en `N ∈ {500, 1000, 1500}`).
   - **Bloque 2 (Subrogados + dHVG):** 7 regímenes, `R=30`, `N=1000`, `iaaft_min_iter=10`, `iaaft_max_iter=250`.
   - **Bloque 4 (Invarianzas):** 12 regímenes, `R=30`, `N=1000`.
   - **Bloque 5 (Mapas de fase `N × SNR`):** `N ∈ [100, 250, 500, 1000]`, `SNR_dB ∈ [inf, 20, 10, 5, 0, -5]`, `R=20`, 8 regímenes representativos (`white_noise`, `brownian`, `arma_correlated`, `quasiperiodic`, `lorenz`, `rossler`, `logistic`, `regime_switching`).

---

## Paso 4: Script maestro nocturno (`scripts/run_paper_suite.sh`) y empaquetado final
1. Crea `scripts/run_paper_suite.sh` (ejecutable, `set -euo pipefail`) que:
   - Ejecute en orden: **Bloque 3 → Bloque 1 → Bloque 2 → Bloque 4 → Bloque 5** con `--profile paper --lang en --workers <auto>`.
   - Registre tiempos por bloque en `outputs/paper_bundle/run_log.txt`.
   - Soporte `--resume` (si un bloque ya tiene su `config_snapshot.json` y `metrics.json` completos en `paper`, lo salta).
   - Al terminar, copie todas las figuras (`.pdf`, `.svg`, `.png`) y tablas (`.csv`) finales a `outputs/paper_bundle/figures/` y `outputs/paper_bundle/tables/` con nombres claros y numerados.

---

## Paso 5: Verificación rápida antes de lanzar la noche
1. Añade un test en `tests/test_paper_infra.py` verificando:
   - Determinismo entre `--workers 1` y `--workers 2`.
   - Existencia de `features_clean.npz`, `oof_predictions.npz` y `config_snapshot.json`.
   - Funcionamiento de `--plot-only --lang en` sin recomputar.
2. Ejecuta `pytest` completo y confirma que toda la suite pasa en verde.
3. **No lances todavía `--profile paper`**; haz solo una prueba seca rápida en `--profile dev --workers 2` de un bloque y dime el comando exacto para lanzar `scripts/run_paper_suite.sh` esta noche.