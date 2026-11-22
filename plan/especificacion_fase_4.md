# Objetivo: Bloques 4 y 5 — Matriz de invarianzas ante transformaciones + Mapas de fase N × SNR

Los Bloques 0, 3, 2 y 1 están cerrados (85 tests en verde). Ahora implementamos los **Bloques 4 y 5** de la ampliación Q1 para cerrar la parte sintética del manuscrito:
- **Bloque 4:** Auditoría empírica de **invarianzas** por familia TS2G frente a transformaciones y perturbaciones de señal.
- **Bloque 5:** **Mapas de fase `N × SNR`** y análisis de modos de fallo (incluyendo topológicos vs. espectrales).

Sigue estrictamente las reglas de arquitectura, estilo y testing de `plan/OLD/gu_a_y_reglas_de_desarrollo_para_cursor.md`. El código fuente reside en `src/tsg/`. Usa solo NumPy, SciPy y scikit-learn.

---

## Paso 1: Módulo de transformaciones y perturbaciones (`src/tsg/generators/perturbations.py`)

Crea `src/tsg/generators/perturbations.py`, determinista por semilla (`rng`), con las siguientes transformaciones aplicables a cualquier serie $x(t)$:
1. **`affine_transform(x, scale=3.5, shift=10.0)`**: $y = a x + b$ (prueba invarianza a escala y desplazamiento).
2. **`monotone_nonlinear(x)`**: transformación estrictamente creciente no lineal tras estandarizar, p. ej. $y = \sinh(1.5 \cdot z)$ o $\text{sign}(z)|z|^{1.8}$ (prueba invarianza ordinal/monótona: `HVG`/`OPN` deberían ser invariantes o casi invariantes, mientras `NVG`/`RN` cambian).
3. **`impulsive_outliers(x, rng, rate=0.02, magnitude=6.0)`**: inyecta picos impulsivos ($\pm 6\sigma$) en el 2% de las muestras.
4. **`quantize_ties(x, n_levels=8)`**: cuantización uniforme a 8 niveles discretos (introduce empates masivos, crítico para `OPN` y `HVG`).
5. **`downsample_series(x, factor=2)`**: submuestreo por factor 2 (o interpolación de vuelta a longitud $N$ si el protocolo requiere misma longitud; documenta la elección).
6. **`add_white_noise_snr(x, snr_db, rng)`**: añade ruido blanco gaussiano calibrado a una relación señal-ruido exacta en decibelios ($\text{SNR}_{\text{dB}} = 10 \log_{10}(\text{Var}(x)/\text{Var}(\epsilon))$).

---

## Paso 2: Bloque 4 — Auditoría de invarianzas (`src/tsg/audit/invariance.py`)

Crea `src/tsg/audit/invariance.py` que, sobre un panel representativo de regímenes (en `dev`: `white_noise`, `brownian`, `arma_correlated`, `quasiperiodic`, `lorenz`, `logistic`, `R=10, N=1000`; en `paper`: los 12 regímenes de `phase2_hard_frontiers`, `R=30, N=2500`):
1. Calcule para cada par **(familia TS2G × transformación)**:
   - **Estabilidad de descriptores (distancia normalizada):** mediana del error absoluto normalizado por la desviación estándar intra-panel (o correlación de Spearman media) entre el vector de descriptores original y el transformado. Permite detectar qué familias son **exactamente invariantes** (error ≈ 0) a `affine` o `monotone_nonlinear`.
   - **Estabilidad clasificatoria ($\Delta \text{Macro-F1}_{\text{OOD}}$):** entrena RF y LR sobre las series **originales** (5-fold CV) y evalúa cada fold de test sobre las series **transformadas** (sin reentrenar), reportando `F1_clean`, `F1_perturbed` y la caída `delta_F1 = F1_perturbed - F1_clean`.
2. Guarde atómicamente en `outputs/reports/phase2_invariance/`:
   - `invariance_metrics.json`
   - `invariance_report.md`
   - Figura resumen `01_invariance_matrix.png/.svg` (doble heatmap: izquierda = error normalizado de descriptores; derecha = caída de Macro-F1 RF por familia × transformación).

---

## Paso 3: Bloque 5 — Mapas de fase `N × SNR` (`src/tsg/audit/phase_maps.py`)

Crea `src/tsg/audit/phase_maps.py` que evalúe la frontera de colapso en el plano `(N, SNR_dB)`:
1. Rejilla de barrido:
   - `--profile dev`: `N ∈ [200, 500, 1000]`, `SNR_dB ∈ [∞ (clean), 20, 10, 5, 0]`, `R=8` sobre 6 regímenes clave (`white_noise`, `arma_correlated`, `quasiperiodic`, `lorenz`, `rossler`, `logistic`).
   - `--profile paper`: `N ∈ [100, 250, 500, 1000, 2500]`, `SNR_dB ∈ [∞, 30, 20, 10, 5, 0, -5]`, `R=20` sobre los 12 regímenes.
2. En cada celda `(N, SNR_dB)`, calcula Macro-F1 (RF y LR, 5-fold CV) para:
   - Las 4 familias (`visibility`, `transition`, `proximity`, `similarity`) + `full`.
   - **Contraste topológico vs. espectral:** separa los descriptores de `full` en dos subconjuntos (`topological_only`: grado, clustering, distancias, eficiencia, direccionalidad; vs. `spectral_only`: `lambda2_g`, `lambda2_lcc`, `lambda2_tilde`, `lambda_max`) y registra también la tasa de `NaN` / desconexión por celda para cuantificar modos de fallo.
3. Guarda atómicamente en `outputs/reports/phase2_phase_maps/`:
   - `phase_maps_metrics.json`
   - `phase_maps_report.md`
   - Figuras vectoriales limpias (`.png` + `.svg`):
     1. `01_phase_maps_families.png/.svg` (grid de heatmaps `N × SNR` para las 5 vistas en RF).
     2. `02_topological_vs_spectral_snr.png/.svg` (curvas de Macro-F1 frente a SNR para `topological_only` vs. `spectral_only` y tasa de fallos/NaN).

---

## Paso 4: Scripts CLI, tests y validación rápida (`--profile dev`)
1. Crea `scripts/run_invariance_audit.py` y `scripts/run_phase_maps_audit.py` (ambos con `--profile dev` y `--profile paper`).
2. Añade `tests/test_invariance_and_phase_maps.py` verificando:
   - `affine_transform` deja invariantes los descriptores de `NVG` y `HVG` (tolerancia numérica).
   - `monotone_nonlinear` deja invariante `HVG` (mismo grafo de visibilidad horizontal) y preserva el orden relativo en `OPN`.
   - `add_white_noise_snr` cumple la relación señal-ruido objetivo dentro de tolerancia estadística.
   - `quantize_ties` produce exactamente ≤ `n_levels` valores únicos.
3. Ejecuta `pytest` completo y confirma que toda la suite pasa en verde.
4. Ejecuta ambos scripts con `--profile dev` y muéstrame:
   - La tabla de **invarianzas** (error normalizado + Macro-F1 RF OOD por familia × transformación).
   - El resumen del **mapa de fase `N × SNR`** (umbral aproximado de SNR donde cae cada familia por debajo de F1 = 0.80 y comparativa `topological_only` vs. `spectral_only`).