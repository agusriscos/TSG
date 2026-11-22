# Objetivo: Bloque 2 — Auditoría de subrogados (Shuffled/FT/IAAFT) + dHVG direccional

El Bloque 3 (`phase2_hard_frontiers`) y los fixes principales de `plotting.py` están cerrados. Ahora implementamos el **Bloque 2** de la ampliación Q1: cuantificar qué fracción del poder discriminativo de cada familia TS2G procede del orden temporal y de la no linealidad genuina, y añadir el grafo de visibilidad horizontal dirigido (`dHVG`) para medir irreversibilidad temporal.

Sigue estrictamente las reglas de arquitectura, estilo y testing de `plan/OLD/gu_a_y_reglas_de_desarrollo_para_cursor.md`. El código fuente reside en `src/tsg/`.

---

## Paso 1: Implementar `dHVG` (Directed Horizontal Visibility Graph)
1. En el módulo de grafos de visibilidad dentro de `src/tsg/graphs/`, añade el transformador **`dHVG`**:
   - Misma condición de visibilidad horizontal que `HVG`, pero con aristas dirigidas siguiendo la flecha del tiempo ($i \to j$ para $i < j$), devolviendo un grafo dirigido (`nx.DiGraph`).
   - Esto reactiva los descriptores direccionales (`i_dir`, etc.) dentro de la familia `visibility`.
   - Añade también la métrica de irreversibilidad temporal basada en `dHVG` (divergencia de Kullback-Leibler suavizada entre la distribución de grado de entrada $k^{\text{in}}$ y de salida $k^{\text{out}}$, integrada limpiamente en `src/tsg/descriptors/directional.py` o donde corresponda).
2. Registra `dHVG` dentro de la familia `visibility` en el factory/registro de métodos, manteniendo 100% de retrocompatibilidad con los configs de Fase 1 y Fase 2 que no lo listen.

---

## Paso 2: Módulo de subrogados (`src/tsg/generators/surrogates.py`)
Crea `src/tsg/generators/surrogates.py` usando solo NumPy/SciPy (sin dependencias externas pesadas), determinista por semilla (`rng`), con tres controles emparejados en longitud para cualquier serie $x(t)$:
1. **`shuffled_surrogate(x, rng)`**: Permutación aleatoria (destruye orden temporal, preserva distribución marginal exacta).
2. **`ft_surrogate(x, rng)`**: Aleatorización de fases de Fourier (preserva el espectro de potencia lineal, destruye correlaciones de fase no lineales).
3. **`iaaft_surrogate(x, rng, max_iter=100, tol=1e-6)`**: *Iterated Amplitude-Adjusted Fourier Transform* (Schreiber & Schmitz), que preserva simultáneamente el espectro de potencia lineal y la distribución exacta de amplitudes, destruyendo la no linealidad de fase.

---

## Paso 3: Pipeline de auditoría de subrogados (`src/tsg/audit/surrogates.py` + CLI)
Crea `src/tsg/audit/surrogates.py` y el script `scripts/run_surrogate_audit.py` (con soporte `--profile dev` [`R=10, N=1000`] y `--profile paper` [`R=30, N=2500`]) que:
1. Sobre un panel representativo (`white_noise`, `arma_correlated`, `quasiperiodic`, `lorenz`, `rossler`, `henon`, `logistic`), genere las series originales y sus 3 subrogados (`shuffled`, `ft`, `iaaft`) y extraiga la matriz $D$ de las 4 familias (incluyendo `dHVG` en `visibility`).
2. Calcule por familia (`visibility`, `proximity`, `transition`, `similarity`, `full`):
   - **Sensibilidad Temporal ($\Delta_T$)**: Caída de Macro-F1 (RF y LR) al clasificar regímenes sobre `shuffled` frente a `original`.
   - **Ganancia de No Linealidad Genuina ($\Delta_{NL}$)**:
     - (a) Clasificación binaria `original` vs `iaaft` por régimen y familia (Macro-F1 / AUROC), para verificar qué familias detectan no linealidad genuina en `lorenz`/`rossler`/`henon`/`logistic` sin falsos positivos en `white_noise`/`arma_correlated` (~0.50).
     - (b) Comparativa de clasificación multiclase sobre `original` vs `ft` vs `iaaft`.
3. Guarde los resultados atómicamente en `outputs/reports/phase2_surrogates/` (`surrogate_metrics.json`, `surrogate_report.md` y al menos 1 figura resumen en `.png` y `.svg`).

---

## Paso 4: Dos micro-ajustes visuales pendientes en `src/tsg/audit/plotting.py`
1. En `01_pca_projection`: aumenta ligeramente el margen entre la leyenda `familia` y el título `B1) Top-8 loadings en PC1` (p. ej. ajustando `bbox_to_anchor` o `pad`) para que no se rocen.
2. En `05_critical_pairs_boxplots`: haz que los regímenes mostrados se lean dinámicamente de `config.critical_pairs` (en lugar de mostrar siempre `white_noise`, `logistic`, `lorenz` fijos).

---

## Paso 5: Tests y corrida de validación rápida (`--profile dev`)
1. Añade tests unitarios en `tests/test_surrogates.py` verificando:
   - `shuffled` conserva el multiconjunto de valores (ordenados idénticos).
   - `ft` conserva el espectro de potencia ($|\text{FFT}|$) dentro de tolerancia numérica.
   - `iaaft` conserva tanto la distribución de amplitudes como el espectro de potencia con error bajo.
   - `dHVG` genera un `DiGraph` válido con `i_dir` no nulo en series asimétricas/caóticas.
2. Ejecuta `pytest` y confirma que toda la suite (previa + nueva) pasa en verde.
3. Ejecuta `scripts/run_surrogate_audit.py --profile dev` y muéstrame la tabla resumen de:
   - Macro-F1 multiclase (`original` vs `shuffled` vs `ft` vs `iaaft`) por familia.
   - Detección binaria `original vs IAAFT` por régimen y familia.