@gu_a_y_reglas_de_desarrollo_para_cursor.md @especificacion_auditoria_discriminalidad_matriz.md @especificaci_n_generador_series_sint_ticas_multirr_gimen.md

# Objetivo: Fase 0 (P0) — Saneamiento integral de la línea base TS2D

Antes de añadir nuevos experimentos, debemos auditar y corregir **5 vulnerabilidades críticas** detectadas en los artefactos de la Fase 1 (`metrics_summary.json` y reportes HTML/MD) para garantizar rigor metodológico, trazabilidad y reproducibilidad.

Sigue estrictamente las reglas de arquitectura, estilo y testing definidas en `gu_a_y_reglas_de_desarrollo_para_cursor.md`. **No modifiques el alcance del flujo directo TS2D** ni añadas todavía bloques nuevos (PID, subrogados, etc.); céntrate exclusivamente en diagnosticar, corregir y verificar los puntos siguientes.

---



## Paso 1: Diagnóstico previo (solo lectura antes de editar)

Inspecciona los siguientes archivos y muéstrame un resumen breve de la causa raíz de cada problema antes de aplicar cambios:

1. `tsg/generators/chaotic.py`, `tsg/generators/periodic.py`, `tsg/generators/stochastic.py`, `tsg/generators/base.py`, `configs/synthetic_suite_phase1.yaml`.
2. `tsg/audit/classification.py`, `tsg/audit/geometry.py`, `tsg/audit/health.py`, `tsg/audit/pipeline.py`, `configs/phase1_audit.yaml`.
3. `tsg/descriptors/degree.py`, `tsg/descriptors/base.py`, `tsg/reporting/audit_report.py`, `scripts/run_phase1_audit.py`.

---



## Paso 2: Correcciones obligatorias



### 1. Erradicar el colapso de varianza intra-clase (Silhouette ≈ 1.000)

- **Problema:** En `metrics_summary.json`, los regímenes `lorenz` (0.9999999967), `logistic` (0.9999999960) y `quasiperiodic` (0.9999999951) presentan varianza intra-clase casi nula entre las `R=30` realizaciones, lo que introduce fuga implícita en 5-fold CV.
- **Acción:** Revisa cómo se generan las `R` realizaciones en `tsg/generators/chaotic.py` y `tsg/generators/periodic.py`. Asegura que cada realización `r` muestree condiciones iniciales (y/o fases y ligeras perturbaciones paramétricas físicamente válidas dentro del mismo régimen) de forma determinista y reproducible a partir de la semilla base (`seed + r` o `SeedSequence`), garantizando trayectorias genuinamente distintas tras descartar el transitorio.



### 2. Auditar y corregir `brownian_mode="increments"`

- **Problema:** `brownian_mode="increments"` genera incrementos i.i.d. (matemáticamente ruido blanco), lo que explica que `transition` confunda `white_noise` y `brownian` (~0.56–0.61 F1), mientras `visibility` y `proximity` obtienen F1 = 1.000.
- **Acción:**
  - Diagnostica en `tsg/generators/stochastic.py` por qué `visibility` y `proximity` separan `white_noise` de `brownian` en modo `increments` (comprueba si hay diferencias de escala/varianza o de normalización).
  - Corrige la configuración por defecto (`configs/phase1_audit.yaml` y `configs/synthetic_suite_phase1.yaml`) para usar movimiento browniano integrado genuino (caminata aleatoria acumulada, no incrementos i.i.d.) y asegúrate de que el preprocesado/normalización de series sea consistente y sin artefactos de escala.



### 3. Incluir la familia `similarity` en la ablación de clasificación

- **Problema:** En `metrics_summary.json`, el bloque `classification.views` evalúa `visibility`, `transition`, `proximity` y `full`, pero **omite** `similarity`.
- **Acción:** Corrige `tsg/audit/classification.py` y/o `tsg/audit/pipeline.py` para que la ablación por vistas evalúe de forma obligatoria las 4 familias (`visibility`, `proximity`, `transition`, `similarity`) más `full`, tanto en Regresión Logística como en Random Forest.



### 4. Corregir bugs de métricas y trazabilidad en `audit/` y `reporting/`

- **Normalización de matriz de confusión:** En `tsg/audit/classification.py`, `confusion_normalized` actualmente normaliza por columnas en lugar de por filas (la fila `white_noise` de `transition` suma `1.067`). Cambia a normalización estricta por filas (`true` labels, recall por clase), de modo que cada fila sume `1.0`.
- **Nombres de variables en PCA:** En `tsg/audit/geometry.py`, `top_loadings_pc1_pc2` guarda `PC1`, `PC2`, `PC3` y `score`, pero **omite el nombre de la feature**. Añade la clave `"feature": <nombre_columna>` a cada entrada.
- **Consistencia de artefactos (112 vs. 128 features):** Asegura en `scripts/run_phase1_audit.py` y `tsg/reporting/audit_report.py` que `metrics_summary.json`, `report_phase1.md`, `report_phase1.html` y `latest.html` se generen atómicamente a partir del mismo diccionario de resultados en la misma ejecución, incluyendo el timestamp y el recuento exacto de features brutas/limpias.



### 5. Gestión explícita de `NaN` y descriptores constantes estructurales

- **Problema:** `transition__STN__assortativity_r` y `similarity__Correlation__assortativity_r` producen `NaN` (típicamente por varianza de grado cero en grafos regulares/completos), y 21 features caen por varianza nula.
- **Acción:** En `tsg/audit/health.py` (y en el cálculo de asortatividad si procede), distingue en el reporte de salud entre:
  - `structural_constant_features` (p. ej. `i_dir` y `self_loop_ratio` en grafos no dirigidos, o `comp`/`lcc` en grafos conexos por construcción como `NVG`/`HVG`), y
  - `numerical_nan_features` (como `assortativity_r` cuando la varianza de grado es 0), documentando el motivo exacto sin silenciar errores reales.

---



## Paso 3: Tests de regresión y verificación

1. Actualiza y amplía la suite en `tests/` (`test_generators.py`, `test_audit.py`, `test_descriptors.py`, `test_reporting.py`) añadiendo assertions explícitas para:
  - Ningún régimen determinista tiene `silhouette > 0.995` por colapso de varianza entre realizaciones.
  - `classification.views` contiene exactamente las 4 familias (`visibility`, `proximity`, `transition`, `similarity`) + `full`.
  - Cada fila de `confusion_normalized` suma `1.0` (con tolerancia `1e-6`).
  - Todos los elementos de `top_loadings_pc1_pc2` contienen la clave `"feature"` no vacía.
  - `n_features_raw` y `n_features_clean` coinciden exactamente entre `metrics_summary.json` y el reporte HTML/MD generado.
2. Ejecuta `pytest` y verifica que todos los tests pasen en verde.
3. Ejecuta una corrida de validación rápida (p. ej., con `R=10, N=1000` o el script `scripts/run_phase1_audit.py`) y muéstrame el resumen de métricas saneadas (salud de descriptores, Silhouette por régimen, Macro-F1 por cada una de las 4 familias + `full`).

