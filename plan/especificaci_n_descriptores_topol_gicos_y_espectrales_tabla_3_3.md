# Especificación Técnica: Descriptores Topológicos y Espectrales (Tabla 3.3)

> **Instrucción para Cursor:** Antes de proponer cambios o escribir código, lee `plan/gu_a_y_reglas_de_desarrollo_para_cursor.md` (o `Plan/cursor_guidelines.md`) y revisa la estructura actual de `src/tsg/graphs/` y los 48 grafos generados en `outputs/graphs/synthetic_phase1/`.

## 1. Contexto y Objetivo

El objetivo de esta tarea es implementar el módulo de extracción de **descriptores topológicos, espectrales y de dinámica direccional** definidos en la **Tabla 3.3** sobre los **48 grafos** generados en la fase anterior (6 series sintéticas $\times$ 8 algoritmos TS2G).

* **Entregable principal (Fase 1):** Generar **6 matrices de descriptores**, una por cada serie sintética (`white_noise`, `brownian`, `sine`, `quasiperiodic`, `lorenz`, `logistic`).
  * **Dimensiones de cada matriz:** **$16 \times 8$** (16 descriptores en filas $\times$ 8 métodos TS2G en columnas: `NVG`, `HVG`, `RN`, `kNN`, `STN`, `OPN`, `Correlation`, `Segment`).
* **Escalabilidad futura:** La arquitectura debe seguir el patrón POO del proyecto para permitir evaluar lotes arbitrarios de grafos (por ejemplo, $N$ realizaciones por régimen en futuras fases) y exportar resultados listos para análisis estadístico o entrenamiento de GNNs.

## 2. Catálogo Completo de los 16 Descriptores (Tabla 3.3)

La Tabla 3.3 agrupa un total de **16 métricas** en **7 categorías**:

| # | Categoría (`Category`) | ID / Columna | Símbolo | Definición Matemática y Justificación Diagnóstica |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **Macro Connectivity** | `comp` | $\text{Comp.}$ | Número total de componentes conexas del grafo. Separa soportes conexos ($\text{Comp.} = 1$) de fragmentados. |
| 2 | **Macro Connectivity** | `lcc` | $\text{LCC}$ | Fracción de nodos en la componente conexa gigante: $\vert V_{\text{LCC}} \vert / n \in [0, 1]$. Distingue grafos dominados por componente gigante ($\text{LCC} \ge 0.8$) de fragmentados ($\text{LCC} < 0.8$). |
| 3 | **Integration & Cohesion** | `c_avg` | $C_{\text{avg}}$ | Coeficiente de agrupamiento (*clustering*) medio. Cuantifica cohesión triádica local. |
| 4 | **Integration & Cohesion** | `e_glob` | $E_{\text{glob}}$ | Eficiencia global: $E_{\text{glob}} = \frac{1}{n(n-1)} \sum_{i \neq j} \frac{1}{d(i,j)}$. Cuantifica capacidad de transferencia paralela incluso en grafos desconectados (donde $1/\infty = 0$). |
| 5 | **Degree Mixing** | `assortativity_r` | $r$ | Asortatividad de grado de Newman. Distingue núcleos *hub-hub* ($r \ge 0.1$) de configuraciones en estrella *hub-leaf* ($r \le -0.1$) que aceleran el *over-smoothing* en GNNs. |
| 6 | **Heterogeneity** | `cv_k` | $\text{CV}(k)$ | Coeficiente de variación del grado: $\sigma_k / \mu_k$. Evalúa dispersión de escala sin requerir ajuste *power-law*. |
| 7 | **Heterogeneity** | `h_pk` | $H(P(k))$ | Entropía de Shannon de la distribución empírica de grados: $-\sum_k P(k) \ln P(k)$ (especificar base natural o $\log_2$ de forma consistente en metadatos). |
| 8 | **Heterogeneity** | `skew_k` | $\text{skew}(k)$ | Asimetría (*skewness*) de la distribución de grados $k$. Evalúa prominencia de *hubs*. |
| 9 | **Path Geometry** | `l_lcc` | $L_{\text{LCC}}$ | Longitud media de camino más corto dentro de la componente conexa gigante ($\text{LCC}$). |
| 10 | **Path Geometry** | `d_lcc` | $D_{\text{LCC}}$ | Diámetro (distancia geodésica máxima) dentro de la $\text{LCC}$. Diagnostica riesgos de *under-reaching* en GNNs. |
| 11 | **Laplacian Spectrum** | `lambda2_g` | $\lambda_2(G)$ | Conectividad algebraica (segundo menor autovalor del Laplaciano) del grafo completo $G$. Vale $0$ si $\text{Comp.} > 1$. |
| 12 | **Laplacian Spectrum** | `lambda2_lcc` | $\lambda_2(\text{LCC})$ | Conectividad algebraica restringida a la componente conexa gigante. Diagnostica *over-squashing* cuando $\lambda_2(\text{LCC}) < 10^{-2}$. |
| 13 | **Laplacian Spectrum** | `lambda2_tilde` | $\tilde{\lambda}_2$ | Conectividad efectiva suave continua: $\tilde{\lambda}_2 = (\text{LCC})^2 \cdot \lambda_2(\text{LCC})$. Evita discontinuidades por puentes simples. |
| 14 | **Laplacian Spectrum** | `lambda_max` | $\lambda_{\max}$ | Mayor autovalor del Laplaciano del grafo completo $G$. |
| 15 | **Directional Dynamics** | `i_dir` | $I_{\text{dir}}(W)$ | Asimetría de flujo direccional: $I_{\text{dir}}(W) = \frac{\| W - W^T \|_F}{\| W + W^T \|_F}$. Cuantifica irreversibilidad temporal en redes de transición (`STN`, `OPN`). |
| 16 | **Directional Dynamics** | `self_loop_ratio` | $\text{Self-loop ratio}$ | Proporción de autotransiciones previa a la eliminación de auto-lazos: $\frac{\sum_u W_{uu}}{\sum_{u,v} W_{uv}}$ (equivalente a $\frac{\sum_u W_{uu}}{N - 1}$ en redes de transición donde el total de transiciones es $N - 1$). Captura persistencia en flujos continuos. |

## 3. Reglas de Cálculo, Simetrización y Casos Borde

Para asegurar rigor matemático y evitar fallos en grafos grandes o degenerados, Cursor debe respetar las siguientes reglas:

### 3.1 Tratamiento de Grafos Dirigidos y Ponderados (`STN`, `OPN`)
1. **Paso 1 — Dinámica Direccional (antes de eliminar auto-lazos):**
   * Extraer la matriz de adyacencia dirigida ponderada original $W$ (con auto-lazos incluidos).
   * **Verificación crítica:** Comprobar si los transformadores `STNTransformer` y `OPNTransformer` actuales en `src/tsg/graphs/transition.py` conservan los auto-lazos $(u \to u)$ al guardar los grafos. Si se hubieran eliminado previamente, actualizarlos o registrar `self_loop_ratio` desde las transiciones originales antes de su eliminación para garantizar que la métrica refleje fielmente $\sum W_{uu} / (N - 1)$.
   * Calcular $I_{\text{dir}}(W) = \| W - W^T \|_F / \| W + W^T \|_F$ (si $\| W + W^T \|_F = 0$, devolver `0.0`).
   * Calcular `self_loop_ratio` $= \sum_u W_{uu} / \sum_{u,v} W_{uv}$ (si el denominador es $0$, devolver `0.0`).
2. **Paso 2 — Proyección Simétrica (sin auto-lazos) para métricas estructurales y espectrales:**
   * Eliminar auto-lazos ($W_{uu} \leftarrow 0$) para calcular las métricas de topología y espectro.
   * Construir la matriz simetrizada $W_{\text{sym}} = \frac{1}{2}(W + W^T)$ para el **espectro Laplaciano** (como exige la Tabla 3.3: *"evaluated over $W_{\text{sym}}$ in STN/OPN"*).
   * Para las métricas topológicas no dirigidas (`Comp.`, `LCC`, `C_avg`, `E_glob`, `r`, `CV(k)`, `H(P(k))`, `skew(k)`, `L_LCC`, `D_LCC`), evaluar sobre el soporte no dirigido sin auto-lazos correspondiente.

### 3.2 Tratamiento de Grafos No Dirigidos Binarios (`NVG`, `HVG`, `RN`, `kNN`, `Correlation`, `Segment`)
* $W$ es simétrica y sin auto-lazos, por lo que `i_dir` $= 0.0$ y `self_loop_ratio` $= 0.0$.

### 3.3 Rendimiento Computacional (Grafos con $N \ge 1000$)
Los grafos `NVG`, `HVG`, `RN` y `kNN` pueden tener varios miles de nodos. Para evitar cuellos de botella:
* **Distancias geodésicas (`E_glob`, `L_LCC`, `D_LCC`):** Usar `scipy.sparse.csgraph.shortest_path(adj_sparse, directed=False, unweighted=True)` en lugar de bucles de Python o `nx.global_efficiency` puro, que es muy lento en grafos grandes.
* **Espectro Laplaciano ($\lambda_2(G)$, $\lambda_2(\text{LCC})$, $\lambda_{\max}$):**
  * Construir el Laplaciano disperso $L = D - W_{\text{sym}}$ (o Laplaciano normalizado si se configura en el YAML, documentando claramente la elección por defecto en `metadata.json` y el reporte HTML).
  * Para grafos pequeños ($n \le 250$, como `STN`, `OPN`, `Correlation`, `Segment`): usar `scipy.linalg.eigh` sobre la matriz densa (exacto y muy rápido).
  * Para grafos grandes ($n > 250$): usar `scipy.sparse.linalg.eigsh` (método *shift-invert* con `sigma=0` o `which='SM'` para $\lambda_2$, y `which='LA'` para $\lambda_{\max}$), con *fallback* controlado y registro en `logging` si la convergencia es lenta.
  * Si $\text{Comp.} > 1$, asignar directamente $\lambda_2(G) = 0.0$ sin coste innecesario de búsqueda de autovalores nulos múltiples.

### 3.4 Manejo Robusto de Casos Borde (sin excepciones no controladas)
* Si el grafo no tiene aristas ($e = 0$): `c_avg = 0.0`, `e_glob = 0.0`, `assortativity_r = np.nan`, `cv_k = 0.0`, `skew_k = 0.0`, `l_lcc = 0.0`, `d_lcc = 0.0`, `lambda2_g = 0.0`, `lambda2_lcc = 0.0`, `lambda2_tilde = 0.0`, `lambda_max = 0.0`.
* Si todos los nodos tienen exactamente el mismo grado ($\sigma_k = 0$): `assortativity_r = np.nan`, `cv_k = 0.0`, `skew_k = 0.0`.
* Si $\vert V_{\text{LCC}} \vert \le 1$: `l_lcc = 0.0`, `d_lcc = 0.0`, `lambda2_lcc = 0.0`, `lambda2_tilde = 0.0`.

## 4. Arquitectura Propuesta en `src/tsg/`

Seguir el mismo estilo modular y orientado a objetos de `src/tsg/generators/` y `src/tsg/graphs/`:

```text
src/tsg/
├── descriptors/
│   ├── __init__.py
│   ├── base.py            # BaseDescriptorExtractor (ABC), DescriptorResult (dataclass)
│   ├── config.py          # DescriptorConfig (YAML / dataclass)
│   ├── connectivity.py    # Comp., LCC, C_avg, E_glob
│   ├── degree.py          # Assortativity r, CV(k), H(P(k)), skew(k)
│   ├── paths.py           # L_LCC, D_LCC (usando scipy.sparse.csgraph)
│   ├── spectral.py        # lambda_2(G), lambda_2(LCC), lambda_2_tilde, lambda_max
│   ├── directional.py     # I_dir(W), Self-loop ratio
│   └── pipeline.py        # Orquestador que calcula los 16 descriptores y construye las 6 matrices (16 x 8)
configs/
└── descriptors_phase1.yaml
scripts/
└── extract_descriptors.py
tests/
└── test_descriptors.py
```

## 5. Formato de Salidas y Persistencia

Guardar los artefactos en `outputs/descriptors/synthetic_phase1/`:
1. **6 Matrices individuales (16 filas $\times$ 8 columnas):**
   * `white_noise_matrix.csv`
   * `brownian_matrix.csv`
   * `sine_matrix.csv`
   * `quasiperiodic_matrix.csv`
   * `lorenz_matrix.csv`
   * `logistic_matrix.csv`
   * Índice de filas: los 16 IDs de descriptores en el orden exacto de la Tabla 3.3.
   * Columnas: los 8 métodos TS2G en el orden de la Tabla 3.2 (`NVG`, `HVG`, `RN`, `kNN`, `STN`, `OPN`, `Correlation`, `Segment`).
2. **Tabla consolidada:** `all_descriptors_long.csv` (con columnas `dataset_id`, `regime`, `method`, `family` + los 16 descriptores) para facilitar comparaciones entre regímenes.
3. **Metadatos:** `metadata.json` con tiempos de cómputo por grafo, tipo de Laplaciano usado, base logarítmica de la entropía, versión y recuento de posibles `NaN`.

## 6. Reporte HTML Autocontenido

Generar el reporte en `outputs/reports/descriptors/<timestamp>_summary.html` y `outputs/reports/descriptors/latest.html` incluyendo:

1. **Resumen Ejecutivo (KPIs):** Total de grafos analizados (48/48), matrices generadas (6/6 de $16 \times 8$), tiempo total de cómputo y estado de integridad (aviso de `NaN` esperados como asortatividad en grafos regulares).
2. **Panel por Serie Sintética (6 secciones):**
   * Tabla HTML completa de la matriz **$16 \times 8$** con formato numérico limpio (4 decimales o notación científica para valores $< 10^{-3}$).
   * **Indicadores diagnósticos visuales (según Tabla 3.3):** resaltar alertas relevantes para GNNs, como fragmentación ($\text{LCC} < 0.8$), riesgo de *over-smoothing* ($r \le -0.1$) y riesgo de *over-squashing* ($\lambda_2(\text{LCC}) < 10^{-2}$).
   * **Heatmap normalizado por fila** (z-score o min-max por descriptor) para comparar rápidamente el perfil estructural de los 8 algoritmos dentro de cada serie.
3. **Comparativa Global entre Regímenes:** Gráficos embebidos comparando descriptores clave (ej. $\tilde{\lambda}_2$, $I_{\text{dir}}(W)$, $E_{\text{glob}}$, $r$) agrupados por régimen (`Stochastic`, `Periodic`, `Chaotic`).

## 7. Tests Unitarios (`tests/test_descriptors.py`)

Implementar pruebas con `pytest` que verifiquen:
1. **Propiedades exactas en grafos sintéticos conocidos:**
   * Grafo completo $K_n$: `comp == 1`, `lcc == 1.0`, `c_avg == 1.0`, `e_glob == 1.0`, `l_lcc == 1.0`, `d_lcc == 1.0`, $\lambda_2(G) = n$ (en Laplaciano combinatorio).
   * Grafo desconectado (dos componentes iguales): `comp == 2`, `lcc == 0.5`, `lambda2_g == 0.0`, y $\tilde{\lambda}_2 = 0.25 \cdot \lambda_2(\text{LCC})$.
   * Grafo dirigido asimétrico con auto-lazos conocidos: comprobar valores exactos de `i_dir` y `self_loop_ratio`, y verificar que los auto-lazos no contaminan `c_avg` ni el espectro.
2. **Dimensiones de salida:** Verificar que cada una de las 6 matrices tiene forma exacta `(16, 8)` sin columnas ni filas faltantes.
3. **Casos borde:** Grafo vacío ($e=0$) y grafo regular sin lanzar excepciones.
4. **Reporte HTML:** Generación correcta de `latest.html` sin errores.