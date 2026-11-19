# Especificación Técnica: Mapeo de Series Temporales a Grafos (TS2G - Tabla 3.2)

> **Instrucción para Cursor:** Antes de proponer cambios o escribir código, lee `plan/gu_a_y_reglas_de_desarrollo_para_cursor.md` (o `Plan/cursor_guidelines.md`) y revisa la arquitectura existente en `src/tsg/` y `outputs/data/synthetic_phase1/`.

## 1. Contexto y Objetivo

El objetivo de esta tarea es implementar el módulo de transformación **Time Series to Graph (TS2G)** para convertir cada una de las 6 series sintéticas generadas en la Fase 1 (`white_noise`, `brownian`, `sine`, `quasiperiodic`, `lorenz`, `logistic`) en grafos utilizando los **8 algoritmos representativos** definidos en la **Tabla 3.2**.

* **Alcance actual (Fase 1):** Ejecutar la matriz completa de **6 series $\times$ 8 métodos = 48 grafos**, guardando cada grafo junto con sus metadatos de construcción para que el siguiente módulo (cálculo de los 14 descriptores topológicos) pueda consumirlos directamente.
* **Escalabilidad futura:** La arquitectura debe estar completamente parametrizada mediante `dataclasses` y archivos YAML para poder aplicar estos mismos transformadores a $N$ series sintéticas o datasets reales con distintos hiperparámetros.

---

## 2. Registro de Algoritmos TS2G (Tabla 3.2)

A continuación se especifican los 8 métodos, sus familias, número esperado de nodos $n$, parámetros por defecto, tipo de grafo y complejidad objetivo:

| Familia (`Family`) | Método (`Method`) | Nodos ($n$) | Parámetros Clave (`Key parameters`) | Tipo de Grafo (`Type`) | Complejidad |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Direct Geometry** | **`NVG`** | $N$ | Geometría *line-of-sight* (Natural Visibility Graph) | No dirigido, binario (`Undirected binary`) | $O(N \log N)$ |
| **Direct Geometry** | **`HVG`** | $N$ | *Horizontal line-of-sight* (Horizontal Visibility Graph) | No dirigido, binario (`Undirected binary`) | $O(N \log N)$ |
| **Phase Space** | **`RN`** | $N - (m-1)\tau$ | $m=3$, $\tau=1$, $w=5$, $\varepsilon = 5\%$ de distancias entre pares | No dirigido, binario (`Undirected binary`) | $O(N^2)$ |
| **Phase Space** | **`kNN`** | $N - (m-1)\tau$ | $m=3$, $\tau=1$, $w=5$, $k=5$, simetrizado por unión (*union symmetrized*) | No dirigido, binario (`Undirected binary`) | $O(N^2)$ |
| **Discr. & Transition** | **`STN`** | $M$ | $M=10$ cuantiles empíricos | Dirigido, ponderado (`Directed weighted`) | $O(N)$ |
| **OPN** | **`OPN`** | $\le m!$ | $m=4$, $\tau=1$ (patrones ordinales de Bandt–Pompe) | Dirigido, ponderado (`Directed weighted`) | $O(N)$ |
| **Similarity & Corr.** | **`Correlation`** | $\lfloor N/L \rfloor$ | $L=20$, $\theta_{\text{corr}} \approx 0.75$ (corrección de Bonferroni) | No dirigido, binario (`Undirected binary`) | $O((N/L)^2)$ |
| **Similarity & Corr.** | **`Segment`** | $\lfloor N/L \rfloor$ | $L=20$, percentil de distancia $\varepsilon = 20\%$ | No dirigido, binario (`Undirected binary`) | $O((N/L)^2)$ |

### Glosario de Símbolos (Nota de la Tabla 3.2)
* $N$: Longitud de la serie temporal.
* $m$: Dimensión de inmersión (*embedding dimension*).
* $\tau$: Retardo temporal (*time delay*).
* $w$: Ventana de exclusión de Theiler (*Theiler exclusion window*): no se permiten aristas entre estados $i, j$ si $|i - j| \le w$.
* $\varepsilon$: Percentil de umbral de distancia para recurrencia (`RN`) o similitud entre segmentos (`Segment`).
* $k$: Número de vecinos más cercanos (`kNN`).
* $M$: Número de cuantiles empíricos para discretización (`STN`).
* $L$: Longitud de bloque/segmento no solapado (`Correlation`, `Segment`).
* $\theta_{\text{corr}}$: Umbral crítico de correlación (`Correlation`).

---

## 3. Detalles de Implementación por Familia

1. **Direct Geometry (`NVG`, `HVG`):**
   * Implementar con complejidad eficiente (p. ej., pila monótona para `HVG` en $O(N)$ / $O(N \log N)$ y algoritmo *divide-and-conquer* o `ts2vg` si se añade como dependencia, o implementación propia vectorizada/optimizada en Python/NumPy para `NVG`).
   * Sin auto-lazos (*self-loops*).
2. **Phase Space (`RN`, `kNN`):**
   * Reconstruir primero la matriz de estados de Takens $X \in \mathbb{R}^{n \times m}$ con $n = N - (m-1)\tau$.
   * Aplicar estrictamente la **ventana de Theiler** ($|i - j| \le w$) excluyendo esos pares tanto del cálculo del percentil $\varepsilon$ como de la selección de vecinos y creación de aristas.
   * En `RN`: calcular distancias euclídeas por pares (fuera de la ventana de Theiler) y conectar pares cuya distancia sea $\le$ percentil 5 %.
   * En `kNN`: buscar los $k=5$ vecinos más cercanos válidos (fuera de la ventana de Theiler) para cada nodo y simetrizar la matriz de adyacencia por **unión** ($A_{\text{sym}} = \max(A, A^T)$).
3. **Discretization & Transition (`STN`):**
   * Discretizar la serie en $M=10$ estados usando cuantiles empíricos (gestionar empates de forma robusta, p. ej., con `np.quantile` o `pd.qcut(..., duplicates="drop")` registrando una advertencia si hubiera colapso de bins).
   * Crear un grafo dirigido (`nx.DiGraph`) donde las aristas $(s_t \to s_{t+1})$ tengan como atributo `weight` el conteo de transiciones (y opcionalmente guardar la probabilidad de transición normalizada por nodo origen).
4. **Ordinal Partition Network (`OPN`):**
   * Extraer vectores de inmersión con $m=4, \tau=1$ y mapear cada vector a su permutación ordinal de Bandt–Pompe (rompiendo empates de forma determinista por orden temporal, como `np.argsort(..., kind="mergesort")`).
   * Construir un `nx.DiGraph` donde los nodos son los patrones ordinales observados ($\le 4! = 24$) y las aristas dirigidas ponderadas representan las transiciones consecutivas entre patrones.
5. **Similarity & Correlation (`Correlation`, `Segment`):**
   * Dividir la serie en $n = \lfloor N/L \rfloor$ segmentos contiguos no solapados de longitud $L=20$ (descartando el resto final si $N$ no es múltiplo exacto de $L$).
   * En `Correlation`: calcular la matriz de correlación de Pearson entre segmentos; conectar $(i, j)$ si $i \neq j$ y $|r_{ij}| \ge \theta_{\text{corr}}$ (permitir configurar $\theta_{\text{corr}}$ explícitamente o calcularlo por umbral de significación con corrección de Bonferroni sobre los $\binom{n}{2}$ pares, usando $\approx 0.75$ como referencia por defecto). Gestionar segmentos de varianza cero sin generar `NaN`.
   * En `Segment`: calcular distancias euclídeas por pares entre segmentos y conectar $(i, j)$ ($i \neq j$) si la distancia es $\le$ percentil 20 %.

---

## 4. Arquitectura Propuesta en `src/tsg/`

Seguir exactamente el patrón POO ya establecido en `src/tsg/generators/`:

```text
src/tsg/
├── graphs/
│   ├── __init__.py
│   ├── base.py            # BaseGraphTransformer (ABC), GraphResult (dataclass)
│   ├── config.py          # Dataclasses de configuración por método + TS2GSuiteConfig
│   ├── registry.py        # Registro de transformadores por nombre (NVG, HVG, RN, kNN, STN, OPN, Correlation, Segment)
│   ├── geometry.py        # NVGTransformer, HVGTransformer
│   ├── phase_space.py     # RecurrenceNetworkTransformer, KNNTransformer
│   ├── transition.py      # STNTransformer, OPNTransformer
│   └── similarity.py      # CorrelationNetworkTransformer, SegmentNetworkTransformer
configs/
└── ts2g_mapping_phase1.yaml
scripts/
└── build_ts2g_graphs.py
tests/
└── test_ts2g.py
```

### Contratos Clave
* **`BaseGraphTransformer(ABC)`:**
  * Método abstracto `transform(self, series: NDArray[np.float64], dataset_id: str) -> GraphResult`.
* **`GraphResult` (`dataclass`):**
  * `dataset_id: str`, `regime: str`, `method: str`, `family: str`, `directed: bool`, `weighted: bool`, `graph: nx.Graph | nx.DiGraph`, `params: dict[str, Any]`, `build_time_sec: float`.
* **Persistencia de grafos (`outputs/graphs/synthetic_phase1/`):**
  * Guardar cada grafo en un formato estándar y lossless para NetworkX (por ejemplo, `outputs/graphs/synthetic_phase1/<dataset_id>/<method>.graphml` o `.gexf` / `.pkl`).
  * Generar un `metadata.json` global con el índice de los 48 grafos, sus rutas, número de nodos, número de aristas, densidad, tiempo de cómputo y parámetros exactos.

---

## 5. Reporte HTML Autocontenido

Generar un reporte en `outputs/reports/ts2g_mapping/<timestamp>_summary.html` y `outputs/reports/ts2g_mapping/latest.html` que incluya:
1. **Resumen Ejecutivo (KPIs):** Total de grafos generados (48/48), tiempo total de ejecución, comprobación de integridad (sin errores ni `NaN`).
2. **Matriz Resumen (6 series $\times$ 8 métodos):** Tabla interactiva o estilizada con nodos ($n$), aristas ($e$), densidad, grado medio y tiempo de construcción para cada combinación.
3. **Verificación de Propiedades Esperadas:** Comprobación visual/tabular de que el número de nodos $n$ coincide con la fórmula teórica de la Tabla 3.2 (por ejemplo, $N$ en NVG/HVG, $N-(m-1)\tau$ en RN/kNN, $\le 24$ en OPN, $\lfloor N/L \rfloor$ en Correlation/Segment).
4. **Visualizaciones Embebidas:**
   * Heatmap de densidad o grado medio (Series vs. Métodos).
   * Distribuciones de grado o visualizaciones representativas de grafos pequeños (`STN`, `OPN`, `Correlation`, `Segment`).

---

## 6. Tests Unitarios (`tests/test_ts2g.py`)

Implementar pruebas con `pytest` que validen:
1. **Número exacto de nodos ($n$)** según cada fórmula de la Tabla 3.2.
2. **Tipo de grafo correcto:** `nx.Graph` (no dirigido) para `NVG`, `HVG`, `RN`, `kNN`, `Correlation`, `Segment`; `nx.DiGraph` (dirigido y con atributo `weight`) para `STN` y `OPN`.
3. **Ventana de Theiler:** En `RN` y `kNN`, verificar que ninguna arista $(i, j)$ cumple $|i - j| \le w$.
4. **Robustez y determinismo:** Misma serie de entrada produce exactamente el mismo grafo; sin `NaN` en pesos ni crashes con series constantes o cortas.
5. **Integración:** Ejecución del pipeline completo y generación sin errores del reporte HTML.