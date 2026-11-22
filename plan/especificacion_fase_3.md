# Objetivo: Bloque 1 — Auditoría teórico-informacional (PID, O-información, SAGE agrupado) y Frontera de Pareto

Los Bloques 0, 3 y 2 están cerrados (62 tests en verde). Ahora implementamos el **Bloque 1** (la contribución metodológica estrella del paper): cuantificar formalmente la **redundancia, información única y sinergia** entre las 4 familias TS2G (`visibility`, `proximity`, `transition`, `similarity`) y trazar la **frontera de Pareto precisión vs. coste computacional** sobre las 15 combinaciones no vacías de familias.

Sigue las reglas de arquitectura, estilo y testing de `plan/OLD/gu_a_y_reglas_de_desarrollo_para_cursor.md`. El código fuente reside en `src/tsg/`. **Usa solo NumPy, SciPy y scikit-learn** (no añadas dependencias pesadas como `dit` o `idtxl`).

---

## Paso 1: Módulo teórico-informacional (`src/tsg/audit/information.py`)

Crea `src/tsg/audit/information.py` con tres bloques bien documentados y reproducibles:

1. **Descomposición de Información Parcial (PID) por pares de familias $(F_i, F_j)$:**
   - Para evitar la maldición de la dimensionalidad sobre bloques de 16–32 descriptores, proyecta cada familia a un resumen compacto libre de fuga de datos: **probabilidades predichas *out-of-fold* (OOF)** de un clasificador calibrado (o `RandomForest`/`LogisticRegression` dentro de CV estratificada), o los primeros componentes discriminantes OOF.
   - Estima la información mutua $I(F_i; Y)$, $I(F_j; Y)$ e $I(F_i, F_j; Y)$ (usando estimador discreto/KNN o Gaussiano regularizado con corrección de sesgo / entropía diferencial) y aplica la descomposición MMI (*Minimum Mutual Information*):
     - $\text{Red}(F_i, F_j; Y) = \min\{I(F_i; Y), I(F_j; Y)\}$
     - $\text{Un}(F_i; Y \mid F_j) = I(F_i; Y) - \text{Red}$
     - $\text{Un}(F_j; Y \mid F_i) = I(F_j; Y) - \text{Red}$
     - $\text{Syn}(F_i, F_j; Y) = I(F_i, F_j; Y) - I(F_i; Y) - I(F_j; Y) + \text{Red}$
   - Calcula la tabla completa para los $\binom{4}{2} = 6$ pares de familias.

2. **O-información ($\Omega$) de orden superior (4 familias):**
   - Implementa el estimador Gaussiano regularizado de **O-información** ($\Omega = (n - 2) H(X) + \sum_i [H(X_i) - H(X_{-i})]$) sobre las representaciones de las 4 familias (tanto global como condicionada por régimen o por grupos dinámicos: estocásticos vs. flujos caóticos vs. mapas discretos) para determinar si el sistema multivista opera dominado por **redundancia ($\Omega > 0$)** o **sinergia ($\Omega < 0$)**.

3. **Atribución agrupada por familia (SAGE / Shapley exacto sobre las 4 familias):**
   - Dado que son solo $M = 4$ familias ($2^4 = 16$ subconjuntos), calcula **valores de Shapley exactos** (sin aproximación Monte Carlo) de Macro-F1 (RF y LR) para cada familia sobre las 15 combinaciones no vacías (`{vis}`, `{prox}`, `{trans}`, `{sim}`, ..., `full`). Esto sustituye con rigor teórico la *Permutation Importance* univariante que colapsaba por colinealidad.

---

## Paso 2: Benchmark de coste computacional y Frontera de Pareto (`src/tsg/audit/pareto.py`)

Crea `src/tsg/audit/pareto.py` que:
1. Mida el tiempo de cómputo empírico (media y desviación estándar por serie, en milisegundos) de construir el grafo + extraer descriptores para cada una de las 4 familias (`visibility`, `transition`, `proximity`, `similarity`) a $N = 1000$ (en `dev`) y escalable a $N \in \{500, 1000, 2500\}$ (en `paper`).
2. Para las **15 combinaciones no vacías de familias**, cruce:
   - **Macro-F1 (RF y LR)** (de la evaluación exacta sobre el panel `phase2_hard_frontiers`), y
   - **Tiempo total por serie** (suma de tiempos de las familias incluidas).
3. Identifique automáticamente el **frente de Pareto óptimo** (combinaciones no dominadas: ninguna otra combinación es a la vez más rápida y más precisa).

---

## Paso 3: Script CLI y artefactos (`scripts/run_information_audit.py`)

Crea `scripts/run_information_audit.py` con soporte `--profile dev` y `--profile paper`:
- **Reutiliza el `ensemble_cache` de `phase2_hard_frontiers`** para no recomputar la matriz $D$ de las 12 clases.
- Genera atómicamente en `outputs/reports/phase2_information/`:
  - `information_metrics.json`
  - `information_report.md`
  - Figuras vectoriales (`.png` + `.svg`) limpias y sin solapamientos:
    1. `01_pid_decomposition_pairs.png/.svg` (barras apiladas de Redundancia / Única / Sinergia por par de familias).
    2. `02_shapley_family_importance.png/.svg` (Shapley exacto por familia en RF y LR).
    3. `03_pareto_frontier_cost_f1.png/.svg` (Macro-F1 vs. tiempo de cómputo en escala log, destacando las combinaciones en la frontera de Pareto).

---

## Paso 4: Tests y corrida rápida (`--profile dev`)
1. Añade `tests/test_information.py` verificando:
   - Propiedades matemáticas de PID (no negatividad de Red/Un y conservación exacta $I_{joint} = \text{Red} + \text{Un}_1 + \text{Un}_2 + \text{Syn}$).
   - Eficiencia de Shapley exacto (la suma de los 4 valores de Shapley iguala exactamente $\text{Score}(\text{full}) - \text{Score}(\emptyset)$).
   - Identificación correcta del frente de Pareto sobre un caso sintético conocido.
2. Ejecuta `pytest` completo y confirma que toda la suite pasa en verde.
3. Ejecuta `scripts/run_information_audit.py --profile dev` y muéstrame:
   - Tabla PID de los 6 pares de familias (Red, Un1, Un2, Syn).
   - Valores de Shapley exactos por familia (RF y LR) y O-información global.
   - Las combinaciones que forman la **frontera de Pareto** (Macro-F1 vs. tiempo ms/serie).