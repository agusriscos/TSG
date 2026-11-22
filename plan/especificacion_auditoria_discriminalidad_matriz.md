# Especificación Técnica — Fase 1: Auditoría de Discriminabilidad de la Matriz D

## 1. Objetivo
Verificar empíricamente si la matriz de descriptores topológicos actual $D \in \mathbb{R}^{p \times M}$ separa de forma inequívoca los 6 regímenes dinámicos sintéticos en el espacio de características. Este hito cierra el alcance actual de la investigación (generación → TS2G → \(D\) → auditoría)[cite: 1].

El script debe detectar de forma explícita si existen **colapsos de identificabilidad (degeneración)** en los pares críticos documentados:
1. Mapa logístico ($f(x) = 4x(1-x)$) vs. Ruido blanco[cite: 1].
2. Señal cuasi-periódica vs. Sistema de Lorenz[cite: 1].
3. Caos determinista (Lorenz / Mapa logístico) vs. Ruido blanco al usar distribuciones de grado aisladas[cite: 1].

---

## 2. Datos de Entrada y Estandarización del Banco Sintético
- **Regímenes (6 clases):** `white_noise`, `brownian`, `sine`, `quasi_periodic`, `lorenz`, `logistic`[cite: 1].
- **Longitud de cada serie:** $N = 5000$ observaciones (coherente con las fases previas del proyecto; aporta mayor margen frente a sesgos de tamaño finito en el test 0–1, $n \le N/10$, y en el conteo de patrones ordinales, $N \gg m!$)[cite: 1].
- **Ensamble:** Generar $R = 30$ (configurable a 50) realizaciones independientes por régimen usando semillas controladas (total: $6 \times R$ series)[cite: 1].
- **Configuración explícita para Movimiento Browniano:** Añadir un flag en configuración (`brownian_mode: "levels" | "increments" | "stl_residuals"`) que documente y aplique si las redes de transición (QG y OPTN) se calculan sobre la serie en niveles o sobre incrementos/residuos STL[cite: 1].
- **Estructura de entrada por realización:**
  - Matriz $D_r \in \mathbb{R}^{p \times M}$ con $p$ descriptores topológicos (filas) y $M$ algoritmos TS2G (columnas) agrupados por familia[cite: 1]:
    - **Visibilidad:** NVG, HVG, LPHVG[cite: 1].
    - **Transición:** QG, OPTN[cite: 1].
    - **Proximidad:** RN, $k$-NN[cite: 1].
  - Vectorizar cada $D_r$ en una fila de características de longitud $p \cdot M$ (nombrando cada columna como `{familia}__{algoritmo}__{descriptor}`), construyendo la tabla maestra `X` de forma $(6R, p \cdot M)$ y el vector de etiquetas `y`.

---

## 3. Pipeline de Análisis a Implementar

### 3.1. Limpieza y diagnóstico de salud de descriptores
- Detectar y reportar descriptores con valores `NaN`, infinitos o varianza cero/casi cero (por ejemplo, métricas que colapsan en algún algoritmo o régimen).
- Estandarizar características con `StandardScaler` (ajustado dentro de cada fold en validación cruzada para evitar fuga de datos).

### 3.2. Análisis no supervisado y geometría de clases
- **PCA:** Calcular componentes principales sobre `X` estandarizada, guardando varianza explicada acumulada y *loadings* (cargas) por descriptor[cite: 1].
- **Métricas de separabilidad geométrica:**
  - *Silhouette Score* global y desglosado por régimen.
  - Matriz de distancias euclídeas (y de Mahalanobis o coseno) entre centroides de los 6 regímenes, destacando los 3 pares críticos.

### 3.3. Clasificación supervisada y mini-ablación por familias
- Evaluar mediante validación cruzada estratificada (`StratifiedKFold`, $k = 5$ o $10$) dos clasificadores complementarios (siguiendo el enfoque de verificación de separabilidad tipo NetF)[cite: 1]:
  1. **Lineal:** Regresión Logística multinomial (L2) o SVM lineal (mide separabilidad lineal directa).
  2. **No lineal:** Random Forest (mide separabilidad con interacciones entre descriptores).
- Ejecutar el mismo esquema de evaluación sobre **4 vistas** para medir el aporte real de cada familia[cite: 1]:
  - Solo **Visibilidad**[cite: 1]
  - Solo **Transición**[cite: 1]
  - Solo **Proximidad**[cite: 1]
  - **Matriz $D$ completa** (multivista)[cite: 1]
- Calcular: Accuracy, Macro-F1, F1 por clase, Matriz de Confusión agregada (out-of-fold) e Importancia por Permutación (*Permutation Importance*).

---

## 4. Visualizaciones Obligatorias para el Reporte
Generar todas las figuras en alta resolución (`300 dpi`, formato `.png` y/o `.svg`) con paleta consistente por régimen y guardarlas en `reports/figures/`:

1. **`01_pca_projection.png` (Proyección PCA + Biplot):**
   - Panel A: Scatter PC1 vs. PC2 (y opcional PC1 vs. PC3) con elipses de confianza del 95 % para cada uno de los 6 regímenes.
   - Panel B: Vectores de carga (*loadings*) de los 10 descriptores con mayor peso en PC1–PC2.
   - *Qué permite interpretar:* Si las nubes de Ruido blanco y Logístico (o Lorenz y Cuasi-periódico) se solapan, existe degeneración directa en el espacio actual.

2. **`02_confusion_matrices_grid.png` (Matrices de confusión 2×2):**
   - Rejilla con 4 heatmaps normalizados (en porcentaje): Solo Visibilidad, Solo Transición, Solo Proximidad y Matriz $D$ completa.
   - *Qué permite interpretar:* Muestra exactamente qué pares de regímenes se confunden en cada familia y si la combinación multivista elimina esos errores.

3. **`03_regime_fingerprints_heatmap.png` (Huella media estandarizada por régimen):**
   - Heatmap de 6 filas (regímenes) × columnas (descriptores agrupados visualmente por familia TS2G), mostrando la media en *z-score*.
   - *Qué permite interpretar:* Permite inspeccionar visualmente qué descriptores distinguen cada régimen y detectar si dos regímenes comparten un perfil casi idéntico.

4. **`04_family_ablation_scores.png` (Comparativa de rendimiento por familia):**
   - Gráfico de barras agrupadas con Macro-F1 y *Silhouette Score* (± desviación estándar entre folds) para Visibilidad, Transición, Proximidad y Matriz completa, más un subpanel con el F1 individual de cada uno de los 6 regímenes en la matriz completa.
   - *Qué permite interpretar:* Identifica rápidamente si algún régimen específico (ej. Logístico o Browniano) queda por debajo del umbral aceptable.

5. **`05_critical_pairs_boxplots.png` (Auditoría de pares críticos en descriptores de grado):**
   - Boxplots/violin plots comparando las distribuciones de los descriptores globales clásicos (grado medio, exponente/entropía de grado, clustering, modularidad) específicamente para **Ruido blanco vs. Logístico vs. Lorenz**[cite: 1].
   - *Qué permite interpretar:* Valida empíricamente la advertencia de la literatura sobre si las estadísticas de grado actuales colapsan caos determinista y ruido[cite: 1].

6. **`06_feature_importance_top15.png` (Ranking de discriminabilidad):**
   - Barras horizontales con los 15 descriptores más importantes según *Permutation Importance*, coloreados por familia TS2G (Visibilidad / Transición / Proximidad).
   - *Qué permite interpretar:* Indica qué columnas sostienen la clasificación y qué familias están aportando poco valor o redundancia.

---

## 5. Criterios Automáticos de Decisión (Semáforo Go / No-Go)
El reporte debe incluir una tabla de veredicto automático con las siguientes reglas:
- **VERDE (Matriz $D$ discriminativa):** Macro-F1 ≥ 0.95 en la matriz completa, F1 ≥ 0.90 en todos los regímenes individuales y tasa de confusión < 5 % en los tres pares críticos (Logístico–Ruido, Lorenz–Ruido, Lorenz–Cuasi-periódico).
- **AMARILLO (Suficiencia frágil):** Macro-F1 ≥ 0.90 pero algún par crítico supera el 5 % de confusión o el modelo lineal cae significativamente frente al no lineal (indica dependencias no lineales o pares ambiguos).
- **ROJO (Degeneración detectada — obligatorio enriquecer $D$):** Confusión ≥ 10 % en algún par crítico o F1 < 0.85 en algún régimen. El reporte debe listar automáticamente qué par falló y recomendar añadir invariantes de orden superior (motivos HVG $m=4\text{--}5$, asimetría dirigida, plano Fisher–Shannon, entropía de transición, OPTN)[cite: 1].

---

## 6. Estructura de Salida Esperada
```text
project/
├── configs/
│   └── phase1_audit.yaml          # Semillas, N=5000, R=30, brownian_mode, rutas
├── src/
│   ├── audit_discriminability.py  # Script principal reproducible
│   └── plotting_audit.py          # Módulo de generación de las 6 figuras
└── reports/
    └── phase1_audit/
        ├── report_phase1.md       # Reporte Markdown con tablas, figuras embebidas y veredicto Go/No-Go
        ├── metrics_summary.json   # Todas las métricas numéricas para reproducibilidad
        └── figures/               # Las 6 visualizaciones (.png)