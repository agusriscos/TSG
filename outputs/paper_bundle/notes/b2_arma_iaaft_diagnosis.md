# Diagnóstico: `arma_correlated` original vs IAAFT

No se ha modificado el algoritmo IAAFT ni se ha relanzado el pipeline de 9 métodos. Las importancias salen de `records.pkl`. El error espectral se mide regenerando solo las series ARMA con las semillas de `phase2_surrogates` (perfil `paper`, R=30, N=1000, seed=42, `iaaft_max_iter=250`, `min_iter=10`, `tol=1e-06`).

- Tiempo del diagnóstico: 107.5 s.

## AUROC reproducido (original vs IAAFT, matrices guardadas)

| Vista | RF | Logística L2 |
|---|---:|---:|
| transition | 0.783 | 0.794 |
| visibility | 0.717 | 0.694 |

Referencia en `auroc_binary.csv`: transition RF = 0.783, visibility RF = 0.717.

## 1. Features que separan original de IAAFT

Clasificador ajustado sobre las 60 filas (30 original + 30 IAAFT) después del mismo filtro de varianza que la auditoría. Coeficiente L2 en el espacio estandarizado (positivo ⇒ más alto en el original). Gini del random forest. `d` de Cohen = (media original − media IAAFT) / desviación agrupada.

### transition

| rank | feature | Gini | coef L2 | d | media orig | media IAAFT |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `transition__OPN__cv_k` | 0.105 | 0.371 | 1.15 | 0.4043 | 0.3274 |
| 2 | `transition__OPN__self_loop_ratio` | 0.100 | 0.934 | 0.90 | 0.7735 | 0.6999 |
| 3 | `transition__OPN__skew_k` | 0.095 | -0.384 | 0.69 | 0.4268 | 0.112 |
| 4 | `transition__OPN__c_avg` | 0.069 | 0.963 | 0.31 | 0.1925 | 0.1675 |
| 5 | `transition__OPN__assortativity_r` | 0.068 | -0.781 | -0.97 | -0.3038 | -0.1735 |
| 6 | `transition__OPN__i_dir` | 0.046 | -0.365 | -0.74 | 0.06899 | 0.09103 |
| 7 | `transition__STN__i_dir` | 0.043 | -0.661 | -0.21 | 0.01296 | 0.01571 |
| 8 | `transition__OPN__l_lcc` | 0.043 | 0.560 | 0.45 | 2.157 | 2.106 |

### visibility

| rank | feature | Gini | coef L2 | d | media orig | media IAAFT |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `visibility__HVG__h_pk` | 0.075 | 0.385 | -0.77 | 1.426 | 1.497 |
| 2 | `visibility__HVG__skew_k` | 0.071 | 0.213 | 0.50 | 1.373 | 1.182 |
| 3 | `visibility__dHVG__h_pk` | 0.062 | 0.385 | -0.77 | 1.426 | 1.497 |
| 4 | `visibility__dHVG__skew_k` | 0.059 | 0.213 | 0.50 | 1.373 | 1.182 |
| 5 | `visibility__NVG__cv_k` | 0.055 | -1.837 | -0.95 | 0.4034 | 0.47 |
| 6 | `visibility__NVG__lambda_max` | 0.038 | -0.118 | -0.25 | 95.11 | 102.1 |
| 7 | `visibility__HVG__assortativity_r` | 0.038 | 0.178 | 0.41 | 0.05341 | 0.03974 |
| 8 | `visibility__NVG__assortativity_r` | 0.035 | 0.804 | 0.44 | 0.1714 | 0.1432 |

## 2. ¿Converge IAAFT en `arma_correlated`?

- Realizaciones: 30. Todas coinciden con `iaaft_surrogate`.
- Motivo de parada: `{'plateau': 30}`.
- Iteración de parada: mediana 45 (mín 25, máx 123). Ninguna llega a `max_iter=250` si el motivo es `plateau`.
- Error espectral relativo al parar: mediana 6.580e-03, media 6.487e-03, máx 1.373e-02. Tolerancia pedida: 1e-06. Fracción con error < 1e-3: 0.00.
- El mismo error al forzar las 250 iteraciones: mediana 6.580e-03. Mejora mediana respecto a la parada: 1.031e-06.
- Error relativo en el 20% de bins graves (baja frecuencia): mediana 3.414e-03. En el 20% agudo: mediana 2.187e-01.
- Fracción de la energía del residual en baja frecuencia: mediana 0.331; en alta frecuencia: mediana 0.108.
- `rho` del AR(2): mediana 0.978 (rango 0.964–0.990).
- Salto de borde `|x0-xN|/mediana|Δx|`: original 10.64, IAAFT 0.96 (razón IAAFT/original mediana 0.085).
- Rugosidad MSSD: razón IAAFT/original mediana 1.092 (media 1.187).

Correlación de Spearman entre el error espectral de cada realización y (feature original − feature IAAFT) de los tres primeros Gini:

| vista | feature | rho | p |
|---|---|---:|---:|
| transition | `transition__OPN__cv_k` | -0.39 | 3.36e-02 |
| transition | `transition__OPN__self_loop_ratio` | -0.54 | 2.14e-03 |
| transition | `transition__OPN__skew_k` | -0.32 | 8.14e-02 |
| visibility | `visibility__HVG__h_pk` | 0.71 | 1.36e-05 |
| visibility | `visibility__HVG__skew_k` | -0.50 | 4.65e-03 |
| visibility | `visibility__dHVG__h_pk` | 0.71 | 1.36e-05 |

## Causa raíz

El IAAFT de producción no alcanza `tol=1e-06`. En el 100% de las realizaciones se detiene por la cláusula de meseta (`|Δ error| < tol`) hacia la iteración 45, con un error espectral relativo mediano de 6.580e-03 (ninguna serie baja de 1e-3). Forzar las 250 iteraciones solo mejora ese error en 1.031e-06 (mediana pareada): el residual es un punto fijo del algoritmo, no un presupuesto de iteraciones corto.

El pico AR está bien imitado en términos relativos: el 20% grave del espectro (`rho` mediano 0.978) queda a 3.414e-03. Ese pico domina la norma L2, así que aun así concentra una fracción 0.33 de la energía absoluta del residual. El 20% agudo tiene un error relativo mediano de 2.187e-01 (denominador pequeño: la serie original casi no tiene potencia ahí) y una fracción 0.11 de la energía del residual. El hecho geométrico es el borde periódico. `|x0 − xN|` vale 10.6 veces el paso interior, pero el grafo no es circular y no ve ese salto. El IAAFT, obligado a realizar el periodograma de esa serie con empalme, reduce el salto a 0.96 (razón mediana 0.085) y reparte la energía como rugosidad interior (MSSD IAAFT/original = 1.092).

Un ARMA gaussiano lineal debería ser indistinguible de un IAAFT convergido (AUROC ≈ 0.50). El AUROC de 0.783 en `transition` y 0.717 en `visibility` mide esa rugosidad, no no-linealidad. En transición las features que más separan son `transition__OPN__cv_k`, `transition__OPN__self_loop_ratio`, `transition__OPN__skew_k`: el original tiene más auto-bucles ordinales y más heterogeneidad de grado que su IAAFT. En visibilidad, `visibility__HVG__h_pk`, `visibility__HVG__skew_k`, `visibility__dHVG__h_pk`: `HVG` y `dHVG` salen duplicados porque `i_dir` está saturado en 1 y el resto de descriptores se calcula sobre el grafo simetrizado, así que la vista de visibilidad no está usando la flecha direccional. No se modifica el algoritmo IAAFT en este cierre. Si se quiere acercar el nulo lineal a 0.50, el siguiente experimento es tratar el borde (ventana o series con extremos emparejados) y volver a medir solo este régimen.

## 3. Control de cierre: end-matching (solo `arma_correlated`)

Verificación acotada: se regeneran las series ARMA con la misma semilla (`seed=42`, R=30, N=1000), se aplica *end-matching* (rampa lineal que fuerza `y[0]=y[-1]`) **antes** de IAAFT, y se extraen únicamente las vistas `transition` / `visibility` (`methods=['NVG', 'HVG', 'dHVG', 'STN', 'OPN']`). No se relanza el panel de 9 métodos ni se modifica `iaaft_surrogate`.

- Tiempo del control: 86.9 s.
- Salto de borde mediano `|x0-xN|/med|Δx|`: raw 10.64 → end-matched 0 → IAAFT(end-matched) 0.91.

### AUROC binario original(end-matched) vs IAAFT(end-matched)

| Vista | RF (baseline) | RF (end-match) | LR (baseline) | LR (end-match) |
|---|---:|---:|---:|---:|
| transition | 0.783 | 0.742 | 0.794 | 0.750 |
| visibility | 0.717 | 0.644 | 0.694 | 0.800 |

Interpretación: un nulo lineal bien emparejado en el borde periódico debería situar el AUROC cerca de 0.50. Tras end-matching, transition RF = 0.742 y visibility RF = 0.644 (sigue por encima de 0.60: el borde no basta por sí solo para explicar toda la separación).
