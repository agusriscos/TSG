# Objetivo: Bloque 3 — Fronteras dinámicas difíciles + perfiles dev/paper

La Fase 0 está cerrada y en verde. Ahora vamos a construir el **Bloque 3** de la ampliación Q1: desaturar el benchmark añadiendo regímenes dinámicos de frontera (estocásticos correlacionados, caos continuo/discreto cercano y cambio de régimen) sin romper los tests ni la configuración de Fase 1.

Sigue estrictamente las reglas de arquitectura, estilo y testing de `plan/OLD/gu_a_y_reglas_de_desarrollo_para_cursor.md`. El código fuente reside en `src/tsg/`.

---

## Paso 1: Nuevos generadores en `src/tsg/generators/`

Amplía `src/tsg/generators/stochastic.py`, `src/tsg/generators/chaotic.py` (y crea `src/tsg/generators/nonstationary.py` si encaja mejor con la arquitectura actual) respetando la clase base, la generación determinista por semilla (`realization_seed`) y la diversidad intra-clase por realización (condiciones iniciales + perturbaciones paramétricas físicamente válidas):

1. **Estocásticos correlacionados (`stochastic.py`):**
   - **fBm (`fbm_antipersistent`, `fbm_persistent`)**: Movimiento Browniano Fraccional con exponente de Hurst $H \approx 0.2$ (anti-persistente, p. ej. $H \sim U(0.15, 0.25)$) y $H \approx 0.8$ (persistente, p. ej. $H \sim U(0.75, 0.85)$), implementable con Davies-Harte / FFT o Cholesky (usa método exacto o Davies-Harte en $O(N \log N)$ usando solo NumPy/SciPy para no añadir dependencias pesadas).
   - **ARMA lineal calibrado (`arma_correlated`)**: Proceso AR(2)/ARMA(2,1) estacionario con fuerte autocorrelación positiva (p. ej. raíces cercanas a la unidad que emulen el tiempo de autocorrelación de un flujo caótico) para poner a prueba si las familias distinguen autocorrelación lineal de caos determinista.

2. **Caos y fronteras paramétricas (`chaotic.py`):**
   - **Rössler (`rossler`)**: Flujo caótico continuo ($a \sim U(0.18, 0.22)$, $b \sim U(0.18, 0.22)$, $c \sim U(5.5, 5.9)$), con descarte de transitorio y condiciones iniciales perturbadas.
   - **Hénon (`henon`)**: Mapa caótico 2D ($a \sim U(1.38, 1.41)$, $b = 0.3$), con descarte de transitorio y `x0, y0` perturbados dentro de la cuenca de atracción (asegurando que no diverja).
   - **Logístico intermitente / caos débil (`logistic_weak`)**: Mapa logístico en la frontera del caos desarrollado ($r \sim U(3.58, 3.65)$, evitando ventanas periódicas triviales) para contrastar frente al `logistic` desarrollado ($r = 4.0$).

3. **No estacionariedad (`nonstationary.py` o `periodic.py`):**
   - **Cambio de régimen (`regime_switching`)**: Serie que transita de dinámica cuasiperiódica a caótica (p. ej. Lorenz o Rössler normalizados en media/varianza) en un punto de cambio aleatorio dentro del tramo central `[0.35*N, 0.65*N]`, con transición suave o abrupta bien documentada.

Registra todos los nuevos generadores en el registro/factory de `src/tsg/generators/` para que sean seleccionables desde YAML.

---

## Paso 2: Configuración del benchmark difícil y perfiles `dev` / `paper`

1. Crea `configs/phase2_hard_frontiers.yaml` con un panel de fronteras difíciles (manteniendo `configs/phase1_audit.yaml` intacto):
   - Regímenes: `white_noise`, `fbm_antipersistent`, `brownian`, `fbm_persistent`, `arma_correlated`, `quasiperiodic`, `lorenz`, `rossler`, `henon`, `logistic_weak`, `logistic`, `regime_switching` (12 clases).
   - `brownian_mode: levels` (aplicado de forma consistente a trayectorias integradas).
   - Define soporte de perfiles en el YAML y en el CLI de `scripts/run_phase1_audit.py` (o un nuevo script `scripts/run_audit.py` retrocompatible):
     - `--profile dev`: `n_realizations: 10`, `n_steps: 1000`
     - `--profile paper`: `n_realizations: 30`, `n_steps: 2500`
   - Define `critical_pairs` específicos de frontera:
     - `["arma_correlated", "lorenz"]`
     - `["lorenz", "rossler"]`
     - `["fbm_persistent", "brownian"]`
     - `["logistic_weak", "logistic"]`
     - `["regime_switching", "quasiperiodic"]`
2. Asegura que el `ensemble_cache` incluya en su hash el listado de regímenes y el perfil (`n_realizations`, `n_steps`), para no mezclar nunca cachés de Fase 1 y Fase 2.

---

## Paso 3: Micro-ajuste de reporte (pendiente menor de Fase 0)
- En `src/tsg/reporting/audit_report.py`, incluye explícitamente en la sección *Salud de descriptores* la línea de `low_variance_features` (no estructurales, como `transition__OPN__d_lcc`) para que la suma de `numerical_nan + structural_constant + low_variance_non_structural` coincida exactamente con el total de `dropped` en el Markdown.

---

## Paso 4: Tests y corrida de validación rápida (`--profile dev`)
1. Añade tests en `tests/test_generators.py` verificando que los 6 nuevos generadores producen series finitas (sin `NaN`/`Inf`), reproducibles con la misma semilla, distintas entre realizaciones y con varianza no nula.
2. Verifica que `pytest` pase al 100% (incluyendo todos los tests previos de Fase 0).
3. Ejecuta una corrida de validación en modo `--profile dev` sobre `configs/phase2_hard_frontiers.yaml` y muéstrame:
   - Macro-F1 (RF y LR) para las 5 vistas (`visibility`, `proximity`, `transition`, `similarity`, `full`).
   - Silhouette global y los 3 pares críticos con mayor tasa de confusión.
   - Tiempo aproximado de ejecución en modo `dev`.