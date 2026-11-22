# Especificación Técnica: Generador de Series Temporales Sintéticas Multirrégimen

## 1. Contexto y Objetivo del Proyecto
El objetivo de esta tarea es implementar un módulo generador de series temporales sintéticas que abarque distintos regímenes dinámicos (**Estocástico**, **Periódico** y **Caótico**), basándose en el registro definido en la **Tabla 3.1 (Multiregime dataset registry)**.

### Estrategia de Implementación y Escalabilidad
* **Reutilización de código base:** Se dispone de una implementación previa en otro proyecto (ver sección de instrucciones para Cursor). Se debe extraer, adaptar y refactorizar toda la lógica relacionada con la generación de estas dinámicas.
* **Fase 1 (Alcance actual):** Generar **1 serie temporal determinista/controlada por cada tipo** (`Dataset ID`) utilizando los parámetros base definidos en la tabla de especificaciones.
* **Fase 2 (Diseño preparado para futuro):** La arquitectura del código (clases, funciones, configuración mediante diccionarios/YAML o `dataclasses`) debe estar desacoplada y parametrizada desde el inicio para permitir generar **$N$ series aleatorias por régimen** en el futuro, variando hiperparámetros (frecuencias, condiciones iniciales, pasos de integración, niveles de ruido, semillas, etc.).

---

## 2. Registro de Series Sintéticas (Tabla 3.1)

A continuación se detallan las especificaciones matemáticas y dinámicas exactas para cada uno de los 6 datasets a implementar:

| Dataset ID | Régimen (`Regime`) | Sistema / Ecuación (`Source / System`) | Variable | Notas Técnicas y Restricciones |
| :--- | :--- | :--- | :--- | :--- |
| **`white_noise`** | Stochastic | $\mathcal{N}(0, 1)$ | `Value` | Control gaussiano no correlacionado y sin memoria. |
| **`brownian`** | Stochastic | Caminata aleatoria sintética (*Random walk*) | `Value` | Acumulación no estacionaria de pasos independientes (ej. suma acumulada de $\mathcal{N}(0, 1)$). |
| **`sine`** | Periodic | $y(t) = \sin(2\pi f t)$, con $f = 0.02$ | `Value` | Órbita cerrada de frecuencia única. **Importante:** Incluir *jitter* gaussiano leve o parametrización inconmensurable para evitar clones de estado exactos en reconstrucciones de espacio de fases. |
| **`quasiperiodic`** | Periodic | $y(t) = \sin(2\pi f_1 t) + \sin(2\pi f_2 t)$<br>con $f_1 = 0.02$ y $f_2 = f_1 / \phi$ | `Value` | Superposición aditiva de frecuencias inconmensurables (donde $\phi = \frac{1 + \sqrt{5}}{2}$ es la proporción áurea) para evitar *aliasing* discreto. |
| **`lorenz`** | Chaotic | Sistema de Lorenz:<br>$\dot{x} = \sigma(y - x)$<br>$\dot{y} = x(\rho - z) - y$<br>$\dot{z} = xy - \beta z$<br>con $\sigma=10, \rho=28, \beta=8/3$ | `x-value` | Atractor caótico continuo. Debe integrarse mediante **Runge-Kutta de 4º orden (RK4)** con paso temporal $\Delta t = 0.02$. Descartar transitorio inicial (*burn-in*) si aplica. |
| **`logistic`** | Chaotic | $x_{t+1} = 4 x_t (1 - x_t)$, con $x_0 = 0.1$ | `Value` | Mapa discreto en régimen de caos total ($r=4$). **Control obligatorio:** Validar que las condiciones iniciales eviten los puntos fijos/triviales $x_0 \notin \{0, 0.5, 1\}$. |

---

## 3. Requisitos de Arquitectura del Código

Para facilitar el mantenimiento y la futura expansión a $N$ series con hiperparámetros aleatorios, el módulo debe seguir esta estructura:

1. **Configuración Centralizada (`config` / `Params`):**
   * Definir un esquema de configuración (ej. `pydantic`, `dataclasses` o `YAML`) donde cada `Dataset ID` tenga sus parámetros por defecto (los de la Tabla 3.1) y rangos de muestreo definidos para el futuro.
   * Control estricto de semillas (`seed` / `np.random.Generator`) para garantizar reproducibilidad total en la Fase 1 y muestreo controlado en la Fase 2.
2. **Interfaz Uniforme de Generación:**
   * Todos los generadores deben devolver un formato estandarizado (por ejemplo, un `DataFrame` de Pandas, un diccionario estructurado o un objeto `TimeSeries` con metadatos: `dataset_id`, `regime`, `params`, `length`, `values`).
   * Parámetros comunes a nivel global: `n_steps` (longitud de la serie), `burn_in` (pasos iniciales a descartar, especialmente relevante en sistemas dinámicos como `lorenz` o `logistic`), y `seed`.
3. **Validaciones Integradas:**
   * En `logistic`: Aserción o ajuste automático si $x_0 \in \{0, 0.5, 1\}$ o $x_0 \notin (0, 1)$.
   * En `sine`: Opción configurable para activar/desactivar el *jitter* gaussiano (con $\sigma_{\text{jitter}}$ pequeño, ej. $10^{-3}$) o el ajuste inconmensurable.

---

## 4. Prompt / Instrucciones para Cursor

> **Instrucción para pegar en el chat de Cursor (junto con referencia `@` a la carpeta/archivos del proyecto antiguo):**
>
> "Actúa como un ingeniero de software científico experto en sistemas dinámicos y Python. Necesito implementar el módulo generador de series sintéticas multirrégimen descrito en este documento (`synthetic_generator_spec.md`).
>
> **Paso 1 (Auditoría y Migración):** Revisa el código existente en `@[RUTA_DEL_PROYECTO_ANTERIOR]` y extrae toda la lógica relacionada con la generación de ruido blanco, movimiento browniano, ondas senoidales/cuasiperiódicas, el atractor de Lorenz (integrador RK4) y el mapa logístico.
>
> **Paso 2 (Refactorización y Diseño):** Adapta ese código a la arquitectura de este nuevo proyecto asegurando que:
> 1. Se cumplan estrictamente las ecuaciones, parámetros y notas técnicas de la **Tabla 3.1** (incluyendo el control de puntos fijos en el mapa logístico, el jitter/inconmensurabilidad en el seno y la integración RK4 con $\Delta t = 0.02$ en Lorenz).
> 2. El diseño esté parametrizado (usando `dataclasses` o archivos de configuración) para que ahora mismo ejecutemos un script que genere **1 serie por cada `Dataset ID`**, pero que permita escalar trivialmente a generar **$N$ series aleatorias** variando rangos de hiperparámetros en el futuro.
> 3. Se incluyan metadatos claros con cada serie generada (régimen, parámetros exactos usados, semilla) y se guarden en el formato estándar del proyecto.
>
> Antes de escribir todo el código, muéstrame brevemente qué archivos vas a reutilizar del proyecto anterior y propón la estructura de archivos que crearás aquí."
```

### Sugerencias rápidas antes de lanzarlo en Cursor:
* **Completa la ruta:** En la sección 4, sustituye `@[RUTA_DEL_PROYECTO_ANTERIOR]` utilizando el autocompletado `@` de Cursor para apuntar directamente a los archivos o carpetas donde ya tienes implementados Lorenz, RK4 o las señales periódicas.
* **Definición de longitud (`n_steps`):** Si en tu proyecto base ya tienes una longitud estándar para las series (por ejemplo, $T = 1000$ o $T = 5000$ puntos tras el *burn-in*), indícaselo en el chat para que los 6 datasets salgan alineados en tamaño desde la primera ejecución.