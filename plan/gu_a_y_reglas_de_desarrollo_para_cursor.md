# Guía de Desarrollo y Reglas del Proyecto para Cursor

Este documento establece los principios de arquitectura, estilo de código, generación de reportes y flujo de trabajo que **Cursor** debe seguir en todo el desarrollo del proyecto de doctorado. 

> **Nota para Cursor:** Todos los documentos de especificación, arquitectura y planificación de tareas se encuentran en la carpeta `Plan/`. Antes de implementar cualquier nueva funcionalidad, revisa siempre este documento y la especificación concreta de la tarea en `Plan/`.

---

## 1. Filosofía y Arquitectura en Python (POO)

El proyecto debe desarrollarse en **Python 3.10+** siguiendo un enfoque estricto de **Programación Orientada a Objetos (POO)**, priorizando la modularidad, la legibilidad y la extensibilidad a futuro.

* **Clases Base Abstractas (`ABC`):** Define interfaces claras usando `abc.ABC` y `@abstractmethod` para los componentes principales (por ejemplo, `BaseSeriesGenerator`, `BaseTransformer`, `BaseAnalyzer`, `BaseReportBuilder`). Esto permitirá añadir nuevos sistemas dinámicos, mapeos o métricas sin romper el código existente.
* **Configuración tipada:** Usa `dataclasses` (o `pydantic.BaseModel`) para encapsular hiperparámetros, configuraciones de experimentos y metadatos de las series. Evita pasar diccionarios sueltos sin validar o demasiados argumentos posicionales.
* **Tipado estático estricto:** Todas las funciones, métodos y atributos deben incluir *type hints* completos (`numpy.typing.NDArray`, `Path`, `Optional`, `dict[str, Any]`, etc.).
* **Principio de Responsabilidad Única (SRP):** Separa claramente:
  1. **Generación / Carga de datos** (lógica matemática y numérica).
  2. **Análisis / Transformación** (cálculo de métricas, grafos, descriptores).
  3. **Visualización y Reportes** (construcción de gráficos y exportación a HTML).
  4. **Orquestación** (scripts principales o *pipelines* que conectan las piezas).
* **Reproducibilidad garantizada:** Nunca uses estado aleatorio global sin control. Utiliza generadores explícitos (`np.random.default_rng(seed)`) inyectados a través de la configuración para asegurar que cualquier experimento sea 100 % reproducible.
* **Documentación y Logging:**
  * Incluye *docstrings* en formato **NumPy** o **Google** en todas las clases y métodos públicos.
  * Usa el módulo estándar `logging` en lugar de `print()` para trazar la ejecución, advertencias y errores.

---

## 2. Sistema de Reportes y Resúmenes en HTML

Cada ejecución de análisis, validación de generadores o experimento debe producir un **reporte en formato HTML autocontenido** para facilitar la inspección visual rápida en el navegador sin necesidad de abrir notebooks ni ejecutar scripts adicionales.

### Requisitos del módulo de reportes (`HTMLReportBuilder`):
* **Autocontenido (Portable):** El archivo `.html` debe poder abrirse directamente en cualquier navegador sin depender de rutas rotas.
  * Para gráficos interactivos: usar **Plotly** (con `include_plotlyjs='cdn'` o embebido según preferencia) para poder hacer zoom sobre las series temporales y revisar detalles.
  * Para gráficos estáticos (Matplotlib/Seaborn): codificar las imágenes en **Base64** directamente dentro de etiquetas `<img>`.
* **Estructura estándar del reporte:**
  1. **Cabecera y Metadatos:** Título del experimento, fecha/hora de ejecución, semilla utilizada y resumen de parámetros de configuración.
  2. **Resumen Ejecutivo (KPIs / Tarjetas):** Resumen rápido con métricas clave (ej. número de series, longitud, comprobación de ausencia de `NaN`/`Inf`, estadísticas básicas como media, desviación estándar, mínimo y máximo).
  3. **Tablas Comparativas:** Tablas HTML limpias y legibles (por ejemplo, exportadas desde `pandas.DataFrame.to_html()` con estilos CSS modernos) que resuman los resultados por `Dataset ID` o régimen.
  4. **Visualizaciones:** Gráficos ordenados por sección o régimen (series temporales, histogramas, espectros, atractores, etc.).
* **Plantillas y Diseño:** Utiliza **Jinja2** (o un generador modular con CSS limpio embebido) para mantener el diseño visual separado de la lógica de análisis. El diseño debe ser limpio, responsivo y con buena tipografía para lectura científica.
* **Gestión de salidas:** Guarda los reportes en una estructura ordenada, por ejemplo: `outputs/reports/<nombre_modulo_o_experimento>/<timestamp>_summary.html`, y mantén opcionalmente un enlace o copia `latest.html` para consulta rápida.

---

## 3. Estructura Recomendada del Proyecto

Cursor debe respetar y mantener la siguiente organización de directorios (adaptándola si es necesario al código base importado):

```text
proyecto_doctorado/
├── Plan/                       # Documentación, especificaciones y prompts (NO borrar)
│   ├── cursor_guidelines.md    # Este documento de reglas generales
│   └── synthetic_generator_spec.md
├── configs/                    # Archivos YAML/JSON de configuración de experimentos
├── src/                        # Código fuente principal (paquete Python)
│   ├── generators/             # Clases generadoras de series sintéticas
│   ├── analysis/               # Módulos de transformación y cálculo de descriptores
│   ├── reporting/              # Generador de reportes HTML y plantillas Jinja2
│   └── utils/                  # Logging, I/O, utilidades matemáticas y semillas
├── scripts/                    # Puntos de entrada CLI para lanzar pipelines
├── outputs/                    # Artefactos generados (añadir a .gitignore si pesan mucho)
│   ├── data/                   # Series guardadas (CSV, Parquet, NPZ)
│   └── reports/                # Reportes HTML autocontenidos
├── tests/                      # Tests unitarios con pytest
├── pyproject.toml              # Dependencias y configuración del proyecto
└── README.md                   # Documentación general y comandos de ejecución
```

---

## 4. Tips de Flujo de Trabajo para Trabajar con Cursor

Para sacar el máximo partido a Cursor (Composer / Agent) durante el desarrollo:

1. **Referencia siempre la carpeta `Plan/`:** Al iniciar una tarea en Cursor, menciona explícitamente los archivos de contexto usando `@`:
   > *"Lee `@Plan/cursor_guidelines.md` y `@Plan/synthetic_generator_spec.md`. Antes de escribir código, propón un plan de implementación y la estructura de clases."*
2. **Aplica el flujo *Plan → Implementar → Validar*:** Pide primero la arquitectura de clases y nombres de archivos. Solo cuando valides el diseño, deja que Cursor genere o refactorice el código.
3. **Reutilización limpia del proyecto anterior:** Al traer código de otro proyecto, indica a Cursor que **no haga copiar-pegar literal**, sino que lo refactorice para ajustarlo a las clases base, *type hints* y estructura de `src/` de este nuevo repositorio.
4. **Tests unitarios obligatorios:** Al terminar cada módulo, pide a Cursor que cree tests en `tests/` usando `pytest` para comprobar:
   * Longitud exacta y tipo de datos de las series.
   * Ausencia de valores `NaN` o infinitos.
   * Reproducibilidad (misma semilla = misma serie; distinta semilla = distinta serie).
   * Generación correcta del archivo HTML de resumen sin errores.
5. **Control de alcance:** Evita que Cursor implemente futuras fases antes de tiempo, pero asegúrate de que deje las interfaces preparadas (por ejemplo, aceptar una lista de configuraciones o un parámetro `n_samples` aunque ahora se ejecute con `n=1`).

---

## 5. Bloque Recomendado para `.cursorrules` (Opcional)

Si quieres que Cursor recuerde estas normas automáticamente en cada chat sin tener que repetirlas, puedes crear un archivo `.cursorrules` en la raíz del proyecto con el siguiente contenido:

```text
Eres un ingeniero de software científico experto en Python.
Reglas obligatorias del proyecto:
1. Consulta siempre la carpeta `Plan/` (especialmente `Plan/cursor_guidelines.md`) antes de diseñar o modificar código.
2. Usa Python 3.10+, Programación Orientada a Objetos (clases base ABC, dataclasses para configuración, type hints estrictos y docstrings estilo NumPy/Google).
3. Garantiza reproducibilidad usando `np.random.default_rng(seed)` y evita el uso de `print()` (usa `logging`).
4. Todos los análisis y pipelines deben generar un reporte HTML autocontenido en `outputs/reports/` con resumen de parámetros, métricas clave, tablas y gráficos embebidos (Plotly o Matplotlib en Base64).
5. Escribe tests unitarios con `pytest` para cada nuevo módulo en `tests/`.
6. Antes de modificar múltiples archivos, presenta brevemente el plan de cambios.
```