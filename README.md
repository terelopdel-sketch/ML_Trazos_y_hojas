<div align="center">

<img src="src/img/logo.png" width="380">

<h1 style="color:#4A6035;">Predicción de Inventario — Trazos y Hojas</h1>

**Proyecto de Machine Learning · Bootcamp de Data Science**
> Predicción de la demanda diaria por producto para una papelería en Madrid

</div>

---

<h2 style="color:#4A6035;">Índice</h2>

1. [Motivación](#motivación)
2. [Descripción del proyecto](#1-descripción-del-proyecto)
3. [Arquitectura del proyecto](#2-arquitectura-del-proyecto)
4. [Estructura de carpetas](#3-estructura-de-carpetas)
5. [Estado del repositorio y ramas](#4-estado-del-repositorio-y-ramas)
6. [Datos](#5-datos)
7. [Pipeline end-to-end](#6-pipeline-end-to-end)
8. [Modelado](#7-modelado)
9. [Interpretabilidad](#8-interpretabilidad)
10. [Instalación y ejecución](#9-instalación-y-ejecución)
11. [Dependencias](#10-dependencias)
12. [Resultados](#11-resultados)
13. [Equipo](#12-equipo)

---

<h2 style="color:#4A6035;">Motivación</h2>

Una papelería necesita anticipar cuántas unidades de cada producto va a vender cada día. Sin esa previsión, el negocio se mueve entre dos extremos costosos: la **rotura de stock** (venta perdida, cliente insatisfecho) y el **sobre-stock** (capital inmovilizado en productos que no rotan). Disponer de un modelo que estime la demanda diaria por producto permite planificar reposición y compras con criterio, en lugar de intuición.

---

<h2 style="color:#4A6035;">1. Descripción del proyecto</h2>

Este proyecto predice, para cada producto y cada día, **cuántas unidades se venderán** en la papelería *Trazos y Hojas*, a partir del histórico de ventas registrado en el TPV (punto de venta). Es un problema de **regresión supervisada sobre una serie temporal** de ventas por producto.

El dataset combina ventas en tienda física y a través de Glovo, e incluye información de producto, categoría, precio, método de pago y tipo de venta.

---

<h2 style="color:#4A6035;">2. Arquitectura del proyecto</h2>

El proyecto sigue un flujo secuencial de tres fases, cada una desarrollada por un miembro del equipo en su propia rama:

```
EDA  →  Preprocesado y selección de features  →  Modelado y persistencia
 │                    │                              │
Raquel               Ali                            Tere
```

- **EDA:** exploración del dataset bruto, construcción del target, selección de productos aptos para modelar y análisis descriptivo.
- **Preprocesado:** limpieza final, ingeniería de variables (lags, medias móviles) y 5 técnicas distintas de selección de features, combinadas por consenso (hard voting).
- **Modelado y persistencia:** comparación de algoritmos con validación cruzada temporal, optimización de hiperparámetros, evaluación final sobre test y guardado del modelo final en disco (`joblib`).

---

<h2 style="color:#4A6035;">3. Estructura de carpetas</h2>

Estructura consolidada del proyecto (una vez integradas las ramas de trabajo):

```
ML_Trazos_y_hojas/
│
├── src/
│   ├── data_sample/
│   │   ├── DatosBrutos (1).csv         # Export bruto del TPV: 11.613 filas × 20 columnas
│   │   └── target_final.csv            # Target agregado producto-día, ya filtrado a productos aptos
│   │
│   ├── img/
│   │   ├── logo.png                    # Logo de Trazos y Hojas
│   │   ├── demanda_diaria.png          # Gráficos generados en el EDA
│   │   ├── eda_distribucion_top10.png
│   │   ├── feature_importance.png      # Gráficos generados en el modelado
│   │   ├── residuos.png
│   │   └── shap_summary.png
│   │
│   ├── models/
│   │   ├── modelo_final_lgbm.joblib        # Modelo final (LightGBM)
│   │   └── features_modelo_final.joblib    # Lista de features usadas por el modelo final
│   │
│   ├── notebooks/
│   │   ├── EDA.ipynb                   # Exploración, target y selección de productos
│   │   ├── Pre_procesing_1.ipynb       # Feature engineering y selección de variables
│   │   └── Modeling.ipynb              # Comparativa de modelos, optimización y evaluación
│   │
│   └── utils/
│       ├── model_utils.py              # Evaluación cruzada, encoding, split temporal
│       └── metrics.py                  # Métricas de evaluación (RMSE, etc.)
│
├── main.ipynb
├── presentacion.pdf
├── requirements.txt
└── .gitignore
```

---

<h2 style="color:#4A6035;">4. Estado del repositorio y ramas</h2>

El proyecto se ha desarrollado en paralelo por los tres miembros del equipo, cada uno en su propia rama. `main` y `develop` sirven de esqueleto del repositorio; el trabajo sustantivo vive en las ramas individuales:

| Rama | Contenido | Autor/a |
| --- | --- | --- |
| `main` | Esqueleto del repositorio | — |
| `develop` | Estructura de carpetas base + notebook de preprocesado sin desarrollar | — |
| `Raquel/-eda` | `EDA.ipynb`: exploración inicial, construcción del target y selección de productos | Raquel ([@raquelmg1312](https://github.com/raquelmg1312)) |
| `Ali/-Pre-Processing` | `EDA.ipynb` + `Pre_procesing_1.ipynb`: feature engineering, lags y selección de variables | Ali ([@alimadriz0110](https://github.com/alimadriz0110)) |
| `Tere/modeling` | `Modeling.ipynb`: comparativa de modelos, optimización, evaluación y persistencia | Tere ([@terelopdel-sketch](https://github.com/terelopdel-sketch)) |

---

<h2 style="color:#4A6035;">5. Datos</h2>

**Fuente:** exportación del TPV de la papelería (ventas en tienda física + Glovo).

**Tamaño del dataset bruto:** 11.613 filas × 20 columnas, con el detalle línea a línea de cada venta.

**Periodo cubierto:** ~138 días, divididos de forma temporal (no aleatoria, por tratarse de una serie temporal):
- **Train:** 119 días (10.128 filas)
- **Test:** últimas 3 semanas — 19 días (1.484 filas), completamente aislado durante el desarrollo

<h3 style="color:#8F7C52;">Variables principales</h3>

| Grupo | Variables |
| --- | --- |
| Producto | `Producto`, `Categoría`, `ProductId` |
| Venta | `Cantidad`, `Fecha/Hora`, `Tipo de Venta`, `Método de Pago` |
| Precio | `Precio de Coste`, `Margen`, `Ventas NETAS`, `Ventas TOTALES`, `Impuesto` |
| Descuento | `Valor del Descuento`, `Motivo del Descuento` |

Columnas con más de un 90% de missing (`Nominal`, `A/C Ref`, `Código Fiscal`, `Notas`, `Identificación del Cliente`) se descartan por no aportar señal.

<h3 style="color:#8F7C52;">Reparto de la venta por canal y categoría (sobre train)</h3>

| Método de Pago | Unidades |
| --- | --- |
| Tarjeta | 24.925 |
| Efectivo | 11.044 |
| Credit | 1.025 |
| Glovo | 348 |
| Efectivo/Tarjeta | 85 |

| Tipo de Venta | Unidades |
| --- | --- |
| Eat in | 37.262 |
| Take Out | 165 |

La categoría **Copistería** concentra la inmensa mayoría del volumen (≈ 29.900 unidades), muy por delante del resto del catálogo (Papel, Bolígrafos, Oficina, etc.) — un patrón claramente tipo Pareto.

<h3 style="color:#8F7C52;">Construcción del target</h3>

El target (`Unidades_Vendidas`) se construye agregando `Cantidad` por `Producto` y `Fecha`, completando con 0 las combinaciones producto-día sin venta. Sobre el total de productos, esto produce un **68,8% de días con 0 ventas**, reflejo de la naturaleza intermitente de la demanda a nivel producto.

<h3 style="color:#8F7C52;">Selección de productos aptos para modelar</h3>

De los **1.012 productos** del catálogo, la mayoría se vende de forma demasiado esporádica para modelar de forma fiable. Se aplican 3 reglas de elegibilidad:

1. **Al menos 30 transacciones** en todo el periodo
2. **Venta en al menos 15 de las 25 semanas** del histórico
3. **Coeficiente de variación (CV) de estabilidad** dentro de un umbral razonable

Resultado: **31 productos aptos**, que forman el universo final sobre el que se entrena el modelo.

---

<h2 style="color:#4A6035;">6. Pipeline end-to-end</h2>

<h3 style="color:#8F7C52;">6.1 EDA (`EDA.ipynb`)</h3>

- Auditoría inicial: dimensiones, tipos de dato, missings
- Construcción del target (`Unidades_Vendidas`) a nivel producto-día
- Split temporal train/test por fecha de corte
- Análisis de la demanda diaria: distribución, día de la semana, mes, relación con el precio
- Análisis de la demanda por método de pago, tipo de venta y categoría
- Correlación entre variables numéricas (colinealidad esperada entre Ventas TOTALES/NETAS, Impuesto y Precio de Coste)
- Detección de días con demanda atípica (método IQR)
- Selección de los 31 productos aptos para modelar

<h3 style="color:#8F7C52;">6.2 Preprocesado (`Pre_procesing_1.ipynb`)</h3>

- Limpieza y filtrado a los 31 productos aptos
- **Feature engineering temporal:** `lag_1` (ventas del día anterior), `lag_7` (ventas hace una semana), medias y desviaciones móviles (`media_movil_7`, `media_movil_14`, `std_movil_7`)
- Nuevas variables categóricas derivadas
- Split temporal replicado igual que en el EDA

**Selección de features**, combinando 5 técnicas distintas:

| Técnica | Descripción |
| --- | --- |
| Selección estadística (`lista_1`) | Pearson/Spearman (numéricas) + ANOVA/Kruskal (categóricas) |
| Mutual Information (`lista_2`) | Captura cualquier tipo de dependencia, no solo lineal |
| SelectFromModel (`lista_3`) | Importancia de un `RandomForestRegressor`, umbral = media |
| RFE (`lista_4`) | Eliminación recursiva con `RandomForestRegressor` |
| SFS (`lista_5`) | Selección secuencial hacia adelante, `cv=3` |
| **Consenso — Hard Voting (`lista_6`)** | Combina las 5 anteriores por nº de votos |

Cada lista se codifica por separado (one-hot para categóricas, `StandardScaler` para numéricas, ajustado solo con train), para poder comparar su rendimiento real en el modelado.

<h3 style="color:#8F7C52;">6.3 Modelado (`Modeling.ipynb`)</h3>

- Comparativa inicial: `DummyRegressor` (baseline), `RandomForest`, `CatBoost`, `LightGBM` sobre las 6 listas de features
- **Corrección crítica de la validación cruzada:** el `TimeSeriesSplit` inicial cortaba por posición de fila (mezclando productos entre folds) en vez de por fecha real; se corrige implementando un split temporal por fechas
- Transformación logarítmica del target (`log1p` / `expm1`) para mitigar el efecto de los picos de alto volumen (productos de copistería) y el alto porcentaje de ceros
- Optimización de hiperparámetros: RandomSearch → GridSearch vs. Optuna (optimización bayesiana)
- Reevaluación de la selección de features en función de la importancia de variables tras optimizar
- Evaluación final sobre el conjunto de test, aislado desde el inicio del proyecto
- Interpretabilidad con importancia nativa del modelo y valores SHAP
- Persistencia del modelo final (`joblib`)

---

<h2 style="color:#4A6035;">7. Modelado</h2>

<h3 style="color:#8F7C52;">Elección de modelos</h3>

Se descarta la Regresión Lineal por no cumplir sus hipótesis (linealidad, homocedasticidad, ausencia de multicolinealidad) en un problema de demanda con relaciones no lineales. Se opta por modelos basados en árboles y *ensemble* (`RandomForest`, `CatBoost`, `LightGBM`), con `DummyRegressor` como baseline de referencia.

<h3 style="color:#8F7C52;">Comparativa de modelos (validación cruzada temporal, tras corregir el split)</h3>

El **DummyRegressor** obtiene un RMSE de 30,66 (R² ≈ 0), confirmando que todos los modelos entrenados aprenden un patrón real. **LightGBM con `lista_5`** resulta el mejor de la comparativa inicial (RMSE 24,01 · R² 0,37).

<h3 style="color:#8F7C52;">Optimización de hiperparámetros</h3>

| Método | RMSE (CV) |
| --- | --- |
| LightGBM sin optimizar | 24,43 |
| RandomSearch | 24,35 |
| RandomSearch → GridSearch | 24,19 |
| **Optuna (bayesiana)** | **24,16** |

<h3 style="color:#8F7C52;">Selección del conjunto de features definitivo</h3>

Tras revisar la importancia de variables, se detecta que `lista_5` no incluye ninguna variable temporal (lags/medias móviles). Se optimizan mediante Optuna las tres listas que sí las incluyen, evaluadas sobre **test**:

| Lista | RMSE (CV) | RMSE (test) | MAE (test) | R² (test) |
| --- | --- | --- | --- | --- |
| lista_1 | 24,52 | 24,77 | 4,12 | 0,322 |
| lista_6 | 24,69 | 24,57 | 4,00 | 0,333 |
| **lista_4** | 24,76 | **23,99** | 4,05 | **0,364** |

El orden se invierte por completo entre validación y test: **`lista_4` (RFE)**, la peor en CV, resulta ser la que mejor generaliza — se adopta como conjunto de features definitivo.

<h3 style="color:#8F7C52;">Modelo final: LightGBM + `lista_4`</h3>

```
LightGBM
  - Features: lista_4 (selección por RFE, incluye lags y medias móviles)
  - Target: log1p(Unidades_Vendidas), revertido con expm1 para las métricas
  - Hiperparámetros optimizados con Optuna
```

<h3 style="color:#8F7C52;">Métricas finales sobre test</h3>

| Métrica | Valor |
| --- | --- |
| **RMSE** | 23,99 |
| **MAE** | 4,05 |
| **R²** | 0,364 |

La cercanía entre el RMSE de validación cruzada (24,76) y el de test (23,99) indica que **el modelo no sobreajusta**. El MAE (≈ 4 unidades) es mucho más bajo que el RMSE porque este último penaliza con fuerza los errores en los productos de copistería de alto volumen (picos de hasta 698 unidades/día); el modelo predice con buena precisión el grueso del catálogo.

<h3 style="color:#8F7C52;">Importancia de variables del modelo final</h3>

| Variable | Importancia |
| --- | --- |
| `media_movil_14` | 768 |
| `media_movil_7` | 684 |
| `Dia_semana` | 582 |
| `lag_1` | 533 |
| `std_movil_7` | 362 |
| `lag_7` | 286 |

---

<h2 style="color:#4A6035;">8. Interpretabilidad</h2>

El análisis de importancia de variables y los valores **SHAP** coinciden en la misma lectura: la demanda de la papelería es fundamentalmente **inercial y semanal**.

- **El histórico reciente manda:** las medias móviles (7 y 14 días) y los lags (`lag_1`, `lag_7`) son, con diferencia, las variables más influyentes. Un producto que se ha vendido de forma sostenida tiende a seguir haciéndolo.
- **La volatilidad reciente importa:** `std_movil_7` ayuda al modelo a no sobre-reaccionar ante picos puntuales en productos de demanda errática.
- **Patrón semanal claro:** `Dia_semana` es la única variable de calendario con peso relevante; el resto de variables de calendario apenas aportan, algo esperable dado el corto periodo de observación (~6 meses).

**Lectura de negocio:** un sistema de previsión basado en el histórico reciente de cada producto permite anticipar las necesidades de inventario con un error medio de ≈ 4 unidades por predicción, ayudando a reducir tanto roturas de stock como sobre-stock.

---

<h2 style="color:#4A6035;">9. Instalación y ejecución</h2>

<h3 style="color:#8F7C52;">Requisitos previos</h3>

- Python ≥ 3.12
- `pip` y `venv`

<h3 style="color:#8F7C52;">Instalación</h3>

```bash
# Clonar el repositorio
git clone https://github.com/terelopdel-sketch/ML_Trazos_y_hojas.git
cd ML_Trazos_y_hojas

# Crear y activar el entorno virtual
python -m venv venv
source venv/Scripts/activate   # Windows (Git Bash)
# source venv/bin/activate     # Linux / macOS

# Instalar dependencias
pip install -r requirements.txt
```

<h3 style="color:#8F7C52;">Ejecución de los notebooks</h3>

Cada rama contiene su propio notebook de trabajo; se recomienda ejecutarlos en este orden una vez integrados en una misma rama:

```bash
jupyter notebook src/notebooks/EDA.ipynb
jupyter notebook src/notebooks/Pre_procesing_1.ipynb
jupyter notebook src/notebooks/Modeling.ipynb
```

> **Importante:** los notebooks deben ejecutarse de principio a fin y en orden (`Run All`), ya que celdas posteriores dependen de variables (`train`, `test`, `target_train`, etc.) definidas en celdas anteriores. Ejecutar celdas sueltas o tras un reinicio del kernel sin volver a correr todo desde el principio es la causa más habitual de errores `NameError`.

---

<h2 style="color:#4A6035;">10. Dependencias</h2>

Principales librerías utilizadas (ver `requirements.txt` para el listado completo):

| Paquete | Uso |
| --- | --- |
| `pandas`, `numpy` | Manipulación de datos |
| `matplotlib`, `seaborn` | Visualización en el EDA |
| `scikit-learn` | Preprocesado, selección de features, métricas |
| `lightgbm`, `catboost`, `xgboost` | Modelos de *gradient boosting* |
| `optuna` | Optimización bayesiana de hiperparámetros |
| `shap` | Interpretabilidad del modelo final |
| `joblib` | Persistencia del modelo |

---

<h2 style="color:#4A6035;">11. Resultados</h2>

El modelo final —**LightGBM entrenado sobre la lista de features `lista_4` (RFE), con target transformado con `log1p` e hiperparámetros optimizados vía Optuna**— logra un **RMSE de 23,99**, un **MAE de 4,05 unidades** y un **R² de 0,364** sobre un conjunto de test completamente aislado durante el desarrollo.

El proyecto demuestra que, incluso con un histórico corto (~6 meses) y una demanda muy intermitente (68,8% de días sin venta a nivel producto), es posible construir un sistema de previsión útil apoyado sobre todo en el comportamiento reciente de ventas y el patrón semanal — información suficiente para apoyar decisiones reales de reposición e inventario en la papelería.

---

<h2 style="color:#4A6035;">12. Equipo</h2>

Proyecto desarrollado en equipo durante el bootcamp de Data Science:

- **Raquel** ([@raquelmg1312](https://github.com/raquelmg1312)) — EDA
- **Ali** ([@alimadriz0110](https://github.com/alimadriz0110)) — Preprocesado y selección de features
- **Tere** ([@terelopdel-sketch](https://github.com/terelopdel-sketch)) — Modelado, optimización y persistencia

---

*Proyecto de Machine Learning · Bootcamp de Data Science · Trazos y Hojas*
