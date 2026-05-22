# Contexto del Proyecto para Codex

## Nombre del proyecto

**El Factor Local: Análisis del Impacto de la Sede y las Condiciones del Torneo en los Resultados de la Copa Mundial de Fútbol**

Proyecto final de la asignatura **Machine Learning**.

---

## Idea central

El proyecto busca analizar si las condiciones contextuales de una edición de la Copa Mundial de Fútbol influyen en los resultados deportivos de las selecciones participantes.

La pregunta principal es determinar si variables como el continente sede, la edición del torneo, los equipos participantes, el rendimiento histórico **pre-torneo** y las condiciones del torneo permiten predecir si una selección avanza más allá de la fase de grupos.

Además, se busca agrupar ediciones de mundiales para identificar perfiles o patrones de torneos usando aprendizaje no supervisado.

> **Nota importante:** Para evitar data leakage, el modelo supervisado utilizará **únicamente variables disponibles antes del inicio del torneo** como predictores. Los goles y resultados dentro del torneo se usan exclusivamente para construir la variable objetivo, no como features.

---

## Objetivo general

Analizar el impacto de las condiciones contextuales de la Copa Mundial de Fútbol en los resultados de los equipos participantes, aplicando algoritmos supervisados y no supervisados de Machine Learning sobre datos históricos del torneo.

---

## Objetivos específicos

1. Recopilar, limpiar y documentar datos históricos de la Copa Mundial desde 1930 hasta 2022, combinando múltiples fuentes para obtener información de ronda por partido.
2. Construir un modelo **baseline** (dummy classifier) y un modelo supervisado de **regresión logística** para predecir si una selección avanza más allá de la fase de grupos.
3. Aplicar **K-Means** y **Clustering Jerárquico** para agrupar ediciones mundialistas según perfiles estadísticos.
4. Evaluar los modelos con métricas estándar, reconociendo las limitaciones estadísticas del dataset.
5. Reflexionar sobre una posible implementación real en análisis deportivo.

---

## Dataset

### Fuente principal

**International Football Results from 1872 to 2024** (Kaggle)

El dataset contiene más de 47.000 registros de partidos internacionales. Para este proyecto se filtrarán únicamente los partidos correspondientes a la **Copa Mundial de la FIFA**, aproximadamente 900 registros.

> ⚠️ **Limitación conocida:** Este dataset **no incluye la ronda del partido** (fase de grupos, octavos, cuartos, etc.). Solo contiene score y nombre del torneo. Por eso es necesaria una fuente complementaria.

### Fuente complementaria (obligatoria)

**FIFA World Cup Dataset** (Kaggle) — archivos `WorldCups.csv` y `WorldCupMatches.csv`

Este dataset sí incluye la fase o ronda de cada partido, lo que permite:
- Identificar qué partidos pertenecen a la fase de grupos.
- Determinar qué selecciones avanzaron más allá de esa fase.
- Construir correctamente la variable objetivo `advanced_group_stage`.

El proceso de construcción del dataset final requiere cruzar ambas fuentes usando `(home_team, away_team, date)` o `(home_team, away_team, world_cup_year)` como llave de unión.

### Variables originales relevantes

- `date`: fecha del partido.
- `home_team`: equipo local.
- `away_team`: equipo visitante.
- `home_score`: goles del equipo local.
- `away_score`: goles del equipo visitante.
- `tournament`: nombre del torneo.
- `city`: ciudad donde se jugó el partido.
- `country`: país sede del partido.
- `neutral`: indica si el partido fue en campo neutral.
- `stage` *(fuente complementaria)*: fase o ronda del partido.

### Variables derivadas necesarias

- `host_continent`: continente sede de la edición del mundial.
- `world_cup_year`: año de la edición del mundial.
- `team`: selección analizada.
- `opponent`: rival.
- `team_confederation`: confederación a la que pertenece el equipo.
- `is_host_region`: indica si el equipo pertenece al mismo continente de la sede.
- `historical_win_rate`: tasa de victorias del equipo en mundiales anteriores (calculada solo con ediciones previas para evitar data leakage).
- `historical_avg_goals_scored`: promedio histórico de goles anotados en mundiales anteriores.
- `historical_appearances`: número de mundiales anteriores en los que participó el equipo.
- `advanced_group_stage`: variable objetivo binaria que indica si el equipo avanzó más allá de la fase de grupos.

> ⚠️ **Sobre data leakage:** Las variables históricas deben calcularse usando **solo las ediciones anteriores** a la que se está prediciendo. Por ejemplo, para predecir el mundial 2014, solo se usan datos hasta 2010. Esto respeta la lógica temporal del problema y evita filtración de información futura.

---

## Granularidad del dataset supervisado

El dataset original está a nivel de **partido**. El modelo supervisado requiere una fila por `(equipo, edición)`. Por eso es necesario un paso intermedio de agregación:

1. Filtrar partidos de fase de grupos usando la columna `stage`.
2. Para cada `(team, world_cup_year)`, construir un registro único con:
   - Variables pre-torneo como features (rendimiento histórico, confederación, sede).
   - La variable objetivo `advanced_group_stage` derivada de si el equipo jugó rondas posteriores a la fase de grupos.

> **Consideración:** El número de partidos de fase de grupos por edición varió históricamente (3 partidos desde 1950, con excepciones en formatos anteriores). El proceso de construcción debe manejar estas inconsistencias.

---

## Problema de Machine Learning

### Tipo de problema supervisado

Clasificación binaria.

### Variable objetivo

`advanced_group_stage`

Valores posibles:

- `1`: la selección avanzó más allá de la fase de grupos.
- `0`: la selección fue eliminada en fase de grupos.

### Balance de clases

En el formato moderno de 32 equipos (1998-2022), 16 de 32 selecciones avanzan, lo que produce un balance aproximado de 50/50. Sin embargo, en ediciones anteriores el balance puede variar. Se debe verificar la distribución y, si es necesario, usar estratificación en el split de datos (`stratify=y`).

### Modelo baseline

Antes de la regresión logística se entrenará un **DummyClassifier** que predice siempre la clase mayoritaria. Esto establece el piso mínimo de accuracy y le da contexto real a las métricas del modelo principal.

```python
from sklearn.dummy import DummyClassifier
dummy = DummyClassifier(strategy="most_frequent")
```

### Modelo supervisado principal

**Regresión logística**.

Se eligió porque:

- Es interpretable.
- Sirve para clasificación binaria.
- Permite analizar el peso de las variables contextuales.
- Es adecuada para un proyecto académico de Machine Learning.

> ⚠️ **Limitación estadística reconocida:** Un mismo equipo aparece en múltiples mundiales, lo que introduce dependencia entre observaciones y viola el supuesto de independencia de la regresión logística estándar. Esta limitación debe reconocerse explícitamente en las conclusiones del proyecto.

---

## Problema no supervisado

### Tipo de problema

Agrupamiento de ediciones mundialistas.

### Tamaño del dataset de clustering

Solo hay **22 ediciones** del mundial entre 1930 y 2022 (sin contar 1942 y 1946). Este es un dataset muy pequeño para K-Means.

### Modelos no supervisados

Se aplicarán dos algoritmos complementarios:

1. **K-Means**: permite seleccionar k mediante el método del codo y el coeficiente de silueta. Sensible a outliers y asume clusters esféricos.
2. **Clustering Jerárquico (Agglomerative Clustering)**: más adecuado para datasets pequeños. Permite visualizar un dendrograma para interpretar la estructura de grupos sin asumir k de antemano.

> Con solo 22 observaciones, el clustering jerárquico es preferible como análisis principal. K-Means puede usarse como validación complementaria.

### Objetivo del clustering

Agrupar las ediciones del mundial según su comportamiento estadístico para identificar perfiles de torneos, por ejemplo:

- Mundiales dominados por selecciones europeas.
- Mundiales dominados por selecciones sudamericanas.
- Mundiales equilibrados.
- Mundiales atípicos o de bajo scoring.

### Variables agregadas por edición

- Promedio de goles por partido.
- Representación por confederación (porcentaje de equipos por confederación).
- Rendimiento promedio por confederación (tasa de avance de fase de grupos).
- Rondas promedio alcanzadas.
- Diferencia promedio de goles.
- Índice de sorpresas (porcentaje de equipos "débiles" que avanzaron).

---

## Normalización de nombres históricos de selecciones

El dataset contiene nombres históricos que deben unificarse. Se debe construir un diccionario de mapeo explícito:

```python
TEAM_NAME_MAP = {
    "West Germany": "Germany",
    "Czechoslovakia": "Czech Republic",
    "Yugoslavia": "Serbia",
    "Soviet Union": "Russia",
    "Zaire": "DR Congo",
    "Dutch East Indies": "Indonesia",
    "Republic of Ireland": "Ireland",
    "FR Yugoslavia": "Serbia",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
}
```

Este diccionario debe aplicarse tanto al dataset de partidos como al de clasificación del mundial para que los joins funcionen correctamente.

---

## Metodología técnica

El proyecto debe seguir un flujo típico de ciencia de datos.

### 1. Carga de datos

- Leer el dataset principal de resultados internacionales.
- Leer el dataset complementario con información de rondas (`WorldCupMatches.csv`).
- Validar columnas disponibles en ambas fuentes.
- Filtrar únicamente los partidos de Copa Mundial de la FIFA.

### 2. Limpieza de datos

- Revisar valores nulos.
- Revisar duplicados.
- Aplicar el diccionario de normalización de nombres de selecciones.
- Revisar valores atípicos en goles.
- Validar fechas y años de edición.
- Cruzar ambas fuentes para obtener la columna `stage` en el dataset principal.

### 3. Análisis exploratorio de datos

Generar visualizaciones y tablas para entender:

- Cantidad de partidos por mundial.
- Distribución de goles.
- Países y continentes sede.
- Rendimiento por región o confederación.
- Comparación entre selecciones locales, regionales y visitantes.
- Relación entre sede y avance de fase.
- Distribución de la variable objetivo (balance de clases).

### 4. Ingeniería de características

Crear variables útiles para los modelos. **Todas las features del modelo supervisado deben ser pre-torneo:**

- Año del mundial.
- Continente sede.
- Confederación del equipo.
- Indicador de pertenencia al continente sede (`is_host_region`).
- Tasa histórica de victorias en mundiales **anteriores** (`historical_win_rate`).
- Promedio histórico de goles anotados en mundiales **anteriores**.
- Número de mundiales anteriores en los que participó el equipo.

> Los goles del torneo actual (`goal_difference`, `total_goals`) **no deben usarse como features** del modelo supervisado porque son información concurrent con el target, no previa.

### 5. Preparación para el modelo supervisado

- Construir el dataset agregado a nivel `(equipo, edición)`.
- Definir `X` con variables predictoras pre-torneo.
- Definir `y` con la variable `advanced_group_stage`.
- Codificar variables categóricas con One-Hot Encoding.
- Escalar variables numéricas.
- Dividir los datos en entrenamiento y prueba con `stratify=y` para mantener el balance de clases.
- Respetar la lógica temporal: calcular variables históricas usando solo ediciones pasadas.

### 6. Entrenamiento del modelo supervisado

- Entrenar primero el modelo baseline (`DummyClassifier`).
- Entrenar la regresión logística.
- Ajustar hiperparámetros básicos (`C`, `solver`, `max_iter`).
- Evaluar resultados comparando el modelo principal contra el baseline.
- Analizar coeficientes para interpretar qué variables influyen más.

### 7. Evaluación del modelo supervisado

Métricas requeridas:

- Accuracy (comparado con el baseline).
- Precision.
- Recall.
- F1-score.
- Matriz de confusión.
- Curva ROC y AUC.

### 8. Preparación para clustering

- Agrupar datos por edición del mundial.
- Crear un dataframe con **una fila por mundial** (~22 filas).
- Seleccionar variables estadísticas agregadas.
- Escalar variables antes de aplicar los modelos (StandardScaler).

### 9. Entrenamiento de modelos no supervisados

**K-Means:**

- Aplicar con distintos valores de `k` (2 a 8).
- Usar método del codo para analizar la inercia.
- Usar coeficiente de silueta para elegir el número de clusters.

**Clustering Jerárquico:**

- Aplicar `AgglomerativeClustering` de scikit-learn.
- Visualizar dendrograma con `scipy.cluster.hierarchy.dendrogram`.
- Comparar la partición obtenida con la de K-Means.

### 10. Evaluación del modelo no supervisado

Métricas requeridas:

- Inercia (K-Means).
- Coeficiente de silueta.
- Visualización de clusters (PCA a 2 componentes para scatter plot).
- Dendrograma (clustering jerárquico).
- Interpretación estadística de cada grupo.

---

## Pipeline de Machine Learning

Se construirá un pipeline con `sklearn.pipeline.Pipeline` para las etapas de **transformación y modelado**. Los pasos de carga, filtrado y limpieza se mantienen fuera del pipeline en scripts separados, ya que Pipeline solo encadena transformadores y un estimador final.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(random_state=42, max_iter=1000))
])
```

> **Nota:** Los pasos de carga de datos, filtrado de partidos de Copa Mundial, limpieza y construcción de variables históricas van en scripts de Python separados, no dentro del Pipeline de scikit-learn.

---

## Metadatos del experimento

Se almacenarán metadatos en un archivo `.json` para documentar el contexto de cada ejecución:

```json
{
  "dataset": "results.csv + WorldCupMatches.csv",
  "execution_date": "2024-XX-XX",
  "model_version": "1.0",
  "algorithm": "LogisticRegression",
  "features": ["host_continent", "team_confederation", "is_host_region", "historical_win_rate", "historical_appearances"],
  "target": "advanced_group_stage",
  "model_params": {"C": 1.0, "solver": "lbfgs", "max_iter": 1000},
  "train_size": 0,
  "test_size": 0,
  "metrics": {
    "accuracy": 0,
    "precision": 0,
    "recall": 0,
    "f1_score": 0,
    "auc_roc": 0
  },
  "notes": ""
}
```

---

## Serialización del modelo

Una vez entrenado el modelo supervisado, se guardará en disco con `joblib`:

```python
import joblib
joblib.dump(pipeline, "outputs/models/logistic_regression_model.joblib")
joblib.dump(kmeans_model, "outputs/models/kmeans_model.joblib")
```

Estructura de archivos generados:

```text
outputs/models/
├── logistic_regression_model.joblib
├── kmeans_model.joblib
├── hierarchical_clustering_model.joblib
└── metadata_model.json
```

---

## Resultados esperados

Al finalizar el proyecto se espera obtener:

1. Un pipeline reproducible desde datos crudos hasta datos listos para modelado.
2. Un modelo baseline como punto de comparación.
3. Un modelo de regresión logística con accuracy **superior al baseline** en prueba (referencia orientativa: >65%).
4. Una segmentación de las ediciones del mundial en 3 o 4 clusters interpretables, validada con K-Means y clustering jerárquico.
5. Un análisis sobre si el continente sede tiene un efecto observable en el rendimiento.
6. Visualizaciones claras para presentar los hallazgos.

---

## Visualizaciones esperadas

- Gráficos de distribución de goles.
- Gráficos de barras por continente o confederación.
- Tablas de frecuencia por edición del mundial.
- Matriz de correlación.
- Distribución de la variable objetivo (balance de clases).
- Matriz de confusión del modelo supervisado.
- Curva ROC (modelo supervisado vs. baseline).
- Gráfico del método del codo para K-Means.
- Gráfico del coeficiente de silueta.
- Dendrograma del clustering jerárquico.
- Gráfico de dispersión de clusters (PCA 2D).
- Comparaciones de avance de fase por continente sede.

---

## Estructura del proyecto

```text
proyecto_mundial_ml/
│
├── data/
│   ├── raw/
│   │   ├── results.csv
│   │   ├── WorldCupMatches.csv
│   │   └── WorldCups.csv
│   ├── processed/
│   │   ├── world_cup_matches_clean.csv
│   │   ├── supervised_dataset.csv
│   │   └── world_cup_editions_clusters.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_supervised_model.ipynb
│   └── 04_unsupervised_model.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── supervised_model.py
│   ├── clustering_model.py
│   ├── evaluation.py
│   └── visualization.py
│
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── models/
│
├── requirements.txt
├── README.md
└── main.py
```

---

## Librerías

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
jupyter
joblib
```

Opcionales:

```txt
plotly
```

> `scipy` es ahora obligatoria para el dendrograma del clustering jerárquico.

---

## Posible extensión futura

Una extensión del proyecto podría ser exponer el modelo como una API REST ligera que reciba características **pre-torneo** de una selección y devuelva una probabilidad estimada de avanzar de fase.

```http
POST /predict
```

Entrada:

```json
{
  "team": "Colombia",
  "world_cup_year": 2026,
  "host_continent": "North America",
  "team_confederation": "CONMEBOL",
  "historical_win_rate": 0.40,
  "historical_appearances": 7
}
```

Salida esperada:

```json
{
  "advanced_group_stage_probability": 0.57,
  "prediction": 1
}
```

---

## Limitaciones conocidas del proyecto

Estas limitaciones deben reconocerse explícitamente en las conclusiones:

1. **Independencia entre observaciones:** Un mismo equipo aparece en múltiples mundiales, lo que viola el supuesto de independencia de la regresión logística estándar.
2. **Dataset pequeño para clustering:** Solo ~22 ediciones mundialistas disponibles. Los resultados del clustering deben interpretarse con cautela.
3. **Normalización de selecciones históricas:** Equipos como West Germany, Yugoslavia o Checoslovaquia ya no existen. La fusión con sus sucesores introduce simplificaciones.
4. **Variables pre-torneo limitadas:** El rendimiento histórico como predictor puede ser débil en selecciones con pocas participaciones previas.
5. **Evolución del formato:** El número de equipos y el formato del torneo cambió significativamente entre 1930 y 2022, lo que introduce heterogeneidad histórica en el dataset.

---

## Enfoque final del análisis

El proyecto no debe limitarse a predecir resultados. También debe explicar e interpretar los patrones encontrados.

Las preguntas clave a responder:

- ¿Existe una ventaja observable para el continente sede?
- ¿Las selecciones del continente anfitrión tienen mejores resultados históricos en sus propias ediciones?
- ¿Qué variables contextuales pre-torneo influyen más en el avance de fase?
- ¿Qué tipos de mundiales aparecen al agrupar las ediciones históricas?
- ¿Los clusters encontrados coinciden con patrones futbolísticos interpretables?
- ¿El modelo supera al baseline de forma estadísticamente significativa?

---

## Checklist de implementación

### Preparación

- [ ] Crear estructura de carpetas.
- [ ] Crear `requirements.txt`.
- [ ] Cargar dataset principal (`results.csv`).
- [ ] Cargar dataset complementario (`WorldCupMatches.csv`).
- [ ] Filtrar partidos de Copa Mundial.
- [ ] Validar columnas y tipos de datos en ambas fuentes.

### Limpieza

- [ ] Eliminar duplicados.
- [ ] Revisar nulos.
- [ ] Aplicar diccionario de normalización de nombres de selecciones.
- [ ] Cruzar ambas fuentes para obtener la columna `stage`.
- [ ] Crear columna de año del mundial.
- [ ] Crear columna de continente sede.

### EDA

- [ ] Analizar goles por partido.
- [ ] Analizar partidos por edición.
- [ ] Analizar países y continentes sede.
- [ ] Analizar comportamiento por selección y confederación.
- [ ] Verificar balance de clases en la variable objetivo.
- [ ] Crear visualizaciones principales.

### Modelo supervisado

- [ ] Construir dataset agregado a nivel `(equipo, edición)`.
- [ ] Calcular variables históricas usando solo ediciones previas (sin data leakage).
- [ ] Definir variable objetivo `advanced_group_stage`.
- [ ] Codificar variables categóricas.
- [ ] Dividir datos en train/test con `stratify=y`.
- [ ] Entrenar modelo baseline (`DummyClassifier`).
- [ ] Entrenar regresión logística con Pipeline.
- [ ] Evaluar con accuracy, precision, recall y F1-score.
- [ ] Comparar métricas contra baseline.
- [ ] Generar matriz de confusión.
- [ ] Generar curva ROC.
- [ ] Interpretar coeficientes.

### Modelo no supervisado

- [ ] Crear dataset agregado por edición (~22 filas).
- [ ] Escalar variables.
- [ ] Aplicar K-Means con distintos valores de `k`.
- [ ] Aplicar método del codo.
- [ ] Calcular coeficiente de silueta.
- [ ] Aplicar Clustering Jerárquico.
- [ ] Generar dendrograma.
- [ ] Comparar resultados de ambos algoritmos.
- [ ] Interpretar clusters.

### Entrega final

- [ ] Guardar visualizaciones definitivas en `outputs/figures/`.
- [ ] Guardar métricas en `outputs/metrics/`.
- [ ] Serializar modelos en `outputs/models/`.
- [ ] Guardar metadatos en `outputs/models/metadata_model.json`.
- [ ] Redactar conclusiones respondiendo las preguntas del análisis.
- [ ] Documentar limitaciones reconocidas.
- [ ] Explicar posible escalabilidad para futuros mundiales (2026, 2030).

---

## Nota para desarrollo

Priorizar una implementación clara, sencilla y **defendible académicamente** por encima de una solución demasiado compleja. El objetivo principal es demostrar un buen proceso de Machine Learning: carga y cruce de datos, limpieza, EDA, ingeniería de características sin data leakage, modelado supervisado con baseline, clustering con dos algoritmos, evaluación e interpretación de resultados.
