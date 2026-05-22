# Plan de Ejecución — El Factor Local

**Proyecto final de Machine Learning**
**Asignatura:** Machine Learning
**Dataset:** International Football Results + FIFA World Cup Dataset (Kaggle)

---

## Vista general del plan

```
FASE 0 → FASE 1 → FASE 2 → FASE 3 → FASE 4 → FASE 5 → FASE 6 → FASE 7
 Setup    Datos    Limpieza   EDA    Features  Supervisado  No-Super  Entrega
```

Cada fase produce entregables concretos que la siguiente fase consume. No avanzar a la siguiente fase sin tener los entregables de la anterior validados.

---

## FASE 0 — Configuración del entorno

**Objetivo:** Tener el proyecto listo para trabajar antes de tocar cualquier dato.

### 0.1 Crear la estructura de carpetas

Ejecutar una sola vez desde la raíz del proyecto:

```bash
mkdir -p proyecto_mundial_ml/{data/{raw,processed},notebooks,src,outputs/{figures,metrics,models}}
cd proyecto_mundial_ml
touch README.md main.py requirements.txt
touch src/{__init__,data_loader,preprocessing,feature_engineering,supervised_model,clustering_model,evaluation,visualization}.py
```

### 0.2 Crear `requirements.txt`

```txt
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
scipy>=1.11
jupyter>=1.0
joblib>=1.3
```

Instalar con:

```bash
pip install -r requirements.txt
```

### 0.3 Descargar los datasets

Descargar manualmente desde Kaggle y colocar en `data/raw/`:

| Archivo | Dataset en Kaggle | Uso |
|---|---|---|
| `results.csv` | *International Football Results from 1872 to 2024* | Scores de partidos |
| `WorldCupMatches.csv` | *FIFA World Cup* | Rondas de partidos |
| `WorldCups.csv` | *FIFA World Cup* | Info por edición |

Verificar que los tres archivos están presentes antes de continuar.

### 0.4 Crear notebook de exploración inicial

Crear `notebooks/00_exploracion_raw.ipynb` y ejecutar solo esto para validar que los archivos cargan correctamente:

```python
import pandas as pd

results = pd.read_csv("../data/raw/results.csv")
wc_matches = pd.read_csv("../data/raw/WorldCupMatches.csv")
wc_editions = pd.read_csv("../data/raw/WorldCups.csv")

print(results.shape, results.columns.tolist())
print(wc_matches.shape, wc_matches.columns.tolist())
print(wc_editions.shape, wc_editions.columns.tolist())
```

**✅ Entregables de la Fase 0:**
- Estructura de carpetas creada.
- `requirements.txt` instalado sin errores.
- Tres archivos CSV presentes en `data/raw/`.
- Los tres datasets cargan correctamente en pandas.

---

## FASE 1 — Carga y cruce de datos

**Objetivo:** Construir un único dataset de partidos de Copa Mundial con la columna `stage` incluida.

**Notebook:** `notebooks/01_data_loading.ipynb`
**Script resultado:** `src/data_loader.py`
**Entregable:** `data/processed/world_cup_matches_raw.csv`

### 1.1 Filtrar partidos de Copa Mundial

Desde `results.csv`, conservar solo filas donde `tournament == "FIFA World Cup"`:

```python
wc_results = results[results["tournament"] == "FIFA World Cup"].copy()
print(f"Partidos de Copa Mundial: {len(wc_results)}")  # ~900 esperados
```

### 1.2 Explorar la fuente complementaria

Revisar qué columnas tiene `WorldCupMatches.csv` y cómo se nombran los equipos y las rondas:

```python
print(wc_matches.columns.tolist())
print(wc_matches["Stage"].unique())        # Ver nombres de rondas
print(wc_matches["Home Team Name"].unique()[:20])  # Ver nombres de equipos
```

Columnas esperadas en `WorldCupMatches.csv`: `Year`, `Datetime`, `Stage`, `Stadium`, `City`, `Home Team Name`, `Home Team Goals`, `Away Team Goals`, `Away Team Name`, etc.

### 1.3 Preparar llaves de cruce

Ambos datasets deben tener columnas comparables para hacer el join. La llave más robusta es `(year, home_team, away_team)`:

```python
# En wc_results: extraer año
wc_results["year"] = pd.to_datetime(wc_results["date"]).dt.year

# En wc_matches: limpiar columnas
wc_matches["year"] = wc_matches["Year"].astype(int)
wc_matches = wc_matches.rename(columns={
    "Home Team Name": "home_team",
    "Away Team Name": "away_team",
    "Stage": "stage"
})
```

### 1.4 Aplicar diccionario de normalización ANTES del join

Este paso es crítico — si los nombres no coinciden, el join falla:

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
    "Northern Ireland": "Northern Ireland",  # mantener como está
    "Trinidad and Tobago": "Trinidad and Tobago",
}

for df in [wc_results, wc_matches]:
    df["home_team"] = df["home_team"].replace(TEAM_NAME_MAP)
    df["away_team"] = df["away_team"].replace(TEAM_NAME_MAP)
```

### 1.5 Hacer el join

```python
merged = wc_results.merge(
    wc_matches[["year", "home_team", "away_team", "stage"]],
    on=["year", "home_team", "away_team"],
    how="left"
)

# Verificar cobertura del join
null_stage = merged["stage"].isna().sum()
print(f"Partidos sin stage asignado: {null_stage} de {len(merged)}")
```

> **Si hay muchos nulos:** revisar si los nombres difieren entre fuentes. Usar `set(wc_results["home_team"]) - set(wc_matches["home_team"])` para identificar discrepancias y ampliar el diccionario de normalización.

### 1.6 Guardar dataset crudo cruzado

```python
merged.to_csv("../data/processed/world_cup_matches_raw.csv", index=False)
```

**✅ Entregables de la Fase 1:**
- `data/processed/world_cup_matches_raw.csv` con columna `stage` poblada.
- Cobertura del join superior al 95% (menos de 5% de partidos sin stage).
- `src/data_loader.py` con función `load_world_cup_data()` reutilizable.

---

## FASE 2 — Limpieza de datos

**Objetivo:** Dataset limpio, consistente y listo para el análisis exploratorio.

**Notebook:** `notebooks/02_cleaning.ipynb`
**Script resultado:** `src/preprocessing.py`
**Entregable:** `data/processed/world_cup_matches_clean.csv`

### 2.1 Revisión de nulos

```python
df = pd.read_csv("../data/processed/world_cup_matches_raw.csv")

print("Nulos por columna:")
print(df.isnull().sum())
```

Columnas críticas que no deben tener nulos: `home_team`, `away_team`, `home_score`, `away_score`, `year`, `stage`.

### 2.2 Revisión de duplicados

```python
dupes = df.duplicated(subset=["year", "home_team", "away_team"])
print(f"Duplicados encontrados: {dupes.sum()}")
df = df[~dupes]
```

### 2.3 Validar valores de `stage`

```python
print(df["stage"].value_counts())
```

Valores esperados: `Group X`, `Round of 16`, `Quarter-finals`, `Semi-finals`, `Third place`, `Final`, y variantes históricas.

Crear una columna binaria para identificar partidos de fase de grupos:

```python
group_stage_keywords = ["Group", "group", "First round", "Preliminary round", "Pool"]
df["is_group_stage"] = df["stage"].str.contains("|".join(group_stage_keywords), na=False)
print(df["is_group_stage"].value_counts())
```

### 2.4 Revisar outliers en goles

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df["home_score"], ax=axes[0]).set_title("Distribución goles local")
sns.histplot(df["away_score"], ax=axes[1]).set_title("Distribución goles visitante")
plt.savefig("../outputs/figures/distribucion_goles_raw.png", dpi=150)
```

Verificar si hay marcadores imposibles (negativos, o muy altos como >20).

### 2.5 Crear columnas auxiliares

```python
# Continente sede por año de mundial
HOST_CONTINENT_MAP = {
    1930: "South America", 1934: "Europe", 1938: "Europe",
    1950: "South America", 1954: "Europe", 1958: "Europe",
    1962: "South America", 1966: "Europe", 1970: "North America",
    1974: "Europe", 1978: "South America", 1982: "Europe",
    1986: "North America", 1990: "Europe", 1994: "North America",
    1998: "Europe", 2002: "Asia", 2006: "Europe",
    2010: "Africa", 2014: "South America", 2018: "Europe",
    2022: "Asia"
}
df["host_continent"] = df["year"].map(HOST_CONTINENT_MAP)

# Diferencia y total de goles (solo para EDA, NO para features del modelo)
df["goal_difference"] = df["home_score"] - df["away_score"]
df["total_goals"] = df["home_score"] + df["away_score"]
```

### 2.6 Guardar dataset limpio

```python
df.to_csv("../data/processed/world_cup_matches_clean.csv", index=False)
print(f"Dataset limpio guardado: {df.shape}")
```

**✅ Entregables de la Fase 2:**
- `data/processed/world_cup_matches_clean.csv` sin nulos en columnas críticas.
- Columna `is_group_stage` correctamente calculada.
- Columna `host_continent` mapeada para todas las ediciones.
- `src/preprocessing.py` con función `clean_world_cup_data()`.

---

## FASE 3 — Análisis Exploratorio de Datos (EDA)

**Objetivo:** Entender el dataset a fondo y generar las visualizaciones que acompañarán el reporte final.

**Notebook:** `notebooks/03_eda.ipynb`
**Script resultado:** `src/visualization.py`
**Entregable:** Todas las figuras en `outputs/figures/`

> En esta fase **no se construye ningún modelo**. Solo se explora, visualiza e interpreta.

### 3.1 Análisis de partidos por edición

```python
partidos_por_edicion = df.groupby("year").size().reset_index(name="partidos")
sns.barplot(data=partidos_por_edicion, x="year", y="partidos")
plt.xticks(rotation=45)
plt.title("Partidos por edición del Mundial")
plt.savefig("../outputs/figures/partidos_por_edicion.png", dpi=150)
```

### 3.2 Distribución de goles

```python
# Goles promedio por edición
goles_por_edicion = df.groupby("year")["total_goals"].mean()
sns.lineplot(data=goles_por_edicion)
plt.title("Promedio de goles por partido por edición")
plt.savefig("../outputs/figures/goles_promedio_por_edicion.png", dpi=150)
```

### 3.3 Rendimiento por continente sede

Para cada edición, comparar la tasa de avance de fase de grupos entre equipos del continente sede vs. el resto:

```python
# Solo fase de grupos
grupos = df[df["is_group_stage"]].copy()
# ... construir columna is_host_region, luego agrupar
```

Esta visualización es clave para la pregunta central del proyecto.

### 3.4 Análisis de selecciones más exitosas históricamente

```python
# Top 10 selecciones con más apariciones en fase de grupos
# Top 10 selecciones con más victorias en fase de grupos
```

### 3.5 Matriz de correlación

Solo con variables numéricas del dataset de partidos:

```python
numeric_cols = ["home_score", "away_score", "total_goals", "goal_difference", "year"]
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.savefig("../outputs/figures/matriz_correlacion.png", dpi=150)
```

### 3.6 Verificar balance de clases (anticipar el modelo)

Construir temporalmente la variable objetivo para ver su distribución:

```python
# Equipos que jugaron rondas posteriores a fase de grupos
equipos_avanzaron = set(
    df[~df["is_group_stage"]][["year", "home_team"]].apply(tuple, axis=1).tolist() +
    df[~df["is_group_stage"]][["year", "away_team"]].apply(tuple, axis=1).tolist()
)
```

**Visualizaciones a guardar en `outputs/figures/`:**

| Archivo | Descripción |
|---|---|
| `distribucion_goles_raw.png` | Histograma goles local y visitante |
| `partidos_por_edicion.png` | Barras: partidos por año |
| `goles_promedio_por_edicion.png` | Línea: promedio goles por edición |
| `rendimiento_por_sede.png` | Comparación sede vs. foráneo |
| `top_selecciones_avance.png` | Ranking de selecciones exitosas |
| `matriz_correlacion.png` | Heatmap de correlaciones |
| `balance_clases.png` | Distribución de la variable objetivo |

**✅ Entregables de la Fase 3:**
- Todas las figuras listadas guardadas en `outputs/figures/`.
- Conclusiones preliminares escritas en el notebook (celdas Markdown).
- `src/visualization.py` con funciones reutilizables para cada gráfico.

---

## FASE 4 — Ingeniería de Características

**Objetivo:** Construir el dataset supervisado con una fila por `(equipo, edición)` y solo features pre-torneo.

**Notebook:** `notebooks/04_feature_engineering.ipynb`
**Script resultado:** `src/feature_engineering.py`
**Entregable:** `data/processed/supervised_dataset.csv`

> Esta es la fase técnicamente más delicada. Requiere especial cuidado con el data leakage.

### 4.1 Definir la confederación por equipo

Crear un diccionario de confederaciones:

```python
CONFEDERATION_MAP = {
    # UEFA (Europa)
    "Germany": "UEFA", "France": "UEFA", "Spain": "UEFA", "Italy": "UEFA",
    "England": "UEFA", "Netherlands": "UEFA", "Portugal": "UEFA",
    "Belgium": "UEFA", "Croatia": "UEFA", "Serbia": "UEFA",
    # CONMEBOL (Sudamérica)
    "Brazil": "CONMEBOL", "Argentina": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Colombia": "CONMEBOL", "Chile": "CONMEBOL", "Paraguay": "CONMEBOL",
    "Ecuador": "CONMEBOL", "Peru": "CONMEBOL", "Bolivia": "CONMEBOL",
    # CONCACAF (Norte y Centroamérica)
    "Mexico": "CONCACAF", "United States": "CONCACAF", "Costa Rica": "CONCACAF",
    "Honduras": "CONCACAF", "Jamaica": "CONCACAF", "Cuba": "CONCACAF",
    # CAF (África)
    "Cameroon": "CAF", "Nigeria": "CAF", "Senegal": "CAF",
    "Ghana": "CAF", "Morocco": "CAF", "Egypt": "CAF",
    "DR Congo": "CAF", "Algeria": "CAF", "Ivory Coast": "CAF",
    # AFC (Asia)
    "Japan": "AFC", "South Korea": "AFC", "Iran": "AFC",
    "Saudi Arabia": "AFC", "Australia": "AFC", "Indonesia": "AFC",
    # OFC y otros
    "New Zealand": "OFC",
    # ... completar según equipos en el dataset
}
```

### 4.2 Obtener lista única de `(equipo, edición)`

```python
grupos = df[df["is_group_stage"]].copy()
# Pivotear: cada partido tiene home_team y away_team
home = grupos[["year", "home_team"]].rename(columns={"home_team": "team"})
away = grupos[["year", "away_team"]].rename(columns={"away_team": "team"})
team_editions = pd.concat([home, away]).drop_duplicates()
```

### 4.3 Construir la variable objetivo `advanced_group_stage`

```python
# Un equipo avanzó si apareció en algún partido fuera de fase de grupos
knockout = df[~df["is_group_stage"]].copy()
knockout_home = set(zip(knockout["year"], knockout["home_team"]))
knockout_away = set(zip(knockout["year"], knockout["away_team"]))
advanced = knockout_home.union(knockout_away)

team_editions["advanced_group_stage"] = team_editions.apply(
    lambda row: 1 if (row["year"], row["team"]) in advanced else 0,
    axis=1
)
print(team_editions["advanced_group_stage"].value_counts())
```

### 4.4 Calcular variables históricas SIN data leakage

Para cada `(team, year)`, calcular stats usando **solo ediciones anteriores**:

```python
def compute_historical_features(team_editions_df, all_matches_df):
    """
    Para cada (team, year), calcula estadísticas históricas
    usando SOLO ediciones previas (year < current_year).
    """
    records = []
    editions = sorted(team_editions_df["year"].unique())

    for current_year in editions:
        # Selecciones que participaron en este mundial
        current_teams = team_editions_df[
            team_editions_df["year"] == current_year
        ]["team"].unique()

        # Solo partidos de ediciones ANTERIORES
        past = all_matches_df[
            (all_matches_df["is_group_stage"]) &
            (all_matches_df["year"] < current_year)
        ]

        for team in current_teams:
            # Partidos del equipo en el pasado
            team_past = past[
                (past["home_team"] == team) | (past["away_team"] == team)
            ]

            appearances = past["year"].unique()
            team_years = team_past["year"].unique()
            n_appearances = len(team_years)

            # Win rate histórico en mundiales pasados
            if n_appearances == 0:
                win_rate = 0.0
                avg_goals = 0.0
            else:
                # Calcular victorias
                wins = 0
                goals = 0
                for _, row in team_past.iterrows():
                    if row["home_team"] == team:
                        goals += row["home_score"]
                        if row["home_score"] > row["away_score"]:
                            wins += 1
                    else:
                        goals += row["away_score"]
                        if row["away_score"] > row["home_score"]:
                            wins += 1
                win_rate = wins / len(team_past)
                avg_goals = goals / len(team_past)

            records.append({
                "year": current_year,
                "team": team,
                "historical_appearances": n_appearances,
                "historical_win_rate": round(win_rate, 4),
                "historical_avg_goals_scored": round(avg_goals, 4),
            })

    return pd.DataFrame(records)
```

### 4.5 Agregar variables de contexto del torneo

```python
# Unir con team_editions
features = team_editions.merge(
    compute_historical_features(team_editions, df),
    on=["year", "team"], how="left"
)

# Confederación del equipo
features["team_confederation"] = features["team"].map(CONFEDERATION_MAP).fillna("OTHER")

# Continente sede
features["host_continent"] = features["year"].map(HOST_CONTINENT_MAP)

# ¿El equipo es del mismo continente que la sede?
CONFEDERATION_TO_CONTINENT = {
    "UEFA": "Europe", "CONMEBOL": "South America",
    "CONCACAF": "North America", "CAF": "Africa",
    "AFC": "Asia", "OFC": "Oceania"
}
features["team_continent"] = features["team_confederation"].map(CONFEDERATION_TO_CONTINENT)
features["is_host_region"] = (
    features["team_continent"] == features["host_continent"]
).astype(int)
```

### 4.6 Guardar dataset supervisado

```python
features.to_csv("../data/processed/supervised_dataset.csv", index=False)
print(f"Dataset supervisado: {features.shape}")
print(features.head())
print(features.dtypes)
```

**Columnas finales del dataset supervisado:**

| Columna | Tipo | Rol |
|---|---|---|
| `year` | int | Identificador |
| `team` | str | Identificador |
| `host_continent` | str | Feature categórica |
| `team_confederation` | str | Feature categórica |
| `is_host_region` | int (0/1) | Feature binaria |
| `historical_appearances` | int | Feature numérica |
| `historical_win_rate` | float | Feature numérica |
| `historical_avg_goals_scored` | float | Feature numérica |
| `advanced_group_stage` | int (0/1) | **Target** |

**✅ Entregables de la Fase 4:**
- `data/processed/supervised_dataset.csv` con una fila por `(equipo, edición)`.
- Variable objetivo calculada correctamente.
- Variables históricas calculadas sin data leakage (verificar manualmente para 1 o 2 equipos).
- `src/feature_engineering.py` con función `build_supervised_dataset()`.

---

## FASE 5 — Modelo Supervisado

**Objetivo:** Entrenar, evaluar e interpretar el modelo de regresión logística comparado contra el baseline.

**Notebook:** `notebooks/05_supervised_model.ipynb`
**Script resultado:** `src/supervised_model.py`
**Entregables:** Métricas, figuras y modelo serializado.

### 5.1 Cargar dataset y definir X, y

```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/processed/supervised_dataset.csv")

feature_cols = [
    "host_continent", "team_confederation", "is_host_region",
    "historical_appearances", "historical_win_rate", "historical_avg_goals_scored"
]
target_col = "advanced_group_stage"

X = df[feature_cols]
y = df[target_col]

print(f"Dimensiones X: {X.shape}")
print(f"Balance clases:\n{y.value_counts(normalize=True)}")
```

### 5.2 Split train/test con estratificación

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Mantiene el balance de clases en ambas particiones
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
```

### 5.3 Construir el Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

numerical_features = ["is_host_region", "historical_appearances",
                       "historical_win_rate", "historical_avg_goals_scored"]
categorical_features = ["host_continent", "team_confederation"]

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
])

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(random_state=42, max_iter=1000, C=1.0))
])
```

### 5.4 Entrenar el modelo baseline

```python
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report

dummy = DummyClassifier(strategy="most_frequent", random_state=42)
dummy.fit(X_train, y_train)
y_pred_dummy = dummy.predict(X_test)

print("=== BASELINE ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dummy):.4f}")
print(classification_report(y_test, y_pred_dummy))
```

### 5.5 Entrenar la regresión logística

```python
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("=== REGRESIÓN LOGÍSTICA ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
```

### 5.6 Matriz de confusión

```python
from sklearn.metrics import ConfusionMatrixDisplay

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_dummy, ax=axes[0]).ax_.set_title("Baseline")
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[1]).ax_.set_title("Regresión Logística")
plt.savefig("../outputs/figures/matrices_confusion.png", dpi=150)
```

### 5.7 Curva ROC

```python
from sklearn.metrics import RocCurveDisplay

fig, ax = plt.subplots(figsize=(8, 6))
RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax, name="Regresión Logística")
ax.plot([0, 1], [0, 1], "k--", label="Baseline (random)")
ax.set_title("Curva ROC")
ax.legend()
plt.savefig("../outputs/figures/curva_roc.png", dpi=150)
```

### 5.8 Interpretar coeficientes

```python
import numpy as np

# Extraer nombres de features después del preprocesamiento
ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
cat_feature_names = ohe.get_feature_names_out(categorical_features)
all_feature_names = numerical_features + list(cat_feature_names)

# Coeficientes del modelo
coefs = pipeline.named_steps["classifier"].coef_[0]
coef_df = pd.DataFrame({
    "feature": all_feature_names,
    "coefficient": coefs
}).sort_values("coefficient", ascending=False)

sns.barplot(data=coef_df, x="coefficient", y="feature")
plt.title("Importancia de variables (coeficientes)")
plt.axvline(0, color="black", linewidth=0.8)
plt.savefig("../outputs/figures/coeficientes_modelo.png", dpi=150, bbox_inches="tight")
```

### 5.9 Guardar métricas

```python
import json
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

metrics = {
    "baseline_accuracy": round(accuracy_score(y_test, y_pred_dummy), 4),
    "model_accuracy": round(accuracy_score(y_test, y_pred), 4),
    "precision": round(precision_score(y_test, y_pred), 4),
    "recall": round(recall_score(y_test, y_pred), 4),
    "f1_score": round(f1_score(y_test, y_pred), 4),
    "auc_roc": round(roc_auc_score(y_test, y_prob), 4),
    "train_size": len(X_train),
    "test_size": len(X_test)
}

with open("../outputs/metrics/supervised_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(json.dumps(metrics, indent=2))
```

### 5.10 Serializar modelo

```python
import joblib

joblib.dump(pipeline, "../outputs/models/logistic_regression_model.joblib")
print("Modelo guardado.")
```

### 5.11 Guardar metadatos del experimento

```python
metadata = {
    "dataset": "results.csv + WorldCupMatches.csv",
    "execution_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
    "model_version": "1.0",
    "algorithm": "LogisticRegression",
    "features": feature_cols,
    "target": target_col,
    "model_params": pipeline.named_steps["classifier"].get_params(),
    **metrics
}

with open("../outputs/models/metadata_model.json", "w") as f:
    json.dump(metadata, f, indent=2)
```

**✅ Entregables de la Fase 5:**
- `outputs/figures/matrices_confusion.png`
- `outputs/figures/curva_roc.png`
- `outputs/figures/coeficientes_modelo.png`
- `outputs/metrics/supervised_metrics.json`
- `outputs/models/logistic_regression_model.joblib`
- `outputs/models/metadata_model.json`
- Interpretación escrita de coeficientes en el notebook.

---

## FASE 6 — Modelo No Supervisado (Clustering)

**Objetivo:** Agrupar las ~22 ediciones del mundial e identificar perfiles de torneos.

**Notebook:** `notebooks/06_clustering.ipynb`
**Script resultado:** `src/clustering_model.py`
**Entregables:** Figuras de clustering y modelo serializado.

### 6.1 Construir dataset por edición

```python
df = pd.read_csv("../data/processed/world_cup_matches_clean.csv")

editions_stats = df.groupby("year").agg(
    avg_total_goals=("total_goals", "mean"),
    avg_goal_diff=("goal_difference", lambda x: x.abs().mean()),
    n_matches=("total_goals", "count"),
).reset_index()

# Agregar rendimiento por confederación
# (proporción de equipos UEFA, CONMEBOL, etc. que avanzaron)
# ... lógica de aggregación usando supervised_dataset.csv

print(f"Ediciones disponibles: {len(editions_stats)}")  # ~22 esperadas
```

### 6.2 Escalar features

```python
from sklearn.preprocessing import StandardScaler

cluster_features = ["avg_total_goals", "avg_goal_diff", ...]  # definir columnas finales
X_cluster = editions_stats[cluster_features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)
```

### 6.3 K-Means — Método del codo

```python
from sklearn.cluster import KMeans

inertias = []
K_range = range(2, 9)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertias, marker="o")
plt.xlabel("Número de clusters (k)")
plt.ylabel("Inercia")
plt.title("Método del codo — K-Means")
plt.savefig("../outputs/figures/kmeans_codo.png", dpi=150)
```

### 6.4 K-Means — Coeficiente de silueta

```python
from sklearn.metrics import silhouette_score

silhouettes = []
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouettes.append(score)
    print(f"k={k}: silhouette={score:.4f}")

best_k = K_range[silhouettes.index(max(silhouettes))]
print(f"\nMejor k según silueta: {best_k}")
```

### 6.5 Clustering Jerárquico — Dendrograma

```python
from scipy.cluster.hierarchy import dendrogram, linkage

Z = linkage(X_scaled, method="ward")

plt.figure(figsize=(12, 6))
dendrogram(Z, labels=editions_stats["year"].astype(str).tolist(), leaf_rotation=45)
plt.title("Dendrograma — Clustering Jerárquico (Ward)")
plt.xlabel("Edición del Mundial")
plt.ylabel("Distancia")
plt.savefig("../outputs/figures/dendrograma.png", dpi=150, bbox_inches="tight")
```

### 6.6 Entrenar modelo final

```python
from sklearn.cluster import KMeans, AgglomerativeClustering

# K-Means final con mejor k
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
editions_stats["cluster_kmeans"] = kmeans_final.fit_predict(X_scaled)

# Clustering jerárquico con mismo k
hier = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
editions_stats["cluster_hier"] = hier.fit_predict(X_scaled)

# Comparar asignaciones
print(editions_stats[["year", "cluster_kmeans", "cluster_hier"]])
```

### 6.7 Visualizar clusters con PCA 2D

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

editions_stats["pca1"] = X_pca[:, 0]
editions_stats["pca2"] = X_pca[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, col, title in zip(axes,
                           ["cluster_kmeans", "cluster_hier"],
                           ["K-Means", "Jerárquico"]):
    sns.scatterplot(data=editions_stats, x="pca1", y="pca2",
                    hue=col, palette="Set2", ax=ax, s=100)
    for _, row in editions_stats.iterrows():
        ax.annotate(str(int(row["year"])), (row["pca1"], row["pca2"]),
                    fontsize=8, ha="center", va="bottom")
    ax.set_title(f"Clusters — {title}")

plt.savefig("../outputs/figures/clusters_pca.png", dpi=150, bbox_inches="tight")
```

### 6.8 Interpretar cada cluster

Para cada cluster, calcular estadísticas descriptivas:

```python
for cluster_id in sorted(editions_stats["cluster_kmeans"].unique()):
    subset = editions_stats[editions_stats["cluster_kmeans"] == cluster_id]
    print(f"\n--- Cluster {cluster_id} ---")
    print(f"Ediciones: {sorted(subset['year'].tolist())}")
    print(subset[cluster_features].mean().round(3))
```

Escribir en el notebook una interpretación cualitativa de cada cluster.

### 6.9 Serializar modelos

```python
joblib.dump(kmeans_final, "../outputs/models/kmeans_model.joblib")
joblib.dump(hier, "../outputs/models/hierarchical_clustering_model.joblib")
joblib.dump(scaler, "../outputs/models/cluster_scaler.joblib")
editions_stats.to_csv("../data/processed/world_cup_editions_clusters.csv", index=False)
```

**✅ Entregables de la Fase 6:**
- `outputs/figures/kmeans_codo.png`
- `outputs/figures/dendrograma.png`
- `outputs/figures/clusters_pca.png`
- `outputs/models/kmeans_model.joblib`
- `outputs/models/hierarchical_clustering_model.joblib`
- `data/processed/world_cup_editions_clusters.csv`
- Interpretación escrita de cada cluster en el notebook.

---

## FASE 7 — Integración y Entrega Final

**Objetivo:** Consolidar todos los resultados, redactar conclusiones y preparar el producto final entregable.

**Notebook:** `notebooks/07_resultados_finales.ipynb`
**Script:** `main.py` (pipeline completo ejecutable)

### 7.1 Construir `main.py`

`main.py` debe poder ejecutar todo el pipeline de extremo a extremo:

```python
# main.py
from src.data_loader import load_world_cup_data
from src.preprocessing import clean_world_cup_data
from src.feature_engineering import build_supervised_dataset
from src.supervised_model import train_supervised_model
from src.clustering_model import train_clustering_models
from src.evaluation import generate_all_metrics
from src.visualization import generate_all_figures

if __name__ == "__main__":
    print("1. Cargando datos...")
    data = load_world_cup_data()

    print("2. Limpiando datos...")
    clean_data = clean_world_cup_data(data)

    print("3. Construyendo features...")
    supervised_df = build_supervised_dataset(clean_data)

    print("4. Entrenando modelo supervisado...")
    train_supervised_model(supervised_df)

    print("5. Entrenando modelos de clustering...")
    train_clustering_models(clean_data)

    print("6. Generando métricas y visualizaciones...")
    generate_all_metrics()
    generate_all_figures()

    print("\n✅ Pipeline completo. Resultados en outputs/")
```

### 7.2 Redactar conclusiones

En `notebooks/07_resultados_finales.ipynb`, responder cada pregunta del análisis:

**Pregunta 1:** ¿Existe ventaja observable para el continente sede?
> Comparar `is_host_region=1` vs `is_host_region=0` en términos de tasa de avance de fase de grupos. Tabular resultados por continente sede.

**Pregunta 2:** ¿Qué variables influyen más en el avance de fase?
> Analizar los coeficientes del modelo. Identificar las 3 variables con mayor peso positivo y negativo.

**Pregunta 3:** ¿Qué tipos de mundiales aparecen al agrupar ediciones?
> Describir cada cluster: qué ediciones lo componen, cuál es su perfil estadístico y cómo se interpreta futbolísticamente.

**Pregunta 4:** ¿El modelo supera al baseline?
> Comparar las métricas de ambos modelos en una tabla. Concluir si la mejora es sustancial o marginal.

### 7.3 Documentar limitaciones

Incluir sección explícita en el notebook final con las 5 limitaciones identificadas en el contexto del proyecto.

### 7.4 Verificar que todos los archivos de salida existen

```bash
# Ejecutar desde la raíz del proyecto
ls outputs/figures/
ls outputs/metrics/
ls outputs/models/
ls data/processed/
```

Resultado esperado:

```
outputs/figures/
  distribucion_goles_raw.png
  partidos_por_edicion.png
  goles_promedio_por_edicion.png
  rendimiento_por_sede.png
  balance_clases.png
  matriz_correlacion.png
  matrices_confusion.png
  curva_roc.png
  coeficientes_modelo.png
  kmeans_codo.png
  dendrograma.png
  clusters_pca.png

outputs/metrics/
  supervised_metrics.json

outputs/models/
  logistic_regression_model.joblib
  kmeans_model.joblib
  hierarchical_clustering_model.joblib
  cluster_scaler.joblib
  metadata_model.json

data/processed/
  world_cup_matches_raw.csv
  world_cup_matches_clean.csv
  supervised_dataset.csv
  world_cup_editions_clusters.csv
```

### 7.5 Actualizar `README.md`

El README debe incluir:
- Descripción del proyecto en 2-3 párrafos.
- Instrucciones de instalación (`pip install -r requirements.txt`).
- Instrucciones de descarga de datasets.
- Cómo ejecutar el pipeline completo (`python main.py`).
- Descripción de la estructura de carpetas.
- Resultados obtenidos (accuracy, AUC, número de clusters).
- Limitaciones conocidas.

**✅ Entregables de la Fase 7:**
- `main.py` ejecutable de extremo a extremo.
- `notebooks/07_resultados_finales.ipynb` con conclusiones completas.
- `README.md` actualizado.
- Todos los archivos de salida verificados.

---

## Dependencias entre fases

```
Fase 0 (Setup)
    └── Fase 1 (Carga y cruce)
            └── Fase 2 (Limpieza)
                    ├── Fase 3 (EDA)  ← puede hacerse en paralelo con Fase 4
                    └── Fase 4 (Feature Engineering)
                                └── Fase 5 (Modelo Supervisado)
                    └── Fase 6 (Clustering)  ← depende de Fase 2, independiente de Fase 5
                                └── Fase 7 (Integración)  ← depende de Fase 5 y Fase 6
```

Las Fases 3, 4 y 6 comparten el mismo insumo (dataset limpio de Fase 2) y pueden desarrollarse en paralelo una vez completada la Fase 2.

---

## Puntos de control y validación

Antes de avanzar entre fases críticas, verificar estos puntos:

| Control | Qué verificar | Cómo |
|---|---|---|
| Después de Fase 1 | Join cubrió >95% de partidos | `merged["stage"].isna().sum()` |
| Después de Fase 2 | No hay nulos en columnas críticas | `df[critical_cols].isnull().sum()` |
| Después de Fase 4 | No hay data leakage | Verificar manualmente para Brasil 1970 (sin mundiales anteriores accesibles) |
| Después de Fase 4 | Balance de clases razonable | `y.value_counts(normalize=True)` |
| Después de Fase 5 | Modelo supera baseline | Comparar accuracy y F1 |
| Después de Fase 6 | Clusters interpretables | Revisar qué años caen en cada grupo |

---

## Errores comunes a evitar

| Error | Cómo prevenirlo |
|---|---|
| Usar goles del torneo actual como features | Solo features pre-torneo en `X` |
| Calcular win_rate con todos los años incluido el actual | Filtrar `year < current_year` siempre |
| Join silencioso con muchos nulos | Verificar cobertura después de cada merge |
| Nombres de selecciones distintos entre fuentes | Aplicar `TEAM_NAME_MAP` antes de cualquier join |
| K-Means sin escalar | Siempre `StandardScaler` antes de clustering |
| Comparar modelos sin baseline | Siempre entrenar `DummyClassifier` primero |
| Sobreajuste por split sin estratificación | Usar `stratify=y` en `train_test_split` |

---

## Nota final

El orden de los notebooks refleja el flujo lógico del proyecto, pero durante el desarrollo es normal iterar hacia atrás. Si en la Fase 5 los resultados son muy pobres, puede ser necesario volver a la Fase 4 y revisar las features. Si en la Fase 6 el clustering no es interpretable, puede ser necesario revisar qué variables se agregan por edición. Eso es parte del proceso real de Machine Learning.
