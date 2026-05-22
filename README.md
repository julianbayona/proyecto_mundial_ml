# El Factor Local

Proyecto final de Machine Learning sobre el impacto de la sede y las condiciones
del torneo en los resultados historicos de la Copa Mundial de Futbol.

El proyecto analiza partidos mundialistas entre 1930 y 2014, cruza dos fuentes de
datos para obtener la fase de cada partido y construye un dataset supervisado a
nivel seleccion-edicion. El objetivo supervisado es predecir si una seleccion
avanza mas alla de la fase de grupos usando solo variables disponibles antes del
torneo.

Tambien se realiza clustering por edicion mundialista para identificar perfiles
historicos de torneos segun goles, formato, representacion regional y tasas de
avance por confederacion.

## Alcance de datos

El alcance final es:

```text
FIFA World Cup 1930-2014
```

Se trabaja hasta 2014 porque el archivo complementario `WorldCupMatches.csv`
disponible en este proyecto no contiene la columna `stage` para 2018 y 2022.

## Instalacion

Desde la raiz del proyecto:

```bash
pip install -r requirements.txt
```

Dependencias principales:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
jupyter
joblib
```

Nota para Windows: si la instalacion completa de Jupyter falla por rutas largas,
las fases de modelado pueden ejecutarse instalando al menos:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy joblib
```

## Datasets requeridos

Colocar los archivos crudos en:

```text
data/raw/
```

Archivos requeridos:

```text
data/raw/results.csv
data/raw/WorldCupMatches.csv
data/raw/WorldCups.csv
```

Fuentes:

- International Football Results from 1872 to 2024.
- FIFA World Cup Dataset.

## Ejecucion

Para ejecutar el pipeline completo:

```bash
python main.py
```

El pipeline realiza:

1. Carga y cruce de datos.
2. Limpieza.
3. Feature engineering supervisado.
4. Entrenamiento del modelo supervisado.
5. Clustering por edicion.
6. Generacion de visualizaciones de EDA.
7. Consolidacion final de metricas.

## Estructura del proyecto

```text
data/
  raw/
  processed/
docs/
notebooks/
src/
outputs/
  figures/
  metrics/
  models/
requirements.txt
README.md
main.py
```

## Modulos principales

```text
src/data_loader.py          Carga y cruce inicial de datos
src/preprocessing.py        Limpieza y columnas auxiliares
src/feature_engineering.py  Dataset supervisado sin data leakage
src/supervised_model.py     Baseline y regresion logistica
src/clustering_model.py     K-Means y clustering jerarquico
src/visualization.py        Figuras de EDA
src/evaluation.py           Consolidacion de metricas y verificacion de outputs
```

## Resultados supervisados

Dataset supervisado:

```text
shape = (424, 9)
```

Features usadas:

```text
host_continent
team_confederation
is_host_region
historical_appearances
historical_win_rate
historical_avg_goals_scored
```

Metricas:

```text
Baseline accuracy: 0.5294
Modelo accuracy:   0.6118
Precision:         0.6061
Recall:            0.5000
F1-score:          0.5479
AUC-ROC:           0.6397
```

El modelo de regresion logistica supera al baseline, aunque el desempeno es
moderado y debe interpretarse con cautela.

## Resultados de clustering

Dataset de ediciones:

```text
shape = (20, 24)
```

K-Means:

```text
best_k = 3
best_silhouette = 0.3213
```

Clusters:

```text
Cluster 0: 1934, 1938, 1954, 1958, 1962, 1966, 1970, 1974, 1978, 1982
Cluster 1: 1986, 1990, 1994, 1998, 2002, 2006, 2010, 2014
Cluster 2: 1930, 1950
```

Interpretacion resumida:

- Cluster 0: ediciones intermedias, torneos de menor tamano que los modernos y
  mayor promedio de goles.
- Cluster 1: ediciones modernas, mas equipos, mas partidos y menor promedio de
  goles.
- Cluster 2: ediciones tempranas y atipicas.

## Outputs principales

Datasets procesados:

```text
data/processed/world_cup_matches_raw.csv
data/processed/world_cup_matches_clean.csv
data/processed/supervised_dataset.csv
data/processed/world_cup_editions_clusters.csv
```

Metricas:

```text
outputs/metrics/supervised_metrics.json
outputs/metrics/clustering_metrics.json
outputs/metrics/final_summary.json
```

Modelos:

```text
outputs/models/logistic_regression_model.joblib
outputs/models/kmeans_model.joblib
outputs/models/hierarchical_clustering_model.joblib
outputs/models/cluster_scaler.joblib
outputs/models/cluster_pca.joblib
outputs/models/metadata_model.json
```

Figuras:

```text
outputs/figures/distribucion_goles_raw.png
outputs/figures/partidos_por_edicion.png
outputs/figures/goles_promedio_por_edicion.png
outputs/figures/rendimiento_por_sede.png
outputs/figures/top_selecciones_avance.png
outputs/figures/matriz_correlacion.png
outputs/figures/balance_clases.png
outputs/figures/matrices_confusion.png
outputs/figures/curva_roc.png
outputs/figures/coeficientes_modelo.png
outputs/figures/kmeans_codo.png
outputs/figures/kmeans_silueta.png
outputs/figures/dendrograma.png
outputs/figures/clusters_pca.png
```

## Notebooks

```text
notebooks/01_data_loading.ipynb
notebooks/02_cleaning.ipynb
notebooks/03_eda.ipynb
notebooks/04_feature_engineering.ipynb
notebooks/05_supervised_model.ipynb
notebooks/06_clustering.ipynb
notebooks/07_resultados_finales.ipynb
```

## Documentacion por fase

```text
docs/fases_0_1_setup_y_carga.md
docs/fase_2_limpieza_datos.md
docs/fase_3_eda.md
docs/fase_4_feature_engineering.md
docs/fase_5_modelo_supervisado.md
docs/fase_6_clustering.md
```

## Limitaciones

1. Un mismo equipo aparece en multiples mundiales, por lo que las observaciones
   no son completamente independientes.
2. El dataset de clustering tiene solo 20 ediciones en el alcance 1930-2014.
3. La normalizacion de selecciones historicas simplifica entidades que cambiaron
   politicamente o deportivamente.
4. Las features pre-torneo disponibles son limitadas y pueden ser debiles para
   selecciones con pocas apariciones previas.
5. El formato de la Copa Mundial cambio fuertemente entre 1930 y 2014.

## Conclusion

El pipeline final es reproducible y cubre carga, limpieza, EDA, feature
engineering sin data leakage, modelo supervisado con baseline y clustering con
dos algoritmos. Los resultados muestran senales descriptivas de efecto regional y
experiencia historica, pero no son suficientes para explicar completamente el
avance deportivo de las selecciones.
