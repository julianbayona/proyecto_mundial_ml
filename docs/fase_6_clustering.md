# Documentacion de avance - Fase 6

Este documento resume el trabajo realizado en la **Fase 6 - Modelo No Supervisado
(Clustering)** del proyecto **El Factor Local**.

La fase se ejecuto despues del entrenamiento del modelo supervisado de la Fase 5.
Los insumos utilizados fueron:

```text
data/processed/world_cup_matches_clean.csv
data/processed/supervised_dataset.csv
```

El resultado principal generado fue:

```text
data/processed/world_cup_editions_clusters.csv
```

## Objetivo de la fase

El objetivo de la Fase 6 fue agrupar ediciones mundialistas segun perfiles
estadisticos del torneo.

La granularidad del dataset de clustering es:

```text
una fila por edicion del Mundial
```

Dentro del alcance actual del proyecto, el dataset contiene 20 ediciones entre
1930 y 2014.

## Consideracion metodologica

El dataset de clustering es pequeno:

```text
n_editions = 20
```

Por eso, los resultados deben interpretarse con cautela. K-Means se uso como
analisis complementario y el clustering jerarquico se uso para revisar la
estructura general de similitudes entre torneos.

## Archivos creados o modificados

Se modifico:

```text
src/clustering_model.py
```

Se creo:

```text
notebooks/06_clustering.ipynb
```

Se generaron outputs en:

```text
data/processed/
outputs/figures/
outputs/metrics/
outputs/models/
```

## Funciones implementadas

En `src/clustering_model.py` se implementaron funciones para construir el dataset
agregado por edicion, evaluar K-Means, entrenar modelos y guardar resultados.

Funciones principales:

```python
load_clustering_inputs()
build_world_cup_editions_dataset()
evaluate_kmeans_range()
train_clustering_models()
```

Tambien se implementaron funciones auxiliares para:

- guardar el grafico del metodo del codo
- guardar el grafico de silueta
- guardar el dendrograma
- guardar el scatter de clusters con PCA
- guardar metricas en JSON
- serializar modelos con `joblib`

## Dataset agregado por edicion

Se construyo:

```text
data/processed/world_cup_editions_clusters.csv
```

Salida validada:

```text
shape = (20, 24)
years = 1930 2014
nulls_total = 0
```

## Variables usadas para clustering

Se usaron 19 features de clustering:

```text
avg_total_goals
avg_abs_goal_diff
n_matches
n_teams
group_stage_share
overall_advance_rate
host_region_team_share
share_UEFA
share_CONMEBOL
share_CONCACAF
share_CAF
share_AFC
share_OFC
advance_rate_UEFA
advance_rate_CONMEBOL
advance_rate_CONCACAF
advance_rate_CAF
advance_rate_AFC
advance_rate_OFC
```

### Variables de partidos

Se agregaron desde `world_cup_matches_clean.csv`:

```text
avg_total_goals
avg_abs_goal_diff
n_matches
group_stage_share
```

### Variables de selecciones y avance

Se agregaron desde `supervised_dataset.csv`:

```text
n_teams
overall_advance_rate
host_region_team_share
```

### Representacion por confederacion

Se calcularon proporciones de equipos por confederacion:

```text
share_UEFA
share_CONMEBOL
share_CONCACAF
share_CAF
share_AFC
share_OFC
```

### Rendimiento por confederacion

Se calcularon tasas de avance por confederacion:

```text
advance_rate_UEFA
advance_rate_CONMEBOL
advance_rate_CONCACAF
advance_rate_CAF
advance_rate_AFC
advance_rate_OFC
```

## Decision tecnica sobre indice de sorpresas

El plan mencionaba un posible indice de sorpresas, pero no definia formalmente
que es un equipo "debil" ni como calcularlo de forma reproducible.

Por esa razon, no se implemento un indice de sorpresas en esta fase. Se priorizo
una implementacion clara y defendible con variables agregadas directamente
observables.

## Escalado

Antes de entrenar modelos se aplico:

```python
StandardScaler()
```

Esto es necesario porque las variables tienen escalas distintas, por ejemplo:

- numero de partidos
- tasas entre 0 y 1
- promedios de goles

El scaler se guardo en:

```text
outputs/models/cluster_scaler.joblib
```

## Evaluacion de K-Means

Se evaluaron valores de `k` entre 2 y 8.

Resultado:

```text
k  inertia   silhouette
2  233.0671  0.2902
3  178.7913  0.3213
4  138.9021  0.2968
5  119.9644  0.2904
6  103.4658  0.1908
7   86.8561  0.1903
8   71.8008  0.2138
```

El mejor valor segun silueta fue:

```text
best_k = 3
best_silhouette = 0.3213
```

La silueta no es alta, por lo que los clusters deben interpretarse como perfiles
descriptivos aproximados, no como particiones fuertes o definitivas.

## Modelos entrenados

Se entrenaron dos modelos:

```text
KMeans(n_clusters=3, random_state=42, n_init=10)
AgglomerativeClustering(n_clusters=3, linkage="ward")
```

Modelos guardados:

```text
outputs/models/kmeans_model.joblib
outputs/models/hierarchical_clustering_model.joblib
```

Tambien se guardo el objeto PCA usado para visualizacion:

```text
outputs/models/cluster_pca.joblib
```

## Asignaciones de clusters

Las asignaciones de K-Means y clustering jerarquico coincidieron completamente:

```text
same_assignments = True
```

Asignaciones:

```text
year  cluster_kmeans  cluster_hier
1930               2             2
1934               0             0
1938               0             0
1950               2             2
1954               0             0
1958               0             0
1962               0             0
1966               0             0
1970               0             0
1974               0             0
1978               0             0
1982               0             0
1986               1             1
1990               1             1
1994               1             1
1998               1             1
2002               1             1
2006               1             1
2010               1             1
2014               1             1
```

Conteo por cluster:

```text
cluster 0: 10 ediciones
cluster 1:  8 ediciones
cluster 2:  2 ediciones
```

## Interpretacion preliminar de clusters

### Cluster 0

Ediciones:

```text
1934, 1938, 1954, 1958, 1962, 1966, 1970, 1974, 1978, 1982
```

Promedios principales:

```text
avg_total_goals            3.434
avg_abs_goal_diff          1.729
n_matches                 32.000
n_teams                   16.600
group_stage_share          0.750
overall_advance_rate       0.417
host_region_team_share     0.543
```

Interpretacion:

Este cluster agrupa principalmente ediciones intermedias, con torneos mas pequenos
que los modernos y un promedio de goles relativamente alto. Puede interpretarse
como un perfil de mundiales de formato clasico o de transicion.

### Cluster 1

Ediciones:

```text
1986, 1990, 1994, 1998, 2002, 2006, 2010, 2014
```

Promedios principales:

```text
avg_total_goals            2.485
avg_abs_goal_diff          1.334
n_matches                 59.500
n_teams                   29.000
group_stage_share          0.728
overall_advance_rate       0.562
host_region_team_share     0.270
```

Interpretacion:

Este cluster representa ediciones modernas, con mas equipos, mas partidos y menor
promedio de goles. Tambien muestra una mayor tasa general de avance por la
estructura moderna del torneo.

### Cluster 2

Ediciones:

```text
1930, 1950
```

Promedios principales:

```text
avg_total_goals            3.944
avg_abs_goal_diff          2.470
n_matches                 20.000
n_teams                   13.000
group_stage_share          0.917
overall_advance_rate       0.154
host_region_team_share     0.462
```

Interpretacion:

Este cluster agrupa ediciones tempranas y atipicas. Tienen pocos equipos, pocos
partidos y alta diferencia media de goles. El formato historico de estas ediciones
las separa claramente del resto.

## Figuras generadas

Se generaron cuatro figuras:

```text
outputs/figures/kmeans_codo.png
outputs/figures/kmeans_silueta.png
outputs/figures/dendrograma.png
outputs/figures/clusters_pca.png
```

### Metodo del codo

Archivo:

```text
outputs/figures/kmeans_codo.png
```

Muestra la inercia para distintos valores de `k`.

### Coeficiente de silueta

Archivo:

```text
outputs/figures/kmeans_silueta.png
```

Muestra la silueta para valores de `k` entre 2 y 8.

### Dendrograma

Archivo:

```text
outputs/figures/dendrograma.png
```

Visualiza la estructura jerarquica usando metodo Ward.

### Clusters con PCA

Archivo:

```text
outputs/figures/clusters_pca.png
```

Muestra una visualizacion 2D de los clusters usando PCA.

Varianza explicada por PCA:

```text
PCA 1: 0.3925
PCA 2: 0.2094
```

## Metricas guardadas

Se guardaron metricas en:

```text
outputs/metrics/clustering_metrics.json
```

El archivo incluye:

- `best_k`
- `best_silhouette`
- evaluacion de K-Means para cada valor de `k`
- lista de features usadas
- numero de ediciones
- varianza explicada por PCA

## Validaciones realizadas

Se valido:

```text
syntax_notebook_json_metrics_ok
shape = (20, 24)
years = 1930 2014
nulls_total = 0
best_k = 3
best_silhouette = 0.3213
same_assignments = True
kmeans_clusters = 3
hier_clusters = 3
scaler_features = 19
```

Tambien se verifico que los objetos serializados pueden cargarse con `joblib`.

## Nota tecnica

Durante la ejecucion aparecio una advertencia no bloqueante de `joblib` sobre la
deteccion del numero de nucleos fisicos del sistema.

La advertencia no afecto el entrenamiento ni la generacion de archivos.

## Notebook creado

Se creo:

```text
notebooks/06_clustering.ipynb
```

El notebook:

1. Carga los insumos de clustering.
2. Construye el dataset agregado por edicion.
3. Ejecuta `train_clustering_models()`.
4. Muestra metricas de K-Means.
5. Muestra asignaciones de clusters.
6. Incluye interpretacion preliminar.

## Entregables de Fase 6

Quedaron listos:

```text
src/clustering_model.py
notebooks/06_clustering.ipynb
data/processed/world_cup_editions_clusters.csv
outputs/metrics/clustering_metrics.json
outputs/models/kmeans_model.joblib
outputs/models/hierarchical_clustering_model.joblib
outputs/models/cluster_scaler.joblib
outputs/models/cluster_pca.joblib
outputs/figures/kmeans_codo.png
outputs/figures/kmeans_silueta.png
outputs/figures/dendrograma.png
outputs/figures/clusters_pca.png
```

## Estado antes de Fase 7

La Fase 6 queda cerrada. El proyecto esta listo para iniciar la Fase 7 de
integracion y entrega final.

En la Fase 7 se debera:

1. Integrar el pipeline completo en `main.py`.
2. Crear el notebook de resultados finales.
3. Actualizar el `README.md`.
4. Verificar que todos los outputs esperados existan.
5. Redactar conclusiones y limitaciones.

No se ha avanzado todavia a Fase 7.
