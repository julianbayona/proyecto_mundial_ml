# Documentacion de avance - Fase 4

Este documento resume el trabajo realizado en la **Fase 4 - Ingenieria de
Caracteristicas** del proyecto **El Factor Local**.

La fase se ejecuto despues del analisis exploratorio de datos de la Fase 3. El
insumo utilizado fue:

```text
data/processed/world_cup_matches_clean.csv
```

El resultado principal generado fue:

```text
data/processed/supervised_dataset.csv
```

## Objetivo de la fase

El objetivo de la Fase 4 fue construir el dataset supervisado del proyecto con
una fila por:

```text
team, year
```

Es decir, una fila por seleccion participante en una edicion de la Copa Mundial.

Este dataset sera el insumo de la Fase 5, donde se entrenara el modelo baseline y
la regresion logistica.

## Regla central: no data leakage

La regla mas importante de esta fase fue evitar data leakage.

Para el modelo supervisado, las variables predictoras deben estar disponibles
antes del inicio del torneo. Por eso, las features historicas se calcularon
usando solamente partidos de ediciones anteriores:

```text
match_year < current_year
```

No se incluyeron como features columnas derivadas de resultados del torneo actual,
por ejemplo:

```text
home_score
away_score
goal_difference
total_goals
```

Estas variables pueden usarse para EDA, pero no para entrenar el modelo
supervisado.

## Archivos creados o modificados

Se modifico:

```text
src/feature_engineering.py
```

Se creo:

```text
notebooks/04_feature_engineering.ipynb
```

Se genero:

```text
data/processed/supervised_dataset.csv
```

## Funciones implementadas

En `src/feature_engineering.py` se implementaron funciones reutilizables para
construir el dataset supervisado.

Funciones principales:

```python
load_clean_world_cup_data()
build_team_editions()
compute_historical_features()
add_context_features()
build_supervised_dataset()
```

### load_clean_world_cup_data()

Carga el dataset limpio desde:

```text
data/processed/world_cup_matches_clean.csv
```

### build_team_editions()

Construye una tabla con una fila por seleccion y edicion, usando los partidos de
fase de grupos o etapa inicial equivalente.

Tambien construye la variable objetivo:

```text
advanced_group_stage
```

### compute_historical_features()

Calcula las variables historicas pre-torneo:

```text
historical_appearances
historical_win_rate
historical_avg_goals_scored
```

Para cada seleccion y mundial, esta funcion filtra solo partidos de ediciones
anteriores. Esto garantiza que no se usen resultados del torneo actual para
construir features.

### add_context_features()

Agrega variables contextuales:

```text
team_confederation
is_host_region
```

Estas variables se calculan a partir de mapas explicitos de confederacion y
continente.

### build_supervised_dataset()

Ejecuta todo el proceso:

1. Carga el dataset limpio.
2. Valida columnas requeridas.
3. Construye la tabla seleccion-edicion.
4. Calcula el target.
5. Calcula features historicas sin leakage.
6. Agrega variables contextuales.
7. Ordena las columnas finales.
8. Guarda `data/processed/supervised_dataset.csv`.

## Construccion del target

La variable objetivo es:

```text
advanced_group_stage
```

Definicion:

- `1`: la seleccion jugo al menos un partido posterior a fase de grupos en esa
  edicion.
- `0`: la seleccion solo aparece en fase de grupos o etapa inicial equivalente.

Para construirla se usaron los partidos donde:

```text
is_group_stage == False
```

Luego se identificaron los equipos que aparecieron como local o visitante en esas
rondas posteriores.

## Features historicas calculadas

Se calcularon tres variables historicas:

```text
historical_appearances
historical_win_rate
historical_avg_goals_scored
```

### historical_appearances

Cantidad de ediciones anteriores en las que la seleccion tuvo partidos de fase
inicial.

### historical_win_rate

Tasa historica de victorias en partidos de fase inicial de mundiales anteriores.

Formula conceptual:

```text
victorias historicas / partidos historicos
```

### historical_avg_goals_scored

Promedio historico de goles anotados por la seleccion en partidos de fase inicial
de mundiales anteriores.

Formula conceptual:

```text
goles historicos anotados / partidos historicos
```

## Variables contextuales agregadas

Se agregaron:

```text
host_continent
team_confederation
is_host_region
```

### host_continent

Proviene del dataset limpio de Fase 2.

### team_confederation

Se construyo usando un diccionario explicito:

```python
CONFEDERATION_MAP
```

El mapa cubre las 77 selecciones presentes en el dataset limpio.

Distribucion resultante:

```text
UEFA        231
CONMEBOL     80
CONCACAF     39
CAF          39
AFC          33
OFC           2
```

Decision tecnica:

Australia se codifico como `AFC`, siguiendo la convencion usada en el plan del
proyecto.

### is_host_region

Indicador binario que compara el continente de la confederacion del equipo con
el continente sede:

```text
1 si team_continent == host_continent
0 en caso contrario
```

Para esto se uso:

```python
CONFEDERATION_TO_CONTINENT
```

## Columnas finales del dataset supervisado

El dataset final contiene 9 columnas:

```text
year
team
host_continent
team_confederation
is_host_region
historical_appearances
historical_win_rate
historical_avg_goals_scored
advanced_group_stage
```

No contiene goles ni resultados del torneo actual como features.

## Salida generada

La ejecucion de Fase 4 produjo:

```text
Dataset supervisado: (424, 9)
```

Balance de clases:

```text
advanced_group_stage
0    225
1    199
```

Proporcion:

```text
0    0.5307
1    0.4693
```

El balance es razonable para entrenar modelos supervisados con division
estratificada en la Fase 5.

## Validaciones realizadas

Se validaron los siguientes puntos:

```text
shape = (424, 9)
columns_ok = True
duplicate_team_year = 0
nulls_total = 0
first_year_hist_sum = 0.0
features_no_current_goals = True
syntax_and_notebook_json_ok
```

### Interpretacion de validaciones

- `duplicate_team_year = 0`: hay una sola fila por seleccion y edicion.
- `nulls_total = 0`: no hay valores nulos en el dataset supervisado.
- `first_year_hist_sum = 0.0`: las selecciones de 1930 no tienen historial previo,
  como corresponde.
- `features_no_current_goals = True`: el dataset no contiene columnas de goles o
  resultados del torneo actual como features.

## Verificacion manual de no leakage

Se reviso el caso de Brasil.

Para 1930:

```text
Brazil 1930:
historical_appearances = 0
historical_win_rate = 0.0
historical_avg_goals_scored = 0.0
```

Esto es correcto porque 1930 es la primera edicion del dataset.

Para 1970:

```text
Brazil 1970:
historical_appearances = 8
historical_win_rate = 0.5714
historical_avg_goals_scored = 2.5238
advanced_group_stage = 1
```

Estas features se calcularon usando solamente ediciones anteriores a 1970.

## Notebook creado

Se creo:

```text
notebooks/04_feature_engineering.ipynb
```

El notebook:

1. Carga el dataset limpio.
2. Ejecuta `build_supervised_dataset()`.
3. Muestra las primeras filas.
4. Revisa tipos de datos y nulos.
5. Revisa balance de clases.
6. Incluye verificaciones manuales de no leakage.

## Entregables de Fase 4

Quedaron listos:

```text
src/feature_engineering.py
notebooks/04_feature_engineering.ipynb
data/processed/supervised_dataset.csv
```

## Estado antes de Fase 5

La Fase 4 queda cerrada. El proyecto esta listo para iniciar la Fase 5 de modelo
supervisado usando como insumo:

```text
data/processed/supervised_dataset.csv
```

En la Fase 5 se debera:

1. Separar `X` e `y`.
2. Usar `train_test_split` con `stratify=y`.
3. Entrenar un `DummyClassifier` como baseline.
4. Entrenar una regresion logistica con `Pipeline`.
5. Guardar metricas, figuras y modelo serializado.

No se ha avanzado todavia a Fase 5.
