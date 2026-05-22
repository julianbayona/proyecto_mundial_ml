# Documentacion de avance - Fase 2

Este documento resume el trabajo realizado en la **Fase 2 - Limpieza de datos**
del proyecto **El Factor Local**.

La fase se ejecuto despues de completar la carga y cruce inicial de datos de la
Fase 1. El insumo utilizado fue:

```text
data/processed/world_cup_matches_raw.csv
```

El resultado principal generado fue:

```text
data/processed/world_cup_matches_clean.csv
```

## Objetivo de la fase

El objetivo de la Fase 2 fue construir un dataset limpio, consistente y listo
para el analisis exploratorio y las fases posteriores de feature engineering.

Las tareas principales fueron:

1. Revisar nulos.
2. Revisar duplicados.
3. Validar valores de `stage`.
4. Identificar partidos de fase de grupos.
5. Revisar valores atipicos en goles.
6. Crear columnas auxiliares para el analisis.
7. Guardar el dataset limpio.
8. Guardar una primera figura de distribucion de goles.

## Archivo implementado

La logica de limpieza se implemento en:

```text
src/preprocessing.py
```

Se creo la funcion:

```python
clean_world_cup_data()
```

Esta funcion puede recibir un DataFrame ya cargado o leer directamente el archivo
crudo cruzado desde:

```text
data/processed/world_cup_matches_raw.csv
```

Por defecto, guarda el dataset limpio en:

```text
data/processed/world_cup_matches_clean.csv
```

Y guarda la figura de distribucion de goles en:

```text
outputs/figures/distribucion_goles_raw.png
```

## Notebook creado

Se creo el notebook:

```text
notebooks/02_cleaning.ipynb
```

El notebook no contiene logica compleja. Su proposito es llamar la funcion
`clean_world_cup_data()`, validar las columnas importantes y mostrar resultados
basicos. Esto mantiene la regla de modularidad del proyecto:

- La logica vive en `src/`.
- Los notebooks se usan para ejecutar, revisar e interpretar.

## Validacion de nulos

Se revisaron los nulos en todas las columnas del dataset crudo cruzado.

Resultado:

```text
date          0
home_team     0
away_team     0
home_score    0
away_score    0
tournament    0
city          0
country       0
neutral       0
year          0
stage         0
```

Las columnas criticas definidas para esta fase fueron:

```text
home_team
away_team
home_score
away_score
year
stage
```

Todas quedaron sin valores nulos.

## Revision de duplicados

Se revisaron tres niveles de duplicados:

1. Duplicados exactos de fila completa.
2. Duplicados por identidad completa del partido.
3. Filas con la misma llave `(year, home_team, away_team)`.

Resultado:

```text
Duplicados exactos encontrados: 0
Duplicados por identidad completa del partido: 0
Filas con misma llave (year, home_team, away_team) preservadas como posibles rematches: 24
```

### Decision tecnica sobre duplicados

No se eliminaron filas solo por compartir la llave:

```text
year, home_team, away_team
```

La razon es que en mundiales antiguos existen partidos repetidos entre las mismas
selecciones dentro de la misma edicion, por ejemplo desempates o rematches. Esos
partidos son historicamente validos y deben conservarse.

Por eso, la limpieza solo elimina duplicados exactos o duplicados por identidad
completa del partido, considerando columnas como fecha, equipos, marcador y
etapa. En esta ejecucion no fue necesario eliminar ninguna fila.

## Validacion de stages

Se revisaron los valores de la columna `stage`.

Conteo principal:

```text
Round of 16                 64
Quarter-finals              62
Group 1                     62
Group A                     60
Group B                     60
Group 2                     59
Group 3                     56
Group 4                     55
Group E                     48
Group D                     48
Group F                     48
Group C                     48
Semi-finals                 34
Group H                     30
Group G                     30
Final                       19
Match for third place       15
Group 6                     12
First round                  9
Preliminary round            8
Group 5                      6
Third place                  2
Play-off for third place     1
```

La columna `stage` quedo presente y sin nulos para todos los partidos.

## Creacion de is_group_stage

Se creo la columna:

```text
is_group_stage
```

Esta columna identifica partidos de fase de grupos o primeras rondas equivalentes
en formatos historicos.

Se usaron las siguientes palabras clave:

```text
Group
First round
Preliminary round
Pool
```

Resultado:

```text
True     639
False    197
```

Esto significa que, dentro del alcance 1930-2014:

- 639 partidos se consideran fase de grupos o etapa equivalente.
- 197 partidos se consideran rondas posteriores.

Esta columna sera importante para construir la variable objetivo en fases
posteriores.

## Creacion de host_continent

Se creo la columna:

```text
host_continent
```

La columna se construyo usando un diccionario manual por anio de mundial:

```text
1930 -> South America
1934 -> Europe
1938 -> Europe
1950 -> South America
1954 -> Europe
1958 -> Europe
1962 -> South America
1966 -> Europe
1970 -> North America
1974 -> Europe
1978 -> South America
1982 -> Europe
1986 -> North America
1990 -> Europe
1994 -> North America
1998 -> Europe
2002 -> Asia
2006 -> Europe
2010 -> Africa
2014 -> South America
```

Validacion:

```text
host_continent_nulls = 0
```

Todas las ediciones del alcance 1930-2014 quedaron mapeadas.

## Revision de goles

Se validaron marcadores negativos y valores extremadamente altos.

Resultado:

```text
Marcadores con goles negativos: 0
Marcadores con algun equipo >20 goles: 0
goal_min_max = 0 12
```

No se detectaron marcadores imposibles.

## Columnas auxiliares creadas

Se agregaron dos columnas para analisis exploratorio:

```text
goal_difference
total_goals
```

Definicion:

```text
goal_difference = home_score - away_score
total_goals = home_score + away_score
```

Estas variables se crean para EDA y analisis descriptivo. No deben usarse como
features del modelo supervisado, porque representan informacion del partido ya
jugado y podrian introducir data leakage.

## Figura generada

Se genero la figura:

```text
outputs/figures/distribucion_goles_raw.png
```

La figura contiene histogramas de:

- goles del equipo local
- goles del equipo visitante

Esta visualizacion sirve como primera revision de la distribucion de marcadores.

## Dataset limpio generado

El dataset limpio se guardo en:

```text
data/processed/world_cup_matches_clean.csv
```

Salida validada:

```text
shape = (836, 15)
years = 1930 2014
```

Columnas finales:

```text
date
home_team
away_team
home_score
away_score
tournament
city
country
neutral
year
stage
is_group_stage
host_continent
goal_difference
total_goals
```

## Validacion tecnica

Se valido que:

```text
syntax_and_notebook_json_ok
```

Esto confirma que:

- Los archivos Python tienen sintaxis valida.
- El notebook `notebooks/02_cleaning.ipynb` es JSON valido.

## Entregables de Fase 2

Quedaron listos los siguientes entregables:

```text
src/preprocessing.py
notebooks/02_cleaning.ipynb
data/processed/world_cup_matches_clean.csv
outputs/figures/distribucion_goles_raw.png
```

## Estado antes de Fase 3

La Fase 2 queda cerrada. El proyecto esta listo para iniciar la Fase 3 de analisis
exploratorio usando como insumo:

```text
data/processed/world_cup_matches_clean.csv
```

No se ha avanzado todavia a Fase 3.
