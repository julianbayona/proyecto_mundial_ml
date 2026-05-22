# Documentacion de avance - Fase 3

Este documento resume el trabajo realizado en la **Fase 3 - Analisis Exploratorio
de Datos (EDA)** del proyecto **El Factor Local**.

La fase se ejecuto despues de completar la limpieza de datos de la Fase 2. El
insumo utilizado fue:

```text
data/processed/world_cup_matches_clean.csv
```

## Objetivo de la fase

El objetivo de la Fase 3 fue explorar el dataset limpio, generar visualizaciones
descriptivas y obtener primeras conclusiones sobre los patrones historicos de la
Copa Mundial entre 1930 y 2014.

En esta fase no se entreno ningun modelo. El analisis fue descriptivo y sirvio
como preparacion para las fases posteriores de feature engineering y modelado.

## Archivos creados o modificados

Se modifico:

```text
src/visualization.py
```

Se creo:

```text
notebooks/03_eda.ipynb
```

Tambien se generaron figuras en:

```text
outputs/figures/
```

## Logica implementada en src/visualization.py

Se implementaron funciones reutilizables para generar las visualizaciones
principales del EDA.

Funciones principales:

```python
load_clean_data()
build_team_edition_summary()
plot_goal_distribution()
plot_matches_by_edition()
plot_average_goals_by_edition()
plot_host_region_performance()
plot_top_teams_advance()
plot_correlation_matrix()
plot_class_balance()
generate_all_figures()
```

La funcion `generate_all_figures()` ejecuta todas las visualizaciones de la fase
y devuelve tablas resumen usadas para interpretar los resultados.

## Dataset utilizado

El dataset limpio tiene:

```text
shape = (836, 15)
years = 1930 2014
```

Columnas principales usadas en el EDA:

```text
year
home_team
away_team
home_score
away_score
stage
is_group_stage
host_continent
goal_difference
total_goals
```

## Tabla temporal seleccion-edicion

Para algunas visualizaciones fue necesario construir una tabla temporal con una
fila por seleccion y edicion:

```text
Dataset temporal seleccion-edicion: (424, 6)
```

Esta tabla se genero con `build_team_edition_summary()` y contiene:

```text
year
team
advanced_group_stage
host_continent
team_continent
is_host_region
```

La variable `advanced_group_stage` se construyo solo para exploracion descriptiva
en esta fase. El dataset supervisado definitivo se construira en la Fase 4.

## Decision tecnica: continentes por seleccion

Para calcular el rendimiento de equipos del continente sede fue necesario asignar
un continente a cada seleccion.

Se creo un diccionario explicito:

```python
TEAM_CONTINENT_MAP
```

Este diccionario se uso solo para el EDA. Su objetivo fue comparar:

- equipos de la misma region continental de la sede
- equipos de otras regiones

Esta decision permite analizar descriptivamente la pregunta central del proyecto
sin adelantar todavia la ingenieria formal de variables de la Fase 4.

## Figuras generadas

Se generaron siete figuras:

```text
outputs/figures/distribucion_goles_raw.png
outputs/figures/partidos_por_edicion.png
outputs/figures/goles_promedio_por_edicion.png
outputs/figures/rendimiento_por_sede.png
outputs/figures/top_selecciones_avance.png
outputs/figures/matriz_correlacion.png
outputs/figures/balance_clases.png
```

### 1. Distribucion de goles

Archivo:

```text
outputs/figures/distribucion_goles_raw.png
```

Muestra histogramas de goles anotados por equipos locales y visitantes.

Esta figura permite revisar visualmente la distribucion de marcadores y detectar
valores atipicos.

### 2. Partidos por edicion

Archivo:

```text
outputs/figures/partidos_por_edicion.png
```

Muestra la cantidad de partidos en cada mundial.

Resultado general:

- Las primeras ediciones tienen menos partidos.
- El numero de partidos crece con la expansion historica del torneo.
- Desde 1998 hasta 2014 se observan 64 partidos por edicion.

### 3. Goles promedio por edicion

Archivo:

```text
outputs/figures/goles_promedio_por_edicion.png
```

Muestra la evolucion del promedio de goles por partido.

Valores destacados:

```text
Mayor promedio: 1954 con 5.385 goles por partido
Menor promedio: 1990 con 2.212 goles por partido
```

Esto sugiere diferencias historicas importantes en formatos, estilos de juego y
contextos competitivos.

### 4. Rendimiento por relacion con continente sede

Archivo:

```text
outputs/figures/rendimiento_por_sede.png
```

Compara la tasa de avance de fase entre equipos del continente sede y equipos de
otras regiones.

Resultado:

```text
Otra region:
  advance_rate = 0.436
  equipos-edicion = 259

Region sede:
  advance_rate = 0.521
  equipos-edicion = 165
```

Interpretacion preliminar:

Los equipos del continente sede muestran una mayor tasa de avance en esta muestra
descriptiva. Esto no demuestra causalidad, pero justifica analizar la variable
`is_host_region` en fases posteriores.

### 5. Top selecciones por avances de fase

Archivo:

```text
outputs/figures/top_selecciones_avance.png
```

Ranking generado:

```text
Germany        16 avances / 18 apariciones
Brazil         15 avances / 20 apariciones
Italy          11 avances / 18 apariciones
England        10 avances / 14 apariciones
Argentina      10 avances / 16 apariciones
Netherlands     8 avances / 10 apariciones
Uruguay         8 avances / 12 apariciones
Mexico          8 avances / 15 apariciones
France          7 avances / 14 apariciones
Spain           7 avances / 14 apariciones
```

Este resultado refleja la presencia historica de selecciones tradicionalmente
fuertes en rondas posteriores.

### 6. Matriz de correlacion

Archivo:

```text
outputs/figures/matriz_correlacion.png
```

Variables numericas analizadas:

```text
home_score
away_score
total_goals
goal_difference
year
```

Correlaciones principales:

```text
home_score vs total_goals:        0.729
away_score vs total_goals:        0.643
home_score vs goal_difference:    0.762
away_score vs goal_difference:   -0.689
year vs total_goals:             -0.271
```

Interpretacion preliminar:

La correlacion negativa entre `year` y `total_goals` sugiere que, historicamente,
el promedio de goles por partido tiende a ser menor en ediciones mas recientes.

### 7. Balance de clases

Archivo:

```text
outputs/figures/balance_clases.png
```

Se construyo una variable temporal de avance para revisar el balance de clases.

Resultado:

```text
No avanzo: 225
Avanzo:    199
```

Interpretacion preliminar:

El balance es razonable para una primera aproximacion supervisada. En fases
posteriores se debera usar `stratify=y` al dividir datos de entrenamiento y prueba.

## Conclusiones preliminares agregadas al notebook

El notebook `notebooks/03_eda.ipynb` incluye una seccion Markdown con conclusiones
preliminares:

1. El dataset contiene 836 partidos entre 1930 y 2014.
2. La cantidad de partidos por edicion aumenta con los cambios historicos de
   formato.
3. El promedio de goles fue mas alto en ediciones antiguas y mas bajo en algunos
   torneos modernos.
4. La variable temporal de avance esta razonablemente balanceada.
5. Los equipos de la region sede muestran mayor tasa de avance de forma
   descriptiva.
6. `goal_difference` y `total_goals` son utiles para EDA, pero no deben usarse
   como features supervisadas por riesgo de data leakage.

## Validacion tecnica

Se verifico que:

```text
syntax_and_notebook_json_ok
missing_figures = []
figure_count = 7
```

Esto confirma que:

- Los archivos Python tienen sintaxis valida.
- El notebook `03_eda.ipynb` es JSON valido.
- Todas las figuras esperadas fueron generadas.

## Entregables de Fase 3

Quedaron listos:

```text
src/visualization.py
notebooks/03_eda.ipynb
outputs/figures/distribucion_goles_raw.png
outputs/figures/partidos_por_edicion.png
outputs/figures/goles_promedio_por_edicion.png
outputs/figures/rendimiento_por_sede.png
outputs/figures/top_selecciones_avance.png
outputs/figures/matriz_correlacion.png
outputs/figures/balance_clases.png
```

## Estado antes de Fase 4

La Fase 3 queda cerrada. El proyecto esta listo para iniciar la Fase 4 de
ingenieria de caracteristicas usando como insumo:

```text
data/processed/world_cup_matches_clean.csv
```

No se ha avanzado todavia a Fase 4.
