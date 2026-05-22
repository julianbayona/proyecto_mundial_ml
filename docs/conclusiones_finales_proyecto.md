# Conclusiones finales del proyecto

Este documento resume las conclusiones globales del proyecto **El Factor Local:
Analisis del Impacto de la Sede y las Condiciones del Torneo en los Resultados
de la Copa Mundial de Futbol**.

El objetivo de este documento no es repetir cada fase tecnica, sino interpretar si
el proyecto cumplio su proposito, que resultados son defendibles, que limitaciones
existen y que pudo haber influido en los hallazgos.

## Cumplimiento del objetivo general

El proyecto si cumplio su objetivo general desde una perspectiva academica y
tecnica.

Se logro construir un flujo reproducible de Machine Learning que parte de datos
crudos, cruza fuentes complementarias, limpia los datos, construye variables
sin data leakage, entrena modelos supervisados y aplica tecnicas no supervisadas
para analizar perfiles de ediciones mundialistas.

En concreto, se logro:

1. Cargar y cruzar `results.csv` con `WorldCupMatches.csv`.
2. Incorporar la columna `stage` al dataset de partidos.
3. Construir un dataset limpio de partidos entre 1930 y 2014.
4. Crear un dataset supervisado con una fila por seleccion y edicion.
5. Calcular variables historicas usando solo ediciones anteriores.
6. Entrenar un baseline con `DummyClassifier`.
7. Entrenar una regresion logistica interpretable.
8. Evaluar el modelo supervisado con metricas estandar.
9. Aplicar K-Means y clustering jerarquico a nivel de edicion mundialista.
10. Generar visualizaciones, metricas, modelos serializados y documentacion por fase.

Por tanto, el proyecto no se limito a entrenar un modelo, sino que desarrollo un
pipeline completo y defendible de ciencia de datos.

## Alcance real del analisis

El alcance final fue:

```text
Copa Mundial de la FIFA 1930-2014
```

Inicialmente `results.csv` contenia partidos hasta 2026, pero la fuente
complementaria `WorldCupMatches.csv` disponible en el proyecto solo tenia
informacion de `stage` hasta 2014.

Por esa razon se decidio trabajar hasta 2014. Esta decision fue correcta porque
evito inventar, imputar o asumir etapas de partidos para ediciones no cubiertas
por la fuente complementaria.

El dataset final de partidos limpios tuvo:

```text
world_cup_matches_clean: (836, 15)
```

El dataset supervisado tuvo:

```text
supervised_dataset: (424, 9)
```

El dataset de clustering tuvo:

```text
world_cup_editions_clusters: (20, 24)
```

## Resultado del modelo supervisado

El modelo supervisado buscaba predecir:

```text
advanced_group_stage
```

Es decir, si una seleccion avanzaba mas alla de la fase de grupos.

Las variables usadas fueron pre-torneo:

```text
host_continent
team_confederation
is_host_region
historical_appearances
historical_win_rate
historical_avg_goals_scored
```

No se usaron goles ni resultados del torneo actual como features, lo cual es una
decision metodologica importante para evitar data leakage.

### Comparacion contra baseline

El baseline obtuvo:

```text
accuracy  = 0.5294
f1_score  = 0.0000
auc_roc   = 0.5000
```

La regresion logistica obtuvo:

```text
accuracy  = 0.6118
precision = 0.6061
recall    = 0.5000
f1_score  = 0.5479
auc_roc   = 0.6397
```

El modelo principal supera al baseline en accuracy, F1 y AUC. Esto indica que las
variables contextuales e historicas si aportan informacion predictiva.

Sin embargo, el desempeno es moderado. La accuracy queda por debajo de la
referencia orientativa de 0.65 planteada inicialmente en el contexto del proyecto.

Por tanto, la conclusion correcta no es que el modelo predice con alta precision,
sino que:

```text
Las variables pre-torneo usadas contienen una senal predictiva real, pero limitada.
```

## Interpretacion de variables

Los coeficientes de la regresion logistica sugieren que algunas variables tienen
asociaciones razonables con el avance de fase.

Coeficientes positivos destacados:

```text
team_confederation_CONMEBOL
host_continent_North America
historical_appearances
team_confederation_UEFA
historical_avg_goals_scored
is_host_region
```

La variable `historical_appearances` tiene sentido futbolistico: selecciones con
mayor experiencia historica suelen tener mas probabilidad de avanzar.

El coeficiente positivo de `is_host_region` tambien es coherente con la pregunta
central del proyecto. Sugiere que pertenecer al continente sede podria estar
asociado con mejor rendimiento.

Sin embargo, esta interpretacion debe hacerse con cuidado. Un coeficiente positivo
no prueba causalidad. Solo indica una asociacion dentro del modelo y bajo las
variables disponibles.

## Ventaja del continente sede

El EDA mostro una diferencia descriptiva en la tasa de avance:

```text
Equipos de otra region:
advance_rate = 0.436

Equipos de la region sede:
advance_rate = 0.521
```

Esto sugiere que existe una ventaja observable para equipos del continente sede.

Pero la conclusion debe formularse con prudencia:

```text
Se observa una asociacion descriptiva entre pertenecer a la region sede y avanzar
de fase, pero no se puede afirmar causalidad con este diseno.
```

La diferencia puede estar mezclada con otros factores:

- calidad historica de las selecciones
- cupos por confederacion
- cambios de formato
- distribucion geografica de sedes
- fortaleza relativa de continentes en distintas epocas

## Resultado del clustering

El clustering se hizo a nivel de edicion mundialista. Se usaron variables como:

- promedio de goles
- diferencia media de goles
- numero de partidos
- numero de equipos
- proporcion de fase de grupos
- tasa general de avance
- representacion por confederacion
- tasa de avance por confederacion

El mejor valor de `k` segun silueta fue:

```text
best_k = 3
best_silhouette = 0.3213
```

La silueta no es alta, por lo que los clusters no deben interpretarse como grupos
muy separados. Aun asi, las agrupaciones son interpretables.

### Cluster 0

Ediciones:

```text
1934, 1938, 1954, 1958, 1962, 1966, 1970, 1974, 1978, 1982
```

Interpretacion:

Ediciones intermedias o de formato clasico/transicional. Tienen menos equipos que
los torneos modernos y un promedio de goles relativamente alto.

### Cluster 1

Ediciones:

```text
1986, 1990, 1994, 1998, 2002, 2006, 2010, 2014
```

Interpretacion:

Ediciones modernas, con mas equipos, mas partidos y menor promedio de goles. Este
grupo refleja la evolucion hacia formatos mas amplios y competitivamente mas
cerrados.

### Cluster 2

Ediciones:

```text
1930, 1950
```

Interpretacion:

Ediciones tempranas y atipicas. Tienen pocos equipos, pocos partidos y alta
diferencia media de goles. Sus formatos historicos las separan claramente del
resto.

## Que pudo haber pasado

Los resultados moderados del modelo supervisado y la estructura de clusters pueden
explicarse por varios factores.

### 1. Pocos datos para Machine Learning

Aunque hay 836 partidos, el modelo supervisado se entrena a nivel seleccion-edicion,
lo que produce solo:

```text
424 filas
```

Este es un dataset pequeno para un modelo predictivo.

### 2. Variables pre-torneo limitadas

El proyecto uso variables historicas y contextuales disponibles dentro de los
datasets trabajados.

Pero faltan variables muy importantes para predecir rendimiento deportivo:

- ranking FIFA
- rating Elo
- calidad de jugadores
- edad promedio de plantillas
- experiencia del entrenador
- lesiones
- desempeno reciente antes del torneo
- dificultad del grupo
- distancia geografica real
- dias de descanso

Sin estas variables, el modelo solo puede capturar una parte pequena del fenomeno.

### 3. Cambios historicos del formato

El Mundial cambio mucho entre 1930 y 2014:

- numero de equipos
- numero de grupos
- reglas de clasificacion
- existencia o no de rondas especificas
- cantidad de partidos por seleccion

Esto hace que `advanced_group_stage` no signifique exactamente lo mismo en todas
las epocas.

### 4. Dependencia entre observaciones

Una misma seleccion aparece muchas veces en el dataset. Brasil, Alemania, Italia,
Argentina y otras selecciones aparecen en multiples ediciones.

Eso viola parcialmente el supuesto de independencia de observaciones de una
regresion logistica estandar.

### 5. Normalizacion historica de selecciones

Para poder cruzar fuentes se normalizaron nombres historicos:

- West Germany -> Germany
- Soviet Union -> Russia
- Yugoslavia -> Serbia
- Czechoslovakia -> Czech Republic

Esta decision es practica y necesaria para el analisis, pero introduce
simplificaciones historicas.

## Que conclusiones son defendibles

Las conclusiones mas defendibles son:

1. El pipeline construido es reproducible y cumple buenas practicas de modularidad.
2. El cruce de datos logro incorporar `stage` sin nulos para 1930-2014.
3. El dataset supervisado evita data leakage al calcular historicos solo con
   ediciones previas.
4. La regresion logistica supera al baseline, lo que indica que las variables
   usadas aportan senal predictiva.
5. La experiencia historica y la region sede parecen tener una asociacion positiva
   con el avance de fase.
6. Los clusters separan ediciones tempranas, intermedias y modernas de manera
   interpretable.
7. Los resultados deben interpretarse como evidencia exploratoria, no como prueba
   causal.

## Que conclusiones no se deben afirmar

No seria correcto afirmar:

```text
Ser del continente sede causa avanzar de fase.
```

Tampoco seria correcto afirmar:

```text
El modelo predice con alta precision el avance de fase.
```

Ni:

```text
Los clusters descubren estilos futbolisticos definitivos.
```

La formulacion correcta es mas prudente:

```text
Los resultados sugieren asociaciones entre contexto regional, experiencia
historica y avance de fase, pero la evidencia es moderada y debe interpretarse
con cautela.
```

## Evaluacion final del proyecto

El proyecto cumplio el objetivo de la asignatura porque demuestra un proceso
completo de Machine Learning:

- definicion del problema
- carga y cruce de datos
- limpieza
- EDA
- feature engineering
- control de data leakage
- baseline
- modelo supervisado interpretable
- evaluacion con metricas
- clustering
- visualizaciones
- serializacion de modelos
- documentacion por fases

Desde el punto de vista academico, el valor principal del proyecto no esta en
alcanzar una accuracy muy alta, sino en construir un analisis honesto,
reproducible y metodologicamente correcto.

## Conclusion general

El proyecto muestra que las condiciones contextuales y el rendimiento historico
pre-torneo contienen informacion util para analizar el avance de selecciones en la
Copa Mundial, pero esa informacion no es suficiente para explicar completamente
los resultados deportivos.

La ventaja regional aparece como una senal descriptiva y el modelo supervisado
mejora frente al baseline, pero el desempeno moderado confirma que el futbol es un
fenomeno complejo y que se requieren variables adicionales para una prediccion mas
fuerte.

El clustering, por su parte, identifica perfiles historicos razonables de torneos:
ediciones tempranas atipicas, ediciones intermedias y ediciones modernas. Esto
refuerza la idea de que el formato y la epoca del Mundial influyen de manera
importante en cualquier analisis historico.

En sintesis:

```text
El objetivo del proyecto se cumplio. Los resultados son coherentes, interpretables
y defendibles, siempre que se presenten como evidencia exploratoria y no como
conclusiones causales definitivas.
```
