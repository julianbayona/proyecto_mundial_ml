# Documentacion de avance - Fases 0 y 1

Este documento resume el trabajo realizado antes de iniciar la Fase 2 del proyecto
**El Factor Local: Analisis del Impacto de la Sede y las Condiciones del Torneo
en los Resultados de la Copa Mundial de Futbol**.

El objetivo de estas fases fue dejar el proyecto preparado a nivel de estructura,
dependencias y carga inicial de datos, y construir el primer dataset cruzado de
partidos mundialistas con la columna `stage`.

## Fuentes de referencia revisadas

Antes de escribir codigo se leyeron completamente los dos documentos de referencia
ubicados en la raiz del proyecto:

- `contexto_proyecto_ml_codex.md`
- `plan_ejecucion_proyecto_ml.md`

Estos documentos definieron el alcance tecnico, la estructura del proyecto, las
librerias requeridas, las reglas de no data leakage, la modularidad esperada y
los entregables de cada fase.

## Fase 0 - Configuracion del proyecto

### Estructura creada

Se creo la estructura base definida en el plan:

```text
data/
  raw/
  processed/
notebooks/
src/
outputs/
  figures/
  metrics/
  models/
docs/
```

La carpeta `docs/` se agrego para mantener documentacion incremental del avance
del proyecto sin mezclarla con el `README.md` final.

### Archivos creados

Se crearon los siguientes archivos base:

```text
requirements.txt
README.md
main.py
src/__init__.py
src/data_loader.py
src/preprocessing.py
src/feature_engineering.py
src/supervised_model.py
src/clustering_model.py
src/evaluation.py
src/visualization.py
```

### Dependencias

El archivo `requirements.txt` se creo con las librerias indicadas en el contexto
del proyecto:

```text
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
scipy>=1.11
jupyter>=1.0
joblib>=1.3
```

### main.py

Se creo `main.py` con la estructura base del pipeline. El archivo importa los
modulos de `src/` y deja definidas las llamadas principales que se completaran en
fases posteriores:

1. Carga de datos.
2. Limpieza.
3. Construccion de features.
4. Entrenamiento supervisado.
5. Entrenamiento de clustering.
6. Generacion de metricas y visualizaciones.

Las funciones de fases futuras aun no estan implementadas, pero la estructura ya
refleja el flujo final esperado por el plan.

### Modulos base en src/

Cada archivo en `src/` se creo con un docstring inicial que describe su proposito.
La logica del proyecto queda centralizada en `src/`, mientras que los notebooks
se mantienen como capas de ejecucion e interpretacion.

## Fase 1 - Carga y cruce de datos

### Archivos crudos disponibles

El usuario coloco los datasets en `data/raw/`. Se verifico la presencia de:

```text
data/raw/results.csv
data/raw/WorldCupMatches.csv
data/raw/WorldCups.csv
```

Tambien existen archivos adicionales en `data/raw/`, como `goalscorers.csv` y
`shootouts.csv`, pero no se usaron en esta fase porque no forman parte del cruce
definido para Fase 1.

### Funcion implementada

Se implemento `load_world_cup_data()` en:

```text
src/data_loader.py
```

La funcion realiza los siguientes pasos:

1. Lee `data/raw/results.csv`.
2. Lee `data/raw/WorldCupMatches.csv`.
3. Valida que ambas fuentes tengan las columnas obligatorias.
4. Filtra `results.csv` para conservar solo `tournament == "FIFA World Cup"`.
5. Extrae la columna `year` desde `date`.
6. Normaliza nombres historicos de selecciones con `TEAM_NAME_MAP`.
7. Prepara `WorldCupMatches.csv` para obtener la columna `stage`.
8. Cruza ambas fuentes para asignar la fase del partido.
9. Imprime la cobertura del join.
10. Retorna el DataFrame resultante.

### Normalizacion de nombres

Se aplico el diccionario base definido en el contexto:

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

Durante la validacion del cruce se identificaron diferencias adicionales entre
las fuentes, por ejemplo:

```text
USA vs United States
Germany FR vs Germany
Korea Republic vs South Korea
Korea DPR vs North Korea
IR Iran vs Iran
```

Tambien se encontraron nombres con prefijos corruptos en `WorldCupMatches.csv`,
como `rn">Bosnia and Herzegovina`. La funcion limpia ese prefijo antes de aplicar
el diccionario.

Estas normalizaciones adicionales se agregaron para que el join fuera consistente
entre las dos fuentes.

### Decision de alcance temporal

Inicialmente `results.csv` contenia partidos de Copa Mundial hasta 2026. Sin
embargo, el archivo complementario `WorldCupMatches.csv`, que es obligatorio para
obtener la columna `stage`, solo cubre ediciones hasta 2014.

El primer cruce literal produjo baja cobertura:

```text
Partidos de Copa Mundial: 1036
Partidos sin stage asignado: 476 de 1063
Cobertura del join: 55.22%
```

Despues del diagnostico se confirmo que el problema principal era que
`WorldCupMatches.csv` no contiene 2018, 2022 ni 2026.

El usuario confirmo trabajar hasta 2014. Por tanto, el alcance actual del dataset
procesado queda definido como:

```text
1930-2014
```

Esta decision evita imputar o inventar etapas para torneos no cubiertos por la
fuente complementaria y mantiene trazabilidad con los datos disponibles.

### Llave de cruce

La llave principal definida por el plan es:

```text
year, home_team, away_team
```

Durante la validacion se encontro que algunos partidos tienen local y visitante
invertidos entre fuentes, y que algunos equipos se enfrentaron mas de una vez en
la misma edicion. Para evitar duplicados y asignaciones ambiguas, se uso como
desempate tecnico:

```text
year, home_team, away_team, home_score, away_score
```

Tambien se genero una version invertida del lookup de `WorldCupMatches.csv`, de
modo que el cruce soporte diferencias de orientacion local/visitante entre las
fuentes.

Los goles se usaron solo para resolver el cruce inicial de datos y no como
features del modelo supervisado. La regla de no data leakage sigue vigente para
fases posteriores.

### Notebook creado

Se creo:

```text
notebooks/01_data_loading.ipynb
```

El notebook:

1. Importa `load_world_cup_data()`.
2. Ejecuta la carga y el cruce.
3. Muestra la forma del DataFrame.
4. Valida que la columna `stage` exista.
5. Guarda el resultado en `data/processed/world_cup_matches_raw.csv`.

### Dataset procesado generado

Se genero:

```text
data/processed/world_cup_matches_raw.csv
```

Este archivo contiene partidos de Copa Mundial entre 1930 y 2014 con la columna
`stage` poblada.

## Validaciones realizadas

### Validacion del cruce final

La ejecucion final de `load_world_cup_data()` produjo:

```text
Partidos de Copa Mundial hasta 2014: 836
Partidos excluidos por alcance/fuente complementaria: 200
Partidos sin stage asignado: 0 de 836
Cobertura del join: 100.00%
```

Validaciones adicionales:

```text
shape = (836, 11)
stage_present = True
missing_stage = 0
duplicated_rows_key_score = 0
years = 1930 2014
```

### Distribucion por edicion

El archivo procesado contiene la siguiente cantidad de partidos por mundial:

```text
1930    18
1934    17
1938    18
1950    22
1954    26
1958    35
1962    32
1966    32
1970    32
1974    38
1978    38
1982    52
1986    52
1990    52
1994    52
1998    64
2002    64
2006    64
2010    64
2014    64
```

### Validacion tecnica

Se valido que:

```text
syntax_and_notebook_json_ok
```

Esto confirma que los archivos Python tienen sintaxis valida y que el notebook
`01_data_loading.ipynb` es JSON valido.

## Entregables listos

### Fase 0

- Estructura de carpetas creada.
- `requirements.txt` creado.
- `main.py` con estructura base del pipeline.
- Modulos base en `src/` con docstrings iniciales.
- Datasets crudos colocados en `data/raw/`.

### Fase 1

- `src/data_loader.py` con `load_world_cup_data()` implementada.
- `notebooks/01_data_loading.ipynb` creado.
- `data/processed/world_cup_matches_raw.csv` generado.
- Columna `stage` presente y sin nulos para el alcance 1930-2014.
- Cobertura del join: 100%.

## Estado antes de Fase 2

El proyecto queda listo para iniciar la Fase 2 de limpieza de datos usando como
entrada:

```text
data/processed/world_cup_matches_raw.csv
```

La siguiente fase debe trabajar sobre este archivo, revisar nulos y duplicados,
crear `is_group_stage`, mapear `host_continent` y guardar:

```text
data/processed/world_cup_matches_clean.csv
```

No se ha avanzado todavia a Fase 2.
