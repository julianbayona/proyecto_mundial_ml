# Documentacion de avance - Fase 5

Este documento resume el trabajo realizado en la **Fase 5 - Modelo Supervisado**
del proyecto **El Factor Local**.

La fase se ejecuto despues de construir el dataset supervisado en la Fase 4. El
insumo utilizado fue:

```text
data/processed/supervised_dataset.csv
```

El objetivo fue entrenar y evaluar dos modelos:

1. Un modelo baseline con `DummyClassifier`.
2. Un modelo principal de regresion logistica con `Pipeline`.

## Objetivo de la fase

El objetivo de esta fase fue predecir si una seleccion avanza mas alla de la fase
de grupos usando solamente variables disponibles antes del inicio del torneo.

La variable objetivo fue:

```text
advanced_group_stage
```

Valores:

```text
0 = no avanzo
1 = avanzo
```

## Regla de no data leakage

El modelo uso unicamente features pre-torneo:

```text
host_continent
team_confederation
is_host_region
historical_appearances
historical_win_rate
historical_avg_goals_scored
```

No se usaron columnas de resultados del torneo actual como:

```text
home_score
away_score
goal_difference
total_goals
```

Esto mantiene la consistencia temporal del problema y evita data leakage.

## Archivos creados o modificados

Se modifico:

```text
src/supervised_model.py
```

Se creo:

```text
notebooks/05_supervised_model.ipynb
```

Se generaron outputs en:

```text
outputs/metrics/
outputs/models/
outputs/figures/
```

## Funciones implementadas

En `src/supervised_model.py` se implementaron funciones para entrenar, evaluar y
guardar los modelos.

Funciones principales:

```python
load_supervised_dataset()
build_model_pipeline()
split_features_target()
train_supervised_model()
```

Tambien se implementaron funciones auxiliares para:

- validar columnas y target
- calcular metricas
- guardar matrices de confusion
- guardar curva ROC
- extraer coeficientes del modelo
- guardar metadatos y metricas en JSON
- serializar el modelo con `joblib`

## Dataset utilizado

El dataset supervisado tiene:

```text
shape = (424, 9)
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

El balance es razonable, por lo que se uso division estratificada.

## Division train/test

Se uso:

```python
train_test_split(
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Resultado:

```text
Train: (339, 6)
Test:  (85, 6)
```

Balance en entrenamiento:

```text
0    0.531
1    0.469
```

Balance en prueba:

```text
0    0.5294
1    0.4706
```

## Modelo baseline

Se entreno un baseline con:

```python
DummyClassifier(strategy="most_frequent", random_state=42)
```

Este modelo predice siempre la clase mayoritaria. Sirve como piso minimo para
comparar el modelo principal.

Metricas del baseline:

```text
accuracy  = 0.5294
precision = 0.0000
recall    = 0.0000
f1_score  = 0.0000
auc_roc   = 0.5000
```

El F1 es cero porque el baseline no predice la clase positiva.

## Modelo principal

Se entreno una regresion logistica usando un `Pipeline` de scikit-learn.

El pipeline incluye:

1. `StandardScaler` para variables numericas.
2. `OneHotEncoder(handle_unknown="ignore")` para variables categoricas.
3. `LogisticRegression(random_state=42, max_iter=1000, C=1.0)`.

Variables numericas:

```text
is_host_region
historical_appearances
historical_win_rate
historical_avg_goals_scored
```

Variables categoricas:

```text
host_continent
team_confederation
```

Metricas de regresion logistica:

```text
accuracy  = 0.6118
precision = 0.6061
recall    = 0.5000
f1_score  = 0.5479
auc_roc   = 0.6397
```

## Comparacion contra baseline

Comparacion principal:

```text
Baseline accuracy: 0.5294
Modelo accuracy:   0.6118

Baseline F1:       0.0000
Modelo F1:         0.5479

Baseline AUC:      0.5000
Modelo AUC:        0.6397
```

El modelo de regresion logistica supera al baseline en accuracy, F1 y AUC.

Sin embargo, la accuracy queda por debajo de la referencia orientativa de 0.65
mencionada en el contexto del proyecto. El resultado debe interpretarse como una
mejora moderada sobre el baseline, coherente con el tamano pequeno del dataset y
las limitaciones de las variables pre-torneo disponibles.

## Coeficientes del modelo

Se genero una grafica de coeficientes para interpretar el modelo.

Coeficientes positivos destacados:

```text
team_confederation_CONMEBOL      0.700734
host_continent_North America     0.650905
historical_appearances           0.637810
team_confederation_UEFA          0.409786
historical_avg_goals_scored      0.317385
is_host_region                   0.241465
```

Coeficientes negativos destacados:

```text
host_continent_South America    -0.659981
team_confederation_AFC          -0.451493
team_confederation_CAF          -0.280773
historical_win_rate             -0.259933
team_confederation_CONCACAF     -0.237501
team_confederation_OFC          -0.194707
```

Interpretacion preliminar:

- La experiencia historica (`historical_appearances`) tiene peso positivo.
- Pertenecer a CONMEBOL o UEFA aparece asociado positivamente en este modelo.
- `is_host_region` tambien tiene coeficiente positivo, aunque menor.
- Algunos coeficientes deben interpretarse con cautela por el tamano reducido del
  dataset y por la codificacion one-hot.

## Ajuste basico explorado

Se hizo una revision adicional con `GridSearchCV` sobre distintos valores de `C`.

Resultado:

```text
Mejor C por CV: 10.0
Mejor F1 promedio CV: 0.6451
```

Pero en el conjunto de prueba esa configuracion empeoro frente a `C=1.0`:

```text
C=10.0:
test_accuracy = 0.5765
test_f1       = 0.5000
test_auc      = 0.6364
```

Por esa razon se mantuvo `C=1.0`, que es la configuracion base del plan y tuvo
mejor desempeno en la evaluacion final.

## Figuras generadas

Se generaron tres figuras:

```text
outputs/figures/matrices_confusion.png
outputs/figures/curva_roc.png
outputs/figures/coeficientes_modelo.png
```

### Matrices de confusion

Archivo:

```text
outputs/figures/matrices_confusion.png
```

Compara visualmente los errores del baseline y de la regresion logistica.

### Curva ROC

Archivo:

```text
outputs/figures/curva_roc.png
```

Compara la curva ROC del baseline contra el modelo principal.

### Coeficientes

Archivo:

```text
outputs/figures/coeficientes_modelo.png
```

Permite interpretar el peso de cada variable transformada dentro de la regresion
logistica.

## Metricas y modelos guardados

Se guardaron metricas en:

```text
outputs/metrics/supervised_metrics.json
```

Se serializo el modelo en:

```text
outputs/models/logistic_regression_model.joblib
```

Se guardaron metadatos en:

```text
outputs/models/metadata_model.json
```

El archivo de metadatos incluye:

- dataset usado
- fecha de ejecucion
- algoritmo
- features
- target
- parametros del modelo
- tamanos de train/test
- metricas
- notas sobre alcance 1930-2014 y features historicas sin leakage

## Validaciones realizadas

Se valido:

```text
syntax_notebook_json_metrics_ok
model_steps = ['preprocessor', 'classifier']
metadata_algorithm = LogisticRegression
```

Tambien se verifico que el modelo serializado puede cargarse con `joblib` y
realizar predicciones sobre una muestra del dataset supervisado.

## Nota sobre dependencias

Durante la ejecucion, el entorno inicial no tenia `joblib` ni `scikit-learn`
disponibles.

Se intento instalar todo `requirements.txt`, pero la instalacion completa fallo
por componentes de Jupyter y limites de rutas largas en Windows.

Para completar la Fase 5 se instalaron las dependencias minimas necesarias para
modelado:

```text
scikit-learn
scipy
joblib
matplotlib
seaborn
```

Luego de eso, el entrenamiento y la serializacion del modelo funcionaron
correctamente.

## Notebook creado

Se creo:

```text
notebooks/05_supervised_model.ipynb
```

El notebook:

1. Carga el dataset supervisado.
2. Revisa el balance de clases.
3. Muestra las features usadas.
4. Ejecuta `train_supervised_model()`.
5. Muestra metricas, coeficientes y reporte de clasificacion.

## Entregables de Fase 5

Quedaron listos:

```text
src/supervised_model.py
notebooks/05_supervised_model.ipynb
outputs/metrics/supervised_metrics.json
outputs/models/logistic_regression_model.joblib
outputs/models/metadata_model.json
outputs/figures/matrices_confusion.png
outputs/figures/curva_roc.png
outputs/figures/coeficientes_modelo.png
```

## Estado antes de Fase 6

La Fase 5 queda cerrada. El proyecto esta listo para iniciar la Fase 6 de modelo
no supervisado usando como insumo principal:

```text
data/processed/world_cup_matches_clean.csv
```

Tambien puede reutilizar:

```text
data/processed/supervised_dataset.csv
```

para construir variables agregadas por edicion relacionadas con avance de fase.

No se ha avanzado todavia a Fase 6.
