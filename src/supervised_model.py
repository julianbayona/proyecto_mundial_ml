"""Funciones para entrenar y serializar los modelos supervisados del proyecto."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


RANDOM_STATE = 42
FEATURE_COLUMNS = [
    "host_continent",
    "team_confederation",
    "is_host_region",
    "historical_appearances",
    "historical_win_rate",
    "historical_avg_goals_scored",
]
TARGET_COLUMN = "advanced_group_stage"
NUMERICAL_FEATURES = [
    "is_host_region",
    "historical_appearances",
    "historical_win_rate",
    "historical_avg_goals_scored",
]
CATEGORICAL_FEATURES = ["host_continent", "team_confederation"]


def _resolve_project_path(project_root: str | Path, path: str | Path) -> Path:
    """Construye una ruta relativa a la raiz del proyecto.

    Args:
        project_root: Ruta relativa a la raiz del proyecto.
        path: Ruta relativa que se quiere resolver.

    Returns:
        Ruta compuesta a partir de `project_root` y `path`.
    """
    return Path(project_root) / Path(path)


def load_supervised_dataset(
    project_root: str | Path = ".",
    input_path: str | Path = "data/processed/supervised_dataset.csv",
) -> pd.DataFrame:
    """Carga el dataset supervisado desde disco.

    Args:
        project_root: Ruta relativa a la raiz del proyecto.
        input_path: Ruta relativa del dataset supervisado.

    Returns:
        DataFrame con una fila por seleccion y edicion.

    Raises:
        FileNotFoundError: Si el dataset supervisado no existe.
    """
    resolved_input_path = _resolve_project_path(project_root, input_path)
    if not resolved_input_path.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo requerido: {resolved_input_path}"
        )
    return pd.read_csv(resolved_input_path)


def _validate_supervised_dataset(df: pd.DataFrame) -> None:
    """Valida que el dataset tenga las columnas necesarias para modelado.

    Args:
        df: DataFrame supervisado.

    Returns:
        None.

    Raises:
        ValueError: Si faltan columnas, hay nulos o el target no es binario.
    """
    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Faltan columnas para modelado: {missing}")

    nulls = df[list(required_columns)].isna().sum()
    if nulls.any():
        raise ValueError(
            "Hay nulos en columnas de modelado:\n" f"{nulls[nulls > 0]}"
        )

    target_values = set(df[TARGET_COLUMN].unique())
    if target_values != {0, 1}:
        raise ValueError(
            "El target debe ser binario con valores {0, 1}. "
            f"Valores encontrados: {sorted(target_values)}"
        )


def build_model_pipeline() -> Pipeline:
    """Construye el pipeline de preprocesamiento y regresion logistica.

    Args:
        None.

    Returns:
        Pipeline de scikit-learn con escalado, one-hot encoding y clasificador.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    random_state=RANDOM_STATE,
                    max_iter=1000,
                    C=1.0,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def split_features_target(
    df: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa variables predictoras y target con split estratificado.

    Args:
        df: Dataset supervisado.
        test_size: Proporcion del dataset que se reserva para prueba.

    Returns:
        Tupla `(X_train, X_test, y_train, y_test)`.
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def _classification_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """Calcula metricas de clasificacion binaria.

    Args:
        y_true: Target real.
        y_pred: Predicciones binarias.
        y_prob: Probabilidades estimadas para la clase positiva.

    Returns:
        Diccionario con metricas redondeadas.
    """
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    if y_prob is not None:
        metrics["auc_roc"] = round(roc_auc_score(y_true, y_prob), 4)
    return metrics


def _get_positive_class_probabilities(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Obtiene probabilidades de clase positiva para un clasificador.

    Args:
        model: Modelo entrenado con `predict_proba`.
        X: Variables predictoras.

    Returns:
        Array con probabilidades de la clase `1`.
    """
    classes = list(model.classes_)
    positive_index = classes.index(1)
    return model.predict_proba(X)[:, positive_index]


def _save_confusion_matrices(
    y_test: pd.Series,
    y_pred_dummy: np.ndarray,
    y_pred_model: np.ndarray,
    output_path: str | Path,
) -> None:
    """Guarda matrices de confusion para baseline y modelo principal.

    Args:
        y_test: Target real de prueba.
        y_pred_dummy: Predicciones del baseline.
        y_pred_model: Predicciones de la regresion logistica.
        output_path: Ruta donde se guardara la figura.

    Returns:
        None.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred_dummy, ax=axes[0], colorbar=False
    )
    axes[0].set_title("Baseline")
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred_model, ax=axes[1], colorbar=False
    )
    axes[1].set_title("Regresion logistica")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_roc_curve(
    y_test: pd.Series,
    y_prob_dummy: np.ndarray,
    y_prob_model: np.ndarray,
    output_path: str | Path,
) -> None:
    """Guarda la curva ROC de baseline y regresion logistica.

    Args:
        y_test: Target real de prueba.
        y_prob_dummy: Probabilidades del baseline para la clase positiva.
        y_prob_model: Probabilidades del modelo para la clase positiva.
        output_path: Ruta donde se guardara la figura.

    Returns:
        None.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_predictions(
        y_test, y_prob_dummy, ax=ax, name="Baseline"
    )
    RocCurveDisplay.from_predictions(
        y_test, y_prob_model, ax=ax, name="Regresion logistica"
    )
    ax.plot([0, 1], [0, 1], "k--", label="Azar")
    ax.set_title("Curva ROC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _feature_names_from_pipeline(pipeline: Pipeline) -> list[str]:
    """Extrae los nombres de variables despues del preprocesamiento.

    Args:
        pipeline: Pipeline entrenado.

    Returns:
        Lista de nombres de features transformadas.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    ohe = preprocessor.named_transformers_["cat"]
    categorical_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    return NUMERICAL_FEATURES + categorical_names


def _coefficient_dataframe(pipeline: Pipeline) -> pd.DataFrame:
    """Construye un DataFrame con coeficientes del modelo.

    Args:
        pipeline: Pipeline de regresion logistica entrenado.

    Returns:
        DataFrame con columnas `feature` y `coefficient`.
    """
    feature_names = _feature_names_from_pipeline(pipeline)
    coefficients = pipeline.named_steps["classifier"].coef_[0]
    return pd.DataFrame(
        {"feature": feature_names, "coefficient": coefficients}
    ).sort_values("coefficient", ascending=False)


def _save_coefficients_plot(coef_df: pd.DataFrame, output_path: str | Path) -> None:
    """Guarda un grafico de coeficientes de la regresion logistica.

    Args:
        coef_df: DataFrame con coeficientes.
        output_path: Ruta donde se guardara la figura.

    Returns:
        None.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=coef_df, x="coefficient", y="feature", ax=ax, color="#4c7f73")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Importancia de variables - coeficientes")
    ax.set_xlabel("Coeficiente")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_json(data: dict[str, Any], output_path: str | Path) -> None:
    """Guarda un diccionario como archivo JSON.

    Args:
        data: Diccionario serializable.
        output_path: Ruta donde se guardara el JSON.

    Returns:
        None.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def train_supervised_model(
    data: pd.DataFrame | None = None,
    project_root: str | Path = ".",
    input_path: str | Path = "data/processed/supervised_dataset.csv",
    test_size: float = 0.2,
    save_outputs: bool = True,
) -> dict[str, Any]:
    """Entrena baseline y regresion logistica para predecir avance de fase.

    Args:
        data: Dataset supervisado opcional. Si es `None`, se lee `input_path`.
        project_root: Ruta relativa a la raiz del proyecto.
        input_path: Ruta relativa del dataset supervisado.
        test_size: Proporcion del dataset reservada para prueba.
        save_outputs: Indica si se deben guardar metricas, figuras y modelo.

    Returns:
        Diccionario con modelos entrenados, metricas y objetos de validacion.
    """
    root = Path(project_root)
    df = data.copy() if data is not None else load_supervised_dataset(root, input_path)
    _validate_supervised_dataset(df)

    X_train, X_test, y_train, y_test = split_features_target(df, test_size=test_size)

    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)
    y_prob_dummy = _get_positive_class_probabilities(dummy, X_test)

    pipeline = build_model_pipeline()
    pipeline.fit(X_train, y_train)
    y_pred_model = pipeline.predict(X_test)
    y_prob_model = _get_positive_class_probabilities(pipeline, X_test)

    baseline_metrics = _classification_metrics(y_test, y_pred_dummy, y_prob_dummy)
    model_metrics = _classification_metrics(y_test, y_pred_model, y_prob_model)

    metrics = {
        "baseline_accuracy": baseline_metrics["accuracy"],
        "baseline_precision": baseline_metrics["precision"],
        "baseline_recall": baseline_metrics["recall"],
        "baseline_f1_score": baseline_metrics["f1_score"],
        "baseline_auc_roc": baseline_metrics["auc_roc"],
        "model_accuracy": model_metrics["accuracy"],
        "precision": model_metrics["precision"],
        "recall": model_metrics["recall"],
        "f1_score": model_metrics["f1_score"],
        "auc_roc": model_metrics["auc_roc"],
        "train_size": len(X_train),
        "test_size": len(X_test),
        "random_state": RANDOM_STATE,
    }

    baseline_report = classification_report(
        y_test, y_pred_dummy, zero_division=0, output_dict=True
    )
    model_report = classification_report(
        y_test, y_pred_model, zero_division=0, output_dict=True
    )
    coef_df = _coefficient_dataframe(pipeline)

    metadata = {
        "dataset": "results.csv + WorldCupMatches.csv",
        "execution_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "model_version": "1.0",
        "algorithm": "LogisticRegression",
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "model_params": {
            "C": pipeline.named_steps["classifier"].C,
            "solver": pipeline.named_steps["classifier"].solver,
            "max_iter": pipeline.named_steps["classifier"].max_iter,
            "random_state": pipeline.named_steps["classifier"].random_state,
        },
        "train_size": len(X_train),
        "test_size": len(X_test),
        "metrics": metrics,
        "notes": (
            "Features historicas calculadas solo con ediciones previas. "
            "Dataset limitado a mundiales 1930-2014 por disponibilidad de stage."
        ),
    }

    if save_outputs:
        _save_confusion_matrices(
            y_test,
            y_pred_dummy,
            y_pred_model,
            root / "outputs" / "figures" / "matrices_confusion.png",
        )
        _save_roc_curve(
            y_test,
            y_prob_dummy,
            y_prob_model,
            root / "outputs" / "figures" / "curva_roc.png",
        )
        _save_coefficients_plot(
            coef_df, root / "outputs" / "figures" / "coeficientes_modelo.png"
        )
        _write_json(metrics, root / "outputs" / "metrics" / "supervised_metrics.json")
        _write_json(metadata, root / "outputs" / "models" / "metadata_model.json")

        model_path = root / "outputs" / "models" / "logistic_regression_model.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, model_path)

    print(f"Dataset supervisado: {df.shape}")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print("Balance y_train:")
    print(y_train.value_counts(normalize=True).sort_index().round(4))
    print("Balance y_test:")
    print(y_test.value_counts(normalize=True).sort_index().round(4))
    print("\n=== BASELINE ===")
    print(json.dumps(baseline_metrics, indent=2))
    print("\n=== REGRESION LOGISTICA ===")
    print(json.dumps(model_metrics, indent=2))
    print("\nTop coeficientes positivos:")
    print(coef_df.head(8).to_string(index=False))
    print("\nTop coeficientes negativos:")
    print(coef_df.tail(8).to_string(index=False))

    return {
        "pipeline": pipeline,
        "dummy": dummy,
        "metrics": metrics,
        "metadata": metadata,
        "baseline_report": baseline_report,
        "model_report": model_report,
        "coefficients": coef_df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred_dummy": y_pred_dummy,
        "y_pred_model": y_pred_model,
        "y_prob_dummy": y_prob_dummy,
        "y_prob_model": y_prob_model,
    }
