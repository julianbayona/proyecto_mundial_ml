"""Funciones para consolidar metricas y verificar salidas del proyecto."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


EXPECTED_PROCESSED_FILES = [
    "data/processed/world_cup_matches_raw.csv",
    "data/processed/world_cup_matches_clean.csv",
    "data/processed/supervised_dataset.csv",
    "data/processed/world_cup_editions_clusters.csv",
]

EXPECTED_FIGURES = [
    "outputs/figures/distribucion_goles_raw.png",
    "outputs/figures/partidos_por_edicion.png",
    "outputs/figures/goles_promedio_por_edicion.png",
    "outputs/figures/rendimiento_por_sede.png",
    "outputs/figures/top_selecciones_avance.png",
    "outputs/figures/matriz_correlacion.png",
    "outputs/figures/balance_clases.png",
    "outputs/figures/matrices_confusion.png",
    "outputs/figures/curva_roc.png",
    "outputs/figures/coeficientes_modelo.png",
    "outputs/figures/kmeans_codo.png",
    "outputs/figures/kmeans_silueta.png",
    "outputs/figures/dendrograma.png",
    "outputs/figures/clusters_pca.png",
]

EXPECTED_METRICS = [
    "outputs/metrics/supervised_metrics.json",
    "outputs/metrics/clustering_metrics.json",
]

EXPECTED_MODELS = [
    "outputs/models/logistic_regression_model.joblib",
    "outputs/models/kmeans_model.joblib",
    "outputs/models/hierarchical_clustering_model.joblib",
    "outputs/models/cluster_scaler.joblib",
    "outputs/models/cluster_pca.joblib",
    "outputs/models/metadata_model.json",
]


def _resolve_project_path(project_root: str | Path, path: str | Path) -> Path:
    """Construye una ruta relativa a la raiz del proyecto.

    Args:
        project_root: Ruta relativa a la raiz del proyecto.
        path: Ruta relativa que se quiere resolver.

    Returns:
        Ruta compuesta a partir de `project_root` y `path`.
    """
    return Path(project_root) / Path(path)


def _read_json(path: Path) -> dict[str, Any]:
    """Lee un archivo JSON.

    Args:
        path: Ruta del archivo JSON.

    Returns:
        Diccionario con el contenido del JSON.
    """
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(data: dict[str, Any], path: Path) -> None:
    """Guarda un diccionario como JSON.

    Args:
        data: Diccionario serializable.
        path: Ruta donde se escribira el archivo.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def check_expected_outputs(project_root: str | Path = ".") -> dict[str, list[str]]:
    """Verifica que existan los outputs esperados del proyecto.

    Args:
        project_root: Ruta relativa a la raiz del proyecto.

    Returns:
        Diccionario con listas de archivos faltantes por categoria.
    """
    root = Path(project_root)
    expected_groups = {
        "processed": EXPECTED_PROCESSED_FILES,
        "figures": EXPECTED_FIGURES,
        "metrics": EXPECTED_METRICS,
        "models": EXPECTED_MODELS,
    }
    missing: dict[str, list[str]] = {}
    for group, files in expected_groups.items():
        missing[group] = [
            file_path
            for file_path in files
            if not _resolve_project_path(root, file_path).exists()
        ]
    return missing


def generate_all_metrics(
    project_root: str | Path = ".",
    output_path: str | Path = "outputs/metrics/final_summary.json",
) -> dict[str, Any]:
    """Consolida metricas y validaciones finales del proyecto.

    Args:
        project_root: Ruta relativa a la raiz del proyecto.
        output_path: Ruta relativa donde se guardara el resumen final.

    Returns:
        Diccionario con metricas principales, shapes de datasets y archivos
        faltantes.
    """
    root = Path(project_root)
    supervised_metrics_path = root / "outputs" / "metrics" / "supervised_metrics.json"
    clustering_metrics_path = root / "outputs" / "metrics" / "clustering_metrics.json"

    supervised_metrics = (
        _read_json(supervised_metrics_path)
        if supervised_metrics_path.exists()
        else {}
    )
    clustering_metrics = (
        _read_json(clustering_metrics_path)
        if clustering_metrics_path.exists()
        else {}
    )

    dataset_shapes = {}
    for label, relative_path in {
        "world_cup_matches_raw": "data/processed/world_cup_matches_raw.csv",
        "world_cup_matches_clean": "data/processed/world_cup_matches_clean.csv",
        "supervised_dataset": "data/processed/supervised_dataset.csv",
        "world_cup_editions_clusters": "data/processed/world_cup_editions_clusters.csv",
    }.items():
        resolved_path = root / relative_path
        if resolved_path.exists():
            dataset_shapes[label] = list(pd.read_csv(resolved_path).shape)

    missing_outputs = check_expected_outputs(root)
    all_missing = sum(len(files) for files in missing_outputs.values())

    summary = {
        "execution_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "dataset_scope": "FIFA World Cup 1930-2014",
        "dataset_shapes": dataset_shapes,
        "supervised_metrics": supervised_metrics,
        "clustering_metrics": clustering_metrics,
        "missing_outputs": missing_outputs,
        "all_expected_outputs_present": all_missing == 0,
    }

    resolved_output_path = _resolve_project_path(root, output_path)
    _write_json(summary, resolved_output_path)

    print("Resumen final de metricas:")
    print(json.dumps(summary, indent=2))
    return summary
