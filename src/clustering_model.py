"""Funciones para entrenar modelos no supervisados sobre ediciones mundialistas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


RANDOM_STATE = 42
CONFEDERATIONS = ["UEFA", "CONMEBOL", "CONCACAF", "CAF", "AFC", "OFC"]
BASE_CLUSTER_FEATURES = [
    "avg_total_goals",
    "avg_abs_goal_diff",
    "n_matches",
    "n_teams",
    "group_stage_share",
    "overall_advance_rate",
    "host_region_team_share",
]
SHARE_FEATURES = [f"share_{confed}" for confed in CONFEDERATIONS]
ADVANCE_RATE_FEATURES = [f"advance_rate_{confed}" for confed in CONFEDERATIONS]
CLUSTER_FEATURES = BASE_CLUSTER_FEATURES + SHARE_FEATURES + ADVANCE_RATE_FEATURES


def _resolve_project_path(project_root: str | Path, path: str | Path) -> Path:
    """Construye una ruta relativa a la raiz del proyecto.

    Args:
        project_root: Ruta relativa a la raiz del proyecto.
        path: Ruta relativa que se quiere resolver.

    Returns:
        Ruta compuesta a partir de `project_root` y `path`.
    """
    return Path(project_root) / Path(path)


def load_clustering_inputs(
    project_root: str | Path = ".",
    matches_path: str | Path = "data/processed/world_cup_matches_clean.csv",
    supervised_path: str | Path = "data/processed/supervised_dataset.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga los datasets necesarios para clustering por edicion.

    Args:
        project_root: Ruta relativa a la raiz del proyecto.
        matches_path: Ruta relativa del dataset limpio de partidos.
        supervised_path: Ruta relativa del dataset supervisado.

    Returns:
        Tupla `(matches_df, supervised_df)`.

    Raises:
        FileNotFoundError: Si falta alguno de los archivos requeridos.
    """
    resolved_matches_path = _resolve_project_path(project_root, matches_path)
    resolved_supervised_path = _resolve_project_path(project_root, supervised_path)

    if not resolved_matches_path.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo requerido: {resolved_matches_path}"
        )
    if not resolved_supervised_path.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo requerido: {resolved_supervised_path}"
        )

    return pd.read_csv(resolved_matches_path), pd.read_csv(resolved_supervised_path)


def _validate_clustering_inputs(
    matches_df: pd.DataFrame,
    supervised_df: pd.DataFrame,
) -> None:
    """Valida columnas necesarias para construir variables de clustering.

    Args:
        matches_df: Dataset limpio de partidos.
        supervised_df: Dataset supervisado a nivel seleccion-edicion.

    Returns:
        None.

    Raises:
        ValueError: Si faltan columnas o existen nulos criticos.
    """
    match_columns = {
        "year",
        "total_goals",
        "goal_difference",
        "is_group_stage",
    }
    supervised_columns = {
        "year",
        "team",
        "team_confederation",
        "is_host_region",
        "advanced_group_stage",
    }

    missing_match_columns = match_columns - set(matches_df.columns)
    missing_supervised_columns = supervised_columns - set(supervised_df.columns)
    if missing_match_columns:
        missing = ", ".join(sorted(missing_match_columns))
        raise ValueError(f"Faltan columnas en partidos: {missing}")
    if missing_supervised_columns:
        missing = ", ".join(sorted(missing_supervised_columns))
        raise ValueError(f"Faltan columnas en dataset supervisado: {missing}")

    match_nulls = matches_df[list(match_columns)].isna().sum()
    supervised_nulls = supervised_df[list(supervised_columns)].isna().sum()
    if match_nulls.any():
        raise ValueError(
            "Hay nulos en columnas de partidos:\n"
            f"{match_nulls[match_nulls > 0]}"
        )
    if supervised_nulls.any():
        raise ValueError(
            "Hay nulos en columnas supervisadas:\n"
            f"{supervised_nulls[supervised_nulls > 0]}"
        )


def build_world_cup_editions_dataset(
    matches_df: pd.DataFrame,
    supervised_df: pd.DataFrame,
) -> pd.DataFrame:
    """Construye un dataset agregado con una fila por edicion mundialista.

    Args:
        matches_df: Dataset limpio de partidos mundialistas.
        supervised_df: Dataset supervisado a nivel seleccion-edicion.

    Returns:
        DataFrame con variables agregadas por mundial y columnas de clustering.
    """
    _validate_clustering_inputs(matches_df, supervised_df)

    matches = matches_df.copy()
    teams = supervised_df.copy()
    matches["year"] = pd.to_numeric(matches["year"], errors="raise").astype(int)
    teams["year"] = pd.to_numeric(teams["year"], errors="raise").astype(int)

    editions_stats = (
        matches.groupby("year")
        .agg(
            avg_total_goals=("total_goals", "mean"),
            avg_abs_goal_diff=("goal_difference", lambda values: values.abs().mean()),
            n_matches=("total_goals", "size"),
            group_stage_share=("is_group_stage", "mean"),
        )
        .reset_index()
    )

    team_stats = (
        teams.groupby("year")
        .agg(
            n_teams=("team", "nunique"),
            overall_advance_rate=("advanced_group_stage", "mean"),
            host_region_team_share=("is_host_region", "mean"),
        )
        .reset_index()
    )
    editions_stats = editions_stats.merge(team_stats, on="year", how="left")

    confed_counts = pd.crosstab(teams["year"], teams["team_confederation"])
    confed_counts = confed_counts.reindex(columns=CONFEDERATIONS, fill_value=0)
    confed_shares = confed_counts.div(confed_counts.sum(axis=1), axis=0)
    confed_shares = confed_shares.add_prefix("share_").reset_index()

    confed_advance_rates = (
        teams.pivot_table(
            index="year",
            columns="team_confederation",
            values="advanced_group_stage",
            aggfunc="mean",
        )
        .reindex(columns=CONFEDERATIONS)
        .fillna(0.0)
    )
    confed_advance_rates = confed_advance_rates.add_prefix(
        "advance_rate_"
    ).reset_index()

    editions_stats = editions_stats.merge(confed_shares, on="year", how="left")
    editions_stats = editions_stats.merge(
        confed_advance_rates, on="year", how="left"
    )
    editions_stats[CLUSTER_FEATURES] = editions_stats[CLUSTER_FEATURES].fillna(0.0)

    return editions_stats.sort_values("year").reset_index(drop=True)


def evaluate_kmeans_range(
    X_scaled,
    k_min: int = 2,
    k_max: int = 8,
) -> pd.DataFrame:
    """Evalua K-Means para un rango de k con inercia y silueta.

    Args:
        X_scaled: Matriz de features escaladas.
        k_min: Valor minimo de clusters.
        k_max: Valor maximo de clusters.

    Returns:
        DataFrame con columnas `k`, `inertia` y `silhouette`.
    """
    n_samples = X_scaled.shape[0]
    max_valid_k = min(k_max, n_samples - 1)
    records = []

    for k in range(k_min, max_valid_k + 1):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(X_scaled)
        records.append(
            {
                "k": k,
                "inertia": round(float(model.inertia_), 4),
                "silhouette": round(float(silhouette_score(X_scaled, labels)), 4),
            }
        )

    return pd.DataFrame(records)


def _save_elbow_plot(evaluation_df: pd.DataFrame, output_path: str | Path) -> None:
    """Guarda el grafico del metodo del codo para K-Means.

    Args:
        evaluation_df: DataFrame con inercia por `k`.
        output_path: Ruta donde se guardara la figura.

    Returns:
        None.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=evaluation_df, x="k", y="inertia", marker="o", ax=ax)
    ax.set_title("Metodo del codo - K-Means")
    ax.set_xlabel("Numero de clusters (k)")
    ax.set_ylabel("Inercia")
    ax.set_xticks(evaluation_df["k"])
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_silhouette_plot(evaluation_df: pd.DataFrame, output_path: str | Path) -> None:
    """Guarda el grafico de coeficiente de silueta por `k`.

    Args:
        evaluation_df: DataFrame con silueta por `k`.
        output_path: Ruta donde se guardara la figura.

    Returns:
        None.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=evaluation_df, x="k", y="silhouette", marker="o", ax=ax)
    ax.set_title("Coeficiente de silueta - K-Means")
    ax.set_xlabel("Numero de clusters (k)")
    ax.set_ylabel("Silueta")
    ax.set_xticks(evaluation_df["k"])
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_dendrogram(
    X_scaled,
    years: list[str],
    output_path: str | Path,
) -> None:
    """Guarda el dendrograma de clustering jerarquico.

    Args:
        X_scaled: Matriz de features escaladas.
        years: Etiquetas de anios para las hojas.
        output_path: Ruta donde se guardara la figura.

    Returns:
        None.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    linkage_matrix = linkage(X_scaled, method="ward")

    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(linkage_matrix, labels=years, leaf_rotation=45, ax=ax)
    ax.set_title("Dendrograma - Clustering jerarquico (Ward)")
    ax.set_xlabel("Edicion del Mundial")
    ax.set_ylabel("Distancia")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_clusters_pca_plot(
    editions_stats: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Guarda un scatter PCA 2D comparando clusters K-Means y jerarquico.

    Args:
        editions_stats: DataFrame con coordenadas PCA y etiquetas de clusters.
        output_path: Ruta donde se guardara la figura.

    Returns:
        None.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, column, title in zip(
        axes,
        ["cluster_kmeans", "cluster_hier"],
        ["K-Means", "Jerarquico"],
    ):
        sns.scatterplot(
            data=editions_stats,
            x="pca1",
            y="pca2",
            hue=column,
            palette="Set2",
            s=100,
            ax=ax,
        )
        for _, row in editions_stats.iterrows():
            ax.annotate(
                str(int(row["year"])),
                (row["pca1"], row["pca2"]),
                fontsize=8,
                ha="center",
                va="bottom",
            )
        ax.set_title(f"Clusters - {title}")
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")

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


def train_clustering_models(
    matches_data: pd.DataFrame | None = None,
    supervised_data: pd.DataFrame | None = None,
    project_root: str | Path = ".",
    matches_path: str | Path = "data/processed/world_cup_matches_clean.csv",
    supervised_path: str | Path = "data/processed/supervised_dataset.csv",
    k_min: int = 2,
    k_max: int = 8,
    save_outputs: bool = True,
) -> dict[str, Any]:
    """Entrena K-Means y clustering jerarquico sobre ediciones mundialistas.

    Args:
        matches_data: Dataset limpio opcional de partidos.
        supervised_data: Dataset supervisado opcional.
        project_root: Ruta relativa a la raiz del proyecto.
        matches_path: Ruta relativa del dataset limpio si `matches_data` es `None`.
        supervised_path: Ruta relativa del dataset supervisado si
            `supervised_data` es `None`.
        k_min: Valor minimo de `k` para evaluar K-Means.
        k_max: Valor maximo de `k` para evaluar K-Means.
        save_outputs: Indica si se deben guardar figuras, modelos y CSV.

    Returns:
        Diccionario con datasets, modelos, scaler, PCA y metricas.
    """
    root = Path(project_root)
    if matches_data is None or supervised_data is None:
        loaded_matches, loaded_supervised = load_clustering_inputs(
            root, matches_path, supervised_path
        )
        matches_df = loaded_matches if matches_data is None else matches_data.copy()
        supervised_df = (
            loaded_supervised if supervised_data is None else supervised_data.copy()
        )
    else:
        matches_df = matches_data.copy()
        supervised_df = supervised_data.copy()

    editions_stats = build_world_cup_editions_dataset(matches_df, supervised_df)
    X_cluster = editions_stats[CLUSTER_FEATURES].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    kmeans_evaluation = evaluate_kmeans_range(X_scaled, k_min=k_min, k_max=k_max)
    best_k = int(
        kmeans_evaluation.sort_values(["silhouette", "k"], ascending=[False, True])
        .iloc[0]["k"]
    )

    kmeans_final = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    editions_stats["cluster_kmeans"] = kmeans_final.fit_predict(X_scaled)

    hierarchical_model = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    editions_stats["cluster_hier"] = hierarchical_model.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    editions_stats["pca1"] = X_pca[:, 0]
    editions_stats["pca2"] = X_pca[:, 1]

    metrics = {
        "best_k": best_k,
        "best_silhouette": float(
            kmeans_evaluation.loc[
                kmeans_evaluation["k"] == best_k, "silhouette"
            ].iloc[0]
        ),
        "kmeans_evaluation": kmeans_evaluation.to_dict(orient="records"),
        "cluster_features": CLUSTER_FEATURES,
        "n_editions": int(len(editions_stats)),
        "pca_explained_variance_ratio": [
            round(float(value), 4) for value in pca.explained_variance_ratio_
        ],
    }

    if save_outputs:
        _save_elbow_plot(
            kmeans_evaluation, root / "outputs" / "figures" / "kmeans_codo.png"
        )
        _save_silhouette_plot(
            kmeans_evaluation, root / "outputs" / "figures" / "kmeans_silueta.png"
        )
        _save_dendrogram(
            X_scaled,
            editions_stats["year"].astype(str).tolist(),
            root / "outputs" / "figures" / "dendrograma.png",
        )
        _save_clusters_pca_plot(
            editions_stats, root / "outputs" / "figures" / "clusters_pca.png"
        )

        models_dir = root / "outputs" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(kmeans_final, models_dir / "kmeans_model.joblib")
        joblib.dump(hierarchical_model, models_dir / "hierarchical_clustering_model.joblib")
        joblib.dump(scaler, models_dir / "cluster_scaler.joblib")
        joblib.dump(pca, models_dir / "cluster_pca.joblib")

        output_path = root / "data" / "processed" / "world_cup_editions_clusters.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        editions_stats.to_csv(output_path, index=False)

        _write_json(metrics, root / "outputs" / "metrics" / "clustering_metrics.json")

    print(f"Dataset de ediciones: {editions_stats.shape}")
    print(f"Features de clustering: {len(CLUSTER_FEATURES)}")
    print("Evaluacion K-Means:")
    print(kmeans_evaluation.to_string(index=False))
    print(f"Mejor k segun silueta: {best_k}")
    print("\nAsignaciones de clusters:")
    print(
        editions_stats[["year", "cluster_kmeans", "cluster_hier"]].to_string(
            index=False
        )
    )

    for cluster_id in sorted(editions_stats["cluster_kmeans"].unique()):
        subset = editions_stats[editions_stats["cluster_kmeans"] == cluster_id]
        print(f"\n--- Cluster K-Means {cluster_id} ---")
        print(f"Ediciones: {sorted(subset['year'].astype(int).tolist())}")
        print(subset[BASE_CLUSTER_FEATURES].mean().round(3).to_string())

    return {
        "editions_stats": editions_stats,
        "kmeans_evaluation": kmeans_evaluation,
        "best_k": best_k,
        "kmeans_model": kmeans_final,
        "hierarchical_model": hierarchical_model,
        "scaler": scaler,
        "pca": pca,
        "metrics": metrics,
    }
