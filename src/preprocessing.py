"""Funciones de limpieza y validacion del dataset de partidos mundialistas."""

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


HOST_CONTINENT_MAP = {
    1930: "South America",
    1934: "Europe",
    1938: "Europe",
    1950: "South America",
    1954: "Europe",
    1958: "Europe",
    1962: "South America",
    1966: "Europe",
    1970: "North America",
    1974: "Europe",
    1978: "South America",
    1982: "Europe",
    1986: "North America",
    1990: "Europe",
    1994: "North America",
    1998: "Europe",
    2002: "Asia",
    2006: "Europe",
    2010: "Africa",
    2014: "South America",
}

CRITICAL_COLUMNS = [
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "year",
    "stage",
]

GROUP_STAGE_PATTERN = "Group|First round|Preliminary round|Pool"


def _resolve_project_path(project_root: str | Path, path: str | Path) -> Path:
    """Construye una ruta relativa a la raiz del proyecto.

    Args:
        project_root: Ruta relativa a la raiz del proyecto.
        path: Ruta relativa que se quiere resolver.

    Returns:
        Ruta compuesta a partir de `project_root` y `path`.
    """
    return Path(project_root) / Path(path)


def _plot_goal_distributions(df: pd.DataFrame, figure_path: str | Path) -> None:
    """Guarda histogramas de goles local y visitante.

    Args:
        df: DataFrame limpio con columnas `home_score` y `away_score`.
        figure_path: Ruta relativa donde se guardara la figura.

    Returns:
        None.
    """
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df["home_score"], bins=range(0, int(df["home_score"].max()) + 2), ax=axes[0])
    axes[0].set_title("Distribucion goles local")
    axes[0].set_xlabel("Goles")
    axes[0].set_ylabel("Partidos")

    sns.histplot(df["away_score"], bins=range(0, int(df["away_score"].max()) + 2), ax=axes[1])
    axes[1].set_title("Distribucion goles visitante")
    axes[1].set_xlabel("Goles")
    axes[1].set_ylabel("Partidos")

    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)


def clean_world_cup_data(
    data: pd.DataFrame | None = None,
    project_root: str | Path = ".",
    input_path: str | Path = "data/processed/world_cup_matches_raw.csv",
    output_path: str | Path = "data/processed/world_cup_matches_clean.csv",
    figure_path: str | Path = "outputs/figures/distribucion_goles_raw.png",
    save_output: bool = True,
    save_figure: bool = True,
) -> pd.DataFrame:
    """Limpia el dataset cruzado de partidos mundialistas.

    Args:
        data: DataFrame opcional con partidos ya cargados. Si es `None`, se lee
            `input_path`.
        project_root: Ruta relativa a la raiz del proyecto.
        input_path: Ruta relativa del dataset crudo cruzado.
        output_path: Ruta relativa donde se guardara el dataset limpio.
        figure_path: Ruta relativa donde se guardara el histograma de goles.
        save_output: Indica si se debe guardar el CSV limpio.
        save_figure: Indica si se debe guardar la figura de distribucion de goles.

    Returns:
        DataFrame limpio con columnas auxiliares para EDA y fases posteriores.

    Raises:
        FileNotFoundError: Si no existe `input_path` cuando `data` es `None`.
        ValueError: Si faltan columnas criticas o hay valores invalidos.
    """
    root = Path(project_root)
    if data is None:
        resolved_input_path = _resolve_project_path(root, input_path)
        if not resolved_input_path.exists():
            raise FileNotFoundError(
                f"No se encontro el archivo requerido: {resolved_input_path}"
            )
        df = pd.read_csv(resolved_input_path)
    else:
        df = data.copy()

    missing_columns = set(CRITICAL_COLUMNS) - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Faltan columnas criticas: {missing}")

    print("Nulos por columna:")
    print(df.isnull().sum())

    for column in ["home_team", "away_team", "stage"]:
        df[column] = df[column].astype(str).str.strip()

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")

    exact_duplicates = df.duplicated().sum()
    match_identity_columns = [
        "date",
        "year",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "stage",
    ]
    match_identity_columns = [
        column for column in match_identity_columns if column in df.columns
    ]
    match_duplicates = df.duplicated(subset=match_identity_columns).sum()
    rematch_candidates = df.duplicated(
        subset=["year", "home_team", "away_team"], keep=False
    ).sum()

    print(f"Duplicados exactos encontrados: {exact_duplicates}")
    print(f"Duplicados por identidad completa del partido: {match_duplicates}")
    print(
        "Filas con misma llave (year, home_team, away_team) preservadas como "
        f"posibles rematches: {rematch_candidates}"
    )

    if match_duplicates:
        df = df.drop_duplicates(subset=match_identity_columns).copy()

    critical_nulls = df[CRITICAL_COLUMNS].isnull().sum()
    if critical_nulls.any():
        raise ValueError(
            "Hay nulos en columnas criticas despues de la limpieza:\n"
            f"{critical_nulls[critical_nulls > 0]}"
        )

    negative_scores = ((df["home_score"] < 0) | (df["away_score"] < 0)).sum()
    very_high_scores = ((df["home_score"] > 20) | (df["away_score"] > 20)).sum()
    if negative_scores:
        raise ValueError(f"Se encontraron {negative_scores} marcadores negativos.")

    df["year"] = df["year"].astype(int)
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    df["is_group_stage"] = df["stage"].str.contains(
        GROUP_STAGE_PATTERN, case=False, na=False, regex=True
    )
    df["host_continent"] = df["year"].map(HOST_CONTINENT_MAP)

    missing_host_continent = df["host_continent"].isna().sum()
    if missing_host_continent:
        raise ValueError(
            f"Hay {missing_host_continent} partidos sin host_continent mapeado."
        )

    # Estas variables son utiles para EDA; no deben usarse como features supervisadas.
    df["goal_difference"] = df["home_score"] - df["away_score"]
    df["total_goals"] = df["home_score"] + df["away_score"]

    print("\nValores de stage:")
    print(df["stage"].value_counts())
    print("\nDistribucion is_group_stage:")
    print(df["is_group_stage"].value_counts())
    print(f"\nMarcadores con goles negativos: {negative_scores}")
    print(f"Marcadores con algun equipo >20 goles: {very_high_scores}")
    print(f"Dataset limpio: {df.shape}")

    if save_figure:
        resolved_figure_path = _resolve_project_path(root, figure_path)
        _plot_goal_distributions(df, resolved_figure_path)
        print(f"Figura guardada: {resolved_figure_path}")

    if save_output:
        resolved_output_path = _resolve_project_path(root, output_path)
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(resolved_output_path, index=False)
        print(f"Dataset limpio guardado: {resolved_output_path}")

    return df
