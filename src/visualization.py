"""Funciones para generar y guardar visualizaciones del analisis mundialista."""

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


TEAM_CONTINENT_MAP = {
    "Algeria": "Africa",
    "Angola": "Africa",
    "Argentina": "South America",
    "Australia": "Asia",
    "Austria": "Europe",
    "Belgium": "Europe",
    "Bolivia": "South America",
    "Bosnia-Herzegovina": "Europe",
    "Brazil": "South America",
    "Bulgaria": "Europe",
    "Cameroon": "Africa",
    "Canada": "North America",
    "Chile": "South America",
    "China PR": "Asia",
    "Colombia": "South America",
    "Costa Rica": "North America",
    "Croatia": "Europe",
    "Cuba": "North America",
    "Czech Republic": "Europe",
    "Denmark": "Europe",
    "DR Congo": "Africa",
    "Ecuador": "South America",
    "Egypt": "Africa",
    "El Salvador": "North America",
    "England": "Europe",
    "France": "Europe",
    "German DR": "Europe",
    "Germany": "Europe",
    "Ghana": "Africa",
    "Greece": "Europe",
    "Haiti": "North America",
    "Honduras": "North America",
    "Hungary": "Europe",
    "Indonesia": "Asia",
    "Iran": "Asia",
    "Iraq": "Asia",
    "Ireland": "Europe",
    "Israel": "Europe",
    "Italy": "Europe",
    "Ivory Coast": "Africa",
    "Jamaica": "North America",
    "Japan": "Asia",
    "Kuwait": "Asia",
    "Mexico": "North America",
    "Morocco": "Africa",
    "Netherlands": "Europe",
    "New Zealand": "Oceania",
    "Nigeria": "Africa",
    "North Korea": "Asia",
    "Northern Ireland": "Europe",
    "Norway": "Europe",
    "Paraguay": "South America",
    "Peru": "South America",
    "Poland": "Europe",
    "Portugal": "Europe",
    "Romania": "Europe",
    "Russia": "Europe",
    "Saudi Arabia": "Asia",
    "Scotland": "Europe",
    "Senegal": "Africa",
    "Serbia": "Europe",
    "Slovakia": "Europe",
    "Slovenia": "Europe",
    "South Africa": "Africa",
    "South Korea": "Asia",
    "Spain": "Europe",
    "Sweden": "Europe",
    "Switzerland": "Europe",
    "Togo": "Africa",
    "Trinidad and Tobago": "North America",
    "Tunisia": "Africa",
    "Turkey": "Europe",
    "Ukraine": "Europe",
    "United Arab Emirates": "Asia",
    "United States": "North America",
    "Uruguay": "South America",
    "Wales": "Europe",
}


def _resolve_project_path(project_root: str | Path, path: str | Path) -> Path:
    """Construye una ruta relativa a la raiz del proyecto.

    Args:
        project_root: Ruta relativa a la raiz del proyecto.
        path: Ruta relativa que se quiere resolver.

    Returns:
        Ruta compuesta a partir de `project_root` y `path`.
    """
    return Path(project_root) / Path(path)


def _save_current_figure(path: str | Path) -> None:
    """Guarda la figura activa y libera memoria.

    Args:
        path: Ruta donde se guardara la figura.

    Returns:
        None.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def load_clean_data(
    project_root: str | Path = ".",
    input_path: str | Path = "data/processed/world_cup_matches_clean.csv",
) -> pd.DataFrame:
    """Carga el dataset limpio de partidos mundialistas.

    Args:
        project_root: Ruta relativa a la raiz del proyecto.
        input_path: Ruta relativa del dataset limpio.

    Returns:
        DataFrame con el dataset limpio.

    Raises:
        FileNotFoundError: Si el archivo limpio no existe.
    """
    resolved_input_path = _resolve_project_path(project_root, input_path)
    if not resolved_input_path.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo requerido: {resolved_input_path}"
        )
    return pd.read_csv(resolved_input_path)


def build_team_edition_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Construye una tabla temporal de avance por seleccion y edicion.

    Args:
        df: DataFrame limpio de partidos con `stage` e `is_group_stage`.

    Returns:
        DataFrame con una fila por `(year, team)` y la variable temporal
        `advanced_group_stage`, usada solo para EDA en esta fase.

    Raises:
        ValueError: Si faltan columnas requeridas o equipos sin continente mapeado.
    """
    required_columns = {
        "year",
        "home_team",
        "away_team",
        "is_group_stage",
        "host_continent",
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Faltan columnas para EDA: {missing}")

    groups = df[df["is_group_stage"]].copy()
    home = groups[["year", "home_team"]].rename(columns={"home_team": "team"})
    away = groups[["year", "away_team"]].rename(columns={"away_team": "team"})
    team_editions = pd.concat([home, away], ignore_index=True).drop_duplicates()

    knockout = df[~df["is_group_stage"]].copy()
    advanced_home = set(zip(knockout["year"], knockout["home_team"]))
    advanced_away = set(zip(knockout["year"], knockout["away_team"]))
    advanced = advanced_home.union(advanced_away)

    team_editions["advanced_group_stage"] = team_editions.apply(
        lambda row: int((row["year"], row["team"]) in advanced),
        axis=1,
    )

    host_context = df[["year", "host_continent"]].drop_duplicates()
    team_editions = team_editions.merge(host_context, on="year", how="left")
    team_editions["team_continent"] = team_editions["team"].map(TEAM_CONTINENT_MAP)

    missing_teams = sorted(
        team_editions.loc[team_editions["team_continent"].isna(), "team"].unique()
    )
    if missing_teams:
        raise ValueError(
            "Faltan equipos en TEAM_CONTINENT_MAP: " + ", ".join(missing_teams)
        )

    team_editions["is_host_region"] = (
        team_editions["team_continent"] == team_editions["host_continent"]
    ).astype(int)
    return team_editions


def plot_goal_distribution(
    df: pd.DataFrame,
    output_path: str | Path = "outputs/figures/distribucion_goles_raw.png",
) -> None:
    """Grafica la distribucion de goles de local y visitante.

    Args:
        df: DataFrame limpio de partidos mundialistas.
        output_path: Ruta donde se guardara la figura.

    Returns:
        None.
    """
    max_goals = int(max(df["home_score"].max(), df["away_score"].max()))
    bins = range(0, max_goals + 2)

    plt.figure(figsize=(12, 4))
    axes = plt.subplot(1, 2, 1), plt.subplot(1, 2, 2)
    sns.histplot(df["home_score"], bins=bins, ax=axes[0], color="#2f6f9f")
    axes[0].set_title("Distribucion goles local")
    axes[0].set_xlabel("Goles")
    axes[0].set_ylabel("Partidos")

    sns.histplot(df["away_score"], bins=bins, ax=axes[1], color="#b05d3b")
    axes[1].set_title("Distribucion goles visitante")
    axes[1].set_xlabel("Goles")
    axes[1].set_ylabel("Partidos")

    _save_current_figure(output_path)


def plot_matches_by_edition(
    df: pd.DataFrame,
    output_path: str | Path = "outputs/figures/partidos_por_edicion.png",
) -> pd.DataFrame:
    """Grafica la cantidad de partidos por edicion del Mundial.

    Args:
        df: DataFrame limpio de partidos mundialistas.
        output_path: Ruta donde se guardara la figura.

    Returns:
        DataFrame con columnas `year` y `matches`.
    """
    matches_by_year = df.groupby("year").size().reset_index(name="matches")

    plt.figure(figsize=(11, 5))
    ax = sns.barplot(data=matches_by_year, x="year", y="matches", color="#4c7f73")
    ax.set_title("Partidos por edicion del Mundial")
    ax.set_xlabel("Anio")
    ax.set_ylabel("Partidos")
    ax.tick_params(axis="x", rotation=45)
    _save_current_figure(output_path)

    return matches_by_year


def plot_average_goals_by_edition(
    df: pd.DataFrame,
    output_path: str | Path = "outputs/figures/goles_promedio_por_edicion.png",
) -> pd.DataFrame:
    """Grafica el promedio de goles por partido por edicion.

    Args:
        df: DataFrame limpio de partidos mundialistas.
        output_path: Ruta donde se guardara la figura.

    Returns:
        DataFrame con promedio de goles por edicion.
    """
    goals_by_year = (
        df.groupby("year")["total_goals"].mean().reset_index(name="avg_total_goals")
    )

    plt.figure(figsize=(10, 5))
    ax = sns.lineplot(
        data=goals_by_year,
        x="year",
        y="avg_total_goals",
        marker="o",
        color="#315f8c",
    )
    ax.set_title("Promedio de goles por partido por edicion")
    ax.set_xlabel("Anio")
    ax.set_ylabel("Goles promedio")
    ax.set_xticks(goals_by_year["year"])
    ax.tick_params(axis="x", rotation=45)
    _save_current_figure(output_path)

    return goals_by_year


def plot_host_region_performance(
    team_editions: pd.DataFrame,
    output_path: str | Path = "outputs/figures/rendimiento_por_sede.png",
) -> pd.DataFrame:
    """Compara la tasa de avance de equipos de la region sede contra el resto.

    Args:
        team_editions: DataFrame generado por `build_team_edition_summary`.
        output_path: Ruta donde se guardara la figura.

    Returns:
        DataFrame con tasas de avance por indicador `is_host_region`.
    """
    performance = (
        team_editions.groupby("is_host_region")
        .agg(
            advance_rate=("advanced_group_stage", "mean"),
            teams=("advanced_group_stage", "size"),
        )
        .reset_index()
    )
    performance["region_type"] = performance["is_host_region"].map(
        {0: "Otra region", 1: "Region sede"}
    )

    plt.figure(figsize=(7, 5))
    ax = sns.barplot(
        data=performance,
        x="region_type",
        y="advance_rate",
        hue="region_type",
        palette=["#8b8b8b", "#3d7a57"],
        legend=False,
    )
    ax.set_title("Tasa de avance por relacion con continente sede")
    ax.set_xlabel("")
    ax.set_ylabel("Tasa de avance")
    ax.set_ylim(0, 1)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f")
    _save_current_figure(output_path)

    return performance


def plot_top_teams_advance(
    team_editions: pd.DataFrame,
    output_path: str | Path = "outputs/figures/top_selecciones_avance.png",
    top_n: int = 10,
) -> pd.DataFrame:
    """Grafica las selecciones con mas avances desde fase de grupos.

    Args:
        team_editions: DataFrame generado por `build_team_edition_summary`.
        output_path: Ruta donde se guardara la figura.
        top_n: Numero de selecciones a mostrar.

    Returns:
        DataFrame con ranking de selecciones por avances.
    """
    ranking = (
        team_editions.groupby("team")
        .agg(
            advances=("advanced_group_stage", "sum"),
            appearances=("advanced_group_stage", "size"),
        )
        .reset_index()
    )
    ranking["advance_rate"] = ranking["advances"] / ranking["appearances"]
    ranking = ranking.sort_values(
        ["advances", "advance_rate", "appearances"], ascending=False
    ).head(top_n)

    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=ranking, y="team", x="advances", color="#7b4f9c")
    ax.set_title(f"Top {top_n} selecciones por avances de fase")
    ax.set_xlabel("Avances")
    ax.set_ylabel("")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f")
    _save_current_figure(output_path)

    return ranking


def plot_correlation_matrix(
    df: pd.DataFrame,
    output_path: str | Path = "outputs/figures/matriz_correlacion.png",
) -> pd.DataFrame:
    """Grafica una matriz de correlacion para variables numericas de partidos.

    Args:
        df: DataFrame limpio de partidos mundialistas.
        output_path: Ruta donde se guardara la figura.

    Returns:
        DataFrame de correlaciones.
    """
    numeric_cols = [
        "home_score",
        "away_score",
        "total_goals",
        "goal_difference",
        "year",
    ]
    corr = df[numeric_cols].corr()

    plt.figure(figsize=(7, 5))
    ax = sns.heatmap(corr, annot=True, cmap="vlag", center=0, fmt=".2f")
    ax.set_title("Matriz de correlacion")
    _save_current_figure(output_path)

    return corr


def plot_class_balance(
    team_editions: pd.DataFrame,
    output_path: str | Path = "outputs/figures/balance_clases.png",
) -> pd.DataFrame:
    """Grafica el balance temporal de la variable objetivo de avance.

    Args:
        team_editions: DataFrame generado por `build_team_edition_summary`.
        output_path: Ruta donde se guardara la figura.

    Returns:
        DataFrame con conteo de clases.
    """
    class_balance = (
        team_editions["advanced_group_stage"]
        .value_counts()
        .rename_axis("advanced_group_stage")
        .reset_index(name="count")
        .sort_values("advanced_group_stage")
    )
    class_balance["label"] = class_balance["advanced_group_stage"].map(
        {0: "No avanzo", 1: "Avanzo"}
    )

    plt.figure(figsize=(7, 5))
    ax = sns.barplot(
        data=class_balance,
        x="label",
        y="count",
        hue="label",
        palette=["#9f5a4f", "#3d7a57"],
        legend=False,
    )
    ax.set_title("Balance de clases - avance de fase")
    ax.set_xlabel("")
    ax.set_ylabel("Selecciones-edicion")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f")
    _save_current_figure(output_path)

    return class_balance


def generate_all_figures(
    project_root: str | Path = ".",
    data: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Genera todas las visualizaciones de EDA de la Fase 3.

    Args:
        project_root: Ruta relativa a la raiz del proyecto.
        data: DataFrame limpio opcional. Si es `None`, se lee desde disco.

    Returns:
        Diccionario con tablas resumen usadas para generar las figuras.
    """
    root = Path(project_root)
    df = data.copy() if data is not None else load_clean_data(root)
    team_editions = build_team_edition_summary(df)

    output_dir = root / "outputs" / "figures"
    summaries = {
        "matches_by_year": plot_matches_by_edition(
            df, output_dir / "partidos_por_edicion.png"
        ),
        "goals_by_year": plot_average_goals_by_edition(
            df, output_dir / "goles_promedio_por_edicion.png"
        ),
        "host_region_performance": plot_host_region_performance(
            team_editions, output_dir / "rendimiento_por_sede.png"
        ),
        "top_teams_advance": plot_top_teams_advance(
            team_editions, output_dir / "top_selecciones_avance.png"
        ),
        "correlation_matrix": plot_correlation_matrix(
            df, output_dir / "matriz_correlacion.png"
        ),
        "class_balance": plot_class_balance(
            team_editions, output_dir / "balance_clases.png"
        ),
    }
    plot_goal_distribution(df, output_dir / "distribucion_goles_raw.png")

    print("Figuras de EDA generadas en outputs/figures/")
    print(f"Dataset de partidos: {df.shape}")
    print(f"Dataset temporal seleccion-edicion: {team_editions.shape}")
    print("Balance de clases:")
    print(summaries["class_balance"].to_string(index=False))
    print("Rendimiento region sede:")
    print(summaries["host_region_performance"].to_string(index=False))

    return summaries
