"""Funciones para construir variables pre-torneo y datasets de modelado."""

from pathlib import Path

import pandas as pd


CONFEDERATION_MAP = {
    # UEFA
    "Austria": "UEFA",
    "Belgium": "UEFA",
    "Bosnia-Herzegovina": "UEFA",
    "Bulgaria": "UEFA",
    "Croatia": "UEFA",
    "Czech Republic": "UEFA",
    "Denmark": "UEFA",
    "England": "UEFA",
    "France": "UEFA",
    "German DR": "UEFA",
    "Germany": "UEFA",
    "Greece": "UEFA",
    "Hungary": "UEFA",
    "Ireland": "UEFA",
    "Israel": "UEFA",
    "Italy": "UEFA",
    "Netherlands": "UEFA",
    "Northern Ireland": "UEFA",
    "Norway": "UEFA",
    "Poland": "UEFA",
    "Portugal": "UEFA",
    "Romania": "UEFA",
    "Russia": "UEFA",
    "Scotland": "UEFA",
    "Serbia": "UEFA",
    "Slovakia": "UEFA",
    "Slovenia": "UEFA",
    "Spain": "UEFA",
    "Sweden": "UEFA",
    "Switzerland": "UEFA",
    "Turkey": "UEFA",
    "Ukraine": "UEFA",
    "Wales": "UEFA",
    # CONMEBOL
    "Argentina": "CONMEBOL",
    "Bolivia": "CONMEBOL",
    "Brazil": "CONMEBOL",
    "Chile": "CONMEBOL",
    "Colombia": "CONMEBOL",
    "Ecuador": "CONMEBOL",
    "Paraguay": "CONMEBOL",
    "Peru": "CONMEBOL",
    "Uruguay": "CONMEBOL",
    # CONCACAF
    "Canada": "CONCACAF",
    "Costa Rica": "CONCACAF",
    "Cuba": "CONCACAF",
    "El Salvador": "CONCACAF",
    "Haiti": "CONCACAF",
    "Honduras": "CONCACAF",
    "Jamaica": "CONCACAF",
    "Mexico": "CONCACAF",
    "Trinidad and Tobago": "CONCACAF",
    "United States": "CONCACAF",
    # CAF
    "Algeria": "CAF",
    "Angola": "CAF",
    "Cameroon": "CAF",
    "DR Congo": "CAF",
    "Egypt": "CAF",
    "Ghana": "CAF",
    "Ivory Coast": "CAF",
    "Morocco": "CAF",
    "Nigeria": "CAF",
    "Senegal": "CAF",
    "South Africa": "CAF",
    "Togo": "CAF",
    "Tunisia": "CAF",
    # AFC
    "Australia": "AFC",
    "China PR": "AFC",
    "Indonesia": "AFC",
    "Iran": "AFC",
    "Iraq": "AFC",
    "Japan": "AFC",
    "Kuwait": "AFC",
    "North Korea": "AFC",
    "Saudi Arabia": "AFC",
    "South Korea": "AFC",
    "United Arab Emirates": "AFC",
    # OFC
    "New Zealand": "OFC",
}

CONFEDERATION_TO_CONTINENT = {
    "UEFA": "Europe",
    "CONMEBOL": "South America",
    "CONCACAF": "North America",
    "CAF": "Africa",
    "AFC": "Asia",
    "OFC": "Oceania",
}

SUPERVISED_COLUMNS = [
    "year",
    "team",
    "host_continent",
    "team_confederation",
    "is_host_region",
    "historical_appearances",
    "historical_win_rate",
    "historical_avg_goals_scored",
    "advanced_group_stage",
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


def load_clean_world_cup_data(
    project_root: str | Path = ".",
    input_path: str | Path = "data/processed/world_cup_matches_clean.csv",
) -> pd.DataFrame:
    """Carga el dataset limpio de partidos mundialistas.

    Args:
        project_root: Ruta relativa a la raiz del proyecto.
        input_path: Ruta relativa del dataset limpio.

    Returns:
        DataFrame con partidos mundialistas limpios.

    Raises:
        FileNotFoundError: Si el archivo limpio no existe.
    """
    resolved_input_path = _resolve_project_path(project_root, input_path)
    if not resolved_input_path.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo requerido: {resolved_input_path}"
        )
    return pd.read_csv(resolved_input_path)


def _validate_clean_data(df: pd.DataFrame) -> None:
    """Valida columnas requeridas para construir el dataset supervisado.

    Args:
        df: DataFrame limpio de partidos mundialistas.

    Returns:
        None.

    Raises:
        ValueError: Si faltan columnas o hay nulos en columnas criticas.
    """
    required_columns = {
        "year",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "is_group_stage",
        "host_continent",
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Faltan columnas para feature engineering: {missing}")

    critical_nulls = df[list(required_columns)].isna().sum()
    if critical_nulls.any():
        raise ValueError(
            "Hay nulos en columnas requeridas:\n"
            f"{critical_nulls[critical_nulls > 0]}"
        )


def build_team_editions(df: pd.DataFrame) -> pd.DataFrame:
    """Construye una fila por seleccion y edicion usando partidos de fase inicial.

    Args:
        df: DataFrame limpio con partidos mundialistas.

    Returns:
        DataFrame con columnas `year`, `team`, `host_continent` y
        `advanced_group_stage`.
    """
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
    return team_editions.sort_values(["year", "team"]).reset_index(drop=True)


def compute_historical_features(
    team_editions_df: pd.DataFrame,
    all_matches_df: pd.DataFrame,
) -> pd.DataFrame:
    """Calcula estadisticas historicas usando solo ediciones anteriores.

    Para cada par `(team, year)`, filtra partidos de fase inicial con
    `match_year < year`. Asi se evita usar resultados del torneo actual como
    features del modelo supervisado.

    Args:
        team_editions_df: DataFrame con una fila por seleccion y edicion.
        all_matches_df: DataFrame limpio con todos los partidos.

    Returns:
        DataFrame con features historicas pre-torneo por `(year, team)`.
    """
    records = []
    editions = sorted(team_editions_df["year"].unique())

    for current_year in editions:
        current_teams = team_editions_df.loc[
            team_editions_df["year"] == current_year, "team"
        ].unique()
        past = all_matches_df[
            (all_matches_df["is_group_stage"])
            & (all_matches_df["year"] < current_year)
        ].copy()

        for team in current_teams:
            team_past = past[
                (past["home_team"] == team) | (past["away_team"] == team)
            ]
            historical_years = sorted(team_past["year"].unique())
            n_appearances = len(historical_years)

            if team_past.empty:
                win_rate = 0.0
                avg_goals_scored = 0.0
            else:
                wins = 0
                goals_scored = 0

                for _, row in team_past.iterrows():
                    if row["home_team"] == team:
                        goals_scored += row["home_score"]
                        wins += int(row["home_score"] > row["away_score"])
                    else:
                        goals_scored += row["away_score"]
                        wins += int(row["away_score"] > row["home_score"])

                win_rate = wins / len(team_past)
                avg_goals_scored = goals_scored / len(team_past)

            records.append(
                {
                    "year": current_year,
                    "team": team,
                    "historical_appearances": n_appearances,
                    "historical_win_rate": round(win_rate, 4),
                    "historical_avg_goals_scored": round(avg_goals_scored, 4),
                }
            )

    return pd.DataFrame(records)


def add_context_features(team_editions_df: pd.DataFrame) -> pd.DataFrame:
    """Agrega confederacion e indicador de region sede.

    Args:
        team_editions_df: DataFrame con una fila por seleccion y edicion.

    Returns:
        DataFrame con variables contextuales pre-torneo.

    Raises:
        ValueError: Si alguna seleccion no tiene confederacion mapeada.
    """
    features = team_editions_df.copy()
    features["team_confederation"] = features["team"].map(CONFEDERATION_MAP)

    missing_teams = sorted(
        features.loc[features["team_confederation"].isna(), "team"].unique()
    )
    if missing_teams:
        raise ValueError(
            "Faltan selecciones en CONFEDERATION_MAP: " + ", ".join(missing_teams)
        )

    team_continent = features["team_confederation"].map(CONFEDERATION_TO_CONTINENT)
    features["is_host_region"] = (
        team_continent == features["host_continent"]
    ).astype(int)
    return features


def build_supervised_dataset(
    data: pd.DataFrame | None = None,
    project_root: str | Path = ".",
    input_path: str | Path = "data/processed/world_cup_matches_clean.csv",
    output_path: str | Path = "data/processed/supervised_dataset.csv",
    save_output: bool = True,
) -> pd.DataFrame:
    """Construye el dataset supervisado con features pre-torneo.

    Args:
        data: DataFrame limpio opcional. Si es `None`, se lee `input_path`.
        project_root: Ruta relativa a la raiz del proyecto.
        input_path: Ruta relativa del dataset limpio de entrada.
        output_path: Ruta relativa donde se guardara el dataset supervisado.
        save_output: Indica si se debe guardar el CSV resultante.

    Returns:
        DataFrame con una fila por `(team, year)` y la variable objetivo
        `advanced_group_stage`.
    """
    root = Path(project_root)
    df = data.copy() if data is not None else load_clean_world_cup_data(root, input_path)
    _validate_clean_data(df)

    df["year"] = pd.to_numeric(df["year"], errors="raise").astype(int)
    df["home_score"] = pd.to_numeric(df["home_score"], errors="raise").astype(int)
    df["away_score"] = pd.to_numeric(df["away_score"], errors="raise").astype(int)

    team_editions = build_team_editions(df)
    historical_features = compute_historical_features(team_editions, df)
    features = team_editions.merge(
        historical_features, on=["year", "team"], how="left"
    )
    features = add_context_features(features)
    features = features[SUPERVISED_COLUMNS].sort_values(
        ["year", "team"]
    ).reset_index(drop=True)

    print(f"Dataset supervisado: {features.shape}")
    print("Balance de advanced_group_stage:")
    print(features["advanced_group_stage"].value_counts().sort_index())
    print("Nulos por columna:")
    print(features.isna().sum())
    print("Confederaciones:")
    print(features["team_confederation"].value_counts())

    if save_output:
        resolved_output_path = _resolve_project_path(root, output_path)
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(resolved_output_path, index=False)
        print(f"Dataset supervisado guardado: {resolved_output_path}")

    return features
