"""Carga y cruce inicial de los datasets historicos de la Copa Mundial."""

from pathlib import Path

import pandas as pd


TEAM_NAME_MAP = {
    "West Germany": "Germany",
    "Germany FR": "Germany",
    "Czechoslovakia": "Czech Republic",
    "Yugoslavia": "Serbia",
    "Soviet Union": "Russia",
    "Zaire": "DR Congo",
    "Dutch East Indies": "Indonesia",
    "Republic of Ireland": "Ireland",
    "FR Yugoslavia": "Serbia",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Serbia and Montenegro": "Serbia",
    "USA": "United States",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "IR Iran": "Iran",
    "Cote d'Ivoire": "Ivory Coast",
    "C\u00f4te d'Ivoire": "Ivory Coast",
}

MAX_WORLD_CUP_YEAR = 2014


def _validate_columns(df: pd.DataFrame, required_columns: set[str], dataset_name: str) -> None:
    """Valida que un DataFrame contenga las columnas requeridas.

    Args:
        df: DataFrame que se quiere validar.
        required_columns: Conjunto de columnas obligatorias.
        dataset_name: Nombre del dataset para mostrar errores claros.

    Raises:
        ValueError: Si falta una o mas columnas obligatorias.
    """
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Faltan columnas en {dataset_name}: {missing}")


def _normalize_team_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres historicos de selecciones en columnas home_team y away_team.

    Args:
        df: DataFrame con columnas `home_team` y `away_team`.

    Returns:
        Copia del DataFrame con nombres de selecciones normalizados.
    """
    normalized = df.copy()
    normalized["home_team"] = normalized["home_team"].astype(str).str.replace(
        'rn">', "", regex=False
    )
    normalized["away_team"] = normalized["away_team"].astype(str).str.replace(
        'rn">', "", regex=False
    )
    normalized["home_team"] = normalized["home_team"].str.replace(
        chr(0xFFFD), "o", regex=False
    )
    normalized["away_team"] = normalized["away_team"].str.replace(
        chr(0xFFFD), "o", regex=False
    )
    normalized["home_team"] = normalized["home_team"].str.strip()
    normalized["away_team"] = normalized["away_team"].str.strip()
    normalized["home_team"] = normalized["home_team"].replace(TEAM_NAME_MAP)
    normalized["away_team"] = normalized["away_team"].replace(TEAM_NAME_MAP)
    return normalized


def load_world_cup_data(
    project_root: str | Path = ".",
    max_year: int = MAX_WORLD_CUP_YEAR,
) -> pd.DataFrame:
    """Carga, filtra y cruza los partidos de Copa Mundial con su etapa del torneo.

    Lee `data/raw/results.csv` y `data/raw/WorldCupMatches.csv`, filtra solo partidos
    del torneo `FIFA World Cup` hasta `max_year`, normaliza nombres historicos de
    selecciones y cruza ambas fuentes con la llave principal
    `(year, home_team, away_team)`.

    Los goles se usan como desempate tecnico porque la fuente complementaria tiene
    partidos repetidos entre las mismas selecciones en una misma edicion y algunos
    registros tienen local/visitante invertido frente a `results.csv`.

    Args:
        project_root: Ruta relativa a la raiz del proyecto. Por defecto usa el
            directorio actual.
        max_year: Ultimo anio mundialista que se conserva. Por defecto es 2014,
            porque `WorldCupMatches.csv` en este proyecto llega hasta esa edicion.

    Returns:
        DataFrame de partidos de Copa Mundial con la columna `stage` incorporada.

    Raises:
        FileNotFoundError: Si no existen los archivos crudos requeridos.
        ValueError: Si faltan columnas obligatorias en alguna fuente.
    """
    root = Path(project_root)
    results_path = root / "data" / "raw" / "results.csv"
    wc_matches_path = root / "data" / "raw" / "WorldCupMatches.csv"

    if not results_path.exists():
        raise FileNotFoundError(f"No se encontro el archivo requerido: {results_path}")
    if not wc_matches_path.exists():
        raise FileNotFoundError(f"No se encontro el archivo requerido: {wc_matches_path}")

    results = pd.read_csv(results_path)
    wc_matches = pd.read_csv(wc_matches_path)

    _validate_columns(
        results,
        {"date", "home_team", "away_team", "home_score", "away_score", "tournament"},
        "results.csv",
    )
    _validate_columns(
        wc_matches,
        {
            "Year",
            "Home Team Name",
            "Away Team Name",
            "Home Team Goals",
            "Away Team Goals",
            "Stage",
        },
        "WorldCupMatches.csv",
    )

    wc_results = results[results["tournament"] == "FIFA World Cup"].copy()
    wc_results["year"] = pd.to_datetime(wc_results["date"], errors="coerce").dt.year
    wc_results = wc_results.dropna(subset=["year", "home_team", "away_team"]).copy()
    wc_results["year"] = wc_results["year"].astype(int)
    wc_results["home_score"] = pd.to_numeric(wc_results["home_score"], errors="coerce")
    wc_results["away_score"] = pd.to_numeric(wc_results["away_score"], errors="coerce")

    wc_matches = wc_matches.dropna(
        subset=[
            "Year",
            "Home Team Name",
            "Away Team Name",
            "Home Team Goals",
            "Away Team Goals",
            "Stage",
        ]
    ).copy()
    wc_matches["year"] = wc_matches["Year"].astype(int)
    wc_matches = wc_matches.rename(
        columns={
            "Home Team Name": "home_team",
            "Away Team Name": "away_team",
            "Home Team Goals": "home_score",
            "Away Team Goals": "away_score",
            "Stage": "stage",
        }
    )
    wc_matches["home_score"] = pd.to_numeric(
        wc_matches["home_score"], errors="coerce"
    )
    wc_matches["away_score"] = pd.to_numeric(
        wc_matches["away_score"], errors="coerce"
    )

    wc_results = _normalize_team_names(wc_results)
    wc_matches = _normalize_team_names(wc_matches)

    available_years = set(wc_matches.loc[wc_matches["year"] <= max_year, "year"])
    original_world_cup_matches = len(wc_results)
    wc_results = wc_results[
        (wc_results["year"] <= max_year) & (wc_results["year"].isin(available_years))
    ].copy()
    wc_matches = wc_matches[wc_matches["year"].isin(available_years)].copy()

    join_columns = ["year", "home_team", "away_team", "home_score", "away_score"]
    direct_lookup = wc_matches[join_columns + ["stage"]].copy()
    reverse_lookup = direct_lookup.rename(
        columns={
            "home_team": "away_team",
            "away_team": "home_team",
            "home_score": "away_score",
            "away_score": "home_score",
        }
    )
    stage_lookup = pd.concat([direct_lookup, reverse_lookup], ignore_index=True)
    stage_lookup = stage_lookup.drop_duplicates(subset=join_columns + ["stage"])

    duplicated_keys = stage_lookup.duplicated(subset=join_columns, keep=False).sum()
    if duplicated_keys:
        print(
            "Advertencia: el lookup de etapas contiene "
            f"{duplicated_keys} filas con llaves de cruce duplicadas."
        )
        stage_lookup = stage_lookup.drop_duplicates(subset=join_columns, keep="first")

    merged = wc_results.merge(stage_lookup, on=join_columns, how="left")

    missing_stage = merged["stage"].isna().sum()
    join_coverage = 1 - (missing_stage / len(merged)) if len(merged) else 0
    excluded_matches = original_world_cup_matches - len(wc_results)
    print(f"Partidos de Copa Mundial hasta {max_year}: {len(wc_results)}")
    print(f"Partidos excluidos por alcance/fuente complementaria: {excluded_matches}")
    print(f"Partidos sin stage asignado: {missing_stage} de {len(merged)}")
    print(f"Cobertura del join: {join_coverage:.2%}")

    return merged
