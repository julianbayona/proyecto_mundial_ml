"""Pipeline principal del proyecto mundialista de Machine Learning."""

from pathlib import Path

from src.data_loader import load_world_cup_data
from src import clustering_model
from src import evaluation
from src import feature_engineering
from src import preprocessing
from src import supervised_model
from src import visualization


def main() -> None:
    """Ejecuta el pipeline completo del proyecto de extremo a extremo."""
    print("1. Cargando datos...")
    raw_data = load_world_cup_data()
    raw_output_path = Path("data") / "processed" / "world_cup_matches_raw.csv"
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_data.to_csv(raw_output_path, index=False)
    print(f"Dataset crudo cruzado guardado: {raw_output_path}")

    print("2. Limpiando datos...")
    clean_data = preprocessing.clean_world_cup_data(data=raw_data)

    print("3. Construyendo features...")
    supervised_df = feature_engineering.build_supervised_dataset(data=clean_data)

    print("4. Entrenando modelo supervisado...")
    supervised_model.train_supervised_model(data=supervised_df)

    print("5. Entrenando modelos de clustering...")
    clustering_model.train_clustering_models(
        matches_data=clean_data,
        supervised_data=supervised_df,
    )

    print("6. Generando visualizaciones de EDA...")
    visualization.generate_all_figures(data=clean_data)

    print("7. Consolidando metricas...")
    evaluation.generate_all_metrics()

    print("Pipeline completo. Resultados en outputs/")


if __name__ == "__main__":
    main()
